"""QLoRA fine-tuning of Phi-3-mini on the compliance-extraction set (Task 2B).

Design notes (each is defended in 02_finetune_qlora.ipynb):
* 4-bit NF4 + double quantisation via bitsandbytes; LoRA adapters via PEFT.
* Plain ``transformers.Trainer`` with our own tokenisation instead of TRL's
  SFTTrainer: TRL's argument names have changed across releases (e.g.
  ``max_seq_length`` -> ``max_length``), and doing it ourselves makes the loss
  mask explicit - **loss is computed on the assistant JSON only**; system and
  user tokens are masked to -100 so the model is not trained to reproduce the
  prompt.
* T4 GPUs (compute capability 7.5) have no native bfloat16, so compute dtype
  and mixed precision are fp16 (the blueprint's bf16 would fail or silently
  emulate on Colab free tier).
* Per-epoch train loss is aggregated by a callback (mean of the step losses in
  the epoch) and val loss is computed at each epoch end.
* Merging: ``merge_and_unload`` on a 4-bit model would bake quantisation error
  into the weights, so the adapter is merged into a freshly loaded **fp16** base.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'QLoRA training module for
# Phi-3-mini on T4: NF4 config, LoRA on fused Phi-3 projections, assistant-only
# loss masking, per-epoch loss callback, fp16 merge and push', Date: 2026-10-07
# SOURCE: QLoRA hyperparameter conventions from Dettmers et al., "QLoRA:
# Efficient Finetuning of Quantized LLMs" (2023), https://arxiv.org/abs/2305.14314
"""
from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import torch

import config

log = logging.getLogger(__name__)
IGNORE_INDEX = -100


# --------------------------------------------------------------------------- #
# Model / tokenizer
# --------------------------------------------------------------------------- #
def bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_tokenizer(name: str = config.BASE_MODEL):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None or tok.pad_token_id == tok.eos_token_id:
        # Phi-3 ships <|endoftext|> as both; keep it but make padding explicit.
        tok.pad_token = tok.unk_token or tok.eos_token
    tok.padding_side = "right"            # training; generation switches to left
    return tok


def load_4bit_model(name: str = config.BASE_MODEL):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        name, quantization_config=bnb_config(), device_map={"": 0},
        torch_dtype=torch.float16, attn_implementation="sdpa")


def lora_config():
    from peft import LoraConfig
    return LoraConfig(r=config.LORA_R, lora_alpha=config.LORA_ALPHA,
                      lora_dropout=config.LORA_DROPOUT, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=config.LORA_TARGET_MODULES)


def prepare_peft_model(model):
    from peft import get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, lora_config())
    return model


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _chat_ids(tok, msgs, add_generation_prompt: bool) -> list[int]:
    ids = tok.apply_chat_template(msgs, add_generation_prompt=add_generation_prompt,
                                  tokenize=True, return_dict=False)
    return list(ids)


def tokenize_example(example: dict, tok) -> dict:
    """Full chat -> input_ids; labels = input_ids with the prompt masked.

    The prompt length is the longest common prefix of the prompt-only and the
    full tokenisation, which is robust to tokenizers that merge the boundary
    between "<|assistant|>
" and the first answer token.
    """
    msgs = example["messages"]
    prompt_ids = _chat_ids(tok, msgs[:-1], add_generation_prompt=True)
    full_ids = _chat_ids(tok, msgs, add_generation_prompt=False)
    n_prompt = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        n_prompt += 1
    if tok.eos_token_id is not None and full_ids[-1] != tok.eos_token_id:
        full_ids = full_ids + [tok.eos_token_id]
    full_ids = full_ids[:config.MAX_SEQ_LENGTH]
    labels = ([IGNORE_INDEX] * n_prompt + full_ids[n_prompt:])[:len(full_ids)]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": labels}


def build_datasets(tok):
    from datasets import Dataset
    from src.format_split import load_split
    out = {}
    for split in ("train", "val"):
        rows = load_split(split)
        out[split] = Dataset.from_list(rows).map(
            lambda ex: tokenize_example(ex, tok),
            remove_columns=["id", "clause_type", "messages"])
    return out


def token_length_stats(tok) -> dict:
    from src.format_split import load_split
    lens = [len(tokenize_example(r, tok)["input_ids"])
            for s in ("train", "val", "test") for r in load_split(s)]
    return {"max": max(lens), "mean": sum(lens) / len(lens), "n": len(lens),
            "truncated": sum(l >= config.MAX_SEQ_LENGTH for l in lens)}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def make_epoch_loss_callback():
    from transformers import TrainerCallback

    class EpochLoss(TrainerCallback):
        """Mean training loss per epoch from the per-step logs."""
        def __init__(self):
            self.step_losses, self.per_epoch = [], []

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                self.step_losses.append(logs["loss"])

        def on_epoch_end(self, args, state, control, **kw):
            if self.step_losses:
                self.per_epoch.append({"epoch": round(state.epoch),
                                       "train_loss": sum(self.step_losses) / len(self.step_losses)})
                self.step_losses = []

    return EpochLoss()


def training_arguments(output_dir: str, report_to: str = "none", **overrides):
    """All arguments explicit; ``overrides`` exists only for the CPU smoke test."""
    from transformers import TrainingArguments
    kw = dict(
        output_dir=output_dir,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_BATCH,
        per_device_eval_batch_size=config.PER_DEVICE_BATCH,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        optim=config.OPTIMIZER,
        fp16=True, bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=True,
        seed=config.TRAIN_SEED,
        data_seed=config.TRAIN_SEED,
        report_to=report_to,
        run_name="phi3-mini-compliance-qlora",
        remove_unused_columns=False,
    )
    kw.update(overrides)
    return TrainingArguments(**kw)


def train(model, tok, datasets, output_dir: str = "checkpoints", report_to: str = "none",
          **arg_overrides):
    from transformers import DataCollatorForSeq2Seq, Trainer
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=IGNORE_INDEX,
                                      pad_to_multiple_of=8)
    cb = make_epoch_loss_callback()
    trainer = Trainer(model=model, args=training_arguments(output_dir, report_to, **arg_overrides),
                      train_dataset=datasets["train"], eval_dataset=datasets["val"],
                      data_collator=collator, callbacks=[cb])
    # Val loss of the untrained adapter = epoch-0 baseline for the curve.
    epoch0 = trainer.evaluate()
    result = trainer.train()
    return trainer, cb, epoch0, result


def loss_table(trainer, cb, epoch0: dict):
    import pandas as pd
    evals = [{"epoch": round(h.get("epoch") or 0), "val_loss": h["eval_loss"]}
             for h in trainer.state.log_history if "eval_loss" in h]
    # trainer.evaluate() before training logs epoch 0; keep the last per epoch
    df_eval = pd.DataFrame(evals).groupby("epoch", as_index=False).last()
    df_train = pd.DataFrame(cb.per_epoch)
    df = df_eval.merge(df_train, on="epoch", how="left")
    if 0 not in df["epoch"].values:
        df = pd.concat([pd.DataFrame([{"epoch": 0, "val_loss": epoch0["eval_loss"]}]), df])
    return df.sort_values("epoch").reset_index(drop=True)[["epoch", "train_loss", "val_loss"]]


def plot_losses(trainer, table, path: Path):
    import matplotlib.pyplot as plt
    steps = [(h["epoch"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
    fig, ax = plt.subplots(figsize=(8, 4))
    if steps:
        ax.plot(*zip(*steps), color="#2a78d6", alpha=.35, lw=1, label="train (per step)")
    t = table.dropna(subset=["train_loss"])
    ax.plot(t["epoch"], t["train_loss"], "o-", color="#2a78d6", lw=2, label="train (epoch mean)")
    ax.plot(table["epoch"], table["val_loss"], "s-", color="#eb6834", lw=2, label="validation")
    ax.set(xlabel="epoch", ylabel="loss (assistant tokens only)",
           title="Phi-3-mini QLoRA: train vs validation loss")
    ax.set_xticks(range(0, config.NUM_EPOCHS + 1))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


# --------------------------------------------------------------------------- #
# Merge + publish
# --------------------------------------------------------------------------- #
def free_gpu(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def merge_adapter(adapter_dir: str, merged_dir: str):
    """Merge LoRA into an fp16 (not 4-bit) base and save safetensors shards."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, torch_dtype=torch.float16,
                                                device_map={"": 0})
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="2GB")
    tok = load_tokenizer()
    tok.padding_side = "left"
    tok.save_pretrained(merged_dir)
    return merged


def write_model_card(merged_dir: str, table, metrics: dict | None = None):
    rows = "\n".join(f"| {int(r.epoch)} | {'' if r.train_loss != r.train_loss else f'{r.train_loss:.4f}'} | {r.val_loss:.4f} |"
                     for r in table.itertuples())
    card = f"""---
base_model: {config.BASE_MODEL}
library_name: transformers
tags: [qlora, peft, phi-3, finance, compliance, information-extraction]
license: mit
---
# Phi-3-mini compliance clause extractor (QLoRA, merged)

Fine-tuned from `{config.BASE_MODEL}` with QLoRA (4-bit NF4, r={config.LORA_R},
alpha={config.LORA_ALPHA}) to turn a financial contract/policy clause into JSON:
`{{clause_type, obligation, party_responsible, trigger_condition, risk_flag}}`.

Training data: synthetic clauses generated by `{config.TEACHER_MODEL}` (teacher != student).
Built for the CDAZZDEV Senior MLE assessment (Task 2). Not for production legal use.

| epoch | train loss | val loss |
|---|---|---|
{rows}

Use the system prompt in `prompts/student_system_prompt.txt` of the source repository.
"""
    Path(merged_dir, "README.md").write_text(card, encoding="utf-8")


def push(merged_dir: str, repo_id: str = config.HF_REPO_ID, token: str | None = None):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, private=False)
    api.upload_folder(folder_path=merged_dir, repo_id=repo_id,
                      commit_message="Upload merged QLoRA model")
    return f"https://huggingface.co/{repo_id}"
