"""CPU smoke test of the GPU code paths, using a tiny randomly initialised Phi-3.

Runs the real src/ functions (tokenisation + loss masking, LoRA on Phi-3's fused
modules, Trainer with per-epoch eval, loss table/plot, adapter save + merge,
batched generation with confidence scores, metrics, bootstrap CI, RAG KB) so
API/shape bugs surface locally before spending Colab GPU time. Quantisation
(bitsandbytes) and fp16 need CUDA, so they are the only parts not exercised.

Usage:  python smoke_test_cpu.py

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'CPU smoke test for the QLoRA
# pipeline with a tiny random Phi-3 model', Date: 2026-10-07
"""
import torch  # noqa: F401  (import before sklearn on Windows)
import shutil
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import config
from src import evaluate as E
from src import train as T
from src.format_split import load_split
from src.rag_fallback import ComplianceKB, answer_with_fallback, calibrate_threshold


def tiny_phi3(tok):
    from transformers import Phi3Config, Phi3ForCausalLM
    cfg = Phi3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                     max_position_embeddings=1024, pad_token_id=tok.pad_token_id,
                     eos_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id)
    torch.manual_seed(0)
    return Phi3ForCausalLM(cfg)


def main():
    from peft import PeftModel, get_peft_model
    tmp = Path(tempfile.mkdtemp())
    tok = T.load_tokenizer()
    model = tiny_phi3(tok)
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    assert set(config.LORA_TARGET_MODULES) <= names, names
    model = get_peft_model(model, T.lora_config())
    model.print_trainable_parameters()

    # Small subset so the CPU run takes about a minute.
    from datasets import Dataset
    sub = {s: load_split(s)[:8] for s in ("train", "val")}
    ds = {s: Dataset.from_list(r).map(lambda ex: T.tokenize_example(ex, tok),
                                      remove_columns=["id", "clause_type", "messages"])
          for s, r in sub.items()}
    trainer, cb, epoch0, _ = T.train(
        model, tok, ds, output_dir=str(tmp / "ckpt"),
        num_train_epochs=2, per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=1, optim="adamw_torch", fp16=False,
        gradient_checkpointing=False, learning_rate=1e-3)
    table = T.loss_table(trainer, cb, epoch0)
    print(table)
    assert list(table["epoch"]) == [0, 1, 2] and table["train_loss"].notna().sum() == 2
    T.plot_losses(trainer, table, tmp / "loss.png")

    # Adapter save -> reload onto a base saved to disk -> merge_and_unload
    trainer.model.save_pretrained(tmp / "adapter")
    base_dir = tmp / "base"
    tiny_phi3(tok).save_pretrained(base_dir)
    from transformers import AutoModelForCausalLM
    merged = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_dir),
                                       tmp / "adapter").merge_and_unload()
    assert not any("lora" in n for n, _ in merged.named_parameters())
    T.write_model_card(str(tmp), table)

    # Generation with confidence, scoring, bootstrap
    test = load_split("test")[:3]
    preds = E.predict_rows(merged, tok, test, batch_size=3)
    for p in preds:
        assert {"raw", "mean_logprob", "n_tokens"} <= set(p) and p["mean_logprob"] <= 0
    scores = E.score_predictions(test, preds, with_bertscore=False)
    print("aggregate:", {k: round(v, 3) for k, v in scores["aggregate"].items()})
    print("bootstrap:", E.bootstrap_diff_ci([0.1, 0.2, 0.3], [0.3, 0.4, 0.5]))

    # RAG: KB over train, threshold, fallback path
    kb = ComplianceKB(load_split("train")[:20])
    hit = kb.retrieve(test[0]["messages"][1]["content"])
    assert len(hit["hits"]) == config.RAG_TOP_K and "Label definitions" in hit["context"]
    thr = calibrate_threshold([p["mean_logprob"] for p in preds])
    res = answer_with_fallback(merged, tok, test[0]["messages"][1]["content"], kb, thr, first=preds[0])
    print("RAG used:", res["used_rag"], "| retrieved:", res["retrieved"])
    shutil.rmtree(tmp, ignore_errors=True)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
