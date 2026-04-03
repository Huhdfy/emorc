#!/usr/bin/env python
"""
Compare three models on the same dataset:
1. Baseline label-only model
2. CoT model with explanation before emotion
3. CoT model with explanation after emotion
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.data.data_processor import DataProcessor
from src.data.emotion_taxonomy import TAXONOMY
from src.models.prompt_template import EmotionPromptBuilder, parse_model_output


def sanitize_lm_head(model):
    target_model = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        target_model = model.base_model.model
    elif hasattr(model, "model"):
        target_model = model.model

    if not hasattr(target_model, "lm_head"):
        return

    with torch.no_grad():
        lm_weight = target_model.lm_head.weight.data
        bad_mask = torch.isinf(lm_weight) | torch.isnan(lm_weight)
        if bad_mask.any():
            bad_rows = torch.where(bad_mask.any(dim=1))[0]
            print(f"Zeroing {len(bad_rows)} corrupted LM head rows (vocab: {bad_rows.tolist()})")
            lm_weight[bad_rows] = 0.0


def load_model(model_path: str, lora_path: str | None = None):
    print(f"Loading base model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    sanitize_lm_head(model)

    if lora_path:
        print(f"Loading LoRA weights from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        sanitize_lm_head(model)

    model.eval()
    return model, tokenizer


def build_label_only_prompt(sample) -> str:
    emotion_list = ", ".join(TAXONOMY.emotions)
    history = sample.dialogue_history[-5:] if sample.dialogue_history else []
    history_text = "\n".join(f"Turn {i+1}: {utt}" for i, utt in enumerate(history)) or "[No previous context]"

    return "\n".join(
        [
            "<|im_start|>system",
            "You are an expert emotion recognition assistant. Identify the emotion of the target utterance from the standard taxonomy.",
            f"Standard emotions: {emotion_list}",
            "<|im_end|>",
            "<|im_start|>user",
            "Now analyze this dialogue:",
            "",
            "Dialogue History:",
            history_text,
            "",
            f"Target Utterance: {sample.target_utterance}",
            "",
            "Return only the final emotion label.",
            "<|im_end|>",
            "<|im_start|>assistant",
            "",
        ]
    )


def parse_label_only_output(output: str) -> str:
    cleaned = output.strip().replace("<|im_end|>", "").replace("<|endoftext|>", "").strip().lower()
    if not cleaned:
        return "neutral"

    candidates = [cleaned.splitlines()[0].strip(" -:\t"), cleaned]
    for candidate in candidates:
        for emotion in TAXONOMY.emotions:
            if candidate == emotion:
                return emotion

    for emotion in TAXONOMY.emotions:
        if re.search(rf"\b{re.escape(emotion)}\b", cleaned):
            return emotion

    return "neutral"


def map_prediction(pred_emotion: str) -> str:
    pred_emotion = pred_emotion.lower().strip()
    if pred_emotion in TAXONOMY.emotions:
        return pred_emotion
    for emotion in TAXONOMY.emotions:
        if emotion in pred_emotion or pred_emotion in emotion:
            return emotion
    return "neutral"


def generate_text(model, tokenizer, prompt: str, max_length: int, max_new_tokens: int):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_baseline(model, tokenizer, samples, max_length: int):
    predictions = []
    details = []
    for sample in tqdm(samples, desc="Evaluating baseline"):
        decoded = generate_text(model, tokenizer, build_label_only_prompt(sample), max_length=max_length, max_new_tokens=32)
        pred = parse_label_only_output(decoded)
        gold = sample.emotion.lower()
        predictions.append(pred)
        details.append(
            {
                "sample_id": sample.sample_id,
                "target_utterance": sample.target_utterance,
                "dialogue_history": sample.dialogue_history,
                "gold_emotion": gold,
                "pred_emotion": pred,
                "raw_output": decoded,
                "correct": pred == gold,
            }
        )
    return predictions, details


def evaluate_cot(model, tokenizer, samples, max_length: int, explanation_position: str, label: str):
    builder = EmotionPromptBuilder(use_retrieval=False, explanation_position=explanation_position)
    predictions = []
    details = []
    for sample in tqdm(samples, desc=f"Evaluating {label}"):
        prompt = builder.build_inference_prompt(
            dialogue_history=sample.dialogue_history,
            target_utterance=sample.target_utterance,
        )
        decoded = generate_text(model, tokenizer, prompt, max_length=max_length, max_new_tokens=256)
        parsed = parse_model_output(decoded)
        pred = map_prediction(parsed["emotion"])
        gold = sample.emotion.lower()
        predictions.append(pred)
        details.append(
            {
                "sample_id": sample.sample_id,
                "target_utterance": sample.target_utterance,
                "dialogue_history": sample.dialogue_history,
                "gold_emotion": gold,
                "pred_emotion": pred,
                "pred_explanation": parsed.get("explanation"),
                "raw_output": decoded,
                "correct": pred == gold,
            }
        )
    return predictions, details


def compute_metrics(predictions, ground_truths):
    labels = sorted(set(predictions + ground_truths))
    label2idx = {label: i for i, label in enumerate(labels)}
    pred_indices = [label2idx[p] for p in predictions]
    gold_indices = [label2idx[g] for g in ground_truths]
    return {
        "weighted_f1": f1_score(gold_indices, pred_indices, average="weighted"),
        "macro_f1": f1_score(gold_indices, pred_indices, average="macro"),
        "accuracy": accuracy_score(gold_indices, pred_indices),
    }


def build_confusion_data(predictions, ground_truths, labels):
    return {
        "labels": labels,
        "matrix": confusion_matrix(ground_truths, predictions, labels=labels).tolist(),
    }


def top_confusions(confusion_data, top_k=10):
    labels = confusion_data["labels"]
    matrix = confusion_data["matrix"]
    pairs = []
    for i, gold in enumerate(labels):
        for j, pred in enumerate(labels):
            if i != j and matrix[i][j] > 0:
                pairs.append((matrix[i][j], gold, pred))
    pairs.sort(reverse=True, key=lambda x: x[0])
    return [{"count": c, "gold": g, "pred": p} for c, g, p in pairs[:top_k]]


def print_top_confusions(name: str, items):
    print(f"Top {name} confusions (gold -> pred):")
    for item in items[:5]:
        print(f"  {item['gold']} -> {item['pred']}: {item['count']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_base_model", type=str, required=True)
    parser.add_argument("--baseline_lora_path", type=str, required=True)
    parser.add_argument("--cot_before_base_model", type=str, required=True)
    parser.add_argument("--cot_before_lora_path", type=str, required=True)
    parser.add_argument("--cot_after_base_model", type=str, required=True)
    parser.add_argument("--cot_after_lora_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, default="ERC1/data/json/val_samples_debiased_clean.json")
    parser.add_argument("--output_dir", type=str, default="./outputs/compare_three")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    processor = DataProcessor()
    samples = processor.load_samples(args.test_data)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"Loaded {len(samples)} evaluation samples from {args.test_data}")

    ground_truths = [sample.emotion.lower() for sample in samples]
    matrix_labels = list(TAXONOMY.emotions)

    baseline_model, baseline_tokenizer = load_model(args.baseline_base_model, args.baseline_lora_path)
    baseline_predictions, baseline_details = evaluate_baseline(baseline_model, baseline_tokenizer, samples, args.max_length)
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    before_model, before_tokenizer = load_model(args.cot_before_base_model, args.cot_before_lora_path)
    before_predictions, before_details = evaluate_cot(before_model, before_tokenizer, samples, args.max_length, "before", "cot_before")
    del before_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    after_model, after_tokenizer = load_model(args.cot_after_base_model, args.cot_after_lora_path)
    after_predictions, after_details = evaluate_cot(after_model, after_tokenizer, samples, args.max_length, "after", "cot_after")
    del after_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {
        "baseline": {
            "metrics": compute_metrics(baseline_predictions, ground_truths),
            "confusion": build_confusion_data(baseline_predictions, ground_truths, matrix_labels),
            "details": baseline_details,
        },
        "cot_before": {
            "metrics": compute_metrics(before_predictions, ground_truths),
            "confusion": build_confusion_data(before_predictions, ground_truths, matrix_labels),
            "details": before_details,
        },
        "cot_after": {
            "metrics": compute_metrics(after_predictions, ground_truths),
            "confusion": build_confusion_data(after_predictions, ground_truths, matrix_labels),
            "details": after_details,
        },
    }

    summary = {
        "num_samples": len(samples),
        "dataset": args.test_data,
        "models": {
            "baseline_lora_path": args.baseline_lora_path,
            "cot_before_lora_path": args.cot_before_lora_path,
            "cot_after_lora_path": args.cot_after_lora_path,
        },
        "baseline_metrics": results["baseline"]["metrics"],
        "cot_before_metrics": results["cot_before"]["metrics"],
        "cot_after_metrics": results["cot_after"]["metrics"],
        "baseline_top_confusions": top_confusions(results["baseline"]["confusion"]),
        "cot_before_top_confusions": top_confusions(results["cot_before"]["confusion"]),
        "cot_after_top_confusions": top_confusions(results["cot_after"]["confusion"]),
    }

    print("\n" + "=" * 60)
    print("THREE-WAY COMPARISON RESULTS")
    print("=" * 60)
    print(f"Baseline weighted_f1 : {summary['baseline_metrics']['weighted_f1']:.4f}")
    print(f"Baseline macro_f1    : {summary['baseline_metrics']['macro_f1']:.4f}")
    print(f"Baseline accuracy    : {summary['baseline_metrics']['accuracy']:.4f}")
    print("-" * 60)
    print(f"CoT-before weighted_f1: {summary['cot_before_metrics']['weighted_f1']:.4f}")
    print(f"CoT-before macro_f1   : {summary['cot_before_metrics']['macro_f1']:.4f}")
    print(f"CoT-before accuracy   : {summary['cot_before_metrics']['accuracy']:.4f}")
    print("-" * 60)
    print(f"CoT-after weighted_f1 : {summary['cot_after_metrics']['weighted_f1']:.4f}")
    print(f"CoT-after macro_f1    : {summary['cot_after_metrics']['macro_f1']:.4f}")
    print(f"CoT-after accuracy    : {summary['cot_after_metrics']['accuracy']:.4f}")

    print_top_confusions("baseline", summary["baseline_top_confusions"])
    print_top_confusions("CoT-before", summary["cot_before_top_confusions"])
    print_top_confusions("CoT-after", summary["cot_after_top_confusions"])

    os.makedirs(args.output_dir, exist_ok=True)
    (Path(args.output_dir) / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, payload in results.items():
        (Path(args.output_dir) / f"{name}_details.json").write_text(json.dumps(payload["details"], ensure_ascii=False, indent=2), encoding="utf-8")
        (Path(args.output_dir) / f"{name}_confusion_matrix.json").write_text(json.dumps(payload["confusion"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
