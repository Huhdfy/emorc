#!/usr/bin/env python
"""
Compare a label-only baseline model and a CoT model on the same dataset.

Example:
    python ERC1/scripts/eval_compare_models.py \
      --baseline_base_model /tmp/pretrainmodel/Qwen2.5-7B-Instruct \
      --baseline_lora_path fQwQf/erc-qwen2.5-7b-sota \
      --cot_base_model /tmp/pretrainmodel/Qwen2.5-7B-Instruct \
      --cot_lora_path ./outputs/qwen7b_cot/final_model \
      --test_data ERC1/data/json/val_samples.json \
      --output_dir ./outputs/compare \
      --max_samples 500
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
    """Zero corrupted lm_head rows that can produce inf/nan logits."""
    target_model = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        target_model = model.base_model.model
    elif hasattr(model, "model"):
        target_model = model.model

    if not hasattr(target_model, "lm_head"):
        return

    with torch.no_grad():
        lm_weight = target_model.lm_head.weight.data
        inf_mask = torch.isinf(lm_weight) | torch.isnan(lm_weight)
        if inf_mask.any():
            bad_rows = torch.where(inf_mask.any(dim=1))[0]
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

    parts = [
        "<|im_start|>system",
        "You are an expert emotion recognition assistant. "
        "Identify the emotion of the target utterance from the standard taxonomy.",
        f"Standard emotions: {emotion_list}",
        "<|im_end|>",
        "<|im_start|>user",
        "Now analyze this dialogue:",
        "",
        "Dialogue History:",
        history_text,
        "",
        f"Target Utterance: {sample.target_utterance}",
    ]

    parts.extend(
        [
            "",
            "Return only the final emotion label.",
            "<|im_end|>",
            "<|im_start|>assistant",
            "",
        ]
    )
    return "\n".join(parts)


def parse_label_only_output(output: str) -> str:
    cleaned = output.strip().replace("<|im_end|>", "").replace("<|endoftext|>", "").strip().lower()
    if not cleaned:
        return "neutral"

    cleaned_first_line = cleaned.splitlines()[0].strip(" -:\t")
    candidates = [cleaned_first_line, cleaned]
    for candidate in candidates:
        for emotion in TAXONOMY.emotions:
            if candidate == emotion:
                return emotion

    for emotion in TAXONOMY.emotions:
        pattern = rf"\b{re.escape(emotion)}\b"
        if re.search(pattern, cleaned):
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
        prompt = build_label_only_prompt(sample)
        decoded = generate_text(model, tokenizer, prompt, max_length=max_length, max_new_tokens=32)
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


def evaluate_cot(model, tokenizer, samples, max_length: int, explanation_position: str):
    builder = EmotionPromptBuilder(use_retrieval=False, explanation_position=explanation_position)
    predictions = []
    details = []

    for sample in tqdm(samples, desc="Evaluating CoT"):
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
    """Build serializable confusion matrix payload."""
    cm = confusion_matrix(ground_truths, predictions, labels=labels)
    return {
        "labels": labels,
        "matrix": cm.tolist(),  # rows: gold, cols: pred
    }


def top_confusions(confusion_data, top_k=10):
    """Return top off-diagonal confusions."""
    labels = confusion_data["labels"]
    matrix = confusion_data["matrix"]
    pairs = []
    for i, gold in enumerate(labels):
        for j, pred in enumerate(labels):
            if i == j:
                continue
            count = matrix[i][j]
            if count > 0:
                pairs.append((count, gold, pred))
    pairs.sort(reverse=True, key=lambda x: x[0])
    return [
        {"count": c, "gold": g, "pred": p}
        for c, g, p in pairs[:top_k]
    ]


def print_disagreement_examples(samples, baseline_details, cot_details, limit: int):
    baseline_map = {item["sample_id"]: item for item in baseline_details}
    cot_map = {item["sample_id"]: item for item in cot_details}

    disagreements = []
    for sample in samples:
        sid = sample.sample_id
        b = baseline_map[sid]
        c = cot_map[sid]
        if b["pred_emotion"] != c["pred_emotion"] or b["correct"] != c["correct"]:
            disagreements.append((sample, b, c))

    print("\n" + "=" * 80)
    print(f"DISAGREEMENT EXAMPLES ({min(limit, len(disagreements))} shown)")
    print("=" * 80)
    for sample, baseline_item, cot_item in disagreements[:limit]:
        print("-" * 80)
        print(f"ID: {sample.sample_id}")
        print(f"Gold: {sample.emotion}")
        print(f"Target: {sample.target_utterance}")
        if sample.dialogue_history:
            print(f"History: {' | '.join(sample.dialogue_history[-3:])}")
        print(f"Baseline pred: {baseline_item['pred_emotion']} | correct={baseline_item['correct']}")
        print(f"Baseline raw: {baseline_item['raw_output']}")
        print(f"CoT pred: {cot_item['pred_emotion']} | correct={cot_item['correct']}")
        print(f"CoT explanation: {cot_item.get('pred_explanation')}")
        print(f"CoT raw: {cot_item['raw_output']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_base_model", type=str, required=True, help="Base model for label-only baseline")
    parser.add_argument("--baseline_lora_path", type=str, default=None, help="Optional LoRA path for label-only baseline")
    parser.add_argument("--cot_base_model", type=str, required=True, help="Base model for CoT evaluation")
    parser.add_argument("--cot_lora_path", type=str, required=True, help="LoRA path for CoT model")
    parser.add_argument("--test_data", type=str, default="ERC1/data/json/val_samples_debiased.json")
    parser.add_argument("--output_dir", type=str, default="./outputs/compare")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--show_examples", type=int, default=8)
    parser.add_argument("--cot_explanation_position", type=str, choices=["before", "after"], default="before")
    args = parser.parse_args()

    processor = DataProcessor()
    samples = processor.load_samples(args.test_data)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"Loaded {len(samples)} evaluation samples from {args.test_data}")

    baseline_model, baseline_tokenizer = load_model(args.baseline_base_model, args.baseline_lora_path)
    baseline_predictions, baseline_details = evaluate_baseline(
        baseline_model, baseline_tokenizer, samples, max_length=args.max_length
    )
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cot_model, cot_tokenizer = load_model(args.cot_base_model, args.cot_lora_path)
    cot_predictions, cot_details = evaluate_cot(
        cot_model,
        cot_tokenizer,
        samples,
        max_length=args.max_length,
        explanation_position=args.cot_explanation_position,
    )
    del cot_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ground_truths = [sample.emotion.lower() for sample in samples]
    baseline_metrics = compute_metrics(baseline_predictions, ground_truths)
    cot_metrics = compute_metrics(cot_predictions, ground_truths)
    matrix_labels = list(TAXONOMY.emotions)
    baseline_confusion = build_confusion_data(baseline_predictions, ground_truths, matrix_labels)
    cot_confusion = build_confusion_data(cot_predictions, ground_truths, matrix_labels)
    baseline_top_confusions = top_confusions(baseline_confusion, top_k=10)
    cot_top_confusions = top_confusions(cot_confusion, top_k=10)

    summary = {
        "num_samples": len(samples),
        "dataset": args.test_data,
        "baseline_base_model": args.baseline_base_model,
        "baseline_lora_path": args.baseline_lora_path,
        "cot_base_model": args.cot_base_model,
        "cot_lora_path": args.cot_lora_path,
        "cot_explanation_position": args.cot_explanation_position,
        "baseline_metrics": baseline_metrics,
        "cot_metrics": cot_metrics,
        "delta": {
            "weighted_f1": cot_metrics["weighted_f1"] - baseline_metrics["weighted_f1"],
            "macro_f1": cot_metrics["macro_f1"] - baseline_metrics["macro_f1"],
            "accuracy": cot_metrics["accuracy"] - baseline_metrics["accuracy"],
        },
        "baseline_top_confusions": baseline_top_confusions,
        "cot_top_confusions": cot_top_confusions,
    }

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Baseline weighted_f1: {baseline_metrics['weighted_f1']:.4f}")
    print(f"Baseline macro_f1   : {baseline_metrics['macro_f1']:.4f}")
    print(f"Baseline accuracy   : {baseline_metrics['accuracy']:.4f}")
    print("-" * 60)
    print(f"CoT weighted_f1     : {cot_metrics['weighted_f1']:.4f}")
    print(f"CoT macro_f1        : {cot_metrics['macro_f1']:.4f}")
    print(f"CoT accuracy        : {cot_metrics['accuracy']:.4f}")
    print("-" * 60)
    print(f"Delta weighted_f1   : {summary['delta']['weighted_f1']:+.4f}")
    print(f"Delta macro_f1      : {summary['delta']['macro_f1']:+.4f}")
    print(f"Delta accuracy      : {summary['delta']['accuracy']:+.4f}")

    print("\nTop baseline confusions (gold -> pred):")
    for item in baseline_top_confusions[:5]:
        print(f"  {item['gold']} -> {item['pred']}: {item['count']}")
    print("Top CoT confusions (gold -> pred):")
    for item in cot_top_confusions[:5]:
        print(f"  {item['gold']} -> {item['pred']}: {item['count']}")

    print_disagreement_examples(samples, baseline_details, cot_details, args.show_examples)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "baseline_details.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_details, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "cot_details.json"), "w", encoding="utf-8") as f:
        json.dump(cot_details, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "baseline_confusion_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_confusion, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "cot_confusion_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(cot_confusion, f, ensure_ascii=False, indent=2)

    print(f"\nSaved summary to {os.path.join(args.output_dir, 'comparison_summary.json')}")
    print(f"Saved baseline details to {os.path.join(args.output_dir, 'baseline_details.json')}")
    print(f"Saved CoT details to {os.path.join(args.output_dir, 'cot_details.json')}")
    print(f"Saved baseline confusion matrix to {os.path.join(args.output_dir, 'baseline_confusion_matrix.json')}")
    print(f"Saved CoT confusion matrix to {os.path.join(args.output_dir, 'cot_confusion_matrix.json')}")


if __name__ == "__main__":
    main()
