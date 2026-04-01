#!/usr/bin/env python
"""
Evaluation Script for Fine-Grained Emotion Recognition
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.data.data_processor import DataProcessor
from src.data.emotion_taxonomy import TAXONOMY
from src.models.prompt_template import EmotionPromptBuilder, parse_model_output


def load_model(base_model_path, lora_path=None):
    """Load model with optional LoRA weights"""
    print(f"Loading base model: {base_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if lora_path:
        print(f"Loading LoRA weights from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    
    model.eval()
    return model, tokenizer


def evaluate(model, tokenizer, samples, max_length=512, batch_size=4):
    """Evaluate model on samples"""
    builder = EmotionPromptBuilder(use_retrieval=False)
    predictions = []
    ground_truths = []
    
    for sample in tqdm(samples, desc="Evaluating"):
        # Build prompt
        prompt = builder.build_inference_prompt(
            dialogue_history=sample.dialogue_history,
            target_utterance=sample.target_utterance,
        )
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        
        # Parse
        result = parse_model_output(decoded)
        
        # Map to known emotion or use neutral
        pred_emotion = result["emotion"].lower()
        if pred_emotion not in TAXONOMY.emotions:
            # Try fuzzy matching
            for e in TAXONOMY.emotions:
                if e in pred_emotion or pred_emotion in e:
                    pred_emotion = e
                    break
            else:
                pred_emotion = "neutral"
        
        predictions.append(pred_emotion)
        ground_truths.append(sample.emotion.lower())
    
    return predictions, ground_truths


def compute_metrics(predictions, ground_truths):
    """Compute evaluation metrics"""
    # Get all unique labels
    all_labels = sorted(set(predictions + ground_truths))
    label2idx = {l: i for i, l in enumerate(all_labels)}
    
    pred_indices = [label2idx[p] for p in predictions]
    true_indices = [label2idx[g] for g in ground_truths]
    
    # Compute metrics
    weighted_f1 = f1_score(true_indices, pred_indices, average="weighted")
    macro_f1 = f1_score(true_indices, pred_indices, average="macro")
    accuracy = accuracy_score(true_indices, pred_indices)
    
    # Per-class F1
    per_class = f1_score(true_indices, pred_indices, average=None)
    class_f1 = {all_labels[i]: f1 for i, f1 in enumerate(per_class)}
    
    # Classification report
    report = classification_report(
        true_indices, pred_indices,
        target_names=all_labels,
        output_dict=True,
        zero_division=0,
    )
    
    return {
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "per_class_f1": class_f1,
        "classification_report": report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--test_data", type=str, default="ERC1/data/json/val_samples_debiased.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.base_model, args.model_path)
    
    # Load test data
    print(f"Loading test data from: {args.test_data}")
    processor = DataProcessor()
    samples = processor.load_samples(args.test_data)
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    print(f"Test samples: {len(samples)}")
    
    # Evaluate
    predictions, ground_truths = evaluate(model, tokenizer, samples)
    
    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # SOTA check
    target_f1 = 0.65
    if metrics['weighted_f1'] >= target_f1:
        print(f"\n✓ SOTA TARGET MET! F1 = {metrics['weighted_f1']:.4f} >= {target_f1}")
    else:
        print(f"\n✗ SOTA TARGET NOT MET. F1 = {metrics['weighted_f1']:.4f} < {target_f1}")
        gap = target_f1 - metrics['weighted_f1']
        print(f"  Gap: {gap:.4f}")
    
    # Print per-class F1 (sorted by F1)
    print("\nPer-class F1 (sorted):")
    sorted_f1 = sorted(metrics['per_class_f1'].items(), key=lambda x: x[1])
    for emotion, f1 in sorted_f1[:10]:  # Show worst 10
        print(f"  {emotion}: {f1:.4f}")
    
    # Save results
    output_dir = args.output_dir or os.path.dirname(args.model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        # Convert numpy types to Python types
        serializable_metrics = {
            "weighted_f1": float(metrics["weighted_f1"]),
            "macro_f1": float(metrics["macro_f1"]),
            "accuracy": float(metrics["accuracy"]),
            "per_class_f1": {k: float(v) for k, v in metrics["per_class_f1"].items()},
        }
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return metrics


if __name__ == "__main__":
    main()
