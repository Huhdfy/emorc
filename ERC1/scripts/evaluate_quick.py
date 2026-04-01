#!/usr/bin/env python
"""
Quick evaluation on a subset of data
"""

import os
import sys
import json
from pathlib import Path

# Set Chinese mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from src.data.data_processor import DataProcessor
from src.data.dataset import EmotionDatasetForInference, EmotionDatasetConfig, collate_fn_inference
from src.models.model import load_model_for_inference
from src.models.prompt_template import parse_model_output
from src.retrieval.retriever import Retriever, BatchRetriever
from src.evaluation.evaluator import EmotionEvaluator
from torch.utils.data import DataLoader


def main():
    # Configuration
    config_path = "configs/config.yaml"
    model_path = "outputs/checkpoints/run_20260211_112814/final_model"
    
    # Load config
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load test data
    processor = DataProcessor()
    test_samples = processor.load_samples("./cache/data/test_samples.json")
    
    # Use only first 500 samples for quick evaluation
    test_samples = test_samples[:500]
    print(f"Using {len(test_samples)} samples for quick evaluation")
    
    # Load model
    print(f"Loading model from {model_path}")
    use_flash = config["model"].get("use_flash_attention", False)
    model_wrapper = load_model_for_inference(
        base_model=config["model"]["base_model"],
        lora_path=model_path,
        device_map="auto",
        use_flash_attention=use_flash,
    )
    model = model_wrapper.get_model()
    tokenizer = model_wrapper.get_tokenizer()
    model.eval()
    
    # Build retriever
    print("Building retriever...")
    train_samples = processor.load_samples("./cache/data/train_samples.json")
    retriever = Retriever(
        model_name=config["retrieval"]["embedding_model"],
        cache_dir=config["retrieval"]["cache_dir"],
        device=device,
    )
    
    index_path = os.path.join(config["retrieval"]["cache_dir"], "index")
    if os.path.exists(index_path):
        retriever.load_index(index_path)
    else:
        retriever.build_index(train_samples)
        retriever.save_index(index_path)
    
    # Create dataset config
    dataset_config = EmotionDatasetConfig(
        max_length=config["model"]["max_length"],
        use_retrieval=config["retrieval"]["top_k"] > 0,
        top_k_demonstrations=config["retrieval"]["top_k"],
    )
    
    # Retrieve demonstrations
    print("Retrieving demonstrations...")
    batch_retriever = BatchRetriever(retriever, k=dataset_config.top_k_demonstrations)
    retrieved_examples = batch_retriever.retrieve_batch(test_samples)
    
    # Create dataset
    test_dataset = EmotionDatasetForInference(
        samples=test_samples,
        tokenizer=tokenizer,
        config=dataset_config,
        retrieved_examples=retrieved_examples,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn_inference,
    )
    
    # Run inference
    predictions = []
    ground_truths = []
    sample_ids = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            
            # Generate
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode outputs
            for i, output in enumerate(outputs):
                generated = output[input_ids.shape[1]:]
                decoded = tokenizer.decode(generated, skip_special_tokens=True)
                
                parsed = parse_model_output(decoded)
                predictions.append(parsed["emotion"])
                ground_truths.append(batch["true_emotions"][i])
                sample_ids.append(batch["sample_ids"][i])
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * 8} samples...")
    
    # Evaluate
    evaluator = EmotionEvaluator(output_dir="./outputs/results")
    
    result = evaluator.evaluate(
        predictions=predictions,
        ground_truths=ground_truths,
        sample_ids=sample_ids,
        save_results=True,
    )
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Weighted F1: {result.weighted_f1:.4f}")
    print(f"Macro F1: {result.macro_f1:.4f}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print("="*60)
    
    # SOTA target check
    target_f1 = 0.65
    if result.weighted_f1 >= target_f1:
        print(f"\n✓ SOTA TARGET MET! F1 = {result.weighted_f1:.4f} >= {target_f1}")
    else:
        print(f"\n✗ SOTA TARGET NOT MET. F1 = {result.weighted_f1:.4f} < {target_f1}")
    
    # Save results
    results = {
        "weighted_f1": result.weighted_f1,
        "macro_f1": result.macro_f1,
        "accuracy": result.accuracy,
        "per_class_f1": result.per_class_f1,
        "num_samples": len(test_samples),
    }
    
    output_file = "./outputs/results/quick_evaluation.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
