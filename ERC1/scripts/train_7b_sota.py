#!/usr/bin/env python
"""
Final SOTA Training Script for Qwen2.5-7B
Optimized with QLoRA, Role Filtering, and No-Truncation (768 length)
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import f1_score, accuracy_score

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_processor import DataProcessor, DialogueSample
from src.data.emotion_taxonomy import TAXONOMY
from src.models.prompt_template import EmotionPromptBuilder, parse_model_output

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0
    
    torch.cuda.set_device(gpu)
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://', 
                               world_size=world_size, rank=rank)
    return rank, world_size, gpu

class MultiDatasetEmotionDataset(Dataset):
    def __init__(
        self,
        samples,
        tokenizer,
        max_length=768,
        explanation_position="before",
        mask_cot=False,
        explanation_loss_weight=1.0,
        emotion_loss_weight=1.0,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_cot = mask_cot
        self.explanation_position = explanation_position
        self.explanation_loss_weight = explanation_loss_weight
        self.emotion_loss_weight = emotion_loss_weight
        self.builder = EmotionPromptBuilder(use_retrieval=False, explanation_position=explanation_position)
        self.class_weights = self._calculate_class_weights()
    
    def _calculate_class_weights(self):
        '''根据样本中的情绪类别进行加权，样本越少，权重越高'''
        emotion_counts = Counter([s.emotion for s in self.samples])
        total = len(self.samples)
        weights = {}
        for emotion, count in emotion_counts.items():
            weights[emotion] = 1.0 / np.sqrt(count / total)
        max_w = max(weights.values())
        return {k: v/max_w for k, v in weights.items()}
    
    def __len__(self):
        return len(self.samples)

    def _encode_length(self, text):
        return len(
            self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
        )

    def _build_loss_weights(self, prompt, attention_mask, response_start_idx):
        valid_end_idx = int(attention_mask.sum().item())
        loss_weights = torch.zeros_like(attention_mask, dtype=torch.float32)

        if response_start_idx >= valid_end_idx:
            return loss_weights

        if self.explanation_position == "before":
            emotion_marker = "\n- Emotion:"
            emotion_marker_idx = prompt.rfind(emotion_marker)
            if emotion_marker_idx != -1:
                emotion_start_idx = min(self._encode_length(prompt[:emotion_marker_idx + 1]), valid_end_idx)
            else:
                emotion_start_idx = valid_end_idx

            explanation_end_idx = max(response_start_idx, min(emotion_start_idx, valid_end_idx))
            explanation_weight = 0.0 if self.mask_cot else self.explanation_loss_weight
            if explanation_end_idx > response_start_idx:
                loss_weights[response_start_idx:explanation_end_idx] = explanation_weight
            if valid_end_idx > explanation_end_idx:
                loss_weights[explanation_end_idx:valid_end_idx] = self.emotion_loss_weight
        else:
            explanation_marker = "\n- Explanation:"
            explanation_marker_idx = prompt.rfind(explanation_marker)
            if explanation_marker_idx != -1:
                explanation_start_idx = min(self._encode_length(prompt[:explanation_marker_idx + 1]), valid_end_idx)
            else:
                explanation_start_idx = valid_end_idx

            emotion_end_idx = max(response_start_idx, min(explanation_start_idx, valid_end_idx))
            if emotion_end_idx > response_start_idx:
                loss_weights[response_start_idx:emotion_end_idx] = self.emotion_loss_weight

            explanation_weight = 0.0 if self.mask_cot else self.explanation_loss_weight
            if valid_end_idx > emotion_end_idx:
                loss_weights[emotion_end_idx:valid_end_idx] = explanation_weight

        return loss_weights
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = self.builder.build_training_prompt(
            dialogue_history=sample.dialogue_history,
            target_utterance=sample.target_utterance,
            emotion=sample.emotion,
            speaker=sample.speaker,
            explanation=sample.explanation,
        )
        
        encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_without_res = self.builder.build_inference_prompt(
            dialogue_history=sample.dialogue_history,
            target_utterance=sample.target_utterance,
            force_explanation=bool(sample.explanation),
        )
        response_start_idx = self._encode_length(prompt_without_res)
        loss_weights = self._build_loss_weights(prompt, attention_mask, response_start_idx)

        labels[:response_start_idx] = -100
        
        labels[attention_mask == 0] = -100
        loss_weights[attention_mask == 0] = 0.0
        
        # Safety check: ensure at least one valid label exists
        valid_labels = (labels != -100).sum()
        if valid_labels == 0:
            last_valid = attention_mask.sum() - 1
            if last_valid >= 0:
                labels[last_valid] = input_ids[last_valid]
                loss_weights[last_valid] = self.emotion_loss_weight

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": loss_weights,
            "weight": torch.tensor(self.class_weights.get(sample.emotion, 1.0), dtype=torch.float32),
        }

def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "loss_weights": torch.stack([item["loss_weights"] for item in batch]),
        "weights": torch.stack([item["weight"] for item in batch]),
    }


def map_prediction_to_taxonomy(pred_emotion: str) -> str:
    pred_emotion = str(pred_emotion).strip().lower()
    if pred_emotion in TAXONOMY.emotions:
        return pred_emotion
    for emotion in TAXONOMY.emotions:
        if emotion in pred_emotion or pred_emotion in emotion:
            return emotion
    return "neutral"


def evaluate_generation_metrics(model, tokenizer, samples, gpu, explanation_position, max_length=512):
    builder = EmotionPromptBuilder(use_retrieval=False, explanation_position=explanation_position)
    predictions = []
    ground_truths = []
    eval_model = model.module if hasattr(model, "module") else model

    for sample in samples:
        prompt = builder.build_inference_prompt(
            dialogue_history=sample.dialogue_history,
            target_utterance=sample.target_utterance,
            force_explanation=True,
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = inputs["input_ids"].to(gpu)
        attention_mask = inputs["attention_mask"].to(gpu)

        outputs = eval_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated = outputs[0][input_ids.shape[1]:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        parsed = parse_model_output(decoded)
        predictions.append(map_prediction_to_taxonomy(parsed["emotion"]))
        ground_truths.append(sample.emotion.lower())

    labels = sorted(set(predictions + ground_truths))
    return {
        "weighted_f1": float(f1_score(ground_truths, predictions, labels=labels, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(ground_truths, predictions, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(ground_truths, predictions)),
    }


def compute_weighted_ce_loss(logits, labels, loss_weights):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_loss_weights = loss_weights[:, 1:].contiguous()

    token_losses = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)

    valid_mask = shift_labels.ne(-100)
    effective_weights = shift_loss_weights * valid_mask.float()
    denom = effective_weights.sum()
    if denom.item() == 0:
        return token_losses.new_tensor(0.0)
    return (token_losses * effective_weights).sum() / denom

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/tmp/pretrainmodel/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--val_metric_samples",
        type=int,
        default=200,
        help="Number of validation samples used for generation-based metrics after each epoch.",
    )
    parser.add_argument(
        "--explanation_position",
        type=str,
        choices=["before", "after"],
        default="before",
        help="Whether to place the explanation before or after the emotion field.",
    )
    parser.add_argument("--mask_cot", action="store_true", help="If set, masks the CoT explanation part in the loss calculation.")
    parser.add_argument("--explanation_loss_weight", type=float, default=1.0, help="Loss weight applied to explanation tokens.")
    parser.add_argument("--emotion_loss_weight", type=float, default=1.0, help="Loss weight applied to emotion tokens.")
    parser.add_argument("--log_file", type=str, default="training_metrics.json", help="File to save detailed training metrics")
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "json" / "final" / "annotated_with_explanations_repaired_final_debiased.json"),
        help="Path to debiased annotated training samples with explanations",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "json" / "val_samples_debiased.json"),
        help="Path to debiased validation samples",
    )
    args = parser.parse_args()
    
    rank, world_size, gpu = setup_distributed()
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Starting 7B SOTA Training on {world_size} GPUs")
        print(f"Effective Batch Size: {args.batch_size * world_size * args.grad_acc}")
        print(f"Explanation position: {args.explanation_position}")
        print(f"mask_cot={args.mask_cot}, explanation_loss_weight={args.explanation_loss_weight}, emotion_loss_weight={args.emotion_loss_weight}")

    # We found that even on A100, bfloat16 + nf4 + sdpa causes severe inf issues in your environment.
    # Sticking to the proven float16 configuration from our earlier debugging session.
    compute_dtype = torch.float16
    if rank == 0:
        print(f"Forcing compute dtype: {compute_dtype} (proven stable configuration)")

    # Quantization config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,

        device_map={"": gpu},
        trust_remote_code=True,
        attn_implementation="sdpa", # Use SDPA (built-in PyTorch optimization)
        use_cache=False, # Required for gradient checkpointing
    )
    
    # Avoid PEFT internally re-enabling checkpointing with default kwargs.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    # Fix corrupted LM head rows from bfloat16 → float16 overflow during loading
    # Our previous debugging showed that forcing float16 + zeroing these rows produces perfectly healthy logits.
    with torch.no_grad():
        lm_weight = model.lm_head.weight.data
        inf_mask = torch.isinf(lm_weight)
        if inf_mask.any():
            bad_rows = torch.where(inf_mask.any(dim=1))[0]
            if rank == 0:
                print(f"Zeroing {len(bad_rows)} corrupted LM head rows (vocab: {bad_rows.tolist()})")
            lm_weight[bad_rows] = 0.0
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
    
    # Data Loading (Load from local processed cache)
    processor = DataProcessor()
    
    train_path = args.train_file
    val_path = args.val_file
        
    if rank == 0:
        print(f"Loading train samples from: {train_path}")
        print(f"Loading val samples from: {val_path}")
        
    train_samples = processor.load_samples(str(train_path))
    val_samples = processor.load_samples(str(val_path))

    random.shuffle(train_samples)
    train_samples = train_samples[:args.max_samples]
    
    if len(val_samples) > 2000: # Limit val for speed
        random.shuffle(val_samples)
        val_samples = val_samples[:2000]
    
    train_ds = MultiDatasetEmotionDataset(
        train_samples,
        tokenizer,
        explanation_position=args.explanation_position,
        mask_cot=args.mask_cot,
        explanation_loss_weight=args.explanation_loss_weight,
        emotion_loss_weight=args.emotion_loss_weight,
    )
    val_ds = MultiDatasetEmotionDataset(
        val_samples,
        tokenizer,
        explanation_position=args.explanation_position,
        mask_cot=args.mask_cot,
        explanation_loss_weight=args.explanation_loss_weight,
        emotion_loss_weight=args.emotion_loss_weight,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, 
        sampler=torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None,
        shuffle=(world_size == 1),  # Shuffle when not using DistributedSampler
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        sampler=torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None,
        collate_fn=collate_fn
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=100, 
        num_training_steps=len(train_loader) * args.epochs // args.grad_acc
    )

    best_val_loss = float('inf')
    
    # Initialize metrics dictionary for JSON logging
    metrics_log = {
        "train_loss": [],
        "val_loss": [],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_acc": args.grad_acc,
        "learning_rate": args.lr,
        "explanation_position": args.explanation_position,
        "mask_cot": args.mask_cot,
        "explanation_loss_weight": args.explanation_loss_weight,
        "emotion_loss_weight": args.emotion_loss_weight,
        "val_metrics": [],
    }

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        if world_size > 1: train_loader.sampler.set_epoch(epoch)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=(rank != 0))
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(gpu)
            attention_mask = batch["attention_mask"].to(gpu)
            labels = batch["labels"].to(gpu)
            loss_weights = batch["loss_weights"].to(gpu)
            weights = batch["weights"].to(gpu)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            base_loss = compute_weighted_ce_loss(outputs.logits, labels, loss_weights)
            loss = (base_loss * weights.mean()) / args.grad_acc

            if not torch.isfinite(loss):
                if rank == 0:
                    print(f"Non-finite loss detected ({loss.item()}). Skipping step.")
                optimizer.zero_grad()
                continue

            loss.backward()
            
            if (step + 1) % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.grad_acc
            if rank == 0 and step % 10 == 0:
                current_loss = loss.item() * args.grad_acc
                pbar.set_postfix(loss=current_loss)
                # Log step loss periodically
                if step % 50 == 0:
                    metrics_log["train_loss"].append({
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "loss": current_loss
                    })
            
            # SAVE INTERMEDIATE CHECKPOINT every 1000 steps
            if (step + 1) % 1000 == 0 and rank == 0:
                mid_path = f"{args.output_dir}/checkpoint_step_{step+1}"
                os.makedirs(mid_path, exist_ok=True)
                if world_size > 1:
                    model.module.save_pretrained(mid_path)
                else:
                    model.save_pretrained(mid_path)
                print(f"\n✓ Saved intermediate checkpoint at step {step+1}")
        
        # End of Epoch: SAVE FIRST
        if rank == 0:
            save_path = f"{args.output_dir}/checkpoint_{epoch+1}"
            os.makedirs(save_path, exist_ok=True)
            if world_size > 1:
                model.module.save_pretrained(save_path)
            else:
                model.save_pretrained(save_path)
            print(f"✓ Saved Checkpoint {epoch+1}")

        # Validation
        model.eval()
        torch.cuda.empty_cache() # Clear cache for validation
        epoch_val_loss = 0
        val_steps = 0
        
        # Use a very small batch size for validation
        val_loader_safe = DataLoader(
            val_ds, batch_size=1,
            sampler=torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None,
            collate_fn=collate_fn
        )
        
        try:
            with torch.no_grad():
                for v_batch in tqdm(val_loader_safe, desc="Validating", disable=(rank != 0)):
                    v_input_ids = v_batch["input_ids"].to(gpu)
                    v_attention_mask = v_batch["attention_mask"].to(gpu)
                    v_labels = v_batch["labels"].to(gpu)
                    v_loss_weights = v_batch["loss_weights"].to(gpu)
                    v_outputs = model(input_ids=v_input_ids, attention_mask=v_attention_mask)
                    v_loss = compute_weighted_ce_loss(v_outputs.logits, v_labels, v_loss_weights)
                    if not torch.isnan(v_loss):
                        epoch_val_loss += v_loss.item()
                        val_steps += 1
            
            # Gather validation loss across GPUs
            if world_size > 1:
                val_loss_tensor = torch.tensor([epoch_val_loss, val_steps], device=gpu)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                avg_val_loss = val_loss_tensor[0].item() / max(1, val_loss_tensor[1].item())
            else:
                avg_val_loss = epoch_val_loss / max(1, val_steps)

            if rank == 0:
                print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
                
                # Log validation loss
                metrics_log["val_loss"].append({
                    "epoch": epoch + 1,
                    "val_loss": avg_val_loss
                })

                metric_sample_count = min(len(val_samples), args.val_metric_samples)
                if metric_sample_count > 0:
                    metric_samples = val_samples[:metric_sample_count]
                    generation_metrics = evaluate_generation_metrics(
                        model=model,
                        tokenizer=tokenizer,
                        samples=metric_samples,
                        gpu=gpu,
                        explanation_position=args.explanation_position,
                    )
                    print(
                        f"Epoch {epoch+1} Val Metrics (n={metric_sample_count}): "
                        f"weighted_f1={generation_metrics['weighted_f1']:.4f}, "
                        f"macro_f1={generation_metrics['macro_f1']:.4f}, "
                        f"accuracy={generation_metrics['accuracy']:.4f}"
                    )
                    metrics_log["val_metrics"].append({
                        "epoch": epoch + 1,
                        "num_samples": metric_sample_count,
                        **generation_metrics,
                    })
                
                # Save metrics to file
                log_path = os.path.join(args.output_dir, args.log_file)
                with open(log_path, "w") as f:
                    json.dump(metrics_log, f, indent=4)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_path = f"{args.output_dir}/best_model"
                    os.makedirs(best_path, exist_ok=True)
                    if world_size > 1:
                        model.module.save_pretrained(best_path)
                    else:
                        model.save_pretrained(best_path)
                    print(f"✓ Saved Best Model")
        except Exception as e:
            if rank == 0:
                print(f"Validation failed with error: {e}. Skipping validation scores.")
        
        torch.cuda.empty_cache() # Clear cache after validation


    if rank == 0:
        if world_size > 1:
            model.module.save_pretrained(f"{args.output_dir}/final_model")
        else:
            model.save_pretrained(f"{args.output_dir}/final_model")
        print("Training Complete.")

if __name__ == "__main__":
    main()
