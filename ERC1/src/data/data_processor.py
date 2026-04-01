"""
Data Preprocessing Pipeline for Multi-Dataset Fine-Grained Emotion Recognition

This module handles loading and preprocessing of multiple emotion datasets:
- EmpatheticDialogues
- GoEmotions
- IEMOCAP
- MELD
"""

import os
import json
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm

from .emotion_taxonomy import TAXONOMY, UnifiedEmotion


@dataclass
class DialogueSample:
    """
    A single dialogue sample with emotion label
    [Architecture Manual 3.2]: 统一样本结构。包含对话历史 (dialogue_history)、
    目标句 (target_utterance)、统一后的 28 类情绪标签 (emotion) 及说话人信息。
    通过该数据结构将不同的异构数据集（ED, GoEmotions, MELD）拉齐。
    """
    sample_id: str
    dialogue_history: List[str]  # List of utterances before target
    target_utterance: str
    emotion: str  # Unified emotion label
    emotion_idx: int
    speaker: str  # For speaker identification task
    prev_speaker: Optional[str] = None
    prev_emotion: Optional[str] = None  # For emotion impact prediction
    dataset: str = ""
    raw_data: Dict = field(default_factory=dict)
    explanation: Optional[str] = None  # CoT: brief explanation of why this emotion is expressed
    
    def to_dict(self) -> Dict:
        d = {
            "sample_id": self.sample_id,
            "dialogue_history": self.dialogue_history,
            "target_utterance": self.target_utterance,
            "emotion": self.emotion,
            "emotion_idx": self.emotion_idx,
            "speaker": self.speaker,
            "prev_speaker": self.prev_speaker,
            "prev_emotion": self.prev_emotion,
            "dataset": self.dataset,
        }
        if self.explanation is not None:
            d["explanation"] = self.explanation
        return d


class DataProcessor:
    """
    Process multiple emotion datasets into unified format.
    [Architecture Manual 2.3 & 3]: 这里的 Processor 负责跨多个语料库提取对话字段，
    并依赖 EmotionTaxonomy 映射为统一的分类体系 (28 类包含 neutral)，最终写入 cache json 文件。
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.taxonomy = TAXONOMY
    
    def load_empathetic_dialogues(self, split: str = "train") -> List[DialogueSample]:
        """Load and process EmpatheticDialogues dataset"""
        print(f"Loading EmpatheticDialogues {split} split...")
        
        try:
            dataset = load_dataset("empathetic_dialogues", split=split)
        except Exception as e:
            print(f"Error loading EmpatheticDialogues: {e}")
            return []
        
        samples = []
        dialogue_buffers: Dict[str, List[Dict]] = {}
        
        for item in tqdm(dataset, desc="Processing EmpatheticDialogues"):
            conv_id = str(item.get("conv_id", ""))
            utterance_idx = item.get("utterance_idx", 0)
            
            if conv_id not in dialogue_buffers:
                dialogue_buffers[conv_id] = []
            
            dialogue_buffers[conv_id].append({
                "utterance": item.get("utterance", ""),
                # In raw data, idx 1, 3, 5... are the Speaker (feeling the emotion)
                # idx 2, 4, 6... are the Listener (empathizing)
                "role": "Speaker" if utterance_idx % 2 != 0 else "Listener",
                "emotion": item.get("context", ""),
                "idx": utterance_idx,
            })
        
        # Process dialogues
        for conv_id, turns in dialogue_buffers.items():
            # Sort by index to maintain dialogue flow
            turns.sort(key=lambda x: x["idx"])
            
            for i, turn in enumerate(turns):
                # [Architecture Manual 3.3]: EmpatheticDialogues 只保留 Speaker 轮作为训练目标
                # 因为数据集的 context (emotion label) 是针对 Speaker 的经历的，
                # Listener 轮次的真实情绪通常不是 context 标签本身（而是共情/同情）。
                # ONLY train/eval on the Speaker turns to match the 'context' label
                # Listener turns are removed as their emotion doesn't match the label
                if turn["role"] == "Listener":
                    continue 
                
                # Double check: only keep Speaker turns (1, 3, 5...)
                if turn["idx"] % 2 == 0:
                    continue
                
                if not turn["utterance"].strip():
                    continue
                
                emotion = self.taxonomy.map_emotion(turn["emotion"], "empathetic")
                if emotion is None:
                    emotion = "neutral"
                
                # History includes both Speaker and Listener turns
                history = [t["utterance"] for t in turns[:i]]
                
                sample = DialogueSample(
                    sample_id=f"ed_{conv_id}_{turn['idx']}",
                    dialogue_history=history,
                    target_utterance=turn["utterance"],
                    emotion=emotion,
                    emotion_idx=self.taxonomy.get_emotion_idx(emotion),
                    speaker="Speaker", # Simplified role for model
                    prev_speaker="Listener" if i > 0 else None,
                    prev_emotion=None, # ED doesn't have reliable utterance-level prev_emotion
                    dataset="empathetic",
                    raw_data={"original_emotion": turn["emotion"]},
                )
                samples.append(sample)
        
        print(f"Processed {len(samples)} samples from EmpatheticDialogues")
        return samples
    
    def load_goemotions(self, split: str = "train") -> List[DialogueSample]:
        """Load and process GoEmotions dataset"""
        print(f"Loading GoEmotions {split} split...")
        
        try:
            dataset = load_dataset("go_emotions", split=split)
        except Exception as e:
            print(f"Error loading GoEmotions: {e}")
            return []
        
        # Load label names
        label_names = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness",
            "surprise", "neutral"
        ]
        
        samples = []
        for i, item in enumerate(tqdm(dataset, desc="Processing GoEmotions")):
            # Get primary emotion (first label if multi-label)
            labels = item.get("labels", [])
            if isinstance(labels, int):
                labels = [labels]
            
            if not labels:
                continue
            
            primary_label = label_names[labels[0]]
            emotion = self.taxonomy.map_emotion(primary_label, "goemotions")
            if emotion is None:
                emotion = "neutral"
            
            sample = DialogueSample(
                sample_id=f"ge_{i}",
                # [Architecture Manual 3.3]: GoEmotions 是单文本分类，不具备对话上下文
                dialogue_history=[],  # GoEmotions doesn't have dialogue context
                target_utterance=item.get("text", ""),
                emotion=emotion,
                emotion_idx=self.taxonomy.get_emotion_idx(emotion),
                speaker="Speaker",
                dataset="goemotions",
                raw_data={"original_labels": labels, "label_names": label_names},
            )
            samples.append(sample)
        
        print(f"Processed {len(samples)} samples from GoEmotions")
        return samples
    
    def load_meld(self, split: str = "train") -> List[DialogueSample]:
        """Load and process MELD dataset"""
        print(f"Loading MELD {split} split...")
        
        # Try different dataset names
        dataset_names = ["meld", "meld_s", "anton-l/meld"]
        dataset = None
        
        for ds_name in dataset_names:
            try:
                dataset = load_dataset(ds_name, split=split, trust_remote_code=True)
                print(f"Successfully loaded MELD from '{ds_name}'")
                break
            except Exception as e:
                print(f"Failed to load '{ds_name}': {e}")
                continue
        
        if dataset is None:
            print("Error: Could not load MELD dataset from any source")
            return []
        
        samples = []
        dialogue_buffers: Dict[str, List[Dict]] = {}
        
        for item in tqdm(dataset, desc="Processing MELD"):
            dia_id = str(item.get("Dialogue_ID", ""))
            utt_id = item.get("Utterance_ID", 0)
            
            if dia_id not in dialogue_buffers:
                dialogue_buffers[dia_id] = []
            
            dialogue_buffers[dia_id].append({
                "utterance": item.get("Utterance", ""),
                "speaker": item.get("Speaker", "Unknown"),
                "emotion": item.get("Emotion", "neutral"),
                "utt_id": utt_id,
            })
        
        # Sort and process dialogues
        for dia_id, turns in dialogue_buffers.items():
            turns.sort(key=lambda x: x["utt_id"])
            
            for i, turn in enumerate(turns):
                emotion = self.taxonomy.map_emotion(turn["emotion"], "meld")
                if emotion is None:
                    emotion = "neutral"
                
                # [Architecture Manual 3.3]: MELD 按发生顺序构建完整的历史上下文
                history = [t["utterance"] for t in turns[:i]]
                prev_emotion = None
                
                # 提取前一轮的情绪和说话人信息，用于辅助建模交互或影响
                if i > 0:
                    prev_emotion = self.taxonomy.map_emotion(
                        turns[i-1]["emotion"], "meld"
                    )
                
                sample = DialogueSample(
                    sample_id=f"meld_{dia_id}_{i}",
                    dialogue_history=history,
                    target_utterance=turn["utterance"],
                    emotion=emotion,
                    emotion_idx=self.taxonomy.get_emotion_idx(emotion),
                    speaker=turn["speaker"],
                    prev_speaker=turns[i-1]["speaker"] if i > 0 else None,
                    prev_emotion=prev_emotion,
                    dataset="meld",
                    raw_data={"original_emotion": turn["emotion"]},
                )
                samples.append(sample)
        
        print(f"Processed {len(samples)} samples from MELD")
        return samples
    
    def load_all_datasets(
        self, 
        datasets: List[str] = None,
        split: str = "train"
    ) -> List[DialogueSample]:
        """Load all specified datasets"""
        if datasets is None:
            datasets = ["empathetic", "goemotions", "meld"]
        
        all_samples = []
        
        if "empathetic" in datasets:
            samples = self.load_empathetic_dialogues(split)
            all_samples.extend(samples)
        
        if "goemotions" in datasets:
            samples = self.load_goemotions(split)
            all_samples.extend(samples)
        
        if "meld" in datasets:
            samples = self.load_meld(split)
            all_samples.extend(samples)
        
        print(f"\nTotal samples loaded: {len(all_samples)}")
        return all_samples
    
    def save_samples(self, samples: List[DialogueSample], filepath: str):
        """Save processed samples to file"""
        data = [s.to_dict() for s in samples]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(samples)} samples to {filepath}")
    
    def load_samples(self, filepath: str) -> List[DialogueSample]:
        """Load processed samples from file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            sample = DialogueSample(
                sample_id=item["sample_id"],
                dialogue_history=item["dialogue_history"],
                target_utterance=item["target_utterance"],
                emotion=item["emotion"],
                emotion_idx=item["emotion_idx"],
                speaker=item["speaker"],
                prev_speaker=item.get("prev_speaker"),
                prev_emotion=item.get("prev_emotion"),
                dataset=item.get("dataset", ""),
                explanation=item.get("explanation"),
            )
            samples.append(sample)
        
        print(f"Loaded {len(samples)} samples from {filepath}")
        return samples
    
    def get_class_distribution(self, samples: List[DialogueSample]) -> Dict[str, int]:
        """Get emotion class distribution"""
        distribution = {}
        for sample in samples:
            distribution[sample.emotion] = distribution.get(sample.emotion, 0) + 1
        return dict(sorted(distribution.items(), key=lambda x: -x[1]))
    
    def print_statistics(self, samples: List[DialogueSample]):
        """Print dataset statistics"""
        print("\n" + "="*50)
        print("Dataset Statistics")
        print("="*50)
        print(f"Total samples: {len(samples)}")
        
        # Class distribution
        distribution = self.get_class_distribution(samples)
        print("\nClass Distribution:")
        for emotion, count in distribution.items():
            pct = count / len(samples) * 100
            print(f"  {emotion}: {count} ({pct:.2f}%)")
        
        # Dialogue length stats
        history_lengths = [len(s.dialogue_history) for s in samples]
        print(f"\nDialogue History Stats:")
        print(f"  Average length: {np.mean(history_lengths):.2f}")
        print(f"  Max length: {max(history_lengths)}")
        print(f"  Min length: {min(history_lengths)}")
        print("="*50)
