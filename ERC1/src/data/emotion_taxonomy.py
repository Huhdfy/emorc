"""
Unified Emotion Taxonomy for Multi-Dataset Fine-Grained Emotion Recognition

This module defines a unified emotion label space that maps emotions from
different datasets (EmpatheticDialogues, GoEmotions, IEMOCAP, MELD)
to a common taxonomy.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum


class UnifiedEmotion(Enum):
    """Unified emotion taxonomy with 28 fine-grained categories"""
    # Positive emotions
    JOY = "joy"
    EXCITEMENT = "excitement"
    HAPPINESS = "happiness"
    GRATITUDE = "gratitude"
    PRIDE = "pride"
    RELIEF = "relief"
    HOPE = "hope"
    LOVE = "love"
    CARING = "caring"
    DESIRE = "desire"
    OPTIMISM = "optimism"
    AMUSEMENT = "amusement"
    
    # Negative emotions
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    ANXIETY = "anxiety"
    DISGUST = "disgust"
    SHAME = "shame"
    GUILT = "guilt"
    DISAPPOINTMENT = "disappointment"
    FRUSTRATION = "frustration"
    GRIEF = "grief"
    LONELINESS = "loneliness"
    JEALOUSY = "jealousy"
    
    # Neutral/Other
    NEUTRAL = "neutral"
    SURPRISE = "surprise"
    CONFUSION = "confusion"
    CURIOSITY = "curiosity"
    NERVOUSNESS = "nervousness"


@dataclass
class EmotionTaxonomy:
    """Unified emotion taxonomy with mappings from different datasets"""
    
    # All unified emotions
    emotions: List[str] = None
    
    # Emotion to index mapping
    emotion2idx: Dict[str, int] = None
    idx2emotion: Dict[int, str] = None
    
    # Dataset-specific mappings
    empathetic_mapping: Dict[str, str] = None
    goemotions_mapping: Dict[str, str] = None
    iemocap_mapping: Dict[str, str] = None
    meld_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        self.emotions = [e.value for e in UnifiedEmotion]
        self.emotion2idx = {e: i for i, e in enumerate(self.emotions)}
        self.idx2emotion = {i: e for i, e in enumerate(self.emotions)}
        
        # EmpatheticDialogues (32 classes) mapping
        self.empathetic_mapping = {
            "anticipating": "excitement",
            "afraid": "fear",
            "angry": "anger",
            "annoyed": "frustration",
            "anxious": "anxiety",
            "apprehensive": "nervousness",
            "ashamed": "shame",
            "caring": "caring",
            "confident": "optimism",
            "content": "happiness",
            "devastated": "grief",
            "disappointed": "disappointment",
            "disgusted": "disgust",
            "embarrassed": "shame",
            "excited": "excitement",
            "faithful": "hope",
            "furious": "anger",
            "grateful": "gratitude",
            "guilty": "guilt",
            "hopeful": "hope",
            "impressed": "amusement",
            "jealous": "jealousy",
            "joyful": "joy",
            "lonely": "loneliness",
            "nostalgic": "sadness",
            "prepared": "optimism",
            "proud": "pride",
            "sad": "sadness",
            "sentimental": "sadness",
            "surprised": "surprise",
            "terrified": "fear",
            "trusting": "caring",
        }
        
        # GoEmotions (28 classes) mapping
        self.goemotions_mapping = {
            "admiration": "gratitude",
            "amusement": "amusement",
            "anger": "anger",
            "annoyance": "frustration",
            "approval": "gratitude",
            "caring": "caring",
            "confusion": "confusion",
            "curiosity": "curiosity",
            "desire": "desire",
            "disappointment": "disappointment",
            "disapproval": "disgust",
            "disgust": "disgust",
            "embarrassment": "shame",
            "excitement": "excitement",
            "fear": "fear",
            "gratitude": "gratitude",
            "grief": "grief",
            "joy": "joy",
            "love": "love",
            "nervousness": "nervousness",
            "optimism": "optimism",
            "pride": "pride",
            "realization": "surprise",
            "relief": "relief",
            "remorse": "guilt",
            "sadness": "sadness",
            "surprise": "surprise",
            "neutral": "neutral",
        }
        
        # IEMOCAP (6 classes) mapping
        self.iemocap_mapping = {
            "ang": "anger",
            "dis": "disgust",
            "fea": "fear",
            "hap": "joy",
            "neu": "neutral",
            "sad": "sadness",
        }
        
        # MELD (7 classes) mapping
        self.meld_mapping = {
            "anger": "anger",
            "disgust": "disgust",
            "fear": "fear",
            "joy": "joy",
            "neutral": "neutral",
            "sadness": "sadness",
            "surprise": "surprise",
        }
        

    
    def get_num_classes(self) -> int:
        """Get number of emotion classes"""
        return len(self.emotions)
    
    def map_emotion(self, emotion: str, dataset: str) -> Optional[str]:
        """Map dataset-specific emotion to unified emotion"""
        mapping = {
            "empathetic": self.empathetic_mapping,
            "goemotions": self.goemotions_mapping,
            "iemocap": self.iemocap_mapping,
            "meld": self.meld_mapping,
        }
        
        dataset = dataset.lower()
        if dataset not in mapping:
            return None
        
        return mapping[dataset].get(emotion.lower())
    
    def get_emotion_idx(self, emotion: str) -> int:
        """Get index for unified emotion"""
        return self.emotion2idx.get(emotion, self.emotion2idx["neutral"])
    
    def get_emotion_from_idx(self, idx: int) -> str:
        """Get emotion name from index"""
        return self.idx2emotion.get(idx, "neutral")


# Global taxonomy instance
TAXONOMY = EmotionTaxonomy()


# Emotion groups for hierarchical analysis
EMOTION_GROUPS = {
    "positive": [
        "joy", "excitement", "happiness", "gratitude", "pride", 
        "relief", "hope", "love", "caring", "desire", 
        "optimism", "amusement"
    ],
    "negative": [
        "sadness", "anger", "fear", "anxiety", "disgust", 
        "shame", "guilt", "disappointment", "frustration", 
        "grief", "loneliness", "jealousy"
    ],
    "neutral": ["neutral", "surprise", "confusion", "curiosity", "nervousness"],
}


def get_emotion_group(emotion: str) -> str:
    """Get the group (positive/negative/neutral) for an emotion"""
    for group, emotions in EMOTION_GROUPS.items():
        if emotion in emotions:
            return group
    return "neutral"
