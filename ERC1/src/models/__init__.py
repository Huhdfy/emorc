"""Minimal model package exports for CoT training."""

from .prompt_template import PromptTemplate, EmotionPromptBuilder, parse_model_output

__all__ = ["PromptTemplate", "EmotionPromptBuilder", "parse_model_output"]
