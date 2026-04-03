"""
Prompt templates for explanation-guided fine-grained emotion recognition.
"""

import re
from typing import List, Optional
from dataclasses import dataclass

from ..data.emotion_taxonomy import TAXONOMY


@dataclass
class PromptTemplate:
    """Prompt template for explanation + emotion generation."""

    EXPLANATION_PREFIX: str = "- Explanation: "
    EMOTION_PREFIX: str = "- Emotion: "
    
    system_instruction: str = "You are an expert emotion recognition assistant. Analyze the dialogue and identify the emotion of the target utterance from the standard 28-class taxonomy."

    emotion_list: str = "Standard emotions: joy, excitement, happiness, gratitude, pride, relief, hope, love, caring, desire, optimism, amusement, sadness, anger, fear, anxiety, disgust, shame, guilt, disappointment, frustration, grief, loneliness, jealousy, neutral, surprise, confusion, curiosity, nervousness."

    @staticmethod
    def validate_explanation_position(explanation_position: str) -> str:
        if explanation_position not in {"before", "after"}:
            raise ValueError(f"Unsupported explanation_position: {explanation_position}")
        return explanation_position

    @staticmethod
    def format_dialogue_history(history: List[str], max_turns: int = 5) -> str:
        """Format dialogue history for prompt"""
        if not history:
            return "[No previous context]"
        
        recent_history = history[-max_turns:]
        formatted = []
        for i, utterance in enumerate(recent_history):
            formatted.append(f"Turn {i+1}: {utterance}")
        return "\n".join(formatted)
    
    @staticmethod
    def format_demonstration(
        dialogue_history: List[str],
        target_utterance: str,
        emotion: str,
        explanation: Optional[str] = None,
        explanation_position: str = "before",
    ) -> str:
        """Format a single demonstration example"""
        explanation_position = PromptTemplate.validate_explanation_position(explanation_position)
        history_str = PromptTemplate.format_dialogue_history(dialogue_history)
        explanation_text = (explanation or "The emotion is inferred from the target utterance and dialogue context.").strip()

        if explanation_position == "before":
            analysis = f"- Explanation: {explanation_text}\n- Emotion: {emotion}"
        else:
            analysis = f"- Emotion: {emotion}\n- Explanation: {explanation_text}"

        demo = f"""Example:
Dialogue History:
{history_str}

Target Utterance: {target_utterance}

Analysis:
{analysis}"""
        
        return demo
    
    @staticmethod
    def format_query(
        dialogue_history: List[str],
        target_utterance: str,
        prev_impact: Optional[str] = None,
        explanation_position: str = "before",
    ) -> str:
        """Format the query (input to be predicted)"""
        explanation_position = PromptTemplate.validate_explanation_position(explanation_position)
        history_str = PromptTemplate.format_dialogue_history(dialogue_history)

        if explanation_position == "before":
            output_format = (
                "- Explanation: [brief explanation of why this emotion is expressed, citing specific text cues]\n"
                "- Emotion: [your emotion prediction]"
            )
        else:
            output_format = (
                "- Emotion: [your emotion prediction]\n"
                "- Explanation: [brief explanation of why this emotion is expressed, citing specific text cues]"
            )

        return f"""Now analyze this dialogue:

Dialogue History:
{history_str}

Target Utterance: {target_utterance}

Provide your analysis in the following format:
{output_format}"""
    
    def build_full_prompt(
        self,
        dialogue_history: List[str],
        target_utterance: str,
        demonstrations: Optional[List[dict]] = None,
        include_system: bool = True,
        prev_impact: Optional[str] = None,
        assistant_prefix: str = "",
        explanation_position: str = "before",
    ) -> str:
        """Build complete prompt with optional demonstrations"""
        explanation_position = self.validate_explanation_position(explanation_position)
        parts = []
        
        # System instruction
        if include_system:
            parts.append(f"<|im_start|>system\n{self.system_instruction}\n\n{self.emotion_list}<|im_end|>")
        
        # Demonstrations (retrieved examples)
        if demonstrations:
            demo_parts = []
            for demo in demonstrations[:3]:  # Max 3 demonstrations
                demo_str = self.format_demonstration(
                    dialogue_history=demo.get("dialogue_history", []),
                    target_utterance=demo.get("target_utterance", ""),
                    emotion=demo.get("emotion", "neutral"),
                    explanation=demo.get("explanation"),
                    explanation_position=explanation_position,
                )
                demo_parts.append(demo_str)
            
            if demo_parts:
                parts.append(f"<|im_start|>user\nHere are some examples:\n\n" + "\n\n".join(demo_parts) + "<|im_end|>")
        
        # Query
        query = self.format_query(
            dialogue_history,
            target_utterance,
            prev_impact=prev_impact,
            explanation_position=explanation_position,
        )
        parts.append(f"<|im_start|>user\n{query}<|im_end|>")
        
        # Assistant response start
        parts.append(f"<|im_start|>assistant\n{assistant_prefix}")
        
        return "\n".join(parts)
    
    def build_training_prompt(
        self,
        dialogue_history: List[str],
        target_utterance: str,
        emotion: str,
        speaker: str,
        prev_impact: Optional[str] = None,
        demonstrations: Optional[List[dict]] = None,
        explanation: Optional[str] = None,
        explanation_position: str = "before",
    ) -> str:
        """Build prompt for training with target response"""
        explanation_position = self.validate_explanation_position(explanation_position)
        has_explanation = bool(explanation and explanation.strip())
        assistant_prefix = self.EXPLANATION_PREFIX if explanation_position == "before" else self.EMOTION_PREFIX
        prompt = self.build_full_prompt(
            dialogue_history=dialogue_history,
            target_utterance=target_utterance,
            demonstrations=demonstrations,
            prev_impact=prev_impact,
            assistant_prefix=assistant_prefix if has_explanation else "",
            explanation_position=explanation_position,
        )
        
        if explanation_position == "before":
            response = ""
            if has_explanation:
                response += f"{explanation.strip()}\n"
            response += f"- Emotion: {emotion}"
        else:
            response = f"{emotion}"
            if has_explanation:
                response += f"\n- Explanation: {explanation.strip()}"
        response += "<|im_end|>"
        
        return prompt + response


class EmotionPromptBuilder:
    """Builder class for creating prompts with retrieval augmentation"""
    
    def __init__(self, use_retrieval: bool = True, top_k: int = 3, explanation_position: str = "before"):
        self.template = PromptTemplate()
        self.use_retrieval = use_retrieval
        self.top_k = top_k
        self.explanation_position = self.template.validate_explanation_position(explanation_position)
    
    def build_inference_prompt(
        self,
        dialogue_history: List[str],
        target_utterance: str,
        retrieved_examples: Optional[List[dict]] = None,
        prev_impact: Optional[str] = None,
        force_explanation: bool = True,
    ) -> str:
        """Build prompt for inference"""
        demonstrations = None
        if self.use_retrieval and retrieved_examples:
            demonstrations = retrieved_examples[:self.top_k]
        
        return self.template.build_full_prompt(
            dialogue_history=dialogue_history,
            target_utterance=target_utterance,
            demonstrations=demonstrations,
            prev_impact=prev_impact,
            assistant_prefix=(
                self.template.EXPLANATION_PREFIX
                if self.explanation_position == "before"
                else self.template.EMOTION_PREFIX
            ) if force_explanation else "",
            explanation_position=self.explanation_position,
        )
    
    def build_training_prompt(
        self,
        dialogue_history: List[str],
        target_utterance: str,
        emotion: str,
        speaker: str,
        prev_impact: Optional[str] = None,
        retrieved_examples: Optional[List[dict]] = None,
        explanation: Optional[str] = None,
    ) -> str:
        """Build prompt for training"""
        demonstrations = None
        if self.use_retrieval and retrieved_examples:
            demonstrations = retrieved_examples[:self.top_k]
        
        return self.template.build_training_prompt(
            dialogue_history=dialogue_history,
            target_utterance=target_utterance,
            emotion=emotion,
            speaker=speaker,
            prev_impact=prev_impact,
            demonstrations=demonstrations,
            explanation=explanation,
            explanation_position=self.explanation_position,
        )


def parse_model_output(output: str) -> dict:
    """Parse model output to extract predictions"""
    result = {
        "emotion": "neutral",
        "explanation": None,
    }

    cleaned = output.strip()
    if not cleaned:
        return result

    cleaned = cleaned.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    def extract_last_field(field_name: str):
        pattern = rf"(?:^|\n)\s*-?\s*{re.escape(field_name)}:\s*(.+)"
        matches = list(re.finditer(pattern, cleaned, flags=re.IGNORECASE))
        if not matches:
            return None, None
        match = matches[-1]
        value = match.group(1).strip()
        return value, match

    emotion_raw, emotion_match = extract_last_field("Emotion")
    if emotion_raw:
        emotion_clean = emotion_raw.split("(")[0].strip().lower()
        result["emotion"] = emotion_clean
    else:
        # In explanation-after mode, the assistant may continue from the
        # "- Emotion: " prefix already present in the prompt, so the generated
        # text can start with a bare label like "joy" on the first line.
        first_line = cleaned.splitlines()[0].strip()
        if first_line:
            first_line = first_line.strip(" -:\t")
            candidate = first_line.split("(")[0].strip().lower()
            if candidate in TAXONOMY.emotions:
                result["emotion"] = candidate
                emotion_match = re.search(re.escape(cleaned.splitlines()[0]), cleaned)

    explanation_raw, explanation_match = extract_last_field("Explanation")
    if explanation_raw:
        if emotion_match and explanation_match and explanation_match.start() < emotion_match.start():
            explanation_span = cleaned[explanation_match.end():emotion_match.start()]
            explanation_raw = explanation_span.strip() or explanation_raw
        result["explanation"] = explanation_raw.strip()
    elif emotion_match:
        prefix_text = cleaned[:emotion_match.start()].strip()
        if prefix_text:
            prefix_text = re.sub(r"^\s*-?\s*Explanation:\s*", "", prefix_text, flags=re.IGNORECASE)
            result["explanation"] = prefix_text.strip() or None

    return result
