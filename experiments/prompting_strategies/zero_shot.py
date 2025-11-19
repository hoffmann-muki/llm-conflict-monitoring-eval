"""Zero-shot prompting strategy (current default approach)."""

from typing import Dict, Any, Optional
from .base import PromptingStrategy


class ZeroShotStrategy(PromptingStrategy):
    """Zero-shot direct classification without examples.
    
    This is the current default approach used in the repository.
    It provides category descriptions and asks for direct classification.
    """
    
    def make_prompt(self, note: str) -> str:
        """Generate zero-shot classification prompt.
        
        Args:
            note: Event description text to classify
            
        Returns:
            Formatted prompt requesting direct classification
        """
        return f"""Classify this event: {note}

Categories: V=Violence against civilians, B=Battles, E=Explosions, P=Protests, R=Riots, S=Strategic developments

Answer with JSON only: {{"label": "V", "confidence": 0.9, "logits": [0.9,0.1,0.0,0.0,0.0,0.0]}}"""
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for zero-shot responses.
        
        Returns:
            JSON schema expecting label, confidence, and optional logits
        """
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "confidence": {"type": "number"},
                "logits": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["label", "confidence"]
        }
    
    def get_system_message(self) -> Optional[str]:
        """Get system message for zero-shot (none needed).
        
        Returns:
            None (zero-shot doesn't use system messages)
        """
        return None
