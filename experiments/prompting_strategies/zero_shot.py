"""Zero-shot prompting strategy (current default approach)."""

from typing import Dict, Any, Optional
from .base import PromptingStrategy


class ZeroShotStrategy(PromptingStrategy):
    """Zero-shot direct classification without examples.
    
    This is the current default approach used in the repository.
    It provides category descriptions and asks for direct classification.
    """
    
    def make_prompt(self, event_note: str) -> str:
        """Generate zero-shot classification prompt.
        
        Args:
            event_note: Event description text to classify
            
        Returns:
            Formatted prompt requesting direct classification
        """
        return f"""You are an expert political conflict event analyst.

Classify the following event into one of six categories: {event_note}

Categories (use ONLY these single-letter codes):
- V = Violence against civilians
- B = Battles
- E = Explosions/Remote violence
- P = Protests
- R = Riots
- S = Strategic developments

Return ONLY valid JSON with this structure:
{{
    "label": "<V, B, E, P, R, or S>",
    "confidence": <decimal between 0 and 1>,
    "logits": {{"V": <num>, "B": <num>, "E": <num>, "P": <num>, "R": <num>, "S": <num>}}
}}

CRITICAL: The "label" field must be exactly one of: V, B, E, P, R, S
Do not use numbers, full words, or any other values.

Additional requirements for `logits`:
- Each value in the `logits` object must be a decimal probability between 0 and 1.
- The six logits (V, B, E, P, R, S) must sum to 1.0.
"""
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for zero-shot responses.
        
        Returns:
            JSON schema expecting label, confidence, and optional logits
        """
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["V", "B", "E", "P", "R", "S"]},
                "confidence": {"type": "number"},
                "logits": {
                    "type": "object",
                    "properties": {
                        "V": {"type": "number"},
                        "B": {"type": "number"},
                        "E": {"type": "number"},
                        "P": {"type": "number"},
                        "R": {"type": "number"},
                        "S": {"type": "number"}
                    }
                }
            },
            "required": ["label", "confidence"]
        }
    
    def get_system_message(self) -> Optional[str]:
        """Get system message for zero-shot.
        
        Returns:
            None (zero-shot doesn't use system messages)
        """
        return None
