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

Categories (fixed order):
1. V - Violence against civilians
2. B - Battles
3. E - Explosions
4. P - Protests
5. R - Riots
6. S - Strategic developments

Return ONLY valid JSON with this structure:
{{
  "label": "<category code>",
  "confidence": <decimal>,
  "logits": [<six decimals>]
}}

JSON rules:
- No extra text before or after the JSON.
- "label" must be one of the six category codes.
- "confidence" must be a decimal between 0 and 1.
- "logits" must be six decimal numbers summing to 1.0 in the same order as the category list.
- If unsure about confidence, use a low confidence (e.g. 0.10) rather than inventing details.
- If unsure about logits, use approximate scores.
"""
    
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
        """Get system message for zero-shot.
        
        Returns:
            None (zero-shot doesn't use system messages)
        """
        return None
