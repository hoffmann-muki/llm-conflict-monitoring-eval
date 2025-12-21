"""Explainable prompting strategy with chain-of-thought reasoning.

Uses chain-of-thought prompting to encourage transparent step-by-step reasoning.
"""

from typing import Dict, Any, Optional
from .base import PromptingStrategy


class ExplainableStrategy(PromptingStrategy):
    """Explainable classification with chain-of-thought reasoning.
    
    Asks the model to explain its reasoning before providing classification,
    enabling analysis of how the model arrives at decisions.
    """
    
    def make_prompt(self, event_note: str) -> str:
        """Generate explainable classification prompt with reasoning request.
        
        Args:
            event_note: Event description text to classify
            
        Returns:
            Formatted prompt requesting reasoning and classification
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

Step 1 — Brief structured reasoning (exactly three short items):
- Provide three-numbered, one-line observations only:
  1. Key actors (who)
  2. Key actions (what)
  3. Category rationale (why)
- Each observation must be at most 20 words.
- Do not include extra commentary or chain-of-thought beyond these three lines.

Step 2 — Final answer (valid JSON only):
Return ONLY valid JSON with this structure:
{{
    "reasoning": [<three strings>],
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
        """Get JSON schema for explainable responses.
        
        Returns:
            JSON schema expecting reasoning, label, confidence, and logits
        """
        return {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "array", "items": {"type": "string"}},
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
                "required": ["reasoning", "label", "confidence", "logits"]
        }
    
    def get_system_message(self) -> Optional[str]:
        """Get system message for explainable strategy.
        
        Returns:
            System message emphasizing transparency and reasoning
        """
        return "You are an expert political conflict event analyst. Always explain your reasoning step-by-step before making classifications."
