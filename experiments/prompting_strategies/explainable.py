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

Categories (fixed order):
1. V - Violence against civilians
2. B - Battles
3. E - Explosions
4. P - Protests
5. R - Riots
6. S - Strategic developments

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
  "label": "<category code>",
  "confidence": <decimal>,
  "logits": [<six decimals>]
}}

JSON rules:
- No extra text before or after the JSON.
- "reasoning" must be an array of three short strings exactly matching Step 1.
- "label" must be one of the six category codes.
- "confidence" must be a decimal between 0 and 1.
- "logits" must be six decimal numbers summing to 1.0 in the same order as the category list.
- If unsure about confidence, use a low confidence (e.g. 0.10) rather than inventing details.
- If unsure about logits, use approximate scores.
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
                "label": {"type": "string"},
                "confidence": {"type": "number"},
                "logits": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["reasoning", "label", "confidence", "logits"]
        }
    
    def get_system_message(self) -> Optional[str]:
        """Get system message for explainable strategy.
        
        Returns:
            System message emphasizing transparency and reasoning
        """
        return "You are an expert political conflict event analyst. Always explain your reasoning step-by-step before making classifications."
