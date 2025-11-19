"""Explainable prompting strategy with chain-of-thought reasoning.

TODO: Implement chain-of-thought prompting for transparent reasoning.
"""

from typing import Dict, Any, Optional
from .base import PromptingStrategy


class ExplainableStrategy(PromptingStrategy):
    """Explainable classification with chain-of-thought reasoning.
    
    Asks the model to explain its reasoning before providing classification,
    enabling analysis of how the model arrives at decisions.
    """
    
    def make_prompt(self, note: str) -> str:
        """Generate explainable classification prompt with reasoning request.
        
        Args:
            note: Event description text to classify
            
        Returns:
            Formatted prompt requesting reasoning and classification
        """
        return f"""Classify this event: {note}

Categories: V=Violence against civilians, B=Battles, E=Explosions, P=Protests, R=Riots, S=Strategic developments

First, explain your reasoning step-by-step:
1. Identify the key actors
2. Identify the key actions
3. Determine which category best fits

Then answer with JSON: {{"reasoning": "...", "label": "V", "confidence": 0.9}}"""
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for explainable responses.
        
        Returns:
            JSON schema expecting reasoning, label, and confidence
        """
        return {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "label": {"type": "string"},
                "confidence": {"type": "number"},
                "logits": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["reasoning", "label", "confidence"]
        }
    
    def get_system_message(self) -> Optional[str]:
        """Get system message for explainable strategy.
        
        Returns:
            System message emphasizing transparency and reasoning
        """
        return "You are an expert analyst. Always explain your reasoning step-by-step before making classifications."
