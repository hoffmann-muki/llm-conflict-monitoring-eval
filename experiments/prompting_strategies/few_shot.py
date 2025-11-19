"""Few-shot prompting strategy with example demonstrations.

TODO: Implement few-shot learning with examples for each category.
"""

from typing import Dict, Any, Optional, List
from .base import PromptingStrategy


class FewShotStrategy(PromptingStrategy):
    """Few-shot classification with example demonstrations.
    
    Provides 1-3 examples per category to guide the model's classification.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize few-shot strategy.
        
        Args:
            config: Configuration with optional 'examples' key containing
                   example events for each category
        """
        super().__init__(config)
        self.examples = config.get('examples', {}) if config else {}
    
    def make_prompt(self, note: str) -> str:
        """Generate few-shot classification prompt with examples.
        
        Args:
            note: Event description text to classify
            
        Returns:
            Formatted prompt with examples and classification request
        """
        # TODO: Implement with actual examples
        examples_text = """Examples:
- "Military forces attacked a village, killing 5 civilians" → {{"label": "V", "confidence": 0.95}}
- "Government troops clashed with rebel forces near the border" → {{"label": "B", "confidence": 0.90}}
- "Hundreds gathered to protest against government policies" → {{"label": "P", "confidence": 0.88}}

"""
        return f"""{examples_text}
Now classify this event: {note}

Categories: V=Violence against civilians, B=Battles, E=Explosions, P=Protests, R=Riots, S=Strategic developments

Answer with JSON only: {{"label": "V", "confidence": 0.9}}"""
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for few-shot responses.
        
        Returns:
            JSON schema expecting label and confidence
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
        """Get system message for few-shot.
        
        Returns:
            System message explaining the classification task
        """
        return "You are an expert at classifying political events based on examples."
