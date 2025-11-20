"""Few-shot prompting strategy with example demonstrations.

TODO: Implement few-shot learning with examples for each category.
"""

from typing import Dict, Any, Optional, List
from .base import PromptingStrategy


class FewShotStrategy(PromptingStrategy):
    """Few-shot classification with example demonstrations.
    
    Provides example(s) per category to guide the model's classification.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize few-shot strategy.
        
        Args:
            config: Configuration with optional 'examples_per_category' key
                   specifying number of examples to show (1-5, default: 1)
        """
        super().__init__(config)
        self.examples_per_category = config.get('examples_per_category', 1) if config else 1
        if not 1 <= self.examples_per_category <= 5:
            raise ValueError("examples_per_category must be between 1 and 5")
    
    def _get_examples(self, n_per_category: int) -> str:
        """Generate example classifications for few-shot learning.
        
        Args:
            n_per_category: Number of examples per category (1-5)
            
        Returns:
            Formatted string with example classifications
        """
        # Example pool: up to 5 examples per category with realistic confidence and logits
        # Logits order: [V, B, E, P, R, S]
        example_pool = {
            "V": [
                '"Military forces shot and injured a woman in Nongomadiba when they fired shots at a building they believed to be holding Ambazonian Separatists." → {{"label": "V", "confidence": 0.89, "logits": [0.89, 0.05, 0.02, 0.01, 0.02, 0.01]}}',
                '"Security forces opened fire on civilians during a raid in Bamenda, killing two people." → {{"label": "V", "confidence": 0.94, "logits": [0.94, 0.03, 0.01, 0.01, 0.01, 0.00]}}',
                '"Armed soldiers beat and detained three civilians suspected of supporting separatists in Kumbo." → {{"label": "V", "confidence": 0.86, "logits": [0.86, 0.02, 0.01, 0.02, 0.03, 0.06]}}',
                '"Police forces tortured detainees at a checkpoint in Buea, injuring five individuals." → {{"label": "V", "confidence": 0.91, "logits": [0.91, 0.03, 0.01, 0.01, 0.02, 0.02]}}',
                '"Military personnel looted civilian homes and assaulted residents in Mamfe." → {{"label": "V", "confidence": 0.87, "logits": [0.87, 0.02, 0.01, 0.02, 0.05, 0.03]}}'
            ],
            "B": [
                '"The police forces killed one suspected Boko Haram fighter and arrested another in Aissa Karde village" → {{"label": "B", "confidence": 0.82, "logits": [0.08, 0.82, 0.03, 0.01, 0.02, 0.04]}}',
                '"Military forces clashed with separatist fighters in Belo, resulting in casualties on both sides." → {{"label": "B", "confidence": 0.93, "logits": [0.03, 0.93, 0.02, 0.01, 0.01, 0.00]}}',
                '"Government troops engaged Boko Haram militants near Fotokol, killing several insurgents." → {{"label": "B", "confidence": 0.91, "logits": [0.04, 0.91, 0.03, 0.00, 0.01, 0.01]}}',
                '"Armed forces exchanged fire with rebel groups in the Northwest region for several hours." → {{"label": "B", "confidence": 0.89, "logits": [0.05, 0.89, 0.03, 0.01, 0.01, 0.01]}}',
                '"Security forces raided a separatist hideout in Kumba, killing three fighters." → {{"label": "B", "confidence": 0.85, "logits": [0.06, 0.85, 0.02, 0.01, 0.02, 0.04]}}'
            ],
            "E": [
                '"An IED planted by suspected Ambazonian separatists detonated in Matezen village, Santa subdivision, injuring three people." → {{"label": "E", "confidence": 0.96, "logits": [0.02, 0.01, 0.96, 0.00, 0.01, 0.00]}}',
                '"A roadside bomb exploded near a military convoy in Kolofata, wounding two soldiers." → {{"label": "E", "confidence": 0.95, "logits": [0.02, 0.02, 0.95, 0.00, 0.01, 0.00]}}',
                '"Unidentified militants launched a mortar attack on a police station in Mora." → {{"label": "E", "confidence": 0.92, "logits": [0.03, 0.03, 0.92, 0.00, 0.01, 0.01]}}',
                '"An explosive device detonated at a market in Maroua, killing one civilian and injuring ten." → {{"label": "E", "confidence": 0.94, "logits": [0.03, 0.01, 0.94, 0.01, 0.01, 0.00]}}',
                '"Suspected insurgents fired rockets at a military base in the Far North region." → {{"label": "E", "confidence": 0.93, "logits": [0.02, 0.03, 0.93, 0.00, 0.01, 0.01]}}'
            ],
            "P": [
                '"About a hundred residents demonstrated in Djoum town against the government\'s delay in compensating them after destroying their houses to build the Bikouna-Djoum road." → {{"label": "P", "confidence": 0.88, "logits": [0.02, 0.01, 0.01, 0.88, 0.06, 0.02]}}',
                '"Teachers held a peaceful protest in Yaoundé demanding better salaries and working conditions." → {{"label": "P", "confidence": 0.95, "logits": [0.01, 0.00, 0.00, 0.95, 0.03, 0.01]}}',
                '"Students marched through Douala to protest tuition fee increases at public universities." → {{"label": "P", "confidence": 0.93, "logits": [0.01, 0.01, 0.00, 0.93, 0.04, 0.01]}}',
                '"Civil society groups organized a demonstration in Bamenda calling for dialogue and peace." → {{"label": "P", "confidence": 0.91, "logits": [0.02, 0.01, 0.01, 0.91, 0.04, 0.01]}}',
                '"Healthcare workers staged a sit-in at the Ministry of Health demanding payment of arrears." → {{"label": "P", "confidence": 0.90, "logits": [0.01, 0.01, 0.00, 0.90, 0.05, 0.03]}}'
            ],
            "R": [
                '"Residents beat and killed 1 civilian from Ngouma in Tchika, accusing the victim of witchcraft." → {{"label": "R", "confidence": 0.79, "logits": [0.12, 0.03, 0.01, 0.03, 0.79, 0.02]}}',
                '"A mob attacked and burned shops owned by foreigners in Garoua following a dispute." → {{"label": "R", "confidence": 0.84, "logits": [0.08, 0.02, 0.02, 0.02, 0.84, 0.02]}}',
                '"Angry youths vandalized government buildings in Buea after a controversial election result." → {{"label": "R", "confidence": 0.81, "logits": [0.05, 0.02, 0.02, 0.08, 0.81, 0.02]}}',
                '"Residents clashed with police in Edéa, destroying vehicles and blocking roads." → {{"label": "R", "confidence": 0.83, "logits": [0.06, 0.04, 0.01, 0.04, 0.83, 0.02]}}',
                '"A violent crowd looted stores and set fire to a police post in Nkongsamba." → {{"label": "R", "confidence": 0.80, "logits": [0.09, 0.03, 0.02, 0.03, 0.80, 0.03]}}'
            ],
            "S": [
                '"Military forces arrested several civilians suspected of connection with ISWAP or Boko Haram militants in Djakana." → {{"label": "S", "confidence": 0.77, "logits": [0.10, 0.05, 0.02, 0.02, 0.04, 0.77]}}',
                '"Government troops increased patrols and established new checkpoints in the Anglophone regions." → {{"label": "S", "confidence": 0.85, "logits": [0.04, 0.03, 0.02, 0.02, 0.04, 0.85]}}',
                '"Security forces conducted a cordon-and-search operation in Mokolo, detaining suspected militants." → {{"label": "S", "confidence": 0.81, "logits": [0.07, 0.05, 0.02, 0.01, 0.04, 0.81]}}',
                '"The army deployed additional personnel to the Far North region to counter insurgent threats." → {{"label": "S", "confidence": 0.83, "logits": [0.05, 0.04, 0.03, 0.01, 0.04, 0.83]}}',
                '"Authorities imposed a curfew in several towns following reports of separatist activity." → {{"label": "S", "confidence": 0.79, "logits": [0.06, 0.04, 0.02, 0.03, 0.06, 0.79]}}'
            ]
        }
        
        # Build example string
        examples_lines = ["You are provided the following example classifications:"]
        for category in ["V", "B", "E", "P", "R", "S"]:
            for i in range(min(n_per_category, len(example_pool[category]))):
                examples_lines.append(f"- {example_pool[category][i]}")
        
        return "\n".join(examples_lines)
    
    def make_prompt(self, event_note: str) -> str:
        """Generate few-shot classification prompt with examples.
        
        Args:
            event_note: Event description text to classify
            
        Returns:
            Formatted prompt with examples and classification request
        """
        examples = self._get_examples(self.examples_per_category)

        return f"""You are an expert political conflict event analyst. {examples}

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
        """Get JSON schema for few-shot responses.
        
        Returns:
            JSON schema expecting label, confidence, and logits
        """
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "confidence": {"type": "number"},
                "logits": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["label", "confidence", "logits"]
        }
    
    def get_system_message(self) -> Optional[str]:
        """Get system message for few-shot strategy.
        
        Returns:
            System message explaining the classification task
        """
        return "You are an expert political conflict event analyst. Classify events based on provided examples."
