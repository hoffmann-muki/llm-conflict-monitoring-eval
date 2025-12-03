"""Few-shot prompting strategy with example demonstrations.

Provides example demonstrations for each category before asking for classification.
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
            Formatted string with example input/output pairs
        """
        # Example pool: up to 5 examples per category
        # Each tuple: (event_description, label, confidence, logits_dict)
        example_pool = {
            "V": [
                ("Military forces shot and injured a woman in Nongomadiba when they fired shots at a building they believed to be holding Ambazonian Separatists.", "V", 0.89, {"V": 0.89, "B": 0.05, "E": 0.02, "P": 0.01, "R": 0.02, "S": 0.01}),
                ("Security forces opened fire on civilians during a raid in Bamenda, killing two people.", "V", 0.94, {"V": 0.94, "B": 0.03, "E": 0.01, "P": 0.01, "R": 0.01, "S": 0.00}),
                ("Armed soldiers beat and detained three civilians suspected of supporting separatists in Kumbo.", "V", 0.86, {"V": 0.86, "B": 0.02, "E": 0.01, "P": 0.02, "R": 0.03, "S": 0.06}),
                ("Police forces tortured detainees at a checkpoint in Buea, injuring five individuals.", "V", 0.91, {"V": 0.91, "B": 0.03, "E": 0.01, "P": 0.01, "R": 0.02, "S": 0.02}),
                ("Military personnel looted civilian homes and assaulted residents in Mamfe.", "V", 0.87, {"V": 0.87, "B": 0.02, "E": 0.01, "P": 0.02, "R": 0.05, "S": 0.03}),
            ],
            "B": [
                ("The police forces killed one suspected Boko Haram fighter and arrested another in Aissa Karde village.", "B", 0.82, {"V": 0.08, "B": 0.82, "E": 0.03, "P": 0.01, "R": 0.02, "S": 0.04}),
                ("Military forces clashed with separatist fighters in Belo, resulting in casualties on both sides.", "B", 0.93, {"V": 0.03, "B": 0.93, "E": 0.02, "P": 0.01, "R": 0.01, "S": 0.00}),
                ("Government troops engaged Boko Haram militants near Fotokol, killing several insurgents.", "B", 0.91, {"V": 0.04, "B": 0.91, "E": 0.03, "P": 0.00, "R": 0.01, "S": 0.01}),
                ("Armed forces exchanged fire with rebel groups in the Northwest region for several hours.", "B", 0.89, {"V": 0.05, "B": 0.89, "E": 0.03, "P": 0.01, "R": 0.01, "S": 0.01}),
                ("Security forces raided a separatist hideout in Kumba, killing three fighters.", "B", 0.85, {"V": 0.06, "B": 0.85, "E": 0.02, "P": 0.01, "R": 0.02, "S": 0.04}),
            ],
            "E": [
                ("An IED planted by suspected Ambazonian separatists detonated in Matezen village, Santa subdivision, injuring three people.", "E", 0.96, {"V": 0.02, "B": 0.01, "E": 0.96, "P": 0.00, "R": 0.01, "S": 0.00}),
                ("A roadside bomb exploded near a military convoy in Kolofata, wounding two soldiers.", "E", 0.95, {"V": 0.02, "B": 0.02, "E": 0.95, "P": 0.00, "R": 0.01, "S": 0.00}),
                ("Unidentified militants launched a mortar attack on a police station in Mora.", "E", 0.92, {"V": 0.03, "B": 0.03, "E": 0.92, "P": 0.00, "R": 0.01, "S": 0.01}),
                ("An explosive device detonated at a market in Maroua, killing one civilian and injuring ten.", "E", 0.94, {"V": 0.03, "B": 0.01, "E": 0.94, "P": 0.01, "R": 0.01, "S": 0.00}),
                ("Suspected insurgents fired rockets at a military base in the Far North region.", "E", 0.93, {"V": 0.02, "B": 0.03, "E": 0.93, "P": 0.00, "R": 0.01, "S": 0.01}),
            ],
            "P": [
                ("About a hundred residents demonstrated in Djoum town against the government's delay in compensating them after destroying their houses to build the Bikouna-Djoum road.", "P", 0.88, {"V": 0.02, "B": 0.01, "E": 0.01, "P": 0.88, "R": 0.06, "S": 0.02}),
                ("Teachers held a peaceful protest in Yaoundé demanding better salaries and working conditions.", "P", 0.95, {"V": 0.01, "B": 0.00, "E": 0.00, "P": 0.95, "R": 0.03, "S": 0.01}),
                ("Students marched through Douala to protest tuition fee increases at public universities.", "P", 0.93, {"V": 0.01, "B": 0.01, "E": 0.00, "P": 0.93, "R": 0.04, "S": 0.01}),
                ("Civil society groups organized a demonstration in Bamenda calling for dialogue and peace.", "P", 0.91, {"V": 0.02, "B": 0.01, "E": 0.01, "P": 0.91, "R": 0.04, "S": 0.01}),
                ("Healthcare workers staged a sit-in at the Ministry of Health demanding payment of arrears.", "P", 0.90, {"V": 0.01, "B": 0.01, "E": 0.00, "P": 0.90, "R": 0.05, "S": 0.03}),
            ],
            "R": [
                ("Residents beat and killed 1 civilian from Ngouma in Tchika, accusing the victim of witchcraft.", "R", 0.79, {"V": 0.12, "B": 0.03, "E": 0.01, "P": 0.03, "R": 0.79, "S": 0.02}),
                ("A mob attacked and burned shops owned by foreigners in Garoua following a dispute.", "R", 0.84, {"V": 0.08, "B": 0.02, "E": 0.02, "P": 0.02, "R": 0.84, "S": 0.02}),
                ("Angry youths vandalized government buildings in Buea after a controversial election result.", "R", 0.81, {"V": 0.05, "B": 0.02, "E": 0.02, "P": 0.08, "R": 0.81, "S": 0.02}),
                ("Residents clashed with police in Edéa, destroying vehicles and blocking roads.", "R", 0.83, {"V": 0.06, "B": 0.04, "E": 0.01, "P": 0.04, "R": 0.83, "S": 0.02}),
                ("A violent crowd looted stores and set fire to a police post in Nkongsamba.", "R", 0.80, {"V": 0.09, "B": 0.03, "E": 0.02, "P": 0.03, "R": 0.80, "S": 0.03}),
            ],
            "S": [
                ("Military forces arrested several civilians suspected of connection with ISWAP or Boko Haram militants in Djakana.", "S", 0.77, {"V": 0.10, "B": 0.05, "E": 0.02, "P": 0.02, "R": 0.04, "S": 0.77}),
                ("Government troops increased patrols and established new checkpoints in the Anglophone regions.", "S", 0.85, {"V": 0.04, "B": 0.03, "E": 0.02, "P": 0.02, "R": 0.04, "S": 0.85}),
                ("Security forces conducted a cordon-and-search operation in Mokolo, detaining suspected militants.", "S", 0.81, {"V": 0.07, "B": 0.05, "E": 0.02, "P": 0.01, "R": 0.04, "S": 0.81}),
                ("The army deployed additional personnel to the Far North region to counter insurgent threats.", "S", 0.83, {"V": 0.05, "B": 0.04, "E": 0.03, "P": 0.01, "R": 0.04, "S": 0.83}),
                ("Authorities imposed a curfew in several towns following reports of separatist activity.", "S", 0.79, {"V": 0.06, "B": 0.04, "E": 0.02, "P": 0.03, "R": 0.06, "S": 0.79}),
            ],
        }
        
        # Build example string - simple input/output format matching expected JSON structure
        import json
        examples_lines = []
        for category in ["V", "B", "E", "P", "R", "S"]:
            for i in range(min(n_per_category, len(example_pool[category]))):
                desc, label, conf, logits = example_pool[category][i]
                output = {"label": label, "confidence": conf, "logits": logits}
                examples_lines.append(f"Event: {desc}")
                examples_lines.append(f"{json.dumps(output)}")
                examples_lines.append("")
        
        return "\n".join(examples_lines)
    
    def make_prompt(self, event_note: str) -> str:
        """Generate few-shot classification prompt with examples.
        
        Args:
            event_note: Event description text to classify
            
        Returns:
            Formatted prompt with examples and classification request
        """
        examples = self._get_examples(self.examples_per_category)

        return f"""Examples:

    {examples}
    --- Now classify the following event in the same format ---
    Final Answer: JSON matching the examples above.

    Event: {event_note}
    """
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for few-shot responses.
        
        Returns:
            JSON schema expecting label, confidence, and logits
        """
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["V", "B", "E", "P", "R", "S"]},
                "confidence": {"type": "number"},
                "logits": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["label", "confidence", "logits"]
        }
    
    def get_system_message(self) -> Optional[str]:
        """Get system message for few-shot strategy.
        
        Returns:
            System message explaining the classification task and output format
        """
        return """You are an expert political conflict event analyst. Classify events into one of six categories:
- V = Violence against civilians
- B = Battles
- E = Explosions/Remote violence
- P = Protests
- R = Riots
- S = Strategic developments

Return JSON with: label (V/B/E/P/R/S), confidence (0-1), and logits (probability for each category)."""
