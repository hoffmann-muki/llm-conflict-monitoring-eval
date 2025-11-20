"""Base class for prompting strategies."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class PromptingStrategy(ABC):
    """Abstract base class for all prompting strategies.
    
    Each strategy must implement make_prompt() to generate prompts
    and get_schema() to define expected response structure.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the prompting strategy.
        
        Args:
            config: Optional configuration dictionary for the strategy
        """
        self.config = config or {}
    
    @abstractmethod
    def make_prompt(self, event_note: str) -> str:
        """Generate a prompt for the given event note.
        
        Args:
            event_note: Event description text to classify
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the expected JSON schema for responses.
        
        Returns:
            JSON schema dictionary
        """
        pass
    
    @abstractmethod
    def get_system_message(self) -> Optional[str]:
        """Get the system message for this strategy.
        
        Returns:
            System message string or None
        """
        pass
    
    def get_name(self) -> str:
        """Get the strategy name (used for results organization).
        
        Returns:
            Strategy name (e.g., 'zero_shot', 'few_shot', 'explainable')
        """
        return self.__class__.__name__.replace('Strategy', '').lower()
