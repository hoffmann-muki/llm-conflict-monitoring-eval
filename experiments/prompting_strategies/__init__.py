"""Prompting strategies for classification experiments.

This module defines different prompting approaches:
- zero_shot: Direct classification without examples
- few_shot: Classification with example demonstrations (future)
- explainable: Chain-of-thought reasoning prompts (future)
"""

from .base import PromptingStrategy
from .zero_shot import ZeroShotStrategy

__all__ = ['PromptingStrategy', 'ZeroShotStrategy']
