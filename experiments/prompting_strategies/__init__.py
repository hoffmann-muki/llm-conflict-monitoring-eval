"""Prompting strategies for classification experiments.

This module defines different prompting approaches:
- zero_shot: Direct classification without examples
- few_shot: Classification with example demonstrations
- explainable: Chain-of-thought reasoning prompts
"""

from .base import PromptingStrategy
from .zero_shot import ZeroShotStrategy
from .few_shot import FewShotStrategy
from .explainable import ExplainableStrategy

__all__ = ['PromptingStrategy', 'ZeroShotStrategy', 'FewShotStrategy', 'ExplainableStrategy']
