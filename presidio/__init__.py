"""
Presidio PII Anonymization Module for PromptForge

This module provides PII detection, anonymization, and de-anonymization
capabilities using Microsoft Presidio, integrated with PromptForge's
agent development framework.
"""

from .policies import PIIAction, PIIPolicy, PIIPolicyEngine
from .middleware import PresidioMiddleware
from .quality_checks import PIIQualityEvaluator

__version__ = "1.0.0"
__all__ = [
    "PIIAction",
    "PIIPolicy", 
    "PIIPolicyEngine",
    "PresidioMiddleware",
    "PIIQualityEvaluator"
]