"""Narration: recorded events in, sentences out."""

from .diagnostics import Diagnostic, diagnose, diagnoses, unscaled_features
from .registry import (
    MetricChanged,
    NarrationContext,
    Narrator,
    StepNarration,
    derives,
    narrates,
    narrates_rest,
    summarizes,
)

__all__ = [
    "Diagnostic",
    "MetricChanged",
    "NarrationContext",
    "Narrator",
    "StepNarration",
    "derives",
    "diagnose",
    "diagnoses",
    "narrates",
    "narrates_rest",
    "summarizes",
    "unscaled_features",
]
