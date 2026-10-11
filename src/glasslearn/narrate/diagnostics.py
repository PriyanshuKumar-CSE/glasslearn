"""Rule-based warnings about a training run, each with a suggested fix."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from ..trace import Trace
from .format import number


class Diagnostic(NamedTuple):
    """A problem spotted in a trace.

    Attributes
    ----------
    text : str
        What went wrong, in plain words.
    fix : str
        What to try instead.
    step : int or None
        First step where the problem shows (None: the whole run).
    """

    text: str
    fix: str
    step: int | None = None


Rule = Callable[[Trace], list[Diagnostic]]

_RULES: dict[str, list[Rule]] = {}


def diagnoses(algorithm: str) -> Callable[[Rule], Rule]:
    """Register a rule ``trace -> list[Diagnostic]`` for one algorithm."""

    def register(rule: Rule) -> Rule:
        _RULES.setdefault(algorithm, []).append(rule)
        return rule

    return register


def diagnose(trace: Trace) -> list[Diagnostic]:
    """Run every rule registered for the trace's algorithm."""
    found: list[Diagnostic] = []
    for rule in _RULES.get(trace.algorithm, []):
        found.extend(rule(trace))
    return found


def unscaled_features(trace: Trace, ratio: float = 10.0) -> list[Diagnostic]:
    """Warn when one feature's spread is more than ``ratio`` times another's.

    Distance-based algorithms then mostly see the widest feature.
    """
    if trace.X is None or trace.X.shape[1] < 2:
        return []
    std = trace.X.std(axis=0)
    varying = np.flatnonzero(std > 0)
    if varying.size < 2:
        return []
    widest = varying[np.argmax(std[varying])]
    narrowest = varying[np.argmin(std[varying])]
    times = std[widest] / std[narrowest]
    if times <= ratio:
        return []
    names = trace.feature_names
    return [
        Diagnostic(
            text=(
                f"The features are on very different scales: {names[widest]} "
                f"spreads {number(times)} times more than {names[narrowest]} "
                f"(standard deviation {number(std[widest])} vs "
                f"{number(std[narrowest])}), so distances mostly measure "
                f"{names[widest]}."
            ),
            fix=(
                "Scale the features first, for example with "
                "make_pipeline(StandardScaler(), model)."
            ),
        )
    ]
