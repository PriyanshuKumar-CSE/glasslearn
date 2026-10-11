"""One recorded training step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from .events import Event

#: Allowed values for :attr:`Step.phase`.
PHASES = ("init", "assign", "update", "epoch", "split", "done")

E = TypeVar("E", bound=Event)


@dataclass(frozen=True, eq=False)
class Step:
    """A snapshot of an algorithm between two moments of training.

    Parameters
    ----------
    index : int
        Position of the step in its trace, starting at 0.
    phase : str
        What kind of step this is: one of ``"init"``, ``"assign"``,
        ``"update"``, ``"epoch"``, ``"split"`` or ``"done"``.
    state : dict
        The algorithm's state after the step (read-only numpy arrays).
    metrics : dict
        Scalar measurements after the step, such as ``{"inertia": 12.5}``.
    events : tuple of Event
        What happened during the step.
    """

    index: int
    phase: str
    state: dict[str, Any]
    metrics: dict[str, float]
    events: tuple[Event, ...]

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {self.phase!r}")

    def events_of(self, kind: type[E]) -> tuple[E, ...]:
        """Return this step's events of one type, in recorded order."""
        return tuple(e for e in self.events if isinstance(e, kind))

    def event(self, kind: type[E]) -> E | None:
        """Return the first event of one type, or None."""
        found = self.events_of(kind)
        return found[0] if found else None

    def __repr__(self) -> str:
        metrics = ", ".join(f"{k}={v:.4g}" for k, v in self.metrics.items())
        return (
            f"Step({self.index}, {self.phase!r}, state=[{', '.join(self.state)}], "
            f"metrics={{{metrics}}}, events={len(self.events)})"
        )


def same_value(a: Any, b: Any) -> bool:
    """Compare two state values, treating numpy arrays element-wise."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and bool(np.array_equal(a, b))
    return bool(a == b)
