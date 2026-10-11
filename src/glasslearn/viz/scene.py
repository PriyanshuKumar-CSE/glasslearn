"""What a view draws, independent of any plotting library.

A view turns one recorded step into a :class:`Scene`: an ordered list of
layers. For a given view the number and order of layers never change between
steps, so a backend can update an existing figure in place.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Colour for each cluster or class index (cycled when there are more).
PALETTE = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#EECA3B",
    "#9D755D",
    "#FF9DA6",
    "#79706E",
)
#: Colour for points that have no group yet (for example before the first assignment).
UNASSIGNED = "#B8B8B8"


def color_of(group: int) -> str:
    """Palette colour for a group index; negative means unassigned."""
    return UNASSIGNED if group < 0 else PALETTE[group % len(PALETTE)]


@dataclass
class Points:
    """Data points, coloured by group.

    ``group`` holds one integer per point (-1 for "no group yet"). ``hover``
    is fixed text per point; the backend adds ``"<group_name> <group>"``.
    """

    x: np.ndarray
    y: np.ndarray
    group: np.ndarray
    hover: list[str] | None = None
    group_name: str = "group"
    name: str = "points"
    size: float = 7.0
    opacity: float = 0.75


@dataclass
class Markers:
    """A few large symbols, such as centroids, each with its own group colour."""

    x: np.ndarray
    y: np.ndarray
    group: np.ndarray
    labels: list[str] = field(default_factory=list)
    name: str = "markers"
    symbol: str = "x"
    size: float = 18.0


@dataclass
class Arrows:
    """Arrows from ``(x0, y0)`` to ``(x1, y1)``; may be empty on some steps."""

    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    group: np.ndarray
    name: str = "arrows"


@dataclass
class Lines:
    """Polylines, one per group; ``paths[i]`` is an ``(n, 2)`` array (may be empty)."""

    paths: list[np.ndarray]
    group: np.ndarray
    name: str = "lines"
    opacity: float = 0.35
    dash: str = "dot"


Layer = Points | Markers | Arrows | Lines


@dataclass
class Scene:
    """Everything one step draws.

    Attributes
    ----------
    layers : list
        Drawn in order (first is at the back).
    x_title, y_title : str
        Axis titles.
    x_range, y_range : (float, float) or None
        Fixed axis ranges, so the plot does not jump between steps.
    caption : str
        Short text describing the step (used by static figures).
    """

    layers: list[Layer]
    x_title: str = ""
    y_title: str = ""
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None
    caption: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def padded_range(values: np.ndarray, pad: float = 0.06) -> tuple[float, float]:
    """``(min, max)`` of the finite values, widened by ``pad`` on each side."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo if hi > lo else max(abs(lo), 1.0)
    return (lo - pad * span, hi + pad * span)


def sample_per_group(
    groups: np.ndarray, max_points: int, random_state: int = 0
) -> np.ndarray:
    """Indices of at most ``max_points`` rows, keeping each group's share.

    Every group keeps at least one point. Returns sorted indices; all rows
    when there are no more than ``max_points``.
    """
    n = len(groups)
    if n <= max_points:
        return np.arange(n)
    rng = np.random.RandomState(random_state)
    values, counts = np.unique(groups, return_counts=True)
    quota = np.maximum(1, np.floor(counts * max_points / n)).astype(int)
    # Hand out what rounding left over to the biggest groups.
    for i in np.argsort(-counts)[: max(0, max_points - int(quota.sum()))]:
        quota[i] += 1
    chosen = [
        rng.choice(np.flatnonzero(groups == v), size=min(q, c), replace=False)
        for v, q, c in zip(values, quota, counts, strict=True)
    ]
    return np.sort(np.concatenate(chosen))


_VIEWS: dict[str, type[View]] = {}


def view_for(algorithm: str):
    """Class decorator: register the view that draws one algorithm's traces."""

    def register(cls: type[View]) -> type[View]:
        _VIEWS[algorithm] = cls
        return cls

    return register


def make_view(trace: Any, **options: Any) -> View:
    """Build the registered view for a trace (importing it next to its estimator)."""
    module = trace.meta.get("estimator_module", "")
    if module.startswith("glasslearn."):
        importlib.import_module(module.rsplit(".", 1)[0] + ".view")
    if trace.algorithm not in _VIEWS:
        raise ValueError(f"No view is registered for {trace.algorithm}.")
    return _VIEWS[trace.algorithm](trace, **options)


class View:
    """Base class for algorithm views: one trace in, one :class:`Scene` per step.

    Subclasses implement :meth:`render`. ``max_points`` limits how many data
    points are drawn; the same sample is used for every step.
    """

    def __init__(self, trace: Any, *, max_points: int = 2000) -> None:
        self.trace = trace
        self.max_points = int(max_points)

    @property
    def n_total(self) -> int:
        """Rows in the training data."""
        return 0 if self.trace.X is None else len(self.trace.X)

    @property
    def n_shown(self) -> int:
        """Rows actually drawn."""
        return self.n_total

    def render(self, step: Any, projection: Any) -> Scene:
        raise NotImplementedError
