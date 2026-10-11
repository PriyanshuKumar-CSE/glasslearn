"""Turns recorded events into sentences.

Algorithms register templates with decorators in their ``narrator.py``:

- ``@narrates(EventType)``: ``(event, ctx) -> str | None`` for one event.
- ``@narrates_rest(EventType)``: ``(events, ctx) -> str | None`` to sum up the
  events of that type that did not make the top of the ranking.
- ``@derives("Algorithm")``: ``(ctx) -> list[Event]`` for comparisons with the
  previous step that are not recorded as events (for example a metric drop).
- ``@summarizes("Algorithm")``: ``(ctx) -> str`` for the whole run.

Templates only read the trace: every number in a sentence comes from it.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._config import LEVELS
from ..trace import Event, Step, Trace
from . import format as fmt
from .diagnostics import Diagnostic, diagnose
from .glossary import define, terms_in

Template = Callable[[Any, "NarrationContext"], "str | None"]
RestTemplate = Callable[[list[Any], "NarrationContext"], "str | None"]

#: Events below this salience are only summed up with others of their type.
BACKGROUND = 0.15

_TEMPLATES: dict[tuple[str | None, type], Template] = {}
_REST_TEMPLATES: dict[tuple[str | None, type], RestTemplate] = {}
_DERIVERS: dict[str, Callable[[NarrationContext], list[Event]]] = {}
_SUMMARIES: dict[str, Callable[[NarrationContext], str]] = {}


def narrates(event_type: type[Event], *, algorithm: str | None = None):
    """Register the sentence template for one event type.

    Pass ``algorithm`` when a shared event type (such as
    :class:`MetricChanged`) needs wording specific to one algorithm.
    """

    def register(fn: Template) -> Template:
        _TEMPLATES[(algorithm, event_type)] = fn
        return fn

    return register


def narrates_rest(event_type: type[Event], *, algorithm: str | None = None):
    """Register the template that sums up leftover events of one type."""

    def register(fn: RestTemplate) -> RestTemplate:
        _REST_TEMPLATES[(algorithm, event_type)] = fn
        return fn

    return register


def derives(algorithm: str):
    """Register a function that derives extra events from the step and its previous."""

    def register(fn):
        _DERIVERS[algorithm] = fn
        return fn

    return register


def summarizes(algorithm: str):
    """Register the whole-run summary for one algorithm."""

    def register(fn):
        _SUMMARIES[algorithm] = fn
        return fn

    return register


def _lookup(table: dict, algorithm: str, event_type: type) -> Any:
    for cls in event_type.__mro__:
        for key in ((algorithm, cls), (None, cls)):
            if key in table:
                return table[key]
    return None


def load_narrator(trace: Trace) -> None:
    """Import the ``narrator`` module next to the estimator that made ``trace``."""
    module = trace.meta.get("estimator_module", "")
    if module.startswith("glasslearn."):
        importlib.import_module(module.rsplit(".", 1)[0] + ".narrator")


@dataclass(frozen=True)
class MetricChanged(Event):
    """A metric changed since the previous step (derived by narration, not recorded)."""

    metric: str
    old: float
    new: float


@dataclass(frozen=True)
class StepNarration:
    """The text for one step.

    Attributes
    ----------
    sentences : tuple of str
        What happened, most important first, causes before effects.
    glossary : tuple of (str, str)
        ``(term, "Term: definition")`` for terms used here for the first time.
    """

    sentences: tuple[str, ...]
    glossary: tuple[tuple[str, str], ...] = ()

    @property
    def text(self) -> str:
        """The sentences joined into one paragraph."""
        return " ".join(self.sentences)


class NarrationContext:
    """What a template can use: the step, its predecessor, the level and helpers.

    Parameters
    ----------
    trace : Trace
        The whole run.
    level : {"beginner", "detailed", "math"}
        How technical the wording is.
    step, prev : Step or None
        The step being narrated and the one before it.
    project : callable, optional
        Maps feature-space rows ``(n, n_features)`` to the plotted plane
        ``(n, 2)``. Used for directions such as "up and to the right".
    """

    number = staticmethod(fmt.number)
    signed = staticmethod(fmt.signed)
    percent = staticmethod(fmt.percent)
    plural = staticmethod(fmt.plural)
    join = staticmethod(fmt.join_words)
    numbers = staticmethod(fmt.numbers)
    distinct = staticmethod(fmt.distinct)

    def __init__(
        self,
        trace: Trace,
        level: str = "beginner",
        step: Step | None = None,
        prev: Step | None = None,
        project: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        self.trace = trace
        self.level = level
        self.step = step
        self.prev = prev
        self._project = project

    @property
    def beginner(self) -> bool:
        return self.level == "beginner"

    @property
    def detailed(self) -> bool:
        return self.level == "detailed"

    @property
    def math(self) -> bool:
        return self.level == "math"

    @property
    def feature_names(self) -> list[str]:
        return self.trace.feature_names

    @property
    def params(self) -> dict[str, Any]:
        return self.trace.params

    @property
    def scale(self) -> float:
        """Typical spread of the training data (mean feature standard deviation)."""
        if self.trace.X is None:
            return 1.0
        spread = float(np.mean(self.trace.X.std(axis=0)))
        return spread if spread > 0 else 1.0

    def metric(self, name: str, step: Step | None = None) -> float | None:
        """A metric of ``step`` (default: the current step), or None."""
        step = self.step if step is None else step
        if step is None or name not in step.metrics:
            return None
        return step.metrics[name]

    def to_plane(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Positions of feature-space rows in the plotted plane, shape ``(n, 2)``."""
        rows = np.atleast_2d(np.asarray(points, dtype=float))
        if self._project is not None:
            return np.asarray(self._project(rows), dtype=float)
        if rows.shape[1] == 1:
            return np.column_stack([rows[:, 0], np.zeros(len(rows))])
        return rows[:, :2]

    def direction(self, old: Sequence[float], new: Sequence[float]) -> str:
        """Where a move from ``old`` to ``new`` goes on the plot.

        Returns one of "to the right", "up and to the right", "up", ...,
        or "nowhere" when the plotted position does not change.
        """
        (x0, y0), (x1, y1) = self.to_plane([old, new])
        dx, dy = x1 - x0, y1 - y0
        if math.hypot(dx, dy) <= 1e-12 * self.scale:
            return "nowhere"
        names = [
            "to the right",
            "up and to the right",
            "up",
            "up and to the left",
            "to the left",
            "down and to the left",
            "down",
            "down and to the right",
        ]
        sector = round(math.degrees(math.atan2(dy, dx)) / 45) % 8
        return names[sector]

    def biggest_change(
        self, old: Sequence[float], new: Sequence[float]
    ) -> tuple[str, float]:
        """The feature that changed most between two positions, and by how much."""
        delta = np.asarray(new, dtype=float) - np.asarray(old, dtype=float)
        j = int(np.argmax(np.abs(delta)))
        return self.feature_names[j], float(delta[j])


class Narrator:
    """Explains a trace step by step, in words.

    Parameters
    ----------
    trace : Trace
        A fitted estimator's ``trace_``.
    level : {"beginner", "detailed", "math"}, default="beginner"
        How technical the wording is.
    projection : object or callable, optional
        The plot's projection (anything with ``transform`` or a callable),
        so that directions match what is drawn. Defaults to the first two
        features.
    max_events : int, default=3
        Events narrated individually per step; the rest are summed up.
    """

    def __init__(
        self,
        trace: Trace,
        *,
        level: str = "beginner",
        projection: Any = None,
        max_events: int = 3,
    ) -> None:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        load_narrator(trace)
        self.trace = trace
        self.level = level
        self.max_events = max_events
        if projection is not None and hasattr(projection, "transform"):
            projection = projection.transform
        self._project = projection
        self._cache: dict[int, tuple[str, ...]] = {}
        self._first_use: dict[str, int] | None = None

    def context(
        self, step: Step | None = None, prev: Step | None = None
    ) -> NarrationContext:
        """A template context for one step, or for the whole run if ``step`` is None."""
        return NarrationContext(self.trace, self.level, step, prev, self._project)

    def _step_and_prev(self, step: int | Step) -> tuple[Step, Step | None]:
        if not isinstance(step, Step):
            step = self.trace[step]
        prev = self.trace[step.index - 1] if step.index > 0 else None
        return step, prev

    def sentences(self, step: int | Step) -> tuple[str, ...]:
        """The sentences for one step, without glossary entries."""
        step, prev = self._step_and_prev(step)
        if step.index in self._cache:
            return self._cache[step.index]
        algorithm = self.trace.algorithm
        ctx = self.context(step, prev)

        recorded = sorted(step.events, key=lambda e: -e.salience)  # stable
        derived = _DERIVERS[algorithm](ctx) if algorithm in _DERIVERS else []
        texts: dict[int, str] = {}
        rest: list[Event] = []
        for event in sorted([*recorded, *derived], key=lambda e: -e.salience):
            template = _lookup(_TEMPLATES, algorithm, type(event))
            if template is None:
                continue
            if len(texts) < self.max_events and event.salience >= BACKGROUND:
                text = template(event, ctx)
                if text:  # a template may decline; the next event takes the slot
                    texts[id(event)] = text
            else:
                rest.append(event)

        rest_text = None
        if rest:
            kind = type(rest[0])
            template = _lookup(_REST_TEMPLATES, algorithm, kind)
            if template is not None:
                rest_text = template([e for e in rest if type(e) is kind], ctx)
        # Causes (recorded, by salience), then the line summing up the rest,
        # then effects derived from comparing with the previous step.
        ordered = [texts[id(e)] for e in recorded if id(e) in texts]
        ordered.append(rest_text)
        ordered += [texts[id(e)] for e in derived if id(e) in texts]
        # Math notation is case-sensitive (c_j is a centroid, C_j its cluster).
        tidy = (lambda t: t) if self.level == "math" else fmt.capitalize
        result = tuple(tidy(t) for t in ordered if t)
        self._cache[step.index] = result
        return result

    def _glossary_index(self) -> dict[str, int]:
        if self._first_use is None:
            first: dict[str, int] = {}
            for step in self.trace:
                for term in terms_in(" ".join(self.sentences(step))):
                    first.setdefault(term, step.index)
            self._first_use = first
        return self._first_use

    def narrate(self, step: int | Step) -> StepNarration:
        """Return the narration for one step (an index or a :class:`Step`).

        Glossary entries are attached to the step where each term first appears,
        so the result does not depend on the order steps are viewed in.
        """
        step, _ = self._step_and_prev(step)
        sentences = self.sentences(step)
        glossary: tuple[tuple[str, str], ...] = ()
        if self.level != "math":
            first = self._glossary_index()
            glossary = tuple(
                (term, define(term))
                for term in terms_in(" ".join(sentences))
                if first.get(term) == step.index
            )
        return StepNarration(sentences, glossary)

    def summary(self) -> str:
        """One or two sentences about the whole run."""
        fn = _SUMMARIES.get(self.trace.algorithm)
        if fn is None or len(self.trace) == 0:
            steps = fmt.plural(len(self.trace), "step")
            return f"{self.trace.algorithm} recorded {steps}."
        text = fn(self.context())
        return text if self.level == "math" else fmt.capitalize(text)

    def diagnostics(self, upto: int | None = None) -> list[Diagnostic]:
        """Problems spotted in the run; with ``upto``, those visible by that step."""
        found = diagnose(self.trace)
        if upto is None:
            return found
        return [d for d in found if d.step is None or d.step <= upto]
