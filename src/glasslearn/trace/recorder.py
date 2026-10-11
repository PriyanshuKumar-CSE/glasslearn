"""Collects steps while an algorithm trains."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .events import Event
from .step import Step, same_value
from .trace import Trace

# Smallest first; for each size the unsigned type comes first.
_INT_DTYPES = (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.int64)


def compact_int(a: np.ndarray) -> np.ndarray:
    """Return an integer array cast to the smallest dtype that holds its values."""
    if a.size == 0:
        return a.astype(np.uint8)
    lo, hi = a.min(), a.max()
    for dtype in _INT_DTYPES:
        info = np.iinfo(dtype)
        if info.min <= lo and hi <= info.max:
            return a.astype(dtype)
    return a.copy()


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        arr = compact_int(value) if value.dtype.kind in "iu" else value.copy()
        arr.setflags(write=False)
        return arr
    if isinstance(value, np.generic):
        return value.item()
    return value


class Recorder:
    """Collects the steps of one training run.

    Algorithms call :meth:`step` once per meaningful moment (an assignment pass,
    an epoch, a split). The recorder copies the arrays it is given, so the
    algorithm can keep updating its own buffers.

    Parameters
    ----------
    enabled : bool, default=True
        When False, every method is a no-op and the frozen trace is empty.
    every : int, default=1
        Keep every ``every``-th step of each phase. ``"init"`` and ``"done"``
        steps are always kept.
    max_steps : int, default=500
        Stop keeping steps after this many. The final ``"done"`` step is still
        kept, so the trace always shows how training ended.
    sticky : tuple of str, default=()
        State keys that are stored only when their value changes (for example
        cluster labels). Read them back with :meth:`Trace.latest`.
    """

    def __init__(
        self,
        enabled: bool = True,
        every: int = 1,
        max_steps: int = 500,
        sticky: Iterable[str] = (),
    ) -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        if max_steps < 2:
            raise ValueError(f"max_steps must be >= 2, got {max_steps}")
        self.enabled = bool(enabled)
        self.every = int(every)
        self.max_steps = int(max_steps)
        self.sticky = tuple(sticky)
        self._steps: list[Step] = []
        self._seen: dict[str, int] = {}
        self._last_sticky: dict[str, Any] = {}
        self._notes: dict[str, Any] = {}
        self._n_dropped = 0

    def __len__(self) -> int:
        return len(self._steps)

    def step(
        self,
        phase: str,
        *,
        state: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float] | None = None,
        events: Iterable[Event] = (),
    ) -> None:
        """Record one step (a no-op when the recorder is disabled)."""
        if not self.enabled:
            return
        seen = self._seen.get(phase, 0)
        self._seen[phase] = seen + 1
        always = phase in ("init", "done")
        if not always and (seen % self.every or len(self._steps) >= self.max_steps - 1):
            self._n_dropped += 1
            return

        kept: dict[str, Any] = {}
        for key, value in (state or {}).items():
            if key in self.sticky:
                if key in self._last_sticky and same_value(
                    self._last_sticky[key], value
                ):
                    continue
                value = _freeze_value(np.asarray(value))
                self._last_sticky[key] = value
                kept[key] = value
            else:
                kept[key] = _freeze_value(value)
        self._steps.append(
            Step(
                index=len(self._steps),
                phase=phase,
                state=kept,
                metrics={k: float(v) for k, v in (metrics or {}).items()},
                events=tuple(events),
            )
        )

    def note(self, **values: Any) -> None:
        """Attach run-level facts (for example every restart's inertia) to the trace."""
        if self.enabled:
            self._notes.update(values)

    def spawn(self) -> Recorder:
        """Return an empty recorder with the same settings, for one restart."""
        return Recorder(self.enabled, self.every, self.max_steps, self.sticky)

    def adopt(self, other: Recorder) -> None:
        """Replace this recorder's steps with those of ``other`` (the best restart)."""
        self._steps = list(other._steps)
        self._seen = dict(other._seen)
        self._last_sticky = dict(other._last_sticky)
        self._n_dropped = other._n_dropped
        self._notes = {**other._notes, **self._notes}

    def freeze(
        self,
        *,
        algorithm: str,
        params: Mapping[str, Any],
        feature_names: Iterable[str],
        target_names: Iterable[str] | None = None,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Trace:
        """Return the finished, read-only :class:`Trace`.

        ``X`` and ``y`` are the training data the steps refer to; they are
        copied and only kept when recording is enabled.
        """
        info = dict(meta or {})
        if self.enabled:
            info.update(self._notes)
            info["sticky"] = list(self.sticky)
            info["n_dropped_steps"] = self._n_dropped
        return Trace(
            algorithm=algorithm,
            params={k: _freeze_value(v) for k, v in params.items()},
            feature_names=[str(n) for n in feature_names],
            target_names=None
            if target_names is None
            else [str(n) for n in target_names],
            steps=list(self._steps),
            X=_freeze_value(np.asarray(X)) if self.enabled and X is not None else None,
            y=_freeze_value(np.asarray(y)) if self.enabled and y is not None else None,
            meta=info,
        )
