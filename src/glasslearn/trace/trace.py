"""The frozen record of one training run."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, overload

import numpy as np

from .events import event_from_dict, event_type_name
from .step import Step

_FORMAT_VERSION = 1


@dataclass(eq=False)
class Trace:
    """Everything an estimator recorded while it was fitted.

    A trace is a list of :class:`Step` objects plus what is needed to draw
    them: the training data and the feature names. It is stored on fitted
    estimators as ``trace_``.

    Parameters
    ----------
    algorithm : str
        Estimator class name, for example ``"KMeans"``.
    params : dict
        The estimator's constructor parameters at fit time.
    feature_names : list of str
        One name per column of ``X``.
    steps : list of Step
        The recorded steps, in order.
    target_names : list of str or None, default=None
        Class names for supervised estimators.
    X, y : ndarray or None, default=None
        Training data the steps refer to (None when recording was off).
    meta : dict, default={}
        Run-level facts, such as the inertia of every k-means restart.
    """

    algorithm: str
    params: dict[str, Any]
    feature_names: list[str]
    steps: list[Step]
    target_names: list[str] | None = None
    X: np.ndarray | None = None
    y: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

    @overload
    def __getitem__(self, index: int) -> Step: ...
    @overload
    def __getitem__(self, index: slice) -> list[Step]: ...
    def __getitem__(self, index: int | slice) -> Step | list[Step]:
        return self.steps[index]

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __repr__(self) -> str:
        return f"Trace(algorithm={self.algorithm!r}, steps={len(self.steps)})"

    @property
    def phases(self) -> list[str]:
        """The phase of every step, in order."""
        return [s.phase for s in self.steps]

    def metric(self, name: str) -> np.ndarray:
        """Return one metric for every step, with NaN where a step lacks it."""
        return np.array([s.metrics.get(name, np.nan) for s in self.steps], dtype=float)

    def latest(self, key: str, index: int) -> Any:
        """Return the most recent value of a state key at or before step ``index``.

        Use this for keys that are stored only when they change, such as labels.
        """
        if index < 0:
            index += len(self.steps)
        for step in reversed(self.steps[: index + 1]):
            if key in step.state:
                return step.state[key]
        raise KeyError(f"no step at or before {index} has state {key!r}")

    def to_frame(self):
        """Return one row per step (index, phase, metrics, events) as a DataFrame."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - pandas is a dev dependency
            raise ImportError(
                "Trace.to_frame() needs pandas: pip install pandas"
            ) from exc
        metric_names = list(dict.fromkeys(k for s in self.steps for k in s.metrics))
        frame = pd.DataFrame(
            {
                "phase": self.phases,
                **{name: self.metric(name) for name in metric_names},
                "events": [", ".join(e.kind for e in s.events) for s in self.steps],
            },
            index=pd.RangeIndex(len(self.steps), name="step"),
        )
        return frame

    def save(self, path: str | Path) -> None:
        """Save the trace to a ``.npz`` file (arrays plus a JSON header)."""
        arrays: dict[str, np.ndarray] = {}
        header = {
            "format": _FORMAT_VERSION,
            "algorithm": self.algorithm,
            "params": _encode(self.params, arrays),
            "feature_names": list(self.feature_names),
            "target_names": self.target_names,
            "X": _encode(self.X, arrays),
            "y": _encode(self.y, arrays),
            "meta": _encode(self.meta, arrays),
            "steps": [
                {
                    "phase": s.phase,
                    "state": _encode(s.state, arrays),
                    "metrics": s.metrics,
                    "events": [
                        {"type": event_type_name(type(e)), "fields": e.to_dict()}
                        for e in s.events
                    ],
                }
                for s in self.steps
            ],
        }
        arrays["__header__"] = np.array(json.dumps(header))
        with open(path, "wb") as fh:
            np.savez_compressed(fh, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Trace:
        """Load a trace written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        header = json.loads(str(arrays.pop("__header__")))
        if header.get("format") != _FORMAT_VERSION:
            raise ValueError(f"unsupported trace format {header.get('format')!r}")
        steps = []
        for i, s in enumerate(header["steps"]):
            state = _decode(s["state"], arrays)
            steps.append(
                Step(
                    index=i,
                    phase=s["phase"],
                    state=state,
                    metrics={k: float(v) for k, v in s["metrics"].items()},
                    events=tuple(
                        event_from_dict(e["type"], e["fields"]) for e in s["events"]
                    ),
                )
            )
        return cls(
            algorithm=header["algorithm"],
            params=_decode(header["params"], arrays),
            feature_names=list(header["feature_names"]),
            target_names=header["target_names"],
            steps=steps,
            X=_decode(header["X"], arrays),
            y=_decode(header["y"], arrays),
            meta=_decode(header["meta"], arrays),
        )


def _encode(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    """Turn a value into JSON, moving numpy arrays into ``arrays``."""
    if isinstance(value, np.ndarray):
        key = f"a{len(arrays)}"
        arrays[key] = value
        return {"__ndarray__": key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _encode(v, arrays) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v, arrays) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"__repr__": repr(value)}


def _decode(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, dict):
        if "__ndarray__" in value:
            arr = arrays[value["__ndarray__"]]
            arr.setflags(write=False)
            return arr
        if "__repr__" in value:
            return value["__repr__"]
        return {k: _decode(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v, arrays) for v in value]
    return value
