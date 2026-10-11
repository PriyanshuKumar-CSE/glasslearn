"""Base class for the things an algorithm reports while it trains."""

from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

# Every Event subclass registers itself here as "module:qualname", so a saved
# trace can rebuild its events without importing arbitrary modules.
_EVENT_TYPES: dict[str, type[Event]] = {}


@dataclass(frozen=True)
class Event:
    """Something worth explaining that happened during one training step.

    Subclasses add the numbers that describe what happened. They must stay
    small (scalars and short tuples) because one is stored per step.

    Parameters
    ----------
    salience : float, default=0.5
        How important the event is, from 0 (background detail) to 1 (the
        headline of the step). Narration keeps the most salient events first.
    """

    salience: float = field(default=0.5, kw_only=True)

    #: Short name used by narration and diagnostics. Defaults to the class name.
    kind: ClassVar[str] = "Event"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "kind" not in cls.__dict__:
            cls.kind = cls.__name__
        _EVENT_TYPES[event_type_name(cls)] = cls

    def __post_init__(self) -> None:
        if not 0.0 <= self.salience <= 1.0:
            raise ValueError(f"salience must be in [0, 1], got {self.salience!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return the event's fields as JSON-friendly builtins."""
        return {
            f.name: _to_builtin(getattr(self, f.name)) for f in dataclasses.fields(self)
        }


def event_type_name(cls: type[Event]) -> str:
    """Return the stable "module:qualname" key used to save an event type."""
    return f"{cls.__module__}:{cls.__qualname__}"


def event_from_dict(type_name: str, data: dict[str, Any]) -> Event:
    """Rebuild an event saved with :meth:`Event.to_dict`.

    Event types defined inside glasslearn are imported on demand; any other
    type must already be imported (and therefore registered).
    """
    cls = _EVENT_TYPES.get(type_name)
    if cls is None:
        module, _, _ = type_name.partition(":")
        if module == "glasslearn" or module.startswith("glasslearn."):
            importlib.import_module(module)
            cls = _EVENT_TYPES.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown event type {type_name!r}; import it before loading.")
    return cls(**{k: _to_tuple(v) for k, v in data.items()})


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    return value


def _to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_to_tuple(v) for v in value)
    return value
