"""Global defaults for ``explain()``."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

LEVELS = ("beginner", "detailed", "math")

_config: dict[str, Any] = {
    "level": "beginner",
    "max_points": 2000,
    "static": None,
}


def get_config() -> dict[str, Any]:
    """Return a copy of the current settings."""
    return dict(_config)


def set_config(
    *,
    level: str | None = None,
    max_points: int | None = None,
    static: bool | None = None,
) -> None:
    """Change the defaults used by ``explain()``.

    Parameters
    ----------
    level : {"beginner", "detailed", "math"}, optional
        Default narration level.
    max_points : int, optional
        Most points drawn in a view (sampled per label above that).
    static : bool, optional
        Force the static plotly figure (True) or the live widget (False).
        ``None`` picks automatically.
    """
    if level is not None:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        _config["level"] = level
    if max_points is not None:
        if max_points < 1:
            raise ValueError(f"max_points must be >= 1, got {max_points}")
        _config["max_points"] = int(max_points)
    if static is not None:
        _config["static"] = bool(static)


@contextmanager
def config_context(**changes: Any) -> Iterator[None]:
    """Temporarily change settings inside a ``with`` block."""
    saved = get_config()
    set_config(**changes)
    try:
        yield
    finally:
        _config.clear()
        _config.update(saved)
