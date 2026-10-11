"""Draws a :class:`~glasslearn.viz.scene.Scene` with plotly.

Every layer becomes a fixed number of plotly traces, so moving to another
step only swaps data inside ``fig.batch_update()``.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from ..scene import Arrows, Lines, Markers, Points, Scene, color_of

ARROW_COLOR = "#333333"


def _colors(groups: np.ndarray) -> list[str]:
    return [color_of(int(g)) for g in groups]


def _points(layer: Points) -> dict[str, Any]:
    groups = np.asarray(layer.group)
    trace: dict[str, Any] = {
        "type": "scatter",
        "mode": "markers",
        "name": layer.name,
        "x": np.asarray(layer.x),
        "y": np.asarray(layer.y),
        "marker": {
            "color": _colors(groups),
            "size": layer.size,
            "opacity": layer.opacity,
            "line": {"width": 0},
        },
        "customdata": [
            f"{layer.group_name} {g}" if g >= 0 else f"no {layer.group_name} yet"
            for g in groups.tolist()
        ],
        "hovertemplate": "%{customdata}<extra></extra>",
    }
    if layer.hover is not None:
        trace["text"] = layer.hover
        trace["hovertemplate"] = "%{text}<br>%{customdata}<extra></extra>"
    return trace


def _markers(layer: Markers) -> dict[str, Any]:
    colors = _colors(np.asarray(layer.group))
    return {
        "type": "scatter",
        "mode": "markers+text",
        "name": layer.name,
        "x": np.asarray(layer.x),
        "y": np.asarray(layer.y),
        "text": list(layer.labels),
        "textposition": "top center",
        "textfont": {"size": 14, "color": colors},
        "marker": {
            "symbol": layer.symbol,
            "size": layer.size,
            "color": colors,
            "line": {"width": 1.5, "color": "#222222"},
        },
        "hovertemplate": f"{layer.name.rstrip('s')} %{{text}}<extra></extra>",
    }


def _arrows(layer: Arrows) -> dict[str, Any]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    sizes: list[float] = []
    for x0, y0, x1, y1 in zip(layer.x0, layer.y0, layer.x1, layer.y1, strict=True):
        xs += [float(x0), float(x1), None]
        ys += [float(y0), float(y1), None]
        sizes += [0.0, 12.0, 0.0]
    return {
        "type": "scatter",
        "mode": "lines+markers",
        "name": layer.name,
        "x": xs,
        "y": ys,
        "line": {"color": ARROW_COLOR, "width": 2},
        "marker": {
            "symbol": "arrow",
            "angleref": "previous",
            "size": sizes,
            "color": ARROW_COLOR,
        },
        "hoverinfo": "skip",
    }


def _lines(layer: Lines) -> list[dict[str, Any]]:
    traces = []
    for path, group in zip(layer.paths, np.asarray(layer.group), strict=True):
        path = np.asarray(path, dtype=float).reshape(-1, 2)
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": f"{layer.name} {int(group)}",
                "x": path[:, 0],
                "y": path[:, 1],
                "line": {
                    "color": color_of(int(group)),
                    "width": 1.5,
                    "dash": layer.dash,
                },
                "marker": {"size": 4, "color": color_of(int(group))},
                "opacity": layer.opacity,
                "hoverinfo": "skip",
            }
        )
    return traces


def traces_for(scene: Scene) -> list[dict[str, Any]]:
    """Plotly trace specs for every layer, in drawing order."""
    traces: list[dict[str, Any]] = []
    for layer in scene.layers:
        if isinstance(layer, Points):
            traces.append(_points(layer))
        elif isinstance(layer, Markers):
            traces.append(_markers(layer))
        elif isinstance(layer, Arrows):
            traces.append(_arrows(layer))
        elif isinstance(layer, Lines):
            traces.extend(_lines(layer))
        else:  # pragma: no cover - new layer types must be added here
            raise TypeError(f"Unsupported layer {type(layer).__name__}")
    return traces


def layout_for(scene: Scene, *, height: int = 460) -> dict[str, Any]:
    """Plotly layout for a scene: titles, fixed ranges, quiet styling."""
    return {
        "template": "plotly_white",
        "height": height,
        "margin": {"l": 60, "r": 20, "t": 20, "b": 50},
        "showlegend": False,
        "hovermode": "closest",
        "uirevision": "glasslearn",  # keep the user's zoom while scrubbing
        "xaxis": {
            "title": {"text": scene.x_title},
            "range": scene.x_range,
            "zeroline": False,
        },
        "yaxis": {
            "title": {"text": scene.y_title},
            "range": scene.y_range,
            "zeroline": False,
        },
    }


def build_figure(scene: Scene, *, widget: bool = False, height: int = 460) -> go.Figure:
    """A new figure for ``scene``; a ``FigureWidget`` when ``widget`` is True."""
    cls = go.FigureWidget if widget else go.Figure
    return cls(data=traces_for(scene), layout=layout_for(scene, height=height))


def update_figure(fig: go.Figure, scene: Scene) -> None:
    """Swap the data of an existing figure to show ``scene``, in place.

    Raises ValueError if the scene would need a different number of traces.
    """
    specs = traces_for(scene)
    if len(specs) != len(fig.data):
        raise ValueError(
            f"The scene needs {len(specs)} traces but the figure has {len(fig.data)}; "
            "a view must keep the same layers on every step."
        )
    with fig.batch_update():
        for trace, spec in zip(fig.data, specs, strict=True):
            spec = dict(spec)
            spec.pop("type")
            trace.update(spec)


def wrap(text: str, width: int = 110) -> str:
    """Wrap text into ``<br>``-separated lines for plotly annotations."""
    return "<br>".join(textwrap.wrap(text, width)) if text else ""


def _caption(text: str) -> dict[str, Any]:
    return {
        "text": text,
        "x": 0,
        "y": 1.02,
        "xref": "paper",
        "yref": "paper",
        "xanchor": "left",
        "yanchor": "bottom",
        "align": "left",
        "showarrow": False,
        "font": {"size": 13},
    }


def animated_figure(
    scenes: Sequence[Scene],
    captions: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    title: str = "",
    height: int = 620,
) -> go.Figure:
    """One plotly figure with a frame per step, a slider and play/pause buttons.

    ``captions`` are HTML strings shown above the plot for each frame.
    """
    labels = (
        list(labels) if labels is not None else [str(i) for i in range(len(scenes))]
    )
    fig = go.Figure(
        data=traces_for(scenes[0]), layout=layout_for(scenes[0], height=height)
    )
    fig.frames = [
        go.Frame(
            data=traces_for(scene),
            name=str(i),
            layout={"annotations": [_caption(captions[i])]},
        )
        for i, scene in enumerate(scenes)
    ]
    still = {
        "mode": "immediate",
        "frame": {"duration": 0, "redraw": True},
        "transition": {"duration": 0},
    }
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15}},
        margin={"l": 60, "r": 20, "t": 190, "b": 40},
        annotations=[_caption(captions[0])],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Step: "},
                "pad": {"t": 40},
                "steps": [
                    {"method": "animate", "label": label, "args": [[str(i)], still]}
                    for i, label in enumerate(labels)
                ],
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": -0.06,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 800, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "❚❚ Pause",
                        "method": "animate",
                        "args": [[None], still],
                    },
                ],
            }
        ],
    )
    return fig
