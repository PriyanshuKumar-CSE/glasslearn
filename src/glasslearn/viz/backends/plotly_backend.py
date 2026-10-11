"""Draws a :class:`~glasslearn.viz.scene.Scene` with plotly, in a light "glass" style.

Every layer becomes a fixed number of plotly traces, so moving to another
step only swaps data in place (``fig.batch_update()``, or
``fig.batch_animate()`` when the move should glide).
"""

from __future__ import annotations

import base64
import textwrap
from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from ..scene import Arrows, Lines, Markers, Points, Scene, color_of

# -- the glass theme --------------------------------------------------------------

FONT = "Inter, 'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif"
INK = "#1E293B"  # main text
MUTED = "#64748B"  # ticks and secondary text
EDGE = "rgba(148, 163, 184, 0.45)"  # thin borders
ARROW_COLOR = "rgba(15, 23, 42, 0.55)"
ACCENT = "#4F7CFF"

#: Time for one step to glide into the next, in milliseconds.
GLIDE_MS = 450

# A pale wash of colour behind the plot: soft blue, violet and mint glows under
# a frosted white layer. Inline SVG, so it works offline and in saved files.
_BACKDROP_SVG = """<svg xmlns='http://www.w3.org/2000/svg' width='800' height='500'
 viewBox='0 0 800 500' preserveAspectRatio='none'><defs>
<radialGradient id='a' cx='12%' cy='18%' r='65%'><stop offset='0' stop-color='#BFD0FF'/>
<stop offset='1' stop-color='#BFD0FF' stop-opacity='0'/></radialGradient>
<radialGradient id='b' cx='88%' cy='85%' r='65%'><stop offset='0' stop-color='#DCCBFF'/>
<stop offset='1' stop-color='#DCCBFF' stop-opacity='0'/></radialGradient>
<radialGradient id='c' cx='78%' cy='8%' r='45%'><stop offset='0' stop-color='#C6F0E4'/>
<stop offset='1' stop-color='#C6F0E4' stop-opacity='0'/></radialGradient></defs>
<rect width='800' height='500' fill='#F7F9FF'/><rect width='800' height='500'
 fill='url(#a)'/><rect width='800' height='500' fill='url(#b)'/><rect width='800'
 height='500' fill='url(#c)'/><rect width='800' height='500' fill='#FFFFFF'
 fill-opacity='0.45'/></svg>"""
BACKDROP = (
    "data:image/svg+xml;base64,"
    + base64.b64encode(" ".join(_BACKDROP_SVG.split()).encode()).decode()
)


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _text_on(hex_color: str) -> str:
    """Dark or white text, whichever reads better on ``hex_color``."""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if luminance > 0.6 else "#FFFFFF"


def _colors(groups: np.ndarray) -> list[str]:
    return [color_of(int(g)) for g in groups]


# -- layers -------------------------------------------------------------------------


def _points(layer: Points) -> dict[str, Any]:
    groups = np.asarray(layer.group)
    trace: dict[str, Any] = {
        "type": "scatter",
        "mode": "markers",
        "name": layer.name,
        "x": np.asarray(layer.x),
        "y": np.asarray(layer.y),
        # Glass beads: translucent fill with a bright rim.
        "marker": {
            "color": _colors(groups),
            "size": layer.size + 1,
            "opacity": layer.opacity,
            "line": {"width": 0.8, "color": "rgba(255, 255, 255, 0.9)"},
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


def _markers(layer: Markers) -> list[dict[str, Any]]:
    """Two traces: a soft halo, then a lens with a white rim and the label inside."""
    colors = _colors(np.asarray(layer.group))
    x, y = np.asarray(layer.x), np.asarray(layer.y)
    halo = {
        "type": "scatter",
        "mode": "markers",
        "name": f"{layer.name} halo",
        "x": x,
        "y": y,
        "marker": {
            "size": layer.size * 2.1,
            "color": [_rgba(c, 0.22) for c in colors],
            "line": {"width": 0},
        },
        "hoverinfo": "skip",
    }
    core = {
        "type": "scatter",
        "mode": "markers+text",
        "name": layer.name,
        "x": x,
        "y": y,
        "text": list(layer.labels),
        "textposition": "middle center",
        "textfont": {
            "size": 11,
            "color": [_text_on(c) for c in colors],
            "family": FONT,
        },
        "marker": {
            "symbol": "circle",
            "size": layer.size + 4,
            "color": colors,
            "line": {"width": 2.5, "color": "rgba(255, 255, 255, 0.95)"},
        },
        "hovertemplate": f"{layer.name.rstrip('s')} %{{text}}<extra></extra>",
    }
    return [halo, core]


def _arrows(layer: Arrows) -> dict[str, Any]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    sizes: list[float] = []
    for x0, y0, x1, y1 in zip(layer.x0, layer.y0, layer.x1, layer.y1, strict=True):
        moved = (x0, y0) != (x1, y1)
        xs += [float(x0), float(x1), None]
        ys += [float(y0), float(y1), None]
        sizes += [0.0, 11.0 if moved else 0.0, 0.0]  # no arrowhead on a still arrow
    return {
        "type": "scatter",
        "mode": "lines+markers",
        "name": layer.name,
        "x": xs,
        "y": ys,
        "line": {"color": ARROW_COLOR, "width": 2.2},
        "marker": {
            "symbol": "arrow",
            "angleref": "previous",
            "size": sizes,
            "color": ARROW_COLOR,
            "line": {"width": 0},
        },
        "hoverinfo": "skip",
    }


def _lines(layer: Lines) -> list[dict[str, Any]]:
    traces = []
    for path, group in zip(layer.paths, np.asarray(layer.group), strict=True):
        path = np.asarray(path, dtype=float).reshape(-1, 2)
        color = color_of(int(group))
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": f"{layer.name} {int(group)}",
                "x": path[:, 0],
                "y": path[:, 1],
                "line": {
                    "color": color,
                    "width": 2,
                    "dash": layer.dash,
                    "shape": "spline",
                    "smoothing": 0.6,
                },
                "marker": {
                    "size": 5,
                    "color": color,
                    "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                },
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
            traces.extend(_markers(layer))
        elif isinstance(layer, Arrows):
            traces.append(_arrows(layer))
        elif isinstance(layer, Lines):
            traces.extend(_lines(layer))
        else:  # pragma: no cover - new layer types must be added here
            raise TypeError(f"Unsupported layer {type(layer).__name__}")
    return traces


def glass_axis(title: str, value_range: Any) -> dict[str, Any]:
    """Axis styling for the glass look (muted ticks, soft grid, no lines)."""
    return {
        "title": {"text": title, "font": {"color": MUTED, "size": 13}},
        "range": value_range,
        "zeroline": False,
        "showline": False,
        "ticks": "",
        "tickfont": {"color": MUTED, "size": 11},
        "gridcolor": "rgba(100, 116, 139, 0.13)",
    }


def glass_layout(
    *, height: int, margin: dict[str, int] | None = None
) -> dict[str, Any]:
    """Shared look for glasslearn figures: frosted backdrop, soft grid, quiet text."""
    return {
        "height": height,
        "margin": margin or {"l": 60, "r": 20, "t": 20, "b": 50},
        "font": {"family": FONT, "color": INK, "size": 13},
        "paper_bgcolor": "#FBFCFF",
        "plot_bgcolor": "rgba(255, 255, 255, 0)",
        "showlegend": False,
        "hovermode": "closest",
        "hoverlabel": {
            "bgcolor": "rgba(255, 255, 255, 0.95)",
            "bordercolor": EDGE,
            "font": {"family": FONT, "color": INK, "size": 12},
        },
        "images": [
            {
                "source": BACKDROP,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1,
                "sizex": 1,
                "sizey": 1,
                "sizing": "stretch",
                "layer": "below",
            }
        ],
        "shapes": [
            {
                "type": "rect",
                "xref": "paper",
                "yref": "paper",
                "x0": 0,
                "y0": 0,
                "x1": 1,
                "y1": 1,
                "line": {"color": EDGE, "width": 1},
                "layer": "below",
            }
        ],
    }


def layout_for(scene: Scene, *, height: int = 460) -> dict[str, Any]:
    """Plotly layout for a scene: titles, fixed ranges and the glass look."""
    layout = glass_layout(height=height)
    layout["uirevision"] = "glasslearn"  # keep the user's zoom while scrubbing
    layout["xaxis"] = glass_axis(scene.x_title, scene.x_range)
    layout["yaxis"] = glass_axis(scene.y_title, scene.y_range)
    return layout


def build_figure(scene: Scene, *, widget: bool = False, height: int = 460) -> go.Figure:
    """A new figure for ``scene``; a ``FigureWidget`` when ``widget`` is True."""
    cls = go.FigureWidget if widget else go.Figure
    return cls(data=traces_for(scene), layout=layout_for(scene, height=height))


def update_figure(fig: go.Figure, scene: Scene, *, glide_ms: int = 0) -> None:
    """Swap the data of an existing figure to show ``scene``, in place.

    With ``glide_ms`` > 0 a ``FigureWidget`` animates points and centroids to
    their new places. Raises ValueError if the scene would need a different
    number of traces.
    """
    specs = traces_for(scene)
    if len(specs) != len(fig.data):
        raise ValueError(
            f"The scene needs {len(specs)} traces but the figure has {len(fig.data)}; "
            "a view must keep the same layers on every step."
        )
    batch = (
        fig.batch_animate(duration=glide_ms, easing="cubic-in-out")
        if glide_ms > 0 and isinstance(fig, go.FigureWidget)
        else fig.batch_update()
    )
    with batch:
        for trace, spec in zip(fig.data, specs, strict=True):
            spec = dict(spec)
            spec.pop("type")
            trace.update(spec)


# -- the static, animated figure ------------------------------------------------------


def wrap(text: str, width: int = 110) -> str:
    """Wrap text into ``<br>``-separated lines for plotly annotations."""
    return "<br>".join(textwrap.wrap(text, width)) if text else ""


def _caption(text: str) -> dict[str, Any]:
    """The narration, on a frosted card above the plot."""
    return {
        "text": text,
        "x": 0,
        "y": 1.03,
        "xref": "paper",
        "yref": "paper",
        "xanchor": "left",
        "yanchor": "bottom",
        "align": "left",
        "showarrow": False,
        "font": {"size": 13, "color": INK, "family": FONT},
        "bgcolor": "rgba(255, 255, 255, 0.82)",
        "bordercolor": EDGE,
        "borderwidth": 1,
        "borderpad": 10,
    }


def animated_figure(
    scenes: Sequence[Scene],
    captions: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    active: int = 0,
    title: str = "",
    height: int = 640,
) -> go.Figure:
    """One plotly figure with a frame per step, a slider and play/pause buttons.

    ``captions`` are HTML strings shown above the plot for each frame;
    ``active`` is the step shown first. Moving between steps glides.
    """
    labels = (
        list(labels) if labels is not None else [str(i) for i in range(len(scenes))]
    )
    first = scenes[active]
    fig = go.Figure(data=traces_for(first), layout=layout_for(first, height=height))
    fig.frames = [
        go.Frame(
            data=traces_for(scene),
            name=str(i),
            layout={"annotations": [_caption(captions[i])]},
        )
        for i, scene in enumerate(scenes)
    ]
    glide = {"duration": GLIDE_MS, "easing": "cubic-in-out"}
    jump = {
        "mode": "immediate",
        "frame": {"duration": GLIDE_MS, "redraw": True},
        "transition": glide,
    }
    play = {
        "frame": {"duration": GLIDE_MS + 650, "redraw": True},
        "transition": glide,
        "fromcurrent": True,
    }
    pause = {
        "mode": "immediate",
        "frame": {"duration": 0, "redraw": False},
        "transition": {"duration": 0},
    }
    # Room above the plot for the title and the tallest caption card.
    title_lines = title.count("<br>") + 1 if title else 0
    caption_lines = max(c.count("<br>") + 1 for c in captions)
    top = 60 + 20 * title_lines + 18 * caption_lines
    fig.update_layout(
        title={
            "text": title,
            "x": 0,
            "xanchor": "left",
            "y": 1,
            "yref": "container",
            "yanchor": "top",
            "pad": {"t": 14, "l": 8},
            "font": {"size": 15, "color": INK, "family": FONT},
        },
        height=height + top - 190,
        margin={"l": 60, "r": 20, "t": top, "b": 110},
        annotations=[_caption(captions[active])],
        sliders=[
            {
                "active": active,
                "x": 0.12,
                "len": 0.88,
                "y": 0,
                "yanchor": "top",
                "pad": {"t": 55, "b": 10},
                "bgcolor": "rgba(226, 232, 240, 0.9)",
                "activebgcolor": ACCENT,
                "bordercolor": EDGE,
                "borderwidth": 1,
                "tickcolor": "rgba(100, 116, 139, 0.5)",
                "font": {"size": 10, "color": MUTED},
                "currentvalue": {
                    "prefix": "Step ",
                    "font": {"size": 12, "color": INK},
                },
                "transition": glide,
                "steps": [
                    {"method": "animate", "label": label, "args": [[str(i)], jump]}
                    for i, label in enumerate(labels)
                ],
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.12,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top",
                "pad": {"t": 62, "r": 10},
                "showactive": False,
                "bgcolor": "rgba(255, 255, 255, 0.9)",
                "bordercolor": EDGE,
                "font": {"color": INK, "size": 12},
                "buttons": [
                    {"label": "▶", "method": "animate", "args": [None, play]},
                    {"label": "❚❚", "method": "animate", "args": [[None], pause]},
                ],
            }
        ],
    )
    return fig
