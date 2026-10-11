"""The notebook view that ``explain()`` returns."""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go

from .._config import LEVELS, get_config
from ..narrate import Narrator
from ..trace import Trace
from .backends.plotly_backend import animated_figure, build_figure, update_figure, wrap
from .projection import make_projection
from .scene import make_view

_WARNING_STYLE = (
    "margin-top:8px;padding:6px 10px;border-left:4px solid #E8A33D;"
    "background:rgba(232,163,61,0.14);border-radius:3px"
)
_QUIET_STYLE = "opacity:0.72;font-size:12px;margin-top:6px"


def widgets_supported() -> bool:
    """True inside a Jupyter-style kernel (JupyterLab, VS Code, Colab) with the
    widget packages installed."""
    try:
        import anywidget  # noqa: F401
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return (
        type(shell).__name__ == "ZMQInteractiveShell" or "google.colab" in sys.modules
    )


class ExplainView(widgets.VBox):
    """A narrated, scrubbable picture of a training run, for a notebook cell.

    Live mode (a widget): summary line, header with the current iteration and
    a level switch, play button, slider and step buttons, the main plot, a
    small metric curve, and the narration with any warnings. Static mode: one
    plotly figure with a frame per step and the narration above the plot.

    Usually created by ``model.explain()``.

    Parameters
    ----------
    trace : Trace
        A recorded trace (``model.trace_``).
    step : int, optional
        Step to show first (negative counts from the end). Default: the first.
    projection : "auto", "pca", (str, str) or Projection, default="auto"
        How to draw data with more than two features.
    level : {"beginner", "detailed", "math"}, optional
        Narration level. Default: ``get_config()["level"]``.
    static : bool, optional
        Force the static figure (True) or the widget (False). Default: the
        widget when the notebook supports it.
    max_points : int, optional
        Most points drawn. Default: ``get_config()["max_points"]``.
    """

    def __init__(
        self,
        trace: Trace,
        *,
        step: int | None = None,
        projection: Any = "auto",
        level: str | None = None,
        static: bool | None = None,
        max_points: int | None = None,
    ) -> None:
        super().__init__()
        config = get_config()
        level = config["level"] if level is None else level
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        if static is None:
            static = config["static"]
        if static is None:
            static = not widgets_supported()
        if len(trace) == 0 or trace.X is None:
            raise ValueError(
                "This trace has no recorded steps. Fit with record=True to explain it."
            )
        self.trace = trace
        self.static = bool(static)
        self.view = make_view(
            trace, max_points=config["max_points"] if max_points is None else max_points
        )
        self.projection = make_projection(projection, trace.X, trace.feature_names)
        self._level = level
        self._index = self._normalize(0 if step is None else step)
        self._narrators: dict[str, Narrator] = {}
        self._static_figure: go.Figure | None = None
        if not self.static:
            self._build_widget()

    # -- public API -------------------------------------------------------------

    @property
    def step(self) -> int:
        """Index of the step on screen."""
        return self._index

    @step.setter
    def step(self, index: int) -> None:
        index = self._normalize(index)
        if self.static:
            self._index = index
        else:
            self._slider.value = index  # triggers the update

    @property
    def level(self) -> str:
        """Narration level on screen."""
        return self._level

    @level.setter
    def level(self, level: str) -> None:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        if self.static:
            self._level = level
            self._static_figure = None
        else:
            self._levels.value = level  # triggers the update

    @property
    def figure(self) -> go.Figure:
        """The main plot: a live ``FigureWidget``, or the animated static figure."""
        return self.static_figure() if self.static else self._figure

    def narrator(self, level: str | None = None) -> Narrator:
        """The narrator for a level (default: the current one)."""
        level = self._level if level is None else level
        if level not in self._narrators:
            self._narrators[level] = Narrator(
                self.trace, level=level, projection=self.projection
            )
        return self._narrators[level]

    def header(self, index: int | None = None) -> str:
        """'Iteration 3 of 7 · update' for a step (default: the current one)."""
        index = self._index if index is None else index
        step = self.trace[index]
        iteration = step.state.get("iteration")
        total = self.trace[-1].state.get("iteration")
        if iteration is None or total is None:
            return f"Step {index + 1} of {len(self.trace)} · {step.phase}"
        return f"Iteration {iteration} of {total} · {step.phase}"

    def notes(self) -> list[str]:
        """Fixed remarks about the picture (sampling, projection)."""
        notes = []
        if self.view.n_shown < self.view.n_total:
            notes.append(
                f"Showing {self.view.n_shown:,} of {self.view.n_total:,} points, "
                "sampled per cluster; the computation used all of them."
            )
        if self.projection.note:
            notes.append(self.projection.note)
        return notes

    def narration_html(self, index: int | None = None, level: str | None = None) -> str:
        """The narration panel for a step, as HTML."""
        index = self._index if index is None else index
        narrator = self.narrator(level)
        narration = narrator.narrate(index)
        parts = [f"<div>{html.escape(narration.text)}</div>"]
        if narration.glossary:
            terms = "<br>".join(
                "<b>{}</b>:{}".format(*map(html.escape, text.split(":", 1)))
                for _, text in narration.glossary
            )
            parts.append(f'<div style="{_QUIET_STYLE}">{terms}</div>')
        for d in narrator.diagnostics(upto=index):
            parts.append(
                f'<div style="{_WARNING_STYLE}">⚠ {html.escape(d.text)}<br>'
                f"<i>Try:</i> {html.escape(d.fix)}</div>"
            )
        for note in self.notes():
            parts.append(f'<div style="{_QUIET_STYLE}">{html.escape(note)}</div>')
        return (
            '<div style="font-size:14px;line-height:1.5">' + "".join(parts) + "</div>"
        )

    def static_figure(self) -> go.Figure:
        """A plotly figure with one frame per step (built once per level)."""
        if self._static_figure is None:
            narrator = self.narrator()
            captions = []
            for i in range(len(self.trace)):
                lines = [f"<b>{html.escape(self.header(i))}</b>"]
                lines.append(wrap(html.escape(narrator.narrate(i).text)))
                for d in narrator.diagnostics(upto=i):
                    lines.append(wrap(html.escape(f"⚠ {d.text} Try: {d.fix}")))
                captions.append("<br>".join(lines))
            scenes = [
                self.view.render(i, self.projection) for i in range(len(self.trace))
            ]
            title = wrap(html.escape(narrator.summary()), 120)
            for note in self.notes():
                note = wrap(html.escape(note), 140)
                title += f"<br><span style='font-size:12px'>{note}</span>"
            fig = animated_figure(scenes, captions, active=self._index, title=title)
            self._static_figure = fig
        return self._static_figure

    def save(self, path: str | Path) -> Path:
        """Write a standalone HTML file of the run (the animated figure).

        The file embeds plotly.js, so it opens offline in any browser.
        """
        path = Path(path)
        self.static_figure().write_html(
            str(path), include_plotlyjs=True, full_html=True, auto_play=False
        )
        return path

    def _ipython_display_(self) -> None:
        from IPython.display import display

        if self.static:
            display(self.static_figure())
        else:
            display(self._repr_mimebundle_(), raw=True)

    def __repr__(self) -> str:
        mode = "static" if self.static else "widget"
        return (
            f"ExplainView({self.trace.algorithm}, steps={len(self.trace)}, "
            f"level={self._level!r}, {mode})"
        )

    # -- live widget ------------------------------------------------------------

    def _normalize(self, index: int) -> int:
        n = len(self.trace)
        if not -n <= index < n:
            raise IndexError(f"step {index} is out of range for a trace of {n} steps")
        return index % n

    def _build_widget(self) -> None:
        last = len(self.trace) - 1
        self._summary = widgets.HTML()
        self._header = widgets.HTML()
        self._levels = widgets.ToggleButtons(
            options=list(LEVELS),
            value=self._level,
            style=widgets.ToggleButtonsStyle(button_width="84px"),
        )
        self._play = widgets.Play(
            value=self._index, min=0, max=last, step=1, interval=900, description="Play"
        )
        self._slider = widgets.IntSlider(
            value=self._index,
            min=0,
            max=last,
            continuous_update=True,
            description="step",
            layout=widgets.Layout(width="420px"),
        )
        widgets.jslink((self._play, "value"), (self._slider, "value"))
        self._prev = widgets.Button(
            description="◀",
            tooltip="Previous step",
            layout=widgets.Layout(width="40px"),
        )
        self._next = widgets.Button(
            description="▶",
            tooltip="Next step",
            layout=widgets.Layout(width="40px"),
        )
        self._figure = build_figure(
            self.view.render(self._index, self.projection), widget=True
        )
        self._metric = self._build_metric_figure()
        self._narration = widgets.HTML()

        top = widgets.HBox(
            [self._header, self._levels],
            layout=widgets.Layout(
                justify_content="space-between", align_items="center"
            ),
        )
        controls = widgets.HBox([self._play, self._prev, self._slider, self._next])
        children: list[widgets.Widget] = [self._summary, top, controls, self._figure]
        if self._metric is not None:
            children.append(self._metric)
        children.append(self._narration)
        self.children = children

        self._slider.observe(self._on_step, names="value")
        self._levels.observe(self._on_level, names="value")
        self._prev.on_click(lambda _: self._move(-1))
        self._next.on_click(lambda _: self._move(1))
        self._refresh_text()

    def _metric_name(self) -> str | None:
        name = getattr(self.view, "metric", None)
        if name is None or np.isnan(self.trace.metric(name)).all():
            return None
        return name

    def _metric_marker(self, values: np.ndarray) -> tuple[list[int], list[float]]:
        value = values[self._index]
        return ([], []) if np.isnan(value) else ([self._index], [float(value)])

    def _build_metric_figure(self) -> go.FigureWidget | None:
        name = self._metric_name()
        if name is None:
            return None
        values = self.trace.metric(name)
        steps = np.flatnonzero(~np.isnan(values))
        mx, my = self._metric_marker(values)
        return go.FigureWidget(
            data=[
                go.Scatter(
                    x=steps,
                    y=values[steps],
                    mode="lines+markers",
                    line={"color": "#4C78A8", "width": 2},
                    marker={"size": 4},
                    hovertemplate=f"step %{{x}}<br>{name} %{{y:.4g}}<extra></extra>",
                ),
                go.Scatter(
                    x=mx, y=my, mode="markers", marker={"size": 11, "color": "#E45756"}
                ),
            ],
            layout={
                "template": "plotly_white",
                "height": 170,
                "margin": {"l": 60, "r": 20, "t": 10, "b": 35},
                "showlegend": False,
                "xaxis": {
                    "title": {"text": "step"},
                    "range": [-0.5, len(values) - 0.5],
                },
                "yaxis": {"title": {"text": name}},
            },
        )

    def _move(self, delta: int) -> None:
        self._slider.value = int(
            np.clip(self._slider.value + delta, 0, len(self.trace) - 1)
        )

    def _on_step(self, change: dict[str, Any]) -> None:
        self._index = int(change["new"])
        update_figure(self._figure, self.view.render(self._index, self.projection))
        if self._metric is not None:
            mx, my = self._metric_marker(self.trace.metric(self._metric_name()))
            with self._metric.batch_update():
                self._metric.data[1].x = mx
                self._metric.data[1].y = my
        self._refresh_text()

    def _on_level(self, change: dict[str, Any]) -> None:
        self._level = change["new"]
        self._static_figure = None
        self._refresh_text()

    def _refresh_text(self) -> None:
        self._summary.value = f"<b>{html.escape(self.narrator().summary())}</b>"
        self._header.value = (
            f"<span style='font-size:15px'><b>{html.escape(self.header())}</b></span>"
            f"<span style='{_QUIET_STYLE};margin-left:8px'>step {self._index} of "
            f"{len(self.trace) - 1}</span>"
        )
        self._narration.value = self.narration_html()
