"""The notebook view that ``explain()`` returns."""

from __future__ import annotations

import html
import sys
import time
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go

from .._config import LEVELS, get_config
from ..narrate import Narrator
from ..trace import Trace
from .backends.plotly_backend import (
    ACCENT,
    FONT,
    animated_figure,
    build_figure,
    glass_axis,
    glass_layout,
    update_figure,
    wrap,
)
from .projection import make_projection
from .scene import make_view

# Frosted-glass cards for the text panels. The cards carry their own light
# background, so they stay readable in dark notebook themes too.
_CARD_STYLE = (
    "background:linear-gradient(135deg,rgba(255,255,255,0.88),rgba(238,243,255,0.72));"
    "border:1px solid rgba(148,163,184,0.35);border-radius:14px;"
    "box-shadow:0 6px 20px rgba(15,23,42,0.07),inset 0 1px 0 rgba(255,255,255,0.8);"
    "backdrop-filter:blur(8px);padding:12px 16px;color:#1E293B;"
    f"font-family:{FONT}"
)
_WARNING_STYLE = (
    "margin-top:10px;padding:8px 12px;border:1px solid rgba(232,163,61,0.45);"
    "border-left:4px solid #E8A33D;background:rgba(255,236,204,0.6);"
    "border-radius:10px;color:#6B4400"
)
_QUIET_STYLE = "color:#64748B;font-size:12px;margin-top:8px"

#: Accent colour of each step phase (the pill next to the iteration).
PHASE_COLORS = {
    "init": "#64748B",
    "assign": "#4F7CFF",
    "update": "#9B6BFF",
    "epoch": "#4F7CFF",
    "split": "#FF8A3D",
    "done": "#22B8A0",
}

#: How long a single step glides in the live widget, in milliseconds. Faster
#: scrubbing than this switches to instant updates so nothing queues up.
LIVE_GLIDE_MS = 350


def _pill(text: str, color: str) -> str:
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return (
        f'<span style="display:inline-block;padding:1px 10px;border-radius:999px;'
        f"background:rgba({r},{g},{b},0.14);border:1px solid rgba({r},{g},{b},0.4);"
        f"color:{color};font-size:12px;font-weight:600;margin-left:8px;"
        f'font-family:{FONT}">{html.escape(text)}</span>'
    )


def widgets_supported() -> bool:
    """True inside a Jupyter kernel (JupyterLab, Notebook, VS Code) with the widget
    packages installed.

    Always False in Google Colab: Colab does not reliably render plotly's
    FigureWidget, so ``explain()`` uses the static figure there (which still
    has a slider and a play button). Pass ``static=False`` to try the widget.
    """
    if "google.colab" in sys.modules:
        return False
    try:
        import anywidget  # noqa: F401
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


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
            f'<div style="{_CARD_STYLE};font-size:14px;line-height:1.55">'
            + "".join(parts)
            + "</div>"
        )

    def static_figure(self) -> go.Figure:
        """A plotly figure with one frame per step (built once per level)."""
        if self._static_figure is None:
            narrator = self.narrator()
            captions = []
            for i in range(len(self.trace)):
                phase = self.trace[i].phase
                color = PHASE_COLORS.get(phase, ACCENT)
                lines = [
                    f"<b>{html.escape(self.header(i).rsplit(' · ', 1)[0])}</b> · "
                    f"<span style='color:{color}'><b>{html.escape(phase)}</b></span>"
                ]
                lines.append(wrap(html.escape(narrator.narrate(i).text)))
                for d in narrator.diagnostics(upto=i):
                    lines.append(wrap(html.escape(f"⚠ {d.text} Try: {d.fix}")))
                captions.append("<br>".join(lines))
            scenes = [
                self.view.render(i, self.projection) for i in range(len(self.trace))
            ]
            title = wrap(html.escape(narrator.summary()), 95)
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
            style={"handle_color": ACCENT},
            layout=widgets.Layout(width="420px"),
        )
        widgets.jslink((self._play, "value"), (self._slider, "value"))
        button_style = widgets.ButtonStyle(button_color="#EEF2FF")
        self._prev = widgets.Button(
            description="◀",
            tooltip="Previous step",
            style=button_style,
            layout=widgets.Layout(width="40px"),
        )
        self._next = widgets.Button(
            description="▶",
            tooltip="Next step",
            style=button_style,
            layout=widgets.Layout(width="40px"),
        )
        self._last_move = 0.0
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
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.08
        layout = glass_layout(height=170, margin={"l": 60, "r": 20, "t": 12, "b": 38})
        layout["xaxis"] = glass_axis("step", [-0.5, len(values) - 0.5])
        layout["yaxis"] = glass_axis(name, [lo - pad, hi + pad])
        return go.FigureWidget(
            data=[
                go.Scatter(
                    x=steps,
                    y=values[steps],
                    mode="lines+markers",
                    line={"color": ACCENT, "width": 2.5},
                    marker={"size": 5, "color": ACCENT},
                    fill="tozeroy",
                    fillcolor="rgba(79, 124, 255, 0.10)",
                    hovertemplate=f"step %{{x}}<br>{name} %{{y:.4g}}<extra></extra>",
                ),
                go.Scatter(
                    x=mx,
                    y=my,
                    mode="markers",
                    marker={
                        "size": 12,
                        "color": ACCENT,
                        "line": {"width": 6, "color": "rgba(79, 124, 255, 0.25)"},
                    },
                    hoverinfo="skip",
                ),
            ],
            layout=layout,
        )

    def _move(self, delta: int) -> None:
        self._slider.value = int(
            np.clip(self._slider.value + delta, 0, len(self.trace) - 1)
        )

    def _on_step(self, change: dict[str, Any]) -> None:
        self._index = int(change["new"])
        # Glide one step at a time (play, step buttons, slow dragging); jump
        # instantly on big or rapid moves so animations never queue up.
        now = time.monotonic()
        single = abs(int(change["new"]) - int(change["old"])) == 1
        glide = single and now - self._last_move >= LIVE_GLIDE_MS / 1000
        self._last_move = now
        update_figure(
            self._figure,
            self.view.render(self._index, self.projection),
            glide_ms=LIVE_GLIDE_MS if glide else 0,
        )
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
        summary = html.escape(self.narrator().summary())
        self._summary.value = (
            f'<div style="{_CARD_STYLE};font-weight:600;padding:10px 16px">'
            f"{summary}</div>"
        )
        phase = self.trace[self._index].phase
        title = html.escape(self.header().rsplit(" · ", 1)[0])
        self._header.value = (
            f'<span style="font-family:{FONT};font-size:15px;color:#1E293B">'
            f"<b>{title}</b></span>{_pill(phase, PHASE_COLORS.get(phase, ACCENT))}"
            f'<span style="{_QUIET_STYLE};margin-left:10px;font-family:{FONT}">'
            f"step {self._index} of {len(self.trace) - 1}</span>"
        )
        self._narration.value = self.narration_html()
