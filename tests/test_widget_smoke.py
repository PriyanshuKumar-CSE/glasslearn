"""The explain() view: live widget and static fallback, without a browser."""

import numpy as np
import plotly.graph_objects as go
import pytest

import glasslearn as gl
from glasslearn.cluster import KMeans
from glasslearn.viz.widget import ExplainView, widgets_supported

X = gl.datasets.make_blobs(n_samples=300, centers=4, random_state=0, as_frame=True)


@pytest.fixture(scope="module")
def km():
    return KMeans(n_clusters=4, random_state=0).fit(X)


def test_scrubbing_every_step_updates_in_place(km):
    view = km.explain(static=False)
    assert isinstance(view, ExplainView) and not view.static
    fig = view.figure
    assert isinstance(fig, go.FigureWidget)
    n_traces = len(fig.data)
    centroids, narrations, headers = set(), set(), set()
    for i in range(len(km.trace_)):
        view._slider.value = i
        assert view.step == i
        assert len(fig.data) == n_traces
        assert view.figure is fig  # updated, not replaced
        centroids.add(tuple(np.round(fig.data[-1].x, 9)))
        narrations.add(view._narration.value)
        headers.add(view._header.value)
    assert len(centroids) > 1 and len(narrations) == len(km.trace_)
    assert "Iteration 7 of 7 · done" in view._header.value


def test_widget_parts(km):
    view = km.explain(static=False)
    summary, top, controls, main, metric, narration = view.children
    assert "settled in" in summary.value
    assert view.header(0) == "Iteration 0 of 7 · init"
    assert list(view._levels.options) == ["beginner", "detailed", "math"]
    assert metric.data[0].y[-1] == pytest.approx(km.inertia_)
    view.step = 3
    assert list(metric.data[1].x) == [3]
    assert "switched to a closer centroid" in narration.value


def test_buttons_and_level_switch(km):
    view = km.explain(static=False, step=-1)
    assert view.step == len(km.trace_) - 1
    view._next.click()
    assert view.step == len(km.trace_) - 1  # clamped at the end
    view._prev.click()
    assert view.step == len(km.trace_) - 2
    view.level = "math"
    assert "J" in view._narration.value and view.level == "math"
    with pytest.raises(ValueError, match="level must be one of"):
        view.level = "expert"


def test_static_mode_has_one_frame_per_step(km):
    view = km.explain(static=True, step=2)
    fig = view.figure
    assert isinstance(fig, go.Figure) and not isinstance(fig, go.FigureWidget)
    assert len(fig.frames) == len(km.trace_)
    assert fig.layout.sliders[0].active == 2
    assert "Iteration 1 of 7" in fig.frames[2].layout.annotations[0].text
    assert view.children == ()


def test_save_writes_standalone_html(km, tmp_path):
    for static in (True, False):
        path = km.explain(static=static).save(tmp_path / f"run_{static}.html")
        text = path.read_text(encoding="utf-8")
        assert "<html>" in text[:200] and "plotly" in text.lower()
        assert "Iteration 0 of 7" in text


def test_sampling_note(km):
    view = km.explain(static=False, max_points=50)
    assert view.view.n_shown == 50
    assert "Showing 50 of 300 points" in view._narration.value


def test_projection_note_for_many_features():
    X6, _ = gl.datasets.make_blobs(
        n_samples=200, n_features=6, centers=3, random_state=1
    )
    view = KMeans(n_clusters=3, random_state=0).fit(X6).explain(static=False)
    assert view.projection.axis_titles[0].startswith("PC1")
    assert "PCA" in view._narration.value


def test_explain_with_a_feature_pair(km):
    view = km.explain(static=False, projection=("feature_1", "feature_0"))
    assert view.figure.layout.xaxis.title.text == "feature_1"


def test_explain_draws_given_data(km):
    shifted = X + 100
    view = km.explain(shifted, static=True)
    assert view.trace.X.min() > 50 and km.trace_.X.min() < 50
    with pytest.raises(ValueError, match="training data has shape"):
        km.explain(X.iloc[:10], static=True)


def test_explain_needs_a_recorded_fit():
    with pytest.raises(ValueError, match="record=False"):
        KMeans(n_clusters=2).fit(X, record=False).explain()
    with pytest.raises(gl.NotFittedError):
        KMeans().explain()


def test_defaults_come_from_config(km):
    assert widgets_supported() is False  # no kernel under pytest
    assert km.explain().static is True
    with gl.config_context(static=False, level="detailed", max_points=40):
        view = km.explain()
        assert not view.static and view.level == "detailed" and view.view.n_shown == 40


def test_ipython_display_does_not_fail(km, capsys):
    km.explain(static=True)._ipython_display_()
    km.explain(static=False)._ipython_display_()
    assert "ExplainView(KMeans" in repr(km.explain(static=True))


def test_bad_step(km):
    with pytest.raises(IndexError, match="out of range"):
        km.explain(step=999)


class ZMQInteractiveShell:  # stands in for the shell of a Jupyter kernel
    pass


def test_jupyter_kernel_gets_the_widget(km, monkeypatch):
    import IPython

    monkeypatch.setattr(IPython, "get_ipython", lambda: ZMQInteractiveShell())
    assert widgets_supported() is True
    assert km.explain().static is False


def test_colab_gets_the_static_figure(km, monkeypatch):
    import sys
    import types

    import IPython

    monkeypatch.setattr(IPython, "get_ipython", lambda: ZMQInteractiveShell())
    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
    assert widgets_supported() is False
    view = km.explain()
    assert view.static is True
    assert len(view.figure.frames) == len(km.trace_)
    assert km.explain(static=False).static is False  # still possible on request
