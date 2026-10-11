"""What KMeans records, and the invariants every recording must satisfy."""

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from glasslearn import InvalidParameterError
from glasslearn.cluster import KMeans
from glasslearn.cluster.kmeans.events import (
    CentroidMoved,
    Converged,
    EmptyCluster,
    InitChosen,
    PointsReassigned,
)
from glasslearn.datasets import make_blobs
from glasslearn.trace import Trace

X, _ = make_blobs(n_samples=300, centers=4, random_state=0)


def fitted(**params):
    return KMeans(**{"n_clusters": 4, "random_state": 0, **params}).fit(X)


# -- invariants (hypothesis) ----------------------------------------------------

data = st.integers(5, 40).flatmap(
    lambda n: arrays(
        np.float64,
        st.tuples(st.just(n), st.integers(1, 3)),
        elements=st.floats(-50, 50, allow_nan=False, allow_infinity=False),
    )
)


@settings(max_examples=60, deadline=None)
@given(
    X=data,
    k=st.integers(1, 5),
    init=st.sampled_from(["k-means++", "random"]),
    seed=st.integers(0, 2**16),
)
def test_inertia_never_rises_and_labels_stay_in_range(X, k, init, seed):
    k = min(k, len(X))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=k, init=init, n_init=1, random_state=seed).fit(X)
    trace = km.trace_
    inertia = trace.metric("inertia")
    inertia = inertia[~np.isnan(inertia)]
    scale = max(1.0, inertia[0])
    assert np.all(np.diff(inertia) <= 1e-9 * scale), inertia
    for step in trace:
        if "labels" in step.state:
            labels = step.state["labels"]
            assert labels.min() >= 0 and labels.max() < k
    assert km.labels_.min() >= 0 and km.labels_.max() < k


@settings(max_examples=30, deadline=None)
@given(X=data, k=st.integers(1, 4), seed=st.integers(0, 2**16))
def test_record_false_is_identical(X, k, seed):
    k = min(k, len(X))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = KMeans(n_clusters=k, init="random", random_state=seed).fit(X)
        b = KMeans(n_clusters=k, init="random", random_state=seed).fit(X, record=False)
    np.testing.assert_array_equal(a.cluster_centers_, b.cluster_centers_)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    assert a.inertia_ == b.inertia_ and a.n_iter_ == b.n_iter_
    assert len(b.trace_) == 0


# -- shape of the trace ---------------------------------------------------------


def test_phases_follow_lloyd():
    km = fitted()
    trace = km.trace_
    reason = trace[-1].event(Converged).reason
    assert reason == "labels stable"
    assert trace.phases == ["init"] + ["assign", "update"] * km.n_iter_ + ["done"]
    iterations = [s.state["iteration"] for s in trace]
    assert iterations[0] == 0 and iterations[-1] == km.n_iter_
    assert len(trace.metric("inertia")) == len(trace)


def test_extra_label_pass_when_stopping_on_max_iter():
    km = fitted(max_iter=2)
    trace = km.trace_
    assert trace.phases == [
        "init",
        "assign",
        "update",
        "assign",
        "update",
        "assign",
        "done",
    ]
    assert trace[-2].event(PointsReassigned).final
    assert trace[-1].event(Converged).reason == "max_iter"
    assert trace[-1].event(Converged).n_iter == km.n_iter_ == 2


def test_stopping_on_tol():
    km = fitted(tol=10.0)
    assert km.trace_[-1].event(Converged).reason == "tol"
    assert km.trace_[-2].phase == "assign"


def test_final_state_matches_fitted_attributes():
    km = fitted()
    trace = km.trace_
    np.testing.assert_array_equal(trace[-1].state["centers"], km.cluster_centers_)
    np.testing.assert_array_equal(trace.latest("labels", -1), km.labels_)
    assert trace.metric("inertia")[-1] == km.inertia_
    np.testing.assert_array_equal(trace.X, X)
    assert trace.feature_names == ["x0", "x1"]


def test_labels_stored_compactly_and_only_on_change():
    trace = fitted().trace_
    stored = [s.index for s in trace if "labels" in s.state]
    assert stored[0] == 1  # the first assignment
    assert len(stored) < len(trace)
    assert all(trace[i].state["labels"].dtype == np.uint8 for i in stored)
    n_changed = trace.metric("n_changed")
    for step in trace:
        if step.phase == "assign" and step.index > 1:
            assert ("labels" in step.state) == (n_changed[step.index] > 0)


def test_init_event_records_seed_order():
    km = fitted(init="k-means++")
    init = km.trace_[0]
    event = init.event(InitChosen)
    assert event.method == "k-means++" and event.n_init == 1 and event.run == 0
    assert len(event.seeds) == 4
    np.testing.assert_allclose(init.state["centers"], X[list(event.seeds)])


def test_assign_events():
    trace = fitted().trace_
    first = trace[1].event(PointsReassigned)
    assert first.first and first.n_changed == 300 and first.transitions == ()
    assert sum(first.sizes) == 300
    second = trace[3].event(PointsReassigned)
    matrix = np.array(second.transitions)
    assert matrix.shape == (4, 4) and matrix.sum() == 300
    assert matrix.sum() - np.trace(matrix) == second.n_changed
    np.testing.assert_array_equal(matrix.sum(axis=0), second.sizes)


def test_update_events():
    trace = fitted().trace_
    step = trace[2]
    moves = step.events_of(CentroidMoved)
    assert [m.cluster for m in moves] == [0, 1, 2, 3]
    np.testing.assert_allclose([m.old for m in moves], step.state["old_centers"])
    np.testing.assert_allclose([m.new for m in moves], step.state["centers"])
    shift = sum(m.distance**2 for m in moves)
    assert shift == pytest.approx(step.metrics["center_shift_tot"])
    assert max(m.salience for m in moves) == pytest.approx(0.8)
    assert sum(m.n_gained for m in moves) == 300  # first update: everyone joined


def test_empty_cluster_is_recorded():
    init = np.vstack([X[:3], [[100.0, 100.0]]])
    km = KMeans(n_clusters=4, init=init, n_init=1).fit(X)
    step = km.trace_[2]
    empty = step.event(EmptyCluster)
    assert empty.cluster == 3 and empty.sample >= 0
    assert step.state["labels"][empty.sample] == 3
    np.testing.assert_allclose(step.state["centers"][3], X[empty.sample])
    moved = step.events_of(CentroidMoved)[3]
    assert moved.size == 1 and moved.n_gained == 1


def test_restarts_keep_best_trace_and_every_inertia():
    km = fitted(init="random", n_init=5)
    meta = km.trace_.meta
    assert len(meta["run_inertia"]) == 5
    assert meta["run_inertia"][meta["best_run"]] == pytest.approx(km.inertia_)
    assert km.trace_[0].event(InitChosen).run == meta["best_run"]


def test_trace_round_trips_through_a_file(tmp_path):
    trace = fitted().trace_
    trace.save(tmp_path / "km.npz")
    loaded = Trace.load(tmp_path / "km.npz")
    assert [s.events for s in loaded] == [s.events for s in trace]
    assert loaded.meta == trace.meta


# -- API details ----------------------------------------------------------------


def test_summary_uses_column_names():
    frame = pd.DataFrame(X, columns=["age", "income"])
    km = KMeans(n_clusters=4, random_state=0).fit(frame)
    summary = km.summary()
    assert list(summary.columns) == ["age", "income"]
    assert summary.index.name == "cluster"
    np.testing.assert_array_equal(summary.to_numpy(), km.cluster_centers_)
    assert km.trace_.feature_names == ["age", "income"]


def test_get_feature_names_out():
    km = fitted()
    assert list(km.get_feature_names_out()) == [
        "kmeans0",
        "kmeans1",
        "kmeans2",
        "kmeans3",
    ]
    with pytest.raises(ValueError, match="length equal to number of features"):
        km.get_feature_names_out(["a"])


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"n_clusters": 0}, "'n_clusters' parameter of KMeans must be an int"),
        ({"init": "kmeans++"}, "'init' parameter"),
        ({"n_init": "always"}, "'n_init' parameter"),
        ({"max_iter": 0}, "'max_iter' parameter"),
        ({"tol": -1.0}, "'tol' parameter"),
        ({"random_state": "seed"}, "'random_state' parameter"),
        ({"algorithm": "fast"}, "'algorithm' parameter"),
    ],
)
def test_invalid_params_raise_in_fit_not_init(params, match):
    km = KMeans(**params)
    with pytest.raises(InvalidParameterError, match=match):
        km.fit(X)


def test_too_few_samples():
    with pytest.raises(ValueError, match="n_samples=3 should be >= n_clusters=4"):
        KMeans(n_clusters=4).fit(X[:3])


def test_bad_init_shape():
    with pytest.raises(ValueError, match="does not match the number of clusters 4"):
        KMeans(n_clusters=4, init=X[:3]).fit(X)


def test_explicit_init_with_several_runs_warns():
    with pytest.warns(RuntimeWarning, match="performing only one init"):
        KMeans(n_clusters=4, init=X[:4], n_init=3).fit(X)


def test_callable_init():
    def first_rows(data, k, random_state):
        return data[:k]

    km = KMeans(n_clusters=4, init=first_rows, n_init=1).fit(X)
    assert km.trace_[0].event(InitChosen).method == "callable"
    np.testing.assert_allclose(km.trace_[0].state["centers"], X[:4])


def test_elkan_and_copy_x_are_accepted():
    a = fitted()
    b = fitted(algorithm="elkan", copy_x=False)
    np.testing.assert_array_equal(a.labels_, b.labels_)


def test_input_is_not_modified():
    data = X.copy()
    fitted(copy_x=False)
    KMeans(n_clusters=4, random_state=0, copy_x=False).fit(data)
    np.testing.assert_array_equal(data, X)
