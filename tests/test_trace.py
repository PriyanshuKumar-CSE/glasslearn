from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from glasslearn.trace import Event, Recorder, Step, Trace
from glasslearn.trace.recorder import compact_int

from ._toy import MeanSeeker, Nudged

X = np.random.RandomState(1).normal(size=(30, 2))


def freeze(rec, **kw):
    return rec.freeze(algorithm="Test", params={}, feature_names=["a", "b"], **kw)


def test_disabled_recorder_is_a_no_op():
    rec = Recorder(enabled=False)
    rec.step("init", state={"x": np.ones(3)}, metrics={"m": 1.0})
    rec.note(runs=[1, 2])
    trace = freeze(rec, X=X)
    assert len(trace) == 0
    assert trace.X is None
    assert "runs" not in trace.meta


def test_recorder_copies_arrays_and_makes_them_read_only():
    buf = np.zeros(3)
    rec = Recorder()
    rec.step("update", state={"centers": buf})
    buf[:] = 7
    stored = freeze(rec)[0].state["centers"]
    np.testing.assert_array_equal(stored, np.zeros(3))
    with pytest.raises(ValueError):
        stored[0] = 1


def test_sticky_state_is_stored_only_when_it_changes():
    rec = Recorder(sticky=("labels",))
    labels = np.array([0, 1, 1, 0])
    rec.step("assign", state={"labels": labels})
    rec.step("update", state={"labels": labels.copy()})
    labels2 = np.array([0, 1, 0, 0])
    rec.step("assign", state={"labels": labels2})
    trace = freeze(rec)
    assert ["labels" in s.state for s in trace] == [True, False, True]
    np.testing.assert_array_equal(trace.latest("labels", 1), labels)
    np.testing.assert_array_equal(trace.latest("labels", -1), labels2)
    with pytest.raises(KeyError):
        trace.latest("centers", 2)


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([0, 3, 255], np.uint8),
        ([0, 256], np.uint16),
        ([-1, 100], np.int8),
        ([-1, 40000], np.int32),
        ([0, 2**40], np.int64),
    ],
)
def test_integer_arrays_use_smallest_dtype(values, dtype):
    assert compact_int(np.array(values, dtype=np.int64)).dtype == dtype
    rec = Recorder()
    rec.step("assign", state={"labels": np.array(values)})
    assert freeze(rec)[0].state["labels"].dtype == dtype


def test_every_keeps_init_and_done():
    rec = Recorder(every=3)
    rec.step("init")
    for i in range(7):
        rec.step("epoch", metrics={"i": i})
    rec.step("done")
    trace = freeze(rec)
    assert trace.phases == ["init", "epoch", "epoch", "epoch", "done"]
    np.testing.assert_array_equal(trace.metric("i")[1:4], [0, 3, 6])
    assert trace.meta["n_dropped_steps"] == 4


def test_sticky_change_in_a_dropped_step_reaches_the_next_kept_step():
    rec = Recorder(every=2, sticky=("labels",))
    rec.step("assign", state={"labels": np.array([0, 0])})
    rec.step("assign", state={"labels": np.array([1, 0])})  # dropped
    rec.step("assign", state={"labels": np.array([1, 0])})
    trace = freeze(rec)
    np.testing.assert_array_equal(trace.latest("labels", 1), [1, 0])


def test_max_steps_still_keeps_done():
    rec = Recorder(max_steps=4)
    rec.step("init")
    for _ in range(10):
        rec.step("epoch")
    rec.step("done")
    trace = freeze(rec)
    assert trace.phases == ["init", "epoch", "epoch", "done"]
    assert [s.index for s in trace] == [0, 1, 2, 3]


def test_spawn_and_adopt_keep_the_chosen_run():
    rec = Recorder(sticky=("labels",))
    runs = []
    for k in range(3):
        run = rec.spawn()
        for _ in range(k + 1):
            run.step("epoch")
        run.note(run_id=k)
        runs.append(run)
    rec.adopt(runs[1])
    rec.note(all_runs=[0, 1, 2])
    trace = freeze(rec)
    assert len(trace) == 2
    assert trace.meta["run_id"] == 1
    assert trace.meta["all_runs"] == [0, 1, 2]


def test_metric_has_one_value_per_step():
    trace = MeanSeeker(n_steps=4).fit(X).trace_
    distance = trace.metric("distance")
    assert distance.shape == (len(trace),) == (6,)
    assert np.isnan(distance[0]) and np.isnan(distance[-1])
    assert np.all(np.diff(distance[1:-1]) < 0)
    assert np.isnan(trace.metric("missing")).all()


def test_trace_indexing():
    trace = MeanSeeker().fit(X).trace_
    assert isinstance(trace[0], Step)
    assert trace[-1].phase == "done"
    assert [s.phase for s in trace[1:3]] == ["update", "update"]
    assert len(list(trace)) == len(trace)
    assert repr(trace) == "Trace(algorithm='MeanSeeker', steps=5)"


def test_step_helpers():
    step = MeanSeeker().fit(X).trace_[1]
    assert step.event(Nudged) is step.events[0]
    assert step.events_of(Nudged) == step.events
    assert step.event(Event) is step.events[0]
    assert "update" in repr(step)


def test_events_and_steps_validate_and_are_frozen():
    with pytest.raises(ValueError, match="salience"):
        Nudged(amount=1.0, toward=(0.0,), salience=1.5)
    with pytest.raises(ValueError, match="phase"):
        Step(0, "warmup", {}, {}, ())
    e = Nudged(amount=1.0, toward=(0.0, 1.0))
    with pytest.raises(FrozenInstanceError):
        e.amount = 2.0
    assert e.salience == 0.5
    assert e.kind == "Nudged"


def test_save_load_round_trip(tmp_path):
    est = MeanSeeker().fit(X)
    trace = est.trace_
    trace.meta["runs"] = [1.5, 2.5]
    path = tmp_path / "trace.npz"
    trace.save(path)
    loaded = Trace.load(path)

    assert loaded.algorithm == trace.algorithm
    assert loaded.params == trace.params
    assert loaded.feature_names == trace.feature_names
    assert loaded.meta == trace.meta
    np.testing.assert_array_equal(loaded.X, trace.X)
    assert len(loaded) == len(trace)
    for a, b in zip(loaded, trace, strict=True):
        assert (a.index, a.phase, a.metrics, a.events) == (
            b.index,
            b.phase,
            b.metrics,
            b.events,
        )
        assert a.state.keys() == b.state.keys()
        for key in a.state:
            np.testing.assert_array_equal(a.state[key], b.state[key])
            assert a.state[key].dtype == b.state[key].dtype


def test_save_load_keeps_array_params(tmp_path):
    rec = Recorder()
    rec.step("init")
    init = np.arange(4.0).reshape(2, 2)
    trace = rec.freeze(
        algorithm="T", params={"init": init, "seed": None}, feature_names=["a", "b"]
    )
    trace.save(tmp_path / "t.npz")
    loaded = Trace.load(tmp_path / "t.npz")
    np.testing.assert_array_equal(loaded.params["init"], init)
    assert loaded.params["seed"] is None


def test_to_frame():
    frame = MeanSeeker().fit(X).trace_.to_frame()
    assert list(frame.columns) == ["phase", "distance", "events"]
    assert len(frame) == 5
    assert frame.loc[1, "events"] == "Nudged"
