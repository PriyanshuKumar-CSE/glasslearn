"""KMeans narration, checked word for word against reviewed golden files.

Regenerate with ``uv run pytest --update-golden`` and review the diff.
"""

import re
import warnings

import numpy as np
import pandas as pd
import pytest

import glasslearn as gl
from glasslearn.cluster import KMeans
from glasslearn.narrate import Narrator

LEVELS = ["beginner", "detailed", "math"]


def blobs():
    """The README example: four blobs with named columns."""
    X = gl.datasets.make_blobs(n_samples=300, centers=4, random_state=0, as_frame=True)
    return KMeans(n_clusters=4, random_state=0).fit(X)


def empty_cluster():
    """One starting centroid far from every point, so its cluster empties."""
    X, _ = gl.datasets.make_blobs(n_samples=120, centers=3, random_state=4)
    init = np.vstack([X[:2], [[40.0, 40.0]]])
    return KMeans(n_clusters=3, init=init, n_init=1).fit(X)


def unscaled_max_iter():
    """Restarts, a feature 100x wider than the other, and too few iterations."""
    rng = np.random.RandomState(5)
    X = pd.DataFrame(
        {"rooms": rng.normal(3, 1, 200), "price": rng.normal(250_000, 80_000, 200)}
    )
    return KMeans(
        n_clusters=3, init="random", n_init=3, max_iter=3, random_state=1
    ).fit(X)


def stop_on_tol():
    X, _ = gl.datasets.make_blobs(n_samples=200, centers=5, random_state=6)
    return KMeans(n_clusters=5, tol=0.5, random_state=2).fit(X)


def duplicates():
    """Fewer distinct points than clusters: one cluster can never be filled."""
    X = np.repeat([[0.0, 0.0], [4.0, 1.0]], 6, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return KMeans(n_clusters=3, init=np.array([[0, 0], [4, 1], [9, 9]])).fit(X)


CASES = {
    "blobs": blobs,
    "empty_cluster": empty_cluster,
    "unscaled_max_iter": unscaled_max_iter,
    "stop_on_tol": stop_on_tol,
    "duplicates": duplicates,
}


def render(trace, level):
    narrator = Narrator(trace, level=level)
    lines = [
        f"# {trace.algorithm}, level={level}",
        "",
        f"Summary: {narrator.summary()}",
    ]
    for step in trace:
        narration = narrator.narrate(step)
        lines += [
            "",
            f"[{step.index}] {step.phase}, iteration {step.state['iteration']}",
        ]
        lines += [f"- {s}" for s in narration.sentences]
        lines += [f"  ({definition})" for _, definition in narration.glossary]
    diagnostics = narrator.diagnostics()
    lines += ["", "Diagnostics:" if diagnostics else "Diagnostics: none"]
    for d in diagnostics:
        where = "whole run" if d.step is None else f"step {d.step}"
        lines += [f"- [{where}] {d.text}", f"  Fix: {d.fix}"]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("case", CASES)
def test_golden(case, level, golden):
    golden(f"kmeans_{case}_{level}.txt", render(CASES[case]().trace_, level))


def count_sentences(text):
    return len(re.findall(r"[.!?](?:\s|$)(?=[A-Z0-9]|$)", text))


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("case", CASES)
def test_two_to_four_sentences_per_step(case, level):
    trace = CASES[case]().trace_
    narrator = Narrator(trace, level=level)
    for step in trace:
        text = narrator.narrate(step).text
        assert 1 <= count_sentences(text) <= 4, (step.index, text)
        assert not re.search(r"\bnan\b|\bNone\b(?! of)", text), text


@pytest.mark.parametrize("case", ["blobs", "empty_cluster"])
def test_beginner_steps_have_at_least_two_sentences(case):
    trace = CASES[case]().trace_
    narrator = Narrator(trace)
    for step in trace:
        assert count_sentences(narrator.narrate(step).text) >= 2, step.index


def test_narration_does_not_depend_on_viewing_order():
    trace = blobs().trace_
    forward = [Narrator(trace).narrate(i) for i in range(len(trace))]
    narrator = Narrator(trace)
    backward = [narrator.narrate(i) for i in reversed(range(len(trace)))][::-1]
    assert forward == backward


def test_column_names_appear_in_detailed_narration():
    trace = unscaled_max_iter().trace_
    text = " ".join(
        Narrator(trace, level="detailed").narrate(i).text for i in range(len(trace))
    )
    assert "price" in text
