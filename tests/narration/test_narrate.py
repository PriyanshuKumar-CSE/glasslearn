import numpy as np
import pytest

from glasslearn.cluster import KMeans
from glasslearn.narrate import NarrationContext, Narrator, unscaled_features
from glasslearn.narrate.format import (
    distinct,
    join_words,
    number,
    percent,
    plural,
    signed,
)
from glasslearn.narrate.glossary import define, terms_in

X = np.random.RandomState(0).normal(size=(50, 2))
TRACE = KMeans(n_clusters=2, random_state=0).fit(X).trace_


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0, "0"),
        (3.0, "3"),
        (12.345, "12.3"),
        (0.012345, "0.0123"),
        (999.7, "1,000"),
        (1234.5, "1,234"),
        (250_000.4, "250,000"),
        (-0.5, "-0.5"),
        (0.00001234, "1.23e-05"),
        (312_048_109_704, "312 billion"),
        (2_500_000, "2.5 million"),
        (4.2e16, "4.20e+16"),
    ],
)
def test_number(value, text):
    assert number(value) == text


def test_number_helpers():
    assert signed(1.5) == "+1.5" and signed(-1.5) == "-1.5" and signed(0) == "0"
    assert percent(0.853) == "85%"
    assert plural(1, "point") == "1 point" and plural(3, "point") == "3 points"
    assert plural(2, "index", "indices") == "2 indices"
    assert join_words([]) == "" and join_words(["a"]) == "a"
    assert join_words(["a", "b", "c"]) == "a, b and c"


def test_distinct_adds_digits_only_when_needed():
    assert distinct(812.0, 640.0) == ("812", "640")
    assert distinct(542.31, 541.87) == ("542.3", "541.9")
    assert distinct(1234.5, 1234.4) == ("1,234.5", "1,234.4")
    assert distinct(5.0, 5.0 + 1e-12) == ("5", "5")
    assert distinct(312.05e9, 312.01e9) == ("312.1 billion", "312 billion")


@pytest.mark.parametrize(
    ("new", "words"),
    [
        ((1, 0), "to the right"),
        ((1, 1), "up and to the right"),
        ((0, 1), "up"),
        ((-1, 1), "up and to the left"),
        ((-1, 0), "to the left"),
        ((-1, -1), "down and to the left"),
        ((0, -1), "down"),
        ((1, -1), "down and to the right"),
        ((0, 0), "nowhere"),
    ],
)
def test_direction(new, words):
    ctx = NarrationContext(TRACE)
    assert ctx.direction((0, 0), new) == words


def test_direction_uses_the_projection():
    swap = lambda rows: np.asarray(rows)[:, ::-1]  # noqa: E731
    ctx = NarrationContext(TRACE, project=swap)
    assert ctx.direction((0, 0), (1, 0)) == "up"


def test_biggest_change():
    ctx = NarrationContext(TRACE)
    assert ctx.biggest_change((0, 0), (0.5, -2.0)) == ("x1", -2.0)


def test_glossary():
    assert terms_in("The centroids moved; inertia fell.") == ["centroid", "inertia"]
    assert terms_in("k-means++ chose") == ["k-means++"]
    assert terms_in("k-means stopped") == []
    assert define("inertia").startswith("Inertia: the sum")


def test_glossary_is_skipped_at_math_level():
    narrator = Narrator(TRACE, level="math")
    assert all(not narrator.narrate(i).glossary for i in range(len(TRACE)))


def test_each_term_is_defined_once():
    narrator = Narrator(TRACE)
    terms = [t for i in range(len(TRACE)) for t, _ in narrator.narrate(i).glossary]
    assert len(terms) == len(set(terms)) and "centroid" in terms


def test_bad_level():
    with pytest.raises(ValueError, match="level must be one of"):
        Narrator(TRACE, level="expert")


def test_unscaled_features_diagnostic():
    wide = X * np.array([1.0, 50.0])
    trace = KMeans(n_clusters=2, random_state=0).fit(wide).trace_
    [found] = unscaled_features(trace)
    assert "x1 spreads" in found.text and "StandardScaler" in found.fix
    assert unscaled_features(TRACE) == []


def test_diagnostics_filter_by_step():
    trace = KMeans(n_clusters=2, max_iter=1, random_state=0).fit(X * [1, 50]).trace_
    narrator = Narrator(trace)
    assert len(narrator.diagnostics()) == 2
    assert len(narrator.diagnostics(upto=0)) == 1  # the max_iter one shows at the end
