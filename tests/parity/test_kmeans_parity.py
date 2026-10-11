"""glasslearn's KMeans must match scikit-learn's, number for number."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import make_blobs

from glasslearn.cluster import KMeans

SEEDS = range(5)
RTOL = 1e-7


def blobs_2d():
    return make_blobs(n_samples=300, centers=4, random_state=0)[0]


def blobs_10d():
    return make_blobs(n_samples=500, n_features=10, centers=5, random_state=1)[0]


def uniform_noise():
    return np.random.RandomState(2).uniform(-5, 5, size=(250, 3))


def named_frame():
    X = make_blobs(n_samples=200, centers=3, random_state=3)[0]
    return pd.DataFrame(X, columns=["height_cm", "weight_kg"])


DATASETS = {
    "blobs_2d": (blobs_2d, 4),
    "blobs_10d": (blobs_10d, 5),
    "uniform": (uniform_noise, 6),
    "frame": (named_frame, 3),
}


def assert_same_fit(ours, theirs):
    np.testing.assert_array_equal(ours.labels_, theirs.labels_)
    np.testing.assert_allclose(
        ours.cluster_centers_, theirs.cluster_centers_, rtol=RTOL
    )
    np.testing.assert_allclose(ours.inertia_, theirs.inertia_, rtol=RTOL)
    assert ours.n_iter_ == theirs.n_iter_
    assert ours.labels_.dtype == theirs.labels_.dtype


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", DATASETS)
def test_explicit_init_matches(name, seed):
    make, k = DATASETS[name]
    X = make()
    values = np.asarray(X)
    rows = np.random.RandomState(seed).choice(len(values), size=k, replace=False)
    init = values[rows] + 0.01  # off the data points, so the first pass matters
    ours = KMeans(n_clusters=k, init=init, n_init=1).fit(X)
    theirs = SkKMeans(n_clusters=k, init=init, n_init=1).fit(X)
    assert_same_fit(ours, theirs)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("init", ["k-means++", "random"])
@pytest.mark.parametrize("name", DATASETS)
def test_seeded_init_matches(name, init, seed):
    """Same random_state gives the same seeds, restarts and winner."""
    make, k = DATASETS[name]
    X = make()
    ours = KMeans(n_clusters=k, init=init, random_state=seed).fit(X)
    theirs = SkKMeans(n_clusters=k, init=init, random_state=seed).fit(X)
    assert_same_fit(ours, theirs)


@pytest.mark.parametrize(
    "params",
    [
        {"tol": 0.0},
        {"tol": 1.0},
        {"max_iter": 1},
        {"max_iter": 2},
        {"n_init": 3, "init": "k-means++", "random_state": 7},
        {"n_clusters": 1},
    ],
)
def test_stopping_rules_match(params):
    X = uniform_noise()
    params = {"n_clusters": 6, "random_state": 0, **params}
    assert_same_fit(KMeans(**params).fit(X), SkKMeans(**params).fit(X))


def test_empty_cluster_is_relocated_like_sklearn():
    X = blobs_2d()
    # One centroid far from every point: it gets no points on the first pass.
    init = np.vstack([X[:3], [[100.0, 100.0]]])
    ours = KMeans(n_clusters=4, init=init, n_init=1).fit(X)
    theirs = SkKMeans(n_clusters=4, init=init, n_init=1).fit(X)
    assert_same_fit(ours, theirs)


def test_several_empty_clusters_match():
    X = uniform_noise()
    init = np.vstack([X[:3], np.full((3, 3), 50.0) + np.arange(3)[:, None]])
    ours = KMeans(n_clusters=6, init=init, n_init=1).fit(X)
    theirs = SkKMeans(n_clusters=6, init=init, n_init=1).fit(X)
    assert_same_fit(ours, theirs)


def test_duplicate_points_match():
    X = np.repeat([[0.0, 0.0], [1.0, 1.0]], 10, axis=0)
    init = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = KMeans(n_clusters=3, init=init, n_init=1).fit(X)
        theirs = SkKMeans(n_clusters=3, init=init, n_init=1).fit(X)
    assert_same_fit(ours, theirs)


@pytest.mark.parametrize("name", DATASETS)
def test_methods_match_on_new_data(name):
    make, k = DATASETS[name]
    X = make()
    values = np.asarray(X)
    X_new = values[::7] + np.random.RandomState(0).normal(size=values[::7].shape)
    if isinstance(X, pd.DataFrame):
        X_new = pd.DataFrame(X_new, columns=X.columns)
    ours = KMeans(n_clusters=k, random_state=0).fit(X)
    theirs = SkKMeans(n_clusters=k, random_state=0).fit(X)
    np.testing.assert_array_equal(ours.predict(X_new), theirs.predict(X_new))
    np.testing.assert_allclose(
        ours.transform(X_new), theirs.transform(X_new), rtol=RTOL
    )
    np.testing.assert_allclose(ours.score(X_new), theirs.score(X_new), rtol=RTOL)
    np.testing.assert_array_equal(
        KMeans(n_clusters=k, random_state=0).fit_predict(X),
        SkKMeans(n_clusters=k, random_state=0).fit_predict(X),
    )
    np.testing.assert_allclose(
        KMeans(n_clusters=k, random_state=0).fit_transform(X),
        SkKMeans(n_clusters=k, random_state=0).fit_transform(X),
        rtol=RTOL,
    )
    np.testing.assert_array_equal(
        ours.get_feature_names_out(), theirs.get_feature_names_out()
    )
    if isinstance(X, pd.DataFrame):
        np.testing.assert_array_equal(ours.feature_names_in_, theirs.feature_names_in_)


def test_float32_and_int_input_match():
    X = blobs_2d()
    for data in (X.astype(np.float32), np.round(X * 10).astype(np.int64)):
        ours = KMeans(n_clusters=4, random_state=0).fit(data)
        theirs = SkKMeans(n_clusters=4, random_state=0).fit(data)
        np.testing.assert_array_equal(ours.labels_, theirs.labels_)
        np.testing.assert_allclose(
            ours.cluster_centers_, theirs.cluster_centers_, rtol=1e-5
        )
        assert ours.n_iter_ == theirs.n_iter_


def test_verbose_output_matches_sklearn_format(capsys):
    X = blobs_2d()
    KMeans(n_clusters=4, random_state=0, verbose=1).fit(X)
    ours = capsys.readouterr().out.splitlines()
    SkKMeans(n_clusters=4, random_state=0, verbose=1).fit(X)
    theirs = capsys.readouterr().out.splitlines()

    def shape(lines):
        return [
            " ".join(w for w in line.split() if not w[0].isdigit()) for line in lines
        ]

    assert shape(ours) == shape(theirs)
