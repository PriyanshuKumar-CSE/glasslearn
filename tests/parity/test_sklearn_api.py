"""glasslearn estimators must work wherever scikit-learn estimators do."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from glasslearn.cluster import KMeans

# Checks we expect to fail, each with the reason. Keep this list short.
EXPECTED_FAILED_CHECKS: dict[str, str] = {
    "check_estimators_unfitted": (
        "requires an instance of sklearn.exceptions.NotFittedError. glasslearn "
        "raises its own NotFittedError, which like sklearn's is a ValueError and "
        "an AttributeError, because glasslearn never imports scikit-learn."
    ),
}


@parametrize_with_checks(
    [KMeans()], expected_failed_checks=lambda est: EXPECTED_FAILED_CHECKS
)
def test_check_estimator(estimator, check):
    check(estimator)


X = np.random.RandomState(0).normal(size=(60, 3))


def test_clone_keeps_params_and_drops_fit():
    km = KMeans(n_clusters=3, init="random", random_state=1).fit(X)
    copy = clone(km)
    assert copy.get_params() == km.get_params()
    assert not hasattr(copy, "cluster_centers_")


def test_pipeline():
    pipe = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=0))
    labels = pipe.fit_predict(X)
    assert labels.shape == (60,)
    assert pipe.transform(X).shape == (60, 3)
    assert len(pipe[-1].trace_) > 0


def test_pipeline_routes_record_flag():
    pipe = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=0))
    pipe.fit(X, kmeans__record=False)
    assert len(pipe[-1].trace_) == 0


def test_grid_search_over_n_clusters():
    search = GridSearchCV(KMeans(random_state=0), {"n_clusters": [2, 3, 4]}, cv=3)
    search.fit(X)
    assert search.best_params_["n_clusters"] in (2, 3, 4)
    assert len(search.cv_results_["mean_test_score"]) == 3


def test_cross_val_score():
    scores = cross_val_score(KMeans(n_clusters=3, random_state=0), X, cv=3)
    assert scores.shape == (3,) and np.all(scores < 0)


@pytest.mark.parametrize("module", ["glasslearn", "glasslearn.cluster"])
def test_glasslearn_never_imports_sklearn(module):
    import subprocess
    import sys

    code = (
        f"import sys, numpy as np, {module}\n"
        "from glasslearn.cluster import KMeans\n"
        "km = KMeans(3, random_state=0).fit(np.random.rand(30, 2))\n"
        "km.predict(np.ones((1, 2)))\n"
        "print(any(m == 'sklearn' or m.startswith('sklearn.') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr
