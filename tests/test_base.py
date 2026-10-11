import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.base import clone

from glasslearn._validation import NotFittedError

from ._toy import MeanSeeker, Wrapper

X = np.random.RandomState(0).normal(size=(20, 3))


def test_init_stores_args_unchanged():
    init = np.array([1.0])
    est = MeanSeeker(n_steps="not validated", rate=init)
    assert est.n_steps == "not validated"
    assert est.rate is init
    assert not [k for k in vars(est) if k.endswith("_")]


def test_get_params_and_set_params():
    est = MeanSeeker(n_steps=5)
    assert est.get_params() == {"n_steps": 5, "rate": 0.5}
    assert est.set_params(rate=0.1) is est
    assert est.rate == 0.1
    with pytest.raises(ValueError, match="Invalid parameter 'speed'"):
        est.set_params(speed=2)


def test_nested_params():
    est = Wrapper(estimator=MeanSeeker())
    params = est.get_params()
    assert params["estimator__n_steps"] == 3
    assert "estimator__n_steps" not in est.get_params(deep=False)
    est.set_params(estimator__n_steps=9, flag=False)
    assert est.estimator.n_steps == 9
    assert est.flag is False


def test_repr_shows_changed_params_only():
    assert repr(MeanSeeker()) == "MeanSeeker()"
    assert repr(MeanSeeker(rate=0.25)) == "MeanSeeker(rate=0.25)"
    assert repr(Wrapper(MeanSeeker(n_steps=2))) == (
        "Wrapper(estimator=MeanSeeker(n_steps=2))"
    )


def test_clone():
    est = MeanSeeker(n_steps=4).fit(X)
    copy = clone(est)
    assert copy.get_params() == est.get_params()
    assert not hasattr(copy, "point_")


def test_fit_sets_feature_info_and_trace():
    est = MeanSeeker().fit(X)
    assert est.n_features_in_ == 3
    assert isinstance(est.feature_names_in_, np.ndarray)
    assert est.feature_names_in_.dtype == object
    assert list(est.feature_names_in_) == ["x0", "x1", "x2"]
    assert est.trace_.algorithm == "MeanSeeker"
    assert est.trace_.feature_names == ["x0", "x1", "x2"]
    assert est.trace_.params == {"n_steps": 3, "rate": 0.5}
    assert est.trace_.phases == ["init", "update", "update", "update", "done"]
    np.testing.assert_array_equal(est.trace_.X, X)


def test_dataframe_column_names_are_kept():
    df = pd.DataFrame(X, columns=["age", "income", "height"])
    est = MeanSeeker().fit(df)
    assert list(est.feature_names_in_) == ["age", "income", "height"]
    assert est.trace_.feature_names == ["age", "income", "height"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est.predict(df)
    with pytest.warns(UserWarning, match="does not have valid feature names"):
        est.predict(X)
    with pytest.raises(ValueError, match="must be in the same order"):
        est.predict(df[["income", "age", "height"]])
    with pytest.raises(ValueError, match="unseen at fit time:\n- weight"):
        est.predict(df.rename(columns={"height": "weight"}))


def test_numpy_fit_then_dataframe_predict_warns():
    est = MeanSeeker().fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est.predict(X)
    with pytest.warns(UserWarning, match="fitted without feature names"):
        est.predict(pd.DataFrame(X, columns=["a", "b", "c"]))


def test_mixed_column_name_types_rejected():
    df = pd.DataFrame(X, columns=["a", 1, "c"])
    with pytest.raises(TypeError, match="only supported if all input features"):
        MeanSeeker().fit(df)


def test_record_false_gives_same_model_and_empty_trace():
    a = MeanSeeker().fit(X)
    b = MeanSeeker().fit(X, record=False)
    np.testing.assert_array_equal(a.point_, b.point_)
    assert len(b.trace_) == 0
    assert b.trace_.X is None


def test_trace_does_not_alias_training_data():
    data = X.copy()
    est = MeanSeeker().fit(data)
    data[:] = 0
    assert est.trace_.X.any()
    assert not est.trace_.X.flags.writeable


def test_unfitted_predict_raises():
    with pytest.raises(NotFittedError, match="not fitted yet"):
        MeanSeeker().predict(X)
    assert issubclass(NotFittedError, ValueError)
    assert issubclass(NotFittedError, AttributeError)


def test_predict_checks_feature_count():
    est = MeanSeeker().fit(X)
    with pytest.raises(
        ValueError, match="X has 2 features, but MeanSeeker is expecting 3"
    ):
        est.predict(X[:, :2])


@pytest.mark.parametrize(
    ("bad", "error", "match"),
    [
        (X[:, 0], ValueError, "Expected 2D array, got 1D array"),
        (X[None], ValueError, "Found array with dim 3"),
        (np.empty((0, 3)), ValueError, r"0 sample\(s\)"),
        (np.empty((4, 0)), ValueError, r"0 feature\(s\)"),
        (np.where(X > 1, np.nan, X), ValueError, "contains NaN"),
        (np.where(X > 1, np.inf, X), ValueError, "contains infinity"),
        (X.astype(complex), ValueError, "Complex data not supported"),
        (np.array([["a", "b"]]), ValueError, "bytes/strings"),
        (sp.csr_matrix(X), TypeError, "dense data is required"),
    ],
)
def test_invalid_input_rejected(bad, error, match):
    with pytest.raises(error, match=match):
        MeanSeeker().fit(bad)


def test_integer_and_list_input_accepted():
    est = MeanSeeker().fit([[1, 2], [3, 4], [5, 7]])
    assert est.point_.dtype == np.float64


def test_pickle_round_trip():
    est = MeanSeeker().fit(X)
    copy = pickle.loads(pickle.dumps(est))
    np.testing.assert_array_equal(copy.predict(X), est.predict(X))
    assert len(copy.trace_) == len(est.trace_)
