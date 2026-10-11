"""Input checks shared by all estimators.

Error messages mirror scikit-learn's so that its estimator checks and users'
habits carry over.
"""

from __future__ import annotations

import numbers
import warnings
from typing import Any

import numpy as np


class NotFittedError(ValueError, AttributeError):
    """Raised when a method needs a fitted estimator but ``fit`` was not called."""


class InvalidParameterError(ValueError, TypeError):
    """Raised by ``fit`` when a constructor parameter has a wrong type or value."""


class ConvergenceWarning(UserWarning):
    """Warns that fitting ended in a degenerate or unconverged state."""


def is_arraylike_not_scalar(x: Any) -> bool:
    """True for arrays, lists and similar, but not for strings or numbers."""
    return (
        hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")
    ) and not np.isscalar(x)


def check_param(estimator: Any, name: str, ok: bool, expected: str) -> None:
    """Raise InvalidParameterError (sklearn's wording) unless ``ok``."""
    if not ok:
        raise InvalidParameterError(
            f"The {name!r} parameter of {type(estimator).__name__} must be "
            f"{expected}. Got {getattr(estimator, name)!r} instead."
        )


def check_random_state(seed: Any) -> np.random.RandomState:
    """Turn ``None``, an int or a RandomState into a RandomState (as scikit-learn)."""
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        f"{seed!r} cannot be used to seed a numpy.random.RandomState instance"
    )


def check_is_fitted(estimator: Any) -> None:
    """Raise NotFittedError unless ``fit`` has set a trailing-underscore attribute."""
    fitted = [k for k in vars(estimator) if k.endswith("_") and not k.startswith("__")]
    if not fitted:
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. Call 'fit' "
            "with appropriate arguments before using this estimator."
        )


def _is_sparse(X: Any) -> bool:
    # Duck-typed so scipy is not needed: sparse arrays and matrices have both.
    return hasattr(X, "tocsr") and hasattr(X, "nnz")


def _is_dataframe(X: Any) -> bool:
    return (
        hasattr(X, "columns") and hasattr(X, "to_numpy") and getattr(X, "ndim", 2) == 2
    )


def get_feature_names(X: Any) -> np.ndarray | None:
    """Return the column names of a dataframe if they are all strings, else None."""
    if not _is_dataframe(X):
        return None
    names = np.asarray(list(X.columns), dtype=object)
    if len(names) == 0:
        return None
    types = sorted(t.__qualname__ for t in {type(v) for v in names})
    if len(types) > 1 and "str" in types:
        raise TypeError(
            "Feature names are only supported if all input features have string names, "
            f"but your input has {types} as feature name / column name types. "
            "If you want feature names to be stored and validated, you must convert "
            "them all to strings, by using X.columns = X.columns.astype(str) for "
            "example. Otherwise you can remove feature / column names from your input "
            "data, or convert them all to a non-string data type."
        )
    if types == ["str"]:
        return names
    return None


def check_array(
    X: Any, *, estimator_name: str | None = None, input_name: str = "X"
) -> np.ndarray:
    """Return ``X`` as a finite, non-empty, 2D float64 array."""
    context = f" by {estimator_name}" if estimator_name else ""
    if _is_sparse(X):
        raise TypeError(
            f"Sparse data was passed for {input_name}, but dense data is required. "
            "Use '.toarray()' to convert to a dense numpy array."
        )
    raw = X.to_numpy() if _is_dataframe(X) else np.asarray(X)
    if raw.dtype.kind == "c":
        raise ValueError(f"Complex data not supported\n{raw}\n")
    if raw.ndim == 0:
        raise ValueError(
            f"Expected 2D array, got scalar array instead:\narray={raw}.\n"
            "Reshape your data either using array.reshape(-1, 1) if "
            "your data has a single feature or array.reshape(1, -1) "
            "if it contains a single sample."
        )
    if raw.ndim == 1:
        raise ValueError(
            f"Expected 2D array, got 1D array instead:\narray={raw}.\n"
            "Reshape your data either using array.reshape(-1, 1) if "
            "your data has a single feature or array.reshape(1, -1) "
            "if it contains a single sample."
        )
    if raw.ndim >= 3:
        raise ValueError(
            f"Found array with dim {raw.ndim}, while dim <= 2 is required{context}."
        )
    if raw.dtype.kind in "USV":
        raise ValueError(
            "dtype='numeric' is not compatible with arrays of bytes/strings."
            "Convert your data to numeric values explicitly instead."
        )
    arr = np.ascontiguousarray(raw, dtype=np.float64)
    n_samples, n_features = arr.shape
    if n_samples < 1:
        raise ValueError(
            f"Found array with {n_samples} sample(s) (shape={arr.shape}) while a "
            f"minimum of 1 is required{context}."
        )
    if n_features < 1:
        raise ValueError(
            f"Found array with {n_features} feature(s) (shape={arr.shape}) while a "
            f"minimum of 1 is required{context}."
        )
    if not np.isfinite(arr.sum()):
        if np.isnan(arr).any():
            raise ValueError(f"Input {input_name} contains NaN.")
        if not np.isfinite(arr).all():
            raise ValueError(
                f"Input {input_name} contains infinity or a value too large for "
                "dtype('float64')."
            )
    return arr


def _check_feature_names(estimator: Any, X: Any) -> None:
    """Compare the column names of ``X`` with those seen during fit."""
    names = get_feature_names(X)
    cls = type(estimator).__name__
    given = getattr(estimator, "_feature_names_given", False)
    if names is None and not given:
        return
    if names is not None and not given:
        warnings.warn(
            f"X has feature names, but {cls} was fitted without feature names",
            stacklevel=3,
        )
        return
    if names is None and given:
        warnings.warn(
            f"X does not have valid feature names, but {cls} was fitted with "
            "feature names",
            stacklevel=3,
        )
        return

    fitted = estimator.feature_names_in_
    if len(fitted) == len(names) and np.all(fitted == names):
        return
    message = "The feature names should match those that were passed during fit.\n"
    unexpected = sorted(set(names) - set(fitted))
    missing = sorted(set(fitted) - set(names))

    def add_names(group: list[str]) -> str:
        out = "".join(f"- {n}\n" for n in group[:5])
        return out + ("- ...\n" if len(group) > 5 else "")

    if unexpected:
        message += "Feature names unseen at fit time:\n" + add_names(unexpected)
    if missing:
        message += "Feature names seen at fit time, yet now missing:\n" + add_names(
            missing
        )
    if not unexpected and not missing:
        message += "Feature names must be in the same order as they were in fit.\n"
    raise ValueError(message)


def validate_data(estimator: Any, X: Any, *, reset: bool) -> np.ndarray:
    """Validate ``X`` and set (``reset=True``) or check the fitted feature info.

    On reset, sets ``n_features_in_`` and ``feature_names_in_``. Unlike
    scikit-learn, ``feature_names_in_`` is always set: when ``X`` has no
    column names it holds ``"x0", "x1", ...`` so that narration and plots can
    name every axis.
    """
    cls = type(estimator).__name__
    if reset:
        names = get_feature_names(X)
        arr = check_array(X, estimator_name=cls)
        estimator.n_features_in_ = arr.shape[1]
        estimator._feature_names_given = names is not None
        if names is None:
            names = np.asarray([f"x{i}" for i in range(arr.shape[1])], dtype=object)
        estimator.feature_names_in_ = names
        return arr

    _check_feature_names(estimator, X)
    arr = check_array(X, estimator_name=cls)
    n_features = arr.shape[1]
    if n_features != estimator.n_features_in_:
        raise ValueError(
            f"X has {n_features} features, but {cls} is expecting "
            f"{estimator.n_features_in_} features as input."
        )
    return arr
