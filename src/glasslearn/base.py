"""Base classes shared by every glasslearn estimator."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any, ClassVar, TypeVar

import numpy as np

from ._validation import check_is_fitted, validate_data
from .trace import Recorder

_E = TypeVar("_E", bound="GlassEstimator")


class GlassEstimator:
    """Base class: scikit-learn's parameter API plus a recorded training trace.

    Subclasses store their constructor arguments unchanged in ``__init__`` and
    implement ``_fit(X, y, rec)``, which trains on validated data and reports
    what happens to the :class:`~glasslearn.trace.Recorder` ``rec``.
    ``fit`` then stores the frozen record as ``trace_``.
    """

    #: State keys the recorder stores only when their value changes.
    _sticky_state: ClassVar[tuple[str, ...]] = ()

    # -- parameters (scikit-learn conventions) --------------------------------

    @classmethod
    def _get_param_names(cls) -> list[str]:
        init = cls.__init__
        if init is object.__init__:
            return []
        names = []
        for p in inspect.signature(init).parameters.values():
            if p.name == "self" or p.kind == p.VAR_KEYWORD:
                continue
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    f"{cls.__name__} should not take *args in __init__; "
                    "glasslearn estimators list every parameter explicitly."
                )
            names.append(p.name)
        return sorted(names)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor parameters, as scikit-learn's ``get_params``."""
        out: dict[str, Any] = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                out.update((f"{key}__{k}", v) for k, v in value.get_params().items())
            out[key] = value
        return out

    def set_params(self: _E, **params: Any) -> _E:
        """Set constructor parameters, including nested ``step__param`` keys."""
        if not params:
            return self
        valid = self.get_params(deep=True)
        nested: dict[str, dict[str, Any]] = defaultdict(dict)
        for full_key, value in params.items():
            key, delim, sub_key = full_key.partition("__")
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {self._get_param_names()!r}."
                )
            if delim:
                nested[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid[key] = value
        for key, sub_params in nested.items():
            valid[key].set_params(**sub_params)
        return self

    def __repr__(self) -> str:
        defaults = inspect.signature(type(self).__init__).parameters
        shown = []
        for key, value in self.get_params(deep=False).items():
            default = defaults[key].default
            if default is inspect.Parameter.empty or repr(value) != repr(default):
                text = repr(value)
                if isinstance(value, np.ndarray):
                    text = np.array2string(
                        value, separator=", ", threshold=12, edgeitems=2
                    )
                    text = f"array({text})"
                shown.append(f"{key}={text}")
        return f"{type(self).__name__}({', '.join(shown)})"

    # -- fitting ---------------------------------------------------------------

    def fit(self: _E, X: Any, y: Any = None, *, record: bool = True) -> _E:
        """Fit the model and record every training step in ``trace_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. A pandas DataFrame keeps its column names.
        y : array-like or None
            Targets (ignored by unsupervised estimators).
        record : bool, default=True
            Record the training steps. With False the fitted model is
            identical but ``trace_`` is empty, which is faster.

        Returns
        -------
        self
        """
        X, y = self._validate_fit_data(X, y)
        rec = Recorder(
            enabled=record, sticky=self._sticky_state, **self._record_options()
        )
        self._fit(X, y, rec)
        self.trace_ = rec.freeze(
            algorithm=type(self).__name__,
            params=self.get_params(deep=False),
            feature_names=self.feature_names_in_,
            target_names=self._target_names(),
            X=X,
            y=y,
            meta=self._meta(X, y),
        )
        return self

    def _fit(self, X: np.ndarray, y: np.ndarray | None, rec: Recorder) -> None:
        raise NotImplementedError

    def _validate_fit_data(
        self, X: Any, y: Any
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Validate training data; unsupervised estimators ignore ``y``."""
        return validate_data(self, X, reset=True), None

    def _validate_predict_data(self, X: Any) -> np.ndarray:
        """Check the model is fitted and ``X`` matches the training features."""
        check_is_fitted(self)
        return validate_data(self, X, reset=False)

    def _record_options(self) -> dict[str, Any]:
        """Extra Recorder settings (``every``, ``max_steps``) for this estimator."""
        return {}

    def _target_names(self) -> list[str] | None:
        return None

    def _meta(self, X: np.ndarray, y: np.ndarray | None) -> dict[str, Any]:
        return {"n_samples": int(X.shape[0])}

    # -- scikit-learn integration -----------------------------------------------

    def __sklearn_tags__(self):
        """Describe the estimator to scikit-learn (``check_estimator``, CV tools).

        scikit-learn calls this hook and requires its own ``Tags`` objects, so
        they are imported here, at call time. This is the only place glasslearn
        touches scikit-learn: it is never imported by glasslearn itself, and
        whenever this runs scikit-learn has already been loaded by its caller.
        """
        from sklearn.utils import InputTags, Tags, TargetTags

        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=False),
            transformer_tags=None,
            regressor_tags=None,
            classifier_tags=None,
            input_tags=InputTags(),
        )


class ClusterMixin:
    """Adds ``fit_predict`` to clustering estimators."""

    _estimator_type = "clusterer"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "clusterer"
        return tags

    def fit_predict(self, X: Any, y: Any = None, *, record: bool = True) -> np.ndarray:
        """Fit the model and return the cluster label of each training sample."""
        return self.fit(X, y, record=record).labels_


class TransformerMixin:
    """Adds ``fit_transform`` to estimators that have ``transform``."""

    def __sklearn_tags__(self):
        from sklearn.utils import TransformerTags

        tags = super().__sklearn_tags__()
        tags.transformer_tags = TransformerTags(preserves_dtype=["float64"])
        return tags

    def fit_transform(
        self, X: Any, y: Any = None, *, record: bool = True
    ) -> np.ndarray:
        """Fit the model, then transform the training data."""
        return self.fit(X, y, record=record).transform(X)
