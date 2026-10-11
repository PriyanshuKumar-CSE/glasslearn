"""Ways to show many-feature data on a flat plot.

A projection is fitted once on the training data and then reused for every
step, so the picture does not shift as training goes on.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class Projection:
    """Maps feature-space rows ``(n, n_features)`` to plot positions ``(n, 2)``."""

    #: Axis titles, set by :meth:`fit`.
    axis_titles: tuple[str, str] = ("", "")

    def fit(self, X: np.ndarray, feature_names: Sequence[str]) -> Projection:
        raise NotImplementedError

    def transform(self, X: Any) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, P: Any) -> np.ndarray:
        """Feature-space rows for plot positions (other features at their mean)."""
        raise NotImplementedError

    @property
    def note(self) -> str:
        """A sentence about what the projection hides, or "" if nothing."""
        return ""


class Identity(Projection):
    """Plot the features as they are (data with one or two features)."""

    def fit(self, X: np.ndarray, feature_names: Sequence[str]) -> Identity:
        X = np.asarray(X, dtype=float)
        if X.shape[1] > 2:
            raise ValueError(
                f"Identity needs at most 2 features, got {X.shape[1]}; use 'pca' or "
                "a pair of feature names."
            )
        self.n_features_ = X.shape[1]
        names = list(feature_names)
        self.axis_titles = (names[0], names[1] if len(names) > 1 else "")
        return self

    def transform(self, X: Any) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.n_features_ == 1:
            return np.column_stack([X[:, 0], np.zeros(len(X))])
        return X[:, :2].copy()

    def inverse_transform(self, P: Any) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, dtype=float))
        return P[:, : self.n_features_].copy()


class FeaturePair(Projection):
    """Plot two chosen features against each other.

    Parameters
    ----------
    name_a, name_b : str
        Feature names for the horizontal and vertical axes.
    """

    def __init__(self, name_a: str, name_b: str) -> None:
        self.name_a = name_a
        self.name_b = name_b

    def fit(self, X: np.ndarray, feature_names: Sequence[str]) -> FeaturePair:
        names = [str(n) for n in feature_names]
        for name in (self.name_a, self.name_b):
            if name not in names:
                raise ValueError(f"Unknown feature {name!r}; features are {names}.")
        self.columns_ = (names.index(self.name_a), names.index(self.name_b))
        self.mean_ = np.asarray(X, dtype=float).mean(axis=0)
        self.n_hidden_ = len(names) - 2
        self.axis_titles = (self.name_a, self.name_b)
        return self

    def transform(self, X: Any) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return X[:, list(self.columns_)].copy()

    def inverse_transform(self, P: Any) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, dtype=float))
        out = np.tile(self.mean_, (len(P), 1))
        out[:, list(self.columns_)] = P
        return out

    @property
    def note(self) -> str:
        if self.n_hidden_ <= 0:
            return ""
        other = (
            "feature is" if self.n_hidden_ == 1 else f"{self.n_hidden_} features are"
        )
        return (
            f"Only {self.name_a} and {self.name_b} are drawn; the other {other} hidden."
        )


class PCA(Projection):
    """Principal component analysis with numpy's SVD, keeping two components.

    Attributes
    ----------
    components_ : ndarray of shape (2, n_features)
        Directions of the two axes in feature space.
    mean_ : ndarray of shape (n_features,)
        Training mean, subtracted before projecting.
    explained_variance_ratio_ : ndarray of shape (2,)
        Share of the total variance each axis keeps.
    """

    def fit(self, X: np.ndarray, feature_names: Sequence[str] = ()) -> PCA:
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        # Fix the sign of each axis (largest loading positive) so plots are stable.
        signs = np.sign(vt[np.arange(len(vt)), np.argmax(np.abs(vt), axis=1)])
        signs[signs == 0] = 1
        vt = vt * signs[:, None]
        variance = s**2
        total = variance.sum()
        ratio = variance / total if total > 0 else np.zeros_like(variance)
        k = min(2, vt.shape[0])
        self.components_ = np.zeros((2, X.shape[1]))
        self.components_[:k] = vt[:k]
        self.explained_variance_ratio_ = np.zeros(2)
        self.explained_variance_ratio_[:k] = ratio[:k]
        r1, r2 = (round(100 * v) for v in self.explained_variance_ratio_)
        self.axis_titles = (f"PC1 ({r1}%)", f"PC2 ({r2}%)")
        return self

    def transform(self, X: Any) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, P: Any) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, dtype=float))
        return P @ self.components_ + self.mean_

    @property
    def note(self) -> str:
        kept = round(100 * float(self.explained_variance_ratio_.sum()))
        return (
            f"The plot shows the 2 directions of greatest spread (PCA), which keep "
            f"{kept}% of the variance; distances on screen are approximate."
        )


def make_projection(
    spec: Any, X: np.ndarray, feature_names: Sequence[str]
) -> Projection:
    """Build and fit a projection: ``"auto"``, ``"pca"``, two names or an instance.

    ``"auto"`` draws the raw features when there are at most two, else PCA.
    """
    X = np.asarray(X, dtype=float)
    if isinstance(spec, Projection):
        return spec if _is_fitted(spec) else spec.fit(X, feature_names)
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return FeaturePair(str(spec[0]), str(spec[1])).fit(X, feature_names)
    if spec == "auto":
        spec = "identity" if X.shape[1] <= 2 else "pca"
    if spec == "identity":
        return Identity().fit(X, feature_names)
    if spec == "pca":
        return PCA().fit(X, feature_names)
    raise ValueError(
        f"projection must be 'auto', 'pca', a pair of feature names or a Projection, "
        f"got {spec!r}"
    )


def _is_fitted(projection: Projection) -> bool:
    return any(k.endswith("_") for k in vars(projection))
