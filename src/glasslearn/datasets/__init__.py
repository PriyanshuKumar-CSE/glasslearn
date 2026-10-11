"""Small synthetic datasets for trying out the algorithms."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np

from .._validation import check_random_state

__all__ = ["make_blobs"]


def make_blobs(
    n_samples: int | Sequence[int] = 100,
    n_features: int = 2,
    *,
    centers: int | Any = None,
    cluster_std: float | Sequence[float] = 1.0,
    center_box: tuple[float, float] = (-10.0, 10.0),
    shuffle: bool = True,
    random_state: Any = None,
    return_centers: bool = False,
    as_frame: bool = False,
) -> Any:
    """Generate round Gaussian blobs of points, a classic clustering playground.

    Same arguments and, for the same ``random_state``, the same data as
    ``sklearn.datasets.make_blobs``, plus ``as_frame``.

    Parameters
    ----------
    n_samples : int or sequence of int, default=100
        Total number of points, split evenly between blobs, or the number of
        points in each blob.
    n_features : int, default=2
        Number of features (ignored when ``centers`` is an array).
    centers : int or array of shape (n_centers, n_features), default=None
        Number of blobs (3 when None) or their exact centres.
    cluster_std : float or sequence of float, default=1.0
        Standard deviation of each blob.
    center_box : tuple of float, default=(-10.0, 10.0)
        Range for randomly placed centres.
    shuffle : bool, default=True
        Shuffle the points (otherwise they are ordered by blob).
    random_state : int, RandomState or None, default=None
        Seed for reproducible data.
    return_centers : bool, default=False
        Also return the blob centres.
    as_frame : bool, default=False
        Return only the features, as a pandas DataFrame with columns
        ``feature_0, feature_1, ...``. The blob labels are left out because
        clustering is about finding them; use ``as_frame=False`` to get them.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features) or DataFrame
        The points.
    y : ndarray of shape (n_samples,)
        The blob each point came from (not returned when ``as_frame=True``).
    centers : ndarray of shape (n_centers, n_features)
        Only when ``return_centers=True``.
    """
    generator = check_random_state(random_state)

    if isinstance(n_samples, numbers.Integral):
        if centers is None:
            centers = 3
        if isinstance(centers, numbers.Integral):
            n_centers = int(centers)
            centers = generator.uniform(
                center_box[0], center_box[1], size=(n_centers, n_features)
            )
        else:
            centers = np.asarray(centers, dtype=np.float64)
            n_centers, n_features = centers.shape
        n_per_center = [int(n_samples // n_centers)] * n_centers
        for i in range(n_samples % n_centers):
            n_per_center[i] += 1
    else:
        n_per_center = [int(n) for n in n_samples]
        n_centers = len(n_per_center)
        if centers is None:
            centers = generator.uniform(
                center_box[0], center_box[1], size=(n_centers, n_features)
            )
        centers = np.asarray(centers, dtype=np.float64)
        if centers.ndim != 2 or len(centers) != n_centers:
            raise ValueError(
                "Length of `n_samples` not consistent with number of centers. Got "
                f"n_samples = {n_samples} and centers = {centers}"
            )
        n_features = centers.shape[1]

    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.full(n_centers, float(cluster_std))
    elif len(cluster_std) != n_centers:
        raise ValueError(
            "Length of `clusters_std` not consistent with number of centers. Got "
            f"centers = {centers} and cluster_std = {cluster_std}"
        )

    ends = np.cumsum(n_per_center)
    X = np.empty((int(ends[-1]), n_features), dtype=np.float64)
    y = np.empty(int(ends[-1]), dtype=int)
    for i, (n, std) in enumerate(zip(n_per_center, cluster_std, strict=True)):
        start = ends[i - 1] if i > 0 else 0
        X[start : ends[i]] = generator.normal(
            loc=centers[i], scale=std, size=(n, n_features)
        )
        y[start : ends[i]] = i

    if shuffle:
        order = np.arange(X.shape[0])
        generator.shuffle(order)
        X, y = X[order], y[order]

    if as_frame:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - pandas is a dev dependency
            raise ImportError("as_frame=True needs pandas: pip install pandas") from exc
        frame = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(n_features)])
        return (frame, centers) if return_centers else frame
    return (X, y, centers) if return_centers else (X, y)
