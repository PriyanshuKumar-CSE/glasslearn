"""K-means clustering (Lloyd's algorithm), recorded step by step."""

from __future__ import annotations

import numbers
import warnings
from typing import Any

import numpy as np

from ..._validation import (
    ConvergenceWarning,
    check_array,
    check_is_fitted,
    check_param,
    check_random_state,
    is_arraylike_not_scalar,
)
from ...base import ClusterMixin, GlassEstimator, TransformerMixin
from ...trace import Recorder
from .events import (
    CentroidMoved,
    Converged,
    EmptyCluster,
    InitChosen,
    PointsReassigned,
)


class KMeans(ClusterMixin, TransformerMixin, GlassEstimator):
    """K-means clustering that records every assignment and update step.

    Each iteration has two halves: every point joins its nearest centroid
    (*assign*), then every centroid moves to the mean of its points
    (*update*). Training stops when no point changes cluster, when the
    centroids barely move, or after ``max_iter`` iterations. Call
    ``explain()`` on the fitted model to watch it happen.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters (and centroids).
    init : {"k-means++", "random"}, callable or array of shape \
            (n_clusters, n_features), default="k-means++"
        How to pick the starting centroids. ``"k-means++"`` spreads them out
        (scikit-learn's greedy variant), ``"random"`` picks random rows, an
        array gives them directly, and a callable is called as
        ``init(X, n_clusters, random_state=random_state)``.
    n_init : "auto" or int, default="auto"
        Restarts with different starting centroids; the run with the lowest
        inertia is kept. ``"auto"`` means 1 for ``"k-means++"`` or an array and
        10 for ``"random"`` or a callable.
    max_iter : int, default=300
        Most iterations per run.
    tol : float, default=1e-4
        Stop when the sum of squared centroid moves is at most ``tol`` times
        the mean variance of the features.
    verbose : int, default=0
        Print progress like scikit-learn does.
    random_state : int, RandomState or None, default=None
        Seed for the starting centroids.
    copy_x : bool, default=True
        Accepted for compatibility; ``X`` is never modified either way.
    algorithm : {"lloyd", "elkan"}, default="lloyd"
        Accepted for compatibility; both run Lloyd's algorithm.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Final centroids.
    labels_ : ndarray of shape (n_samples,)
        Cluster of each training point.
    inertia_ : float
        Sum of squared distances from each point to its centroid.
    n_iter_ : int
        Iterations run by the kept restart.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Column names seen during ``fit`` (``"x0", "x1", ...`` for arrays).
    trace_ : Trace
        The recorded steps of the kept restart. ``trace_.meta["run_inertia"]``
        holds the inertia of every restart.

    Differences from scikit-learn
    -----------------------------
    - ``fit`` takes ``record`` (default True) and no ``sample_weight``;
      ``score`` takes no ``sample_weight`` either.
    - Sparse input is rejected, and float32 input is computed in float64.
    - ``algorithm="elkan"`` runs Lloyd's algorithm, which finds the same
      clustering; ``copy_x`` has no effect because ``X`` is never modified.
    - ``feature_names_in_`` is set for numpy input too (``"x0", "x1", ...``).
    - Computation is single-threaded numpy, so results match scikit-learn up
      to floating-point rounding (about 1e-12 relative), with the same labels
      and ``n_iter_``.
    - ``set_output`` is not supported.

    Examples
    --------
    >>> import numpy as np
    >>> from glasslearn.cluster import KMeans
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> km = KMeans(n_clusters=2, random_state=0).fit(X)
    >>> sorted(km.cluster_centers_.tolist())
    [[1.0, 2.0], [10.0, 2.0]]
    """

    _sticky_state = ("labels",)

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        init: Any = "k-means++",
        n_init: str | int = "auto",
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: int = 0,
        random_state: Any = None,
        copy_x: bool = True,
        algorithm: str = "lloyd",
    ) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

    # -- fitting ---------------------------------------------------------------

    def _fit(self, X: np.ndarray, y: Any, rec: Recorder) -> None:
        self._check_params()
        n_samples = X.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}."
            )
        tol = _tolerance(X, self.tol)
        n_init = self._resolve_n_init()
        random_state = check_random_state(self.random_state)

        init = self.init
        init_is_array = is_arraylike_not_scalar(init)
        if init_is_array:
            init = check_array(
                init, estimator_name=type(self).__name__, input_name="init"
            )
            self._validate_center_shape(X, init)

        # As scikit-learn: centre the data for more accurate distances, and
        # shift the centroids back at the end.
        x_mean = X.mean(axis=0)
        X = X - x_mean
        if init_is_array:
            init = init - x_mean
        x_squared_norms = np.einsum("ij,ij->i", X, X)

        best = None
        run_inertia = []
        for run in range(n_init):
            centers_init, seeds = self._init_centroids(
                X, x_squared_norms, init, random_state
            )
            if self.verbose:
                print("Initialization complete")
            run_rec = rec.spawn()
            run_rec.step(
                "init",
                state={"iteration": 0, "centers": centers_init + x_mean},
                events=[
                    InitChosen(
                        method=self._init_method(),
                        seeds=seeds,
                        n_init=n_init,
                        run=run,
                        salience=0.9,
                    )
                ],
            )
            labels, inertia, centers, n_iter = _kmeans_single_lloyd(
                X,
                centers_init,
                max_iter=self.max_iter,
                tol=tol,
                verbose=bool(self.verbose),
                rec=run_rec,
                offset=x_mean,
            )
            run_inertia.append(float(inertia))
            # Same rule as scikit-learn: a later run only wins with a strictly
            # lower inertia and a genuinely different clustering.
            if best is None or (
                inertia < best[1]
                and not _is_same_clustering(labels, best[0], self.n_clusters)
            ):
                best = (labels, inertia, centers, n_iter, run_rec, run)

        labels, inertia, centers, n_iter, best_rec, best_run = best
        rec.adopt(best_rec)
        rec.note(run_inertia=run_inertia, best_run=best_run, tol_scaled=float(tol))

        distinct = len(set(labels.tolist()))
        if distinct < self.n_clusters:
            warnings.warn(
                f"Number of distinct clusters ({distinct}) found smaller than "
                f"n_clusters ({self.n_clusters}). Possibly due to duplicate points "
                "in X.",
                ConvergenceWarning,
                stacklevel=3,
            )

        self.cluster_centers_ = centers + x_mean
        self._n_features_out = self.cluster_centers_.shape[0]
        self.labels_ = labels
        self.inertia_ = float(inertia)
        self.n_iter_ = n_iter

    def _check_params(self) -> None:
        def is_int(v: Any) -> bool:
            return isinstance(v, numbers.Integral)

        def is_real(v: Any) -> bool:
            return isinstance(v, numbers.Real)

        check_param(
            self,
            "n_clusters",
            is_int(self.n_clusters) and self.n_clusters >= 1,
            "an int in the range [1, inf)",
        )
        init = self.init
        check_param(
            self,
            "init",
            (isinstance(init, str) and init in ("k-means++", "random"))
            or callable(init)
            or (not isinstance(init, str) and is_arraylike_not_scalar(init)),
            "a str among {'k-means++', 'random'}, a callable or an array-like",
        )
        check_param(
            self,
            "n_init",
            (isinstance(self.n_init, str) and self.n_init == "auto")
            or (is_int(self.n_init) and self.n_init >= 1),
            "a str among {'auto'} or an int in the range [1, inf)",
        )
        check_param(
            self,
            "max_iter",
            is_int(self.max_iter) and self.max_iter >= 1,
            "an int in the range [1, inf)",
        )
        check_param(
            self,
            "tol",
            is_real(self.tol) and self.tol >= 0,
            "a float in the range [0, inf)",
        )
        check_param(
            self,
            "verbose",
            is_int(self.verbose) and self.verbose >= 0,
            "an int in the range [0, inf) or a bool",
        )
        rs = self.random_state
        check_param(
            self,
            "random_state",
            rs is None or is_int(rs) or isinstance(rs, np.random.RandomState),
            "an int in the range [0, 4294967295], an instance of "
            "'numpy.random.mtrand.RandomState' or None",
        )
        check_param(
            self,
            "copy_x",
            isinstance(self.copy_x, (bool, np.bool_)),
            "an instance of 'bool' or an instance of 'numpy.bool'",
        )
        check_param(
            self,
            "algorithm",
            isinstance(self.algorithm, str) and self.algorithm in ("lloyd", "elkan"),
            "a str among {'elkan', 'lloyd'}",
        )

    def _resolve_n_init(self) -> int:
        if self.n_init == "auto":
            if isinstance(self.init, str):
                n_init = 1 if self.init == "k-means++" else 10
            else:
                n_init = 10 if callable(self.init) else 1
        else:
            n_init = int(self.n_init)
        if is_arraylike_not_scalar(self.init) and n_init != 1:
            warnings.warn(
                "Explicit initial center position passed: performing only one init "
                f"in {type(self).__name__} instead of n_init={n_init}.",
                RuntimeWarning,
                stacklevel=4,
            )
            n_init = 1
        return n_init

    def _init_method(self) -> str:
        if isinstance(self.init, str):
            return self.init
        return "callable" if callable(self.init) else "array"

    def _validate_center_shape(self, X: np.ndarray, centers: np.ndarray) -> None:
        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}."
            )
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}."
            )

    def _init_centroids(
        self,
        X: np.ndarray,
        x_squared_norms: np.ndarray,
        init: Any,
        random_state: np.random.RandomState,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """Return the starting centroids and the rows they were taken from."""
        n_samples = X.shape[0]
        sample_weight = np.ones(n_samples)
        if isinstance(init, str) and init == "k-means++":
            centers, indices = _kmeans_plusplus(
                X, self.n_clusters, x_squared_norms, sample_weight, random_state
            )
            return centers, tuple(int(i) for i in indices)
        if isinstance(init, str) and init == "random":
            seeds = random_state.choice(
                n_samples,
                size=self.n_clusters,
                replace=False,
                p=sample_weight / sample_weight.sum(),
            )
            return X[seeds], tuple(int(i) for i in seeds)
        if callable(init):
            centers = init(X, self.n_clusters, random_state=random_state)
            centers = check_array(
                centers, estimator_name=type(self).__name__, input_name="init"
            )
            self._validate_center_shape(X, centers)
            return centers, ()
        return init.copy(), ()

    # -- using the fitted model ------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        """Return the index of the nearest centroid for each sample."""
        X = self._validate_predict_data(X)
        return _assign(X, self.cluster_centers_)

    def transform(self, X: Any) -> np.ndarray:
        """Return the distance from each sample to every centroid."""
        X = self._validate_predict_data(X)
        return np.sqrt(_squared_distances(X, self.cluster_centers_))

    def score(self, X: Any, y: Any = None) -> float:
        """Return the opposite of the inertia of ``X`` (higher is better)."""
        X = self._validate_predict_data(X)
        labels = _assign(X, self.cluster_centers_)
        return -_inertia(X, self.cluster_centers_, labels)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Names of the ``transform`` outputs: ``kmeans0``, ``kmeans1``, ..."""
        check_is_fitted(self)
        if input_features is not None:
            input_features = np.asarray(input_features, dtype=object)
            if self._feature_names_given and not np.array_equal(
                self.feature_names_in_, input_features
            ):
                raise ValueError("input_features is not equal to feature_names_in_")
            if len(input_features) != self.n_features_in_:
                raise ValueError(
                    "input_features should have length equal to number of "
                    f"features ({self.n_features_in_}), got {len(input_features)}"
                )
        prefix = type(self).__name__.lower()
        return np.asarray(
            [f"{prefix}{i}" for i in range(self._n_features_out)], dtype=object
        )

    def summary(self):
        """Return the centroids as a DataFrame, one column per feature."""
        check_is_fitted(self)
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - pandas is a dev dependency
            raise ImportError(
                "KMeans.summary() needs pandas: pip install pandas"
            ) from exc
        return pd.DataFrame(
            self.cluster_centers_,
            columns=[str(n) for n in self.feature_names_in_],
            index=pd.RangeIndex(self.n_clusters, name="cluster"),
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.transformer_tags.preserves_dtype = ["float64"]
        return tags


# -- the algorithm (mirrors scikit-learn's sklearn/cluster/_kmeans.py) ----------


def _tolerance(X: np.ndarray, tol: float) -> float:
    """Scale ``tol`` by the mean variance of the features."""
    if tol == 0:
        return 0.0
    return float(np.mean(np.var(X, axis=0)) * tol)


def _squared_distances(
    A: np.ndarray, B: np.ndarray, B_squared_norms: np.ndarray | None = None
) -> np.ndarray:
    """Squared distances between rows of A and B, as ``||a||² - 2a·b + ||b||²``."""
    if B_squared_norms is None:
        B_squared_norms = np.einsum("ij,ij->i", B, B)
    distances = -2 * (A @ B.T)
    distances += np.einsum("ij,ij->i", A, A)[:, None]
    distances += B_squared_norms[None, :]
    np.maximum(distances, 0, out=distances)
    return distances


def _assign(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the nearest centre for each row (ties go to the lowest index)."""
    # ||x||² is the same for every centre, so comparing ||c||² - 2x·c is enough.
    scores = np.einsum("ij,ij->i", centers, centers)[None, :] - 2.0 * (X @ centers.T)
    return scores.argmin(axis=1).astype(np.int32)


def _inertia(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """Sum of squared distances from each point to its own centre."""
    return float(((X - centers[labels]) ** 2).sum())


def _kmeans_plusplus(
    X: np.ndarray,
    n_clusters: int,
    x_squared_norms: np.ndarray,
    sample_weight: np.ndarray,
    random_state: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy k-means++: each seed is the best of several distance-weighted draws."""
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    n_local_trials = 2 + int(np.log(n_clusters))

    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
    indices = np.full(n_clusters, -1, dtype=int)
    centers[0] = X[center_id]
    indices[0] = center_id

    closest_dist_sq = _squared_distances(centers[0, np.newaxis], X, x_squared_norms)
    current_pot = closest_dist_sq @ sample_weight

    for c in range(1, n_clusters):
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(
            np.cumsum(sample_weight * closest_dist_sq), rand_vals
        )
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        distance_to_candidates = _squared_distances(
            X[candidate_ids], X, x_squared_norms
        )
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices


def _update_centers(
    X: np.ndarray, centers_old: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, float]]]:
    """Move each centre to the mean of its points.

    As scikit-learn: an empty cluster is moved onto the point farthest from
    its own centre. When every point sits exactly on a centre (more clusters
    than distinct points) it is instead stacked on the biggest cluster.
    Returns the new centres, the cluster sizes after any relocation, and
    ``(cluster, sample, from_cluster, distance)`` for each empty cluster
    (``sample = -1`` and ``from_cluster`` = the biggest cluster when stacked).
    """
    n_clusters, n_features = centers_old.shape
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    sums = np.empty((n_clusters, n_features))
    for f in range(n_features):
        sums[:, f] = np.bincount(labels, weights=X[:, f], minlength=n_clusters)

    empty_clusters = np.flatnonzero(counts == 0)
    empties = []
    if empty_clusters.size:
        n_empty = empty_clusters.size
        distances = ((X - centers_old[labels]) ** 2).sum(axis=1)
        far_from_centers = np.argpartition(distances, -n_empty)[: -n_empty - 1 : -1]
        if np.max(distances) == 0:
            # More clusters than distinct points: relocating is pointless.
            biggest = int(np.argmax(counts))
            empties = [(int(j), -1, biggest, 0.0) for j in empty_clusters]
        else:
            for new_id, far_idx in zip(empty_clusters, far_from_centers, strict=True):
                old_id = labels[far_idx]
                sums[old_id] -= X[far_idx]
                sums[new_id] = X[far_idx]
                counts[new_id] = 1.0
                counts[old_id] -= 1.0
                empties.append(
                    (
                        int(new_id),
                        int(far_idx),
                        int(old_id),
                        float(np.sqrt(distances[far_idx])),
                    )
                )

    # Average in cluster order. A still-empty cluster copies the biggest
    # cluster's row as it is at that moment, which is still a raw sum when
    # the biggest cluster comes later. This mirrors scikit-learn exactly.
    centers_new = sums
    biggest = int(np.argmax(counts))
    for j in range(n_clusters):
        if counts[j] > 0:
            centers_new[j] *= 1.0 / counts[j]
        else:
            centers_new[j] = centers_new[biggest]
    return centers_new, counts, empties


def _kmeans_single_lloyd(
    X: np.ndarray,
    centers_init: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    verbose: bool,
    rec: Recorder,
    offset: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """One run of Lloyd's algorithm on centred data.

    Returns ``(labels, inertia, centers, n_iter)``. ``offset`` is added to
    every recorded position so the trace is in the user's feature space.
    """
    n_samples = X.shape[0]
    n_clusters = centers_init.shape[0]
    centers = centers_init
    labels = np.full(n_samples, -1, dtype=np.int32)
    labels_old = labels.copy()
    log = _IterationLog(rec, X, n_clusters, offset)
    strict_convergence = False
    reason = "max_iter"
    center_shift_tot = 0.0

    for i in range(max_iter):
        labels = _assign(X, centers)
        centers_new, counts, empties = _update_centers(X, centers, labels)
        center_shift = np.sqrt(((centers_new - centers) ** 2).sum(axis=1))
        center_shift_tot = float((center_shift**2).sum())

        if verbose:
            print(f"Iteration {i}, inertia {_inertia(X, centers, labels)}.")
        log.assign(i + 1, centers, labels)
        log.update(i + 1, centers, centers_new, labels, center_shift, empties)

        centers = centers_new

        if np.array_equal(labels, labels_old):
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True
            reason = "labels stable"
            break
        if center_shift_tot <= tol:
            if verbose:
                print(
                    f"Converged at iteration {i}: center shift {center_shift_tot} "
                    f"within tolerance {tol}."
                )
            reason = "tol"
            break
        labels_old = labels

    n_iter = i + 1
    if not strict_convergence:
        # Re-assign so that the returned labels match the final centres.
        labels = _assign(X, centers)
        log.assign(n_iter, centers, labels, final=True)

    inertia = _inertia(X, centers, labels)
    log.done(n_iter, centers, labels, inertia, reason, center_shift_tot, tol)
    return labels, inertia, centers, n_iter


class _IterationLog:
    """Turns Lloyd's iterations into recorded steps (does nothing when disabled).

    Kept apart from the loop so the arithmetic above reads like scikit-learn's.
    """

    def __init__(
        self, rec: Recorder, X: np.ndarray, n_clusters: int, offset: np.ndarray
    ):
        self.rec = rec
        self.X = X
        self.k = n_clusters
        self.offset = offset
        self.shown: np.ndarray | None = None  # labels as last recorded
        self.transitions: np.ndarray | None = None

    def assign(
        self,
        iteration: int,
        centers: np.ndarray,
        labels: np.ndarray,
        final: bool = False,
    ) -> None:
        if not self.rec.enabled:
            return
        k = self.k
        sizes = np.bincount(labels, minlength=k)
        first = self.shown is None
        if first:
            n_changed = labels.size
            self.transitions = None
            transitions: tuple[tuple[int, ...], ...] = ()
        else:
            n_changed = int((self.shown != labels).sum())
            pairs = self.shown.astype(np.int64) * k + labels
            self.transitions = np.bincount(pairs, minlength=k * k).reshape(k, k)
            transitions = tuple(tuple(int(v) for v in row) for row in self.transitions)
        salience = 0.9 if first else (0.6 if n_changed == 0 else 0.8)
        self.rec.step(
            "assign",
            state={
                "iteration": iteration,
                "centers": centers + self.offset,
                "labels": labels,
            },
            metrics={
                "inertia": _inertia(self.X, centers, labels),
                "n_changed": n_changed,
            },
            events=[
                PointsReassigned(
                    n_changed=n_changed,
                    n_samples=int(labels.size),
                    sizes=tuple(int(s) for s in sizes),
                    transitions=transitions,
                    first=first,
                    final=final,
                    salience=salience,
                )
            ],
        )
        self.shown = labels

    def update(
        self,
        iteration: int,
        centers_old: np.ndarray,
        centers_new: np.ndarray,
        labels: np.ndarray,
        center_shift: np.ndarray,
        empties: list[tuple[int, int, int, float]],
    ) -> None:
        if not self.rec.enabled:
            return
        k = self.k
        # Points moved onto empty clusters now belong to them.
        shown = labels.copy()
        for cluster, sample, _, _ in empties:
            if sample >= 0:
                shown[sample] = cluster
        sizes = np.bincount(shown, minlength=k)
        if self.transitions is None:
            gained = np.bincount(labels, minlength=k)
            lost = np.zeros(k, dtype=np.int64)
        else:
            stay = np.diag(self.transitions)
            gained = self.transitions.sum(axis=0) - stay
            lost = self.transitions.sum(axis=1) - stay
        for cluster, sample, from_cluster, _ in empties:
            if sample >= 0:
                gained[cluster] += 1
                lost[from_cluster] += 1

        top = float(center_shift.max()) if center_shift.size else 0.0
        old_pos = centers_old + self.offset
        new_pos = centers_new + self.offset
        events: list[Any] = [
            CentroidMoved(
                cluster=j,
                old=tuple(float(v) for v in old_pos[j]),
                new=tuple(float(v) for v in new_pos[j]),
                distance=float(center_shift[j]),
                n_gained=int(gained[j]),
                n_lost=int(lost[j]),
                size=int(sizes[j]),
                salience=0.2 + 0.6 * float(center_shift[j]) / top if top > 0 else 0.2,
            )
            for j in range(k)
        ]
        events += [
            EmptyCluster(
                cluster=cluster,
                sample=sample,
                from_cluster=from_cluster,
                distance=distance,
                salience=0.95,
            )
            for cluster, sample, from_cluster, distance in empties
        ]
        self.rec.step(
            "update",
            state={
                "iteration": iteration,
                "old_centers": old_pos,
                "centers": new_pos,
                "labels": shown,
            },
            metrics={
                "inertia": _inertia(self.X, centers_new, shown),
                "center_shift_tot": float((center_shift**2).sum()),
            },
            events=events,
        )
        self.shown = shown

    def done(
        self,
        n_iter: int,
        centers: np.ndarray,
        labels: np.ndarray,
        inertia: float,
        reason: str,
        center_shift_tot: float,
        tol: float,
    ) -> None:
        if not self.rec.enabled:
            return
        self.rec.step(
            "done",
            state={
                "iteration": n_iter,
                "centers": centers + self.offset,
                "labels": labels,
            },
            metrics={"inertia": inertia},
            events=[
                Converged(
                    reason=reason,
                    n_iter=n_iter,
                    center_shift_tot=center_shift_tot,
                    tol=float(tol),
                    salience=1.0,
                )
            ],
        )


def _is_same_clustering(
    labels1: np.ndarray, labels2: np.ndarray, n_clusters: int
) -> bool:
    """True when each cluster of ``labels1`` maps to one cluster of ``labels2``."""
    pairs = labels1.astype(np.int64) * n_clusters + labels2
    return np.unique(pairs).size == np.unique(labels1).size
