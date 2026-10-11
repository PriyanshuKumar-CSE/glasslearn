"""What k-means reports while it trains."""

from __future__ import annotations

from dataclasses import dataclass

from ...trace import Event


@dataclass(frozen=True)
class InitChosen(Event):
    """The starting centroids were chosen.

    Attributes
    ----------
    method : str
        ``"k-means++"``, ``"random"``, ``"array"`` (given by the user) or
        ``"callable"``.
    seeds : tuple of int
        Row of ``X`` picked as each starting centroid, in the order they were
        picked (empty for ``"array"`` and ``"callable"``).
    n_init : int
        How many restarts were run in total.
    run : int
        Which restart (from 0) this trace shows: the one with the lowest inertia.
    """

    method: str
    seeds: tuple[int, ...]
    n_init: int
    run: int


@dataclass(frozen=True)
class PointsReassigned(Event):
    """Every point was assigned to its nearest centroid.

    Attributes
    ----------
    n_changed : int
        Points whose cluster changed (all of them on the first pass).
    n_samples : int
        Total number of points.
    sizes : tuple of int
        Points in each cluster after the assignment.
    transitions : tuple of tuple of int
        ``transitions[a][b]`` points moved from cluster ``a`` to ``b`` (empty
        on the first pass).
    first : bool
        This is the first assignment, right after initialisation.
    final : bool
        This is the extra pass k-means makes after stopping on ``tol`` or
        ``max_iter``, so that labels match the final centroids.
    """

    n_changed: int
    n_samples: int
    sizes: tuple[int, ...]
    transitions: tuple[tuple[int, ...], ...]
    first: bool = False
    final: bool = False


@dataclass(frozen=True)
class CentroidMoved(Event):
    """A centroid moved to the mean of its points.

    Attributes
    ----------
    cluster : int
        Cluster index.
    old, new : tuple of float
        Position before and after, in feature space.
    distance : float
        Euclidean distance between ``old`` and ``new``.
    n_gained, n_lost : int
        Points that joined and left the cluster in this iteration.
    size : int
        Points in the cluster now.
    """

    cluster: int
    old: tuple[float, ...]
    new: tuple[float, ...]
    distance: float
    n_gained: int
    n_lost: int
    size: int


@dataclass(frozen=True)
class EmptyCluster(Event):
    """A cluster lost all its points and its centroid was moved.

    Normally the centroid jumps onto the point farthest from its own centroid,
    which then joins the empty cluster. When every point already sits exactly
    on a centroid (more clusters than distinct points) that cannot help, and
    the centroid is stacked on the biggest cluster instead.

    Attributes
    ----------
    cluster : int
        The cluster that became empty.
    sample : int
        Row of ``X`` it was moved onto, or -1 when it was stacked instead.
    from_cluster : int
        The cluster that point came from, or the biggest cluster it was
        stacked on.
    distance : float
        How far that point was from its old centroid (0 when stacked).
    """

    cluster: int
    sample: int
    from_cluster: int
    distance: float


@dataclass(frozen=True)
class Converged(Event):
    """Training stopped.

    Attributes
    ----------
    reason : str
        ``"labels stable"`` (no point changed cluster), ``"tol"`` (the
        centroids moved less than the tolerance) or ``"max_iter"``.
    n_iter : int
        Iterations run, as ``n_iter_``.
    center_shift_tot : float
        Sum of squared centroid moves in the last iteration.
    tol : float
        The tolerance actually used: ``tol`` times the mean feature variance.
    """

    reason: str
    n_iter: int
    center_shift_tot: float
    tol: float
