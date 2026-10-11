"""Sentences for k-means events, at the beginner, detailed and math levels."""

from __future__ import annotations

import math

import numpy as np

from ...narrate import (
    Diagnostic,
    MetricChanged,
    NarrationContext,
    derives,
    diagnoses,
    narrates,
    narrates_rest,
    summarizes,
    unscaled_features,
)
from ...trace import Trace
from .events import (
    CentroidMoved,
    Converged,
    EmptyCluster,
    InitChosen,
    PointsReassigned,
)

_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _sub(j: int) -> str:
    return str(j).translate(_SUBSCRIPTS)


def _rows(ctx: NarrationContext, rows: tuple[int, ...], limit: int = 6) -> str:
    shown = [str(r) for r in rows[:limit]]
    if len(rows) > limit:
        shown.append(f"{len(rows) - limit} more")
    return f"{'row' if len(rows) == 1 else 'rows'} {ctx.join(shown)}"


def _sizes(ctx: NarrationContext, sizes: tuple[int, ...]) -> str:
    return ctx.join(f"{s:,}" for s in sizes)


def _from_to(ctx: NarrationContext, old: float, new: float, arrow: str = " to ") -> str:
    """'from 542.3 to 541.9', with enough digits to show the change."""
    a, b = ctx.distinct(old, new)
    return f"{a}{arrow}{b}"


def _same(ctx: NarrationContext, old: float, new: float) -> bool:
    """True when two values print the same, so the change is too small to show."""
    a, b = ctx.distinct(old, new)
    return a == b


def _change(ctx: NarrationContext, old: float, new: float) -> str:
    """Relative change, such as '-21%'."""
    if old == 0:
        return ctx.signed(new - old)
    return f"{ctx.signed(100 * (new - old) / old)}%"


def _biggest_flow(e: PointsReassigned) -> tuple[int, int, int] | None:
    if not e.transitions:
        return None
    matrix = np.array(e.transitions)
    np.fill_diagonal(matrix, 0)
    if matrix.max() == 0:
        return None
    a, b = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    return int(a), int(b), int(matrix[a, b])


def _cause(ctx: NarrationContext, e: CentroidMoved) -> str:
    parts = []
    if e.n_gained:
        parts.append(f"{ctx.plural(e.n_gained, 'point')} joined it")
    if e.n_lost:
        parts.append(f"{ctx.plural(e.n_lost, 'point')} left")
    return " and ".join(parts) if parts else "no point joined or left it"


def _barely(ctx: NarrationContext, distance: float) -> bool:
    return distance <= 1e-3 * ctx.scale


def _unchanged(e: CentroidMoved) -> bool:
    return e.n_gained == 0 and e.n_lost == 0


def _emptied(ctx: NarrationContext, cluster: int) -> bool:
    """True when the step's EmptyCluster event already explains this centroid."""
    return any(x.cluster == cluster for x in ctx.step.events_of(EmptyCluster))


# -- events --------------------------------------------------------------------


@narrates(InitChosen)
def init_chosen(e: InitChosen, ctx: NarrationContext) -> str:
    k = len(ctx.step.state["centers"])
    trials = 2 + int(math.log(k))
    runs = ctx.trace.meta.get("run_inertia", [])
    kept = ctx.number(runs[e.run]) if runs else ""
    worst = ctx.number(max(runs)) if runs else ""
    rows = _rows(ctx, e.seeds)

    if ctx.math:
        seeds = ", ".join(str(s) for s in e.seeds)
        text = {
            "k-means++": (
                f"c_j = x_i_j with i = ({seeds}); each i_j is the best of {trials} "
                "candidates drawn with P(i) ∝ D(x_i)², the one minimising "
                "Σ_i min_l ‖x_i − c_l‖²."
            ),
            "random": (
                f"c_j = x_i_j with i = ({seeds}), drawn uniformly without replacement."
            ),
            "array": f"c_j for j = 0..{k - 1} given by init.",
            "callable": f"c_j for j = 0..{k - 1} returned by the init function.",
        }[e.method]
        if e.n_init > 1:
            text += f" Best of {e.n_init} restarts: J = {kept} (worst {worst})."
        return text

    if ctx.detailed:
        text = {
            "k-means++": (
                f"Init: k-means++ seeded the {k} centroids at {rows}; each seed was "
                f"the best of {trials} candidates sampled with probability "
                "proportional to squared distance from the centroids already chosen."
            ),
            "random": f"Init: {k} distinct {rows} chosen uniformly at random.",
            "array": f"Init: the {k} centroids passed as init.",
            "callable": f"Init: the {k} centroids returned by your init function.",
        }[e.method]
        if e.n_init > 1:
            text += (
                f" Showing restart {e.run + 1} of {e.n_init}, the one kept (inertia "
                f"{kept} vs {worst} for the worst restart)."
            )
        return text + " Next: assign each point to its nearest centroid."

    text = {
        "k-means++": (
            f"K-means++ chose {k} starting centroids one at a time, each time "
            f"preferring points far from the ones already chosen ({rows})."
        ),
        "random": (
            f"{k} starting centroids were placed on randomly chosen points ({rows})."
        ),
        "array": f"The {k} starting centroids are the positions you passed as init.",
        "callable": f"Your init function chose the {k} starting centroids.",
    }[e.method]
    if e.n_init > 1:
        text += (
            f" This is the best of {e.n_init} restarts, each from different starting "
            "centroids."
        )
    return text + " Next, every point will join its nearest centroid."


@narrates(PointsReassigned)
def points_reassigned(e: PointsReassigned, ctx: NarrationContext) -> str:
    inertia = ctx.metric("inertia")
    before = ctx.metric("inertia", ctx.prev) if ctx.prev is not None else inertia
    now = ctx.number(inertia)
    sizes = _sizes(ctx, e.sizes)
    flow = _biggest_flow(e)
    n, changed = e.n_samples, e.n_changed

    if e.first:
        if ctx.math:
            return (
                f"label(i) = argmin_j ‖x_i − c_j‖² for all {n:,} points; "
                f"|C_j| = ({', '.join(str(v) for v in e.sizes)}). "
                f"J = Σ_i ‖x_i − c_label(i)‖² = {now}."
            )
        if ctx.detailed:
            return (
                f"Assigned all {n:,} points to their nearest centroid; cluster sizes "
                f"{sizes}. Inertia: {now}."
            )
        return (
            f"Every point joined its nearest centroid, giving clusters of {sizes} "
            f"points. The inertia starts at {now}."
        )

    if e.final:
        if ctx.math:
            return (
                "Final E-step with the last centroids: "
                f"{ctx.plural(changed, 'label')} changed; J = {now}."
            )
        if ctx.detailed:
            return (
                "Final assignment pass with the last centroids: "
                f"{ctx.plural(changed, 'point')} changed cluster; inertia "
                f"{_from_to(ctx, before, inertia, ' → ')}."
            )
        moved = "no point changed cluster"
        if changed:
            moved = f"{ctx.plural(changed, 'point')} changed cluster"
        return (
            "After stopping, k-means made one last assignment so that every point "
            f"belongs to its nearest final centroid; {moved}. The final inertia is "
            f"{now}."
        )

    if changed == 0:
        if ctx.math:
            return (
                "No label changed: label(i) = argmin_j ‖x_i − c_j‖² already holds; "
                f"J = {now}."
            )
        if ctx.detailed:
            return f"0 of {n:,} points changed cluster; inertia unchanged at {now}."
        return (
            "No point changed cluster: each one is already with its nearest centroid. "
            f"The inertia stays at {now}."
        )

    if ctx.math:
        text = (
            f"{ctx.plural(changed, 'label')} changed; "
            f"J: {_from_to(ctx, before, inertia, ' → ')} "
            f"(ΔJ = {ctx.signed(inertia - before)})."
        )
        if flow:
            a, b, count = flow
            text += f" Largest transition T{_sub(a)}{_sub(b)} = {count:,}."
        return text
    if ctx.detailed:
        text = (
            f"{changed:,} of {n:,} points ({ctx.percent(changed / n)}) changed "
            f"cluster, taking inertia {_from_to(ctx, before, inertia, ' → ')} "
            f"({_change(ctx, before, inertia)})."
        )
        if flow:
            a, b, count = flow
            text += (
                f" Largest flow: {count:,} from cluster {a} to {b}; sizes now {sizes}."
            )
        return text
    if _same(ctx, before, inertia):
        effect = f"and the inertia stayed at {now}"
    else:
        effect = f"so the inertia fell from {_from_to(ctx, before, inertia)}"
    text = f"{ctx.plural(changed, 'point')} switched to a closer centroid, {effect}."
    if flow:
        a, b, count = flow
        text += (
            f" The biggest move was {ctx.plural(count, 'point')} from cluster {a} to "
            f"cluster {b}."
        )
    return text


@narrates(CentroidMoved)
def centroid_moved(e: CentroidMoved, ctx: NarrationContext) -> str | None:
    if _emptied(ctx, e.cluster):
        return None
    j, dist = e.cluster, ctx.number(e.distance)
    if ctx.math:
        return (
            f"c{_sub(j)} ← mean of C{_sub(j)}, |C{_sub(j)}| = {e.size:,} "
            f"(+{e.n_gained:,}, −{e.n_lost:,}); ‖Δc{_sub(j)}‖ = {dist}."
        )
    assigned = ctx.prev.event(PointsReassigned) if ctx.prev is not None else None
    first = assigned is not None and assigned.first
    if ctx.detailed:
        if _barely(ctx, e.distance):
            return f"Centroid {j} moved only {dist} ({_cause(ctx, e)})."
        feature, delta = ctx.biggest_change(e.old, e.new)
        why = f"because {_cause(ctx, e)} (now {e.size:,} points)"
        if first:
            why = f"to the mean of its {e.size:,} points"
        return (
            f"Centroid {j} moved {dist}, mostly along {feature} "
            f"({ctx.signed(delta)}), {why}."
        )
    if _barely(ctx, e.distance):
        if first:
            members = ctx.plural(e.size, "point")
            return f"Centroid {j} was already in the middle of its {members}."
        return f"Centroid {j} barely moved: {_cause(ctx, e)}, but they balanced out."
    where = ctx.direction(e.old, e.new)
    where = "" if where == "nowhere" else f" {where}"
    if first:
        members = ctx.plural(e.size, "point")
        return f"Centroid {j} moved{where} to the middle of its {members}."
    return (
        f"Centroid {j} moved{where} by {dist} because {_cause(ctx, e)}, so the "
        "average of its points changed."
    )


@narrates_rest(CentroidMoved)
def other_centroids(events: list[CentroidMoved], ctx: NarrationContext) -> str | None:
    events = sorted(
        (e for e in events if not _emptied(ctx, e.cluster)), key=lambda e: e.cluster
    )
    if not events:
        return None
    clusters = ctx.join(str(e.cluster) for e in events)
    everyone = len(events) == len(ctx.step.events_of(CentroidMoved))
    if all(_unchanged(e) for e in events):
        if ctx.math:
            ids = ", ".join(str(e.cluster) for e in events)
            return f"C_j unchanged ⇒ c_j unchanged for j ∈ {{{ids}}}."
        if everyone:
            return (
                f"None of the {ctx.plural(len(events), 'centroid')} moved, because no "
                "point joined or left any cluster."
            )
        if len(events) == 1:
            return f"Centroid {clusters} kept the same points, so it stayed put."
        return f"Centroids {clusters} kept the same points, so they stayed put."

    top = max(e.distance for e in events)
    if ctx.math:
        ids = ", ".join(str(e.cluster) for e in events)
        return f"‖Δc_j‖ ≤ {ctx.number(top)} for j ∈ {{{ids}}}."
    if ctx.detailed:
        label = "Centroid" if len(events) == 1 else "Centroids"
        return f"{label} {clusters} moved {ctx.numbers([e.distance for e in events])}."
    if all(_barely(ctx, e.distance) for e in events):
        if len(events) == 1:
            return f"Centroid {clusters} barely moved."
        return f"Centroids {clusters} barely moved."
    if len(events) == 1:
        return f"Centroid {clusters} moved less ({ctx.number(top)})."
    return f"Centroids {clusters} moved less (at most {ctx.number(top)})."


@narrates(EmptyCluster)
def empty_cluster(e: EmptyCluster, ctx: NarrationContext) -> str:
    c, row = e.cluster, e.sample
    if row < 0:
        if ctx.math:
            return (
                f"C{_sub(c)} = ∅ and max_i ‖x_i − c_label(i)‖ = 0, so no point can be "
                "moved to it."
            )
        return (
            f"Cluster {c} has no points and cannot get any: every point already sits "
            "exactly on a centroid, because there are fewer distinct points than "
            "clusters."
        )
    dist = ctx.number(e.distance)
    if ctx.math:
        return (
            f"C{_sub(c)} = ∅ ⇒ c{_sub(c)} ← x_{row}, where {row} = "
            f"argmax_i ‖x_i − c_label(i)‖ ({dist}); x_{row} moves from "
            f"C{_sub(e.from_cluster)} to C{_sub(c)}."
        )
    if ctx.detailed:
        return (
            f"Cluster {c} was empty; its centroid was moved onto row {row}, which was "
            f"{dist} from its centroid in cluster {e.from_cluster}, farther than any "
            "other point."
        )
    return (
        f"Cluster {c} lost all its points, so its centroid jumped onto the point "
        f"farthest from its own centroid (row {row}, taken from cluster "
        f"{e.from_cluster}) to keep every cluster in use."
    )


@derives("KMeans")
def inertia_change(ctx: NarrationContext) -> list:
    step, prev = ctx.step, ctx.prev
    if step.phase != "update" or prev is None:
        return []
    old, new = ctx.metric("inertia", prev), ctx.metric("inertia")
    if old is None or new is None:
        return []
    return [MetricChanged(metric="inertia", old=old, new=new, salience=0.85)]


@narrates(MetricChanged, algorithm="KMeans")
def inertia_changed(e: MetricChanged, ctx: NarrationContext) -> str:
    if _same(ctx, e.old, e.new):
        if ctx.math:
            return f"ΔJ ≈ 0: J = {ctx.number(e.new)}."
        if ctx.detailed:
            return f"Inertia unchanged at {ctx.number(e.new)}."
        return f"The inertia stayed at {ctx.number(e.new)}."
    change = _from_to(ctx, e.old, e.new, " → ")
    if ctx.math:
        return f"J: {change} (ΔJ = {ctx.signed(e.new - e.old)})."
    if ctx.detailed:
        return f"Inertia {change} ({_change(ctx, e.old, e.new)})."
    lowered = _from_to(ctx, e.old, e.new)
    return f"Together these moves lowered the inertia from {lowered}."


@narrates(Converged)
def converged(e: Converged, ctx: NarrationContext) -> str:
    inertia = ctx.number(ctx.metric("inertia"))
    shift, tol = ctx.distinct(e.center_shift_tot, e.tol)
    iters = ctx.plural(e.n_iter, "iteration")
    if ctx.math:
        return {
            "labels stable": (
                f"Fixed point after n_iter = {e.n_iter}: label⁽ᵗ⁾ = label⁽ᵗ⁻¹⁾. "
                f"J* = {inertia}."
            ),
            "tol": (
                f"Stopped at n_iter = {e.n_iter}: Σ_j ‖Δc_j‖² = {shift} ≤ "
                f"tol · mean(Var x) = {tol}. J* = {inertia}."
            ),
            "max_iter": (
                f"Stopped at n_iter = max_iter = {e.n_iter}; Σ_j ‖Δc_j‖² = {shift} > "
                f"{tol}. J* = {inertia}."
            ),
        }[e.reason]
    if ctx.detailed:
        return {
            "labels stable": (
                f"Converged after {iters}: no label changed in the last one. Final "
                f"inertia {inertia}."
            ),
            "tol": (
                f"Converged after {iters}: total squared centroid shift {shift} ≤ "
                f"tolerance {tol} (tol × mean feature variance). Final inertia "
                f"{inertia}."
            ),
            "max_iter": (
                f"Stopped at max_iter = {e.n_iter} without converging (last shift "
                f"{shift} > tolerance {tol}). Final inertia {inertia}."
            ),
        }[e.reason]
    return {
        "labels stable": (
            f"K-means stopped after {iters} because no point changed cluster, so "
            f"another round would change nothing. The final inertia is {inertia}."
        ),
        "tol": (
            f"K-means stopped after {iters} because the centroids moved less than the "
            f"tolerance ({shift} vs {tol}). The final inertia is {inertia}."
        ),
        "max_iter": (
            f"K-means stopped because it reached max_iter = {e.n_iter} before the "
            f"clusters settled. The final inertia is {inertia}."
        ),
    }[e.reason]


# -- whole run ------------------------------------------------------------------


def _drop_profile(trace: Trace) -> tuple[float, float, int | None, float]:
    """Start and final inertia, and how fast it dropped.

    Returns ``(start, final, n, fraction)``: by the end of iteration ``n`` the
    inertia had made ``fraction`` (at least 80%) of its total drop. ``n`` is
    None when the inertia did not drop.
    """
    assigns = [s for s in trace if s.phase == "assign"]
    start = assigns[0].metrics["inertia"]
    final = trace[-1].metrics["inertia"]
    total = start - final
    if total <= 1e-12 * max(abs(start), 1.0):
        return start, final, None, 0.0
    for s in trace:
        if s.phase in ("update", "done"):
            fraction = (start - s.metrics["inertia"]) / total
            if fraction >= 0.8:
                return start, final, int(s.state["iteration"]), fraction
    return start, final, None, 1.0  # pragma: no cover - the done step reaches 100%


@summarizes("KMeans")
def kmeans_summary(ctx: NarrationContext) -> str:
    trace = ctx.trace
    done = trace[-1].event(Converged)
    init = trace[0].event(InitChosen)
    k = len(trace[0].state["centers"])
    start, final, n, fraction = _drop_profile(trace)
    n_iter = done.n_iter
    fast = n is not None and n < n_iter
    drop = _from_to(ctx, start, final)

    if ctx.math:
        iters = ctx.plural(n_iter, "iteration")
        text = f"Lloyd's algorithm, k = {k}: J fell from {drop} in {iters}"
        if fast:
            text += f" ({ctx.percent(fraction)} of ΔJ by iteration {n})"
        return text + "."

    if ctx.detailed:
        why = {
            "labels stable": "no label changed",
            "tol": "centroid shift below tolerance",
            "max_iter": "hit max_iter, not converged",
        }[done.reason]
        text = (
            f"K-means (k = {k}, init = {init.method}) ran "
            f"{ctx.plural(n_iter, 'iteration')} ({why}). Inertia fell from {drop}"
        )
        if fast:
            text += (
                f"; {ctx.percent(fraction)} of the drop came in the first "
                f"{ctx.plural(n, 'iteration')}"
            )
        text += "."
        if init.n_init > 1:
            text += f" Best of {init.n_init} restarts."
        return text

    if done.reason == "max_iter":
        text = f"K-means used all {n_iter} allowed iterations without fully settling"
    else:
        text = f"K-means settled in {ctx.plural(n_iter, 'iteration')}"
    if fast:
        first = "iteration" if n == 1 else f"{n} iterations"
        text += (
            f"; {ctx.percent(fraction)} of the drop in inertia happened in the first "
            f"{first}."
        )
    elif final < start:
        text += f"; the inertia fell from {drop}."
    else:
        text += "; the starting centroids were already stable."
    if init.n_init > 1:
        text += f" It kept the best of {init.n_init} restarts."
    return text


# -- diagnostics ----------------------------------------------------------------


@diagnoses("KMeans")
def kmeans_diagnostics(trace: Trace) -> list[Diagnostic]:
    found = unscaled_features(trace)
    for step in trace:
        empty = step.event(EmptyCluster)
        if empty is None:
            continue
        if empty.sample < 0:
            text = (
                f"Cluster {empty.cluster} stayed empty: the data has fewer distinct "
                "points than clusters."
            )
            fix = "Use a smaller n_clusters or remove duplicate points."
        else:
            text = (
                f"Cluster {empty.cluster} lost all its points in iteration "
                f"{step.state.get('iteration')}, so its centroid had to be moved."
            )
            fix = (
                "Empty clusters usually mean poor starting centroids or too many "
                "clusters: try init='k-means++', more restarts (n_init=10) or a "
                "smaller n_clusters."
            )
        found.append(Diagnostic(text=text, fix=fix, step=step.index))
        break
    done = trace[-1].event(Converged) if len(trace) else None
    if done is not None and done.reason == "max_iter":
        found.append(
            Diagnostic(
                text=(
                    f"K-means hit max_iter = {done.n_iter} before converging, so the "
                    "clusters may not be final."
                ),
                fix=(
                    "Increase max_iter, or scale the features so k-means converges "
                    "faster."
                ),
                step=len(trace) - 1,
            )
        )
    return found
