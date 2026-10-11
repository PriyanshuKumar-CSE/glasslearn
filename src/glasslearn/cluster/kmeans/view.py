"""How a k-means step is drawn."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...narrate.format import number
from ...trace import Step, Trace
from ...viz.projection import Projection
from ...viz.scene import (
    Arrows,
    Lines,
    Markers,
    Points,
    Scene,
    View,
    padded_range,
    sample_per_group,
    view_for,
)


@view_for("KMeans")
class KMeansView(View):
    """Points coloured by cluster, centroids as large X markers, arrows from old to
    new centroid on update steps, and a faint trail of each centroid's path.

    Parameters
    ----------
    trace : Trace
        A fitted KMeans ``trace_``.
    max_points : int, default=2000
        Most points drawn, sampled per final cluster; the same rows are shown
        at every step.
    """

    #: Metric drawn under the main plot.
    metric = "inertia"

    def __init__(self, trace: Trace, *, max_points: int = 2000) -> None:
        super().__init__(trace, max_points=max_points)
        if trace.X is None or len(trace) == 0:
            raise ValueError("This trace has no recorded steps; fit with record=True.")
        try:
            final = np.asarray(trace.latest("labels", -1))
        except KeyError:
            final = np.zeros(len(trace.X), dtype=int)
        self.rows = sample_per_group(final, self.max_points)
        self._centers = [np.asarray(step.state["centers"]) for step in trace]
        self._prepared: dict[int, dict[str, Any]] = {}

    @property
    def n_shown(self) -> int:
        return len(self.rows)

    def _prepare(self, projection: Projection) -> dict[str, Any]:
        key = id(projection)
        if key not in self._prepared:
            X = self.trace.X[self.rows]
            points = projection.transform(X)
            centers = [projection.transform(c) for c in self._centers]
            every_x = np.concatenate([points[:, 0], *(c[:, 0] for c in centers)])
            every_y = np.concatenate([points[:, 1], *(c[:, 1] for c in centers)])
            names = self.trace.feature_names
            shown = range(min(len(names), 6))
            hover = [
                f"row {row}<br>"
                + "<br>".join(f"{names[j]} = {number(values[j])}" for j in shown)
                for row, values in zip(self.rows.tolist(), X, strict=True)
            ]
            self._prepared[key] = {
                "points": points,
                "centers": centers,
                "x_range": padded_range(every_x),
                "y_range": padded_range(every_y),
                "hover": hover,
            }
        return self._prepared[key]

    def render(self, step: int | Step, projection: Projection) -> Scene:
        """The scene for one step, drawn with an already fitted projection."""
        index = step.index if isinstance(step, Step) else int(step)
        if index < 0:
            index += len(self.trace)
        step = self.trace[index]
        data = self._prepare(projection)
        centers = data["centers"][index]
        k = len(centers)
        clusters = np.arange(k)

        try:
            labels = np.asarray(self.trace.latest("labels", index))[self.rows].astype(
                int
            )
        except KeyError:  # the init step: nobody is assigned yet
            labels = np.full(len(self.rows), -1)

        trails = []
        for j in clusters:
            path = np.array([c[j] for c in data["centers"][: index + 1]])
            keep = np.ones(len(path), dtype=bool)
            keep[1:] = np.any(path[1:] != path[:-1], axis=1)
            trails.append(path[keep])

        if step.phase == "update":
            old = projection.transform(step.state["old_centers"])
            arrows = Arrows(
                old[:, 0], old[:, 1], centers[:, 0], centers[:, 1], clusters
            )
        else:
            empty = np.empty(0)
            arrows = Arrows(empty, empty, empty, empty, np.empty(0, dtype=int))

        x_title, y_title = projection.axis_titles
        iteration = step.state.get("iteration", 0)
        return Scene(
            layers=[
                Lines(trails, clusters, name="trail"),
                Points(
                    data["points"][:, 0],
                    data["points"][:, 1],
                    labels,
                    hover=data["hover"],
                    group_name="cluster",
                ),
                arrows,
                Markers(
                    centers[:, 0],
                    centers[:, 1],
                    clusters,
                    labels=[str(j) for j in clusters],
                    name="centroids",
                ),
            ],
            x_title=x_title,
            y_title=y_title,
            x_range=data["x_range"],
            y_range=data["y_range"],
            caption=f"Iteration {iteration} · {step.phase}",
        )
