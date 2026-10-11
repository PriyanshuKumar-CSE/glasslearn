"""A tiny estimator used to test the shared machinery without a real algorithm."""

from dataclasses import dataclass

import numpy as np

from glasslearn.base import GlassEstimator
from glasslearn.trace import Event


@dataclass(frozen=True)
class Nudged(Event):
    amount: float
    toward: tuple[float, ...]


class MeanSeeker(GlassEstimator):
    """Moves a point from the origin toward the data mean, ``rate`` of the way
    per step, and labels samples by which side of the point they fall."""

    _sticky_state = ("labels",)

    def __init__(self, n_steps=3, rate=0.5):
        self.n_steps = n_steps
        self.rate = rate

    def _fit(self, X, y, rec):
        point = np.zeros(X.shape[1])
        target = X.mean(axis=0)
        rec.step("init", state={"point": point, "labels": self._labels(X, point)})
        for _ in range(self.n_steps):
            new = point + self.rate * (target - point)
            rec.step(
                "update",
                state={"point": new, "labels": self._labels(X, new)},
                metrics={"distance": float(np.linalg.norm(target - new))},
                events=[
                    Nudged(
                        amount=float(np.linalg.norm(new - point)),
                        toward=tuple(target),
                        salience=0.7,
                    )
                ],
            )
            point = new
        rec.step("done", state={"point": point})
        self.point_ = point

    @staticmethod
    def _labels(X, point):
        return (X[:, 0] > point[0]).astype(np.int64)

    def predict(self, X):
        X = self._validate_predict_data(X)
        return np.linalg.norm(X - self.point_, axis=1)


class Wrapper(GlassEstimator):
    def __init__(self, estimator=None, flag=True):
        self.estimator = estimator
        self.flag = flag
