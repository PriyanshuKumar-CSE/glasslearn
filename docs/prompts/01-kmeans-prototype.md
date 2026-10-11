# Task 01: project foundations + KMeans prototype (milestones M0 + M1)

Build the shared core plus KMeans, end to end. Do not build other algorithms yet,
but design the core so LinearRegression, LogisticRegression, KNN and DecisionTree
slot in. Read `CLAUDE.md` and `docs/PLAN.md` first.

## Setup

- Python >= 3.10, src layout, hatchling build, managed with uv. Tools: ruff,
  pytest, hypothesis, nbmake. GitHub Actions CI on Python 3.10-3.13.
- Runtime deps: numpy, plotly>=5, ipywidgets>=8. pandas optional (duck-typed:
  anything with `.columns` and `.to_numpy()`). Dev/test deps: scikit-learn, pandas,
  pytest, hypothesis, nbmake, ruff, jupyter.
- `pyproject.toml` metadata:
  - name `glasslearn`, version `0.1.0a1`
  - description "scikit-learn you can see through: ML algorithms that explain each training step"
  - license `BSD-3-Clause` (add a LICENSE file, copyright 2026 Priyanshu Kumar)
  - authors `Priyanshu Kumar`, requires-python `>=3.10`
  - urls: Repository `https://github.com/PriyanshuKumar-CSE/glasslearn`,
    Issues `https://github.com/PriyanshuKumar-CSE/glasslearn/issues`
  - classifiers: Development Status :: 3 - Alpha, Intended Audience :: Education,
    Intended Audience :: Science/Research, Topic :: Scientific/Engineering :: Artificial Intelligence
- `.github/workflows/ci.yml`: ruff + pytest on 3.10-3.13 for every push and PR.
- `.github/workflows/release.yml`: on tag `v*`, `uv build`, then publish with
  `pypa/gh-action-pypi-publish@release/v1` using trusted publishing
  (environment `pypi`, `permissions: id-token: write`). Add a separate
  manually-triggered job or workflow that publishes to TestPyPI
  (environment `testpypi`, `repository-url: https://test.pypi.org/legacy/`).
- `.gitignore` for Python, uv, Jupyter checkpoints, `dist/`.

## Layout

```
src/glasslearn/
  __init__.py  base.py  _validation.py  _config.py
  trace/      recorder.py  step.py  events.py  trace.py
  narrate/    registry.py  format.py  diagnostics.py  glossary.py
  viz/        scene.py  projection.py  widget.py  static.py  backends/plotly_backend.py
  cluster/__init__.py          (exports KMeans)
  cluster/kmeans/  estimator.py  events.py  narrator.py  view.py
  datasets/__init__.py         (numpy-only make_blobs with as_frame=True option)
tests/  test_base.py  test_trace.py  parity/test_kmeans_parity.py
        narration/test_kmeans_golden.py  test_widget_smoke.py
examples/01_kmeans.ipynb
```

## Core contracts

1. **GlassEstimator** (`base.py`): `__init__` stores args unchanged;
   `get_params`/`set_params` by introspecting the `__init__` signature (sklearn's
   convention, without importing sklearn); sklearn-style `__repr__`.
   `fit(X, y=None, *, record=True)`: validate input, set `feature_names_in_`
   ("x0", "x1"... when no names) and `n_features_in_`, create a Recorder, call
   `self._fit(X, y, rec)`, then `self.trace_ = rec.freeze(meta)`.
   `explain(...)` returns an ExplainView. Add `ClusterMixin` with `fit_predict`.
2. **Event**: frozen dataclass base with `salience: float` (0..1).
   **Step**: frozen dataclass (`index, phase, state: dict, metrics: dict, events: tuple`).
   **Recorder**(`enabled, every, max_steps`): `step(...)` is a no-op when disabled
   and copies arrays it gets. **Trace**: `algorithm, params, feature_names, steps`;
   `__len__`, `__getitem__`, `metric(name) -> np.ndarray`, `to_frame()` (lazy pandas
   import), `save`/`load` (.npz + JSON).
3. **Narration**: registry with an `@narrates(EventType)` decorator mapping event
   types to functions `(event, ctx) -> str`. `NarrationContext` provides `level`
   ("beginner" | "detailed" | "math"), feature names, `direction(old, new)` in the
   plotted 2D plane ("up and to the right"), `biggest_change(old, new) -> (feature,
   delta)`, number formatting and plurals. `Narrator.narrate(step, prev)` returns the
   top 3 events by salience as sentences, each linking effect to cause.
   `Narrator.summary(trace)` gives a whole-run summary. Glossary adds a one-line
   definition the first time a term appears. Diagnostics: functions
   `trace -> list[(text, suggested_fix)]`.
4. **Scene** (`viz/scene.py`): backend-neutral dataclasses Points, Markers, Arrows,
   Lines, axis titles. A View has `render(step, projection) -> Scene`.
   `plotly_backend` turns a Scene into traces and updates an existing FigureWidget
   in place inside `fig.batch_update()`; trace count and order stay fixed across
   steps so an update only swaps data.
5. **Projection**: Identity (<= 2 features), FeaturePair(name_a, name_b), PCA (own
   numpy SVD, fitted once on training X; `transform`, `inverse_transform`,
   `explained_variance_ratio_`). `projection="auto"` chooses. Axis titles like "PC1 (61%)".
6. **ExplainView** (`viz/widget.py`), an ipywidgets VBox: summary line; header with
   "Iteration 3 of 7 · update" and a level ToggleButtons; Play jslinked to an
   IntSlider plus prev/next buttons; main FigureWidget; small metric FigureWidget
   (inertia curve with a marker at the current step); HTML narration panel with a
   highlighted warning box for diagnostics. Display via `_ipython_display_`.
   `static=True`, or no widget support detected: one plotly figure with frames and
   a built-in slider, narration as a per-frame annotation. `.save("x.html")` works
   in both modes. Draw at most `max_points=2000` points (sampled per label); the
   narration says "showing N of M points" when sampling.

## KMeans (`cluster/kmeans`)

- `KMeans(n_clusters=8, *, init="k-means++", n_init="auto", max_iter=300,
  tol=1e-4, random_state=None)`. `init`: "k-means++", "random" or an
  `(n_clusters, n_features)` array. `n_init="auto"` = 1 for k-means++ or an array,
  10 for "random" (as in current sklearn). Keep only the best run's trace; record
  every run's inertia in meta.
- Fitted: `cluster_centers_, labels_, inertia_, n_iter_, feature_names_in_,
  n_features_in_, trace_`. Methods: `fit, predict, fit_predict, transform`
  (distances), `score` (= -inertia, as sklearn), `summary()` (centroids as a
  DataFrame with feature-name columns).
- Algorithm: Lloyd. **Before writing it**, open `sklearn/cluster/_kmeans.py` in the
  installed scikit-learn and mirror `_kmeans_single_lloyd` and `_tolerance`
  exactly: tol scaled by the mean per-feature variance; stop when labels are
  unchanged or the sum of squared centre shifts <= scaled tol; a final label pass
  when not strictly converged; empty clusters relocated to the farthest points;
  the same `n_iter_` meaning. k-means++ follows sklearn's greedy variant.
- Recording: an `init` step (InitChosen: method, and for k-means++ the order the
  seeds were picked). Per iteration an `assign` step (state: centers, labels;
  PointsReassigned with n_changed and a k x k transition matrix) and an `update`
  step (state: old and new centers; CentroidMoved per cluster with distance,
  n_gained, n_lost; EmptyCluster when relocated). Metrics: inertia, n_changed,
  center_shift_tot. A final `done` step with Converged(reason = "labels stable" |
  "tol" | "max_iter", n_iter). Store labels in the smallest int dtype, and only
  when they changed.
- View: points coloured by label, centroids as large X markers, arrows from old to
  new centroid on update steps, a faint trail of each centroid's past positions.
- Narration: templates for every event at all three levels, plus a summary such as
  "K-means settled in 7 iterations; 85% of the drop in inertia happened in the first 2."
- Diagnostics: unscaled features (largest feature std / smallest > 10), an empty
  cluster occurred, max_iter reached without converging.

## Tests (all must pass)

- Parity vs `sklearn.cluster.KMeans`: 5 seeds x datasets (2D blobs, 10D blobs,
  uniform noise, a pandas DataFrame with named columns). Pass the same explicit
  init array with `n_init=1`; assert `cluster_centers_`, `labels_`, `inertia_`
  (rtol 1e-7) and `n_iter_` match. Include an empty-cluster case.
- `sklearn.utils.estimator_checks.check_estimator(KMeans())` passes; any skipped
  check is listed with a reason.
- `clone()`, `Pipeline` and `GridSearchCV` over `n_clusters` run without errors.
- Hypothesis: inertia never rises across update steps; labels lie in [0, k).
- Trace: len, `metric("inertia")` shape, save/load round trip; `record=False` gives
  identical results and an empty trace.
- Golden narration files per level under `tests/narration/golden/`, regenerated
  with `pytest --update-golden`.
- Widget smoke test: build ExplainView, set the slider to every step, check figure
  data changes and the number of traces stays the same; static mode has
  `len(frames) == len(trace)`.
- `examples/01_kmeans.ipynb` runs under nbmake.

## Definition of done

- `uv run pytest` green and `uv run ruff check .` clean.
- In a fresh notebook this shows the widget, scrubbing is smooth, and each step
  shows 2-4 sentences that use the real column names:

  ```python
  import glasslearn as gl
  from glasslearn.cluster import KMeans

  X = gl.datasets.make_blobs(n_samples=300, centers=4, random_state=0, as_frame=True)
  km = KMeans(n_clusters=4, random_state=0).fit(X)
  km.explain()
  ```

- README with a 10-line quickstart.

## Build order (commit after each)

1. pyproject, LICENSE, .gitignore, CI and release workflows
2. base + trace (+ tests)
3. KMeans algorithm + parity tests
4. narration + golden tests
5. scene, projection, plotly backend
6. widget + static fallback (+ smoke test)
7. example notebook
8. README
