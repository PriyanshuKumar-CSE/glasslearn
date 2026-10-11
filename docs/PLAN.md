# Glasslearn design plan

Classic ML algorithms rewritten from scratch in numpy, behind the scikit-learn API.
Every `fit` records a trace; `.explain()` turns it into a narrated visual with a
slider inside the notebook cell. v0.1 ships five algorithms: KMeans,
LinearRegression, LogisticRegression, KNN (KNeighborsClassifier/Regressor) and
DecisionTree (Classifier/Regressor).

## Scope

- **API parity from day one:** import paths, constructor params, fitted attributes,
  `get_params`/`set_params`, works in sklearn `Pipeline`, `GridSearchCV`,
  `cross_val_score`. Users swap one word in an import line.
- **Algorithm coverage in tiers:**
  - Tier 1 (v0.1): the five above.
  - Tier 2: Ridge, Lasso, Naive Bayes, linear SVM (SGD), PCA, DBSCAN, GMM,
    Random Forest, Gradient Boosting, MLP.
  - Tier 3: utilities with explain views: StandardScaler, train_test_split,
    cross_val_score, confusion matrix, Pipeline.

## Architecture: record, then render

```
Your code -> fit() wrapper -> algorithm _fit loop -> Recorder
                                                        | freeze()
                                                        v
                                               model.trace_ (Trace)
                                                        | .explain()
                         +------------------------------+-------------------+
                         v                              v                   v
                     Narrator                  View (per algorithm)     Projection
               events -> sentences             state -> Scene layers   fitted once on X
                         +------------------------------+-------------------+
                                                        v
                                   ExplainView widget in the notebook cell
```

### What one step means

| Algorithm | One step | State snapshot | Key events |
| --- | --- | --- | --- |
| KMeans | half iteration: assign, then update | centers, labels | PointsReassigned, CentroidMoved, EmptyCluster, Converged |
| LinearRegression `solver="normal"` | one matrix stage: XᵀX, solve, residuals | coef, intercept, residuals | CoefficientSolved, LargestResidual |
| Linear/LogisticRegression `solver="gd"` | one epoch | coef, intercept, loss, grad norm | LossDropped, LossRose, CoefficientChanged, Converged |
| DecisionTree | one node expansion | node, depth, chosen split, top 5 candidates, class counts | SplitChosen, LeafCreated |
| KNN | one query point, recorded at predict | distances, neighbour idx, votes | NeighborsFound, VoteDecided, Tie |

### Core data structures

```python
@dataclass(frozen=True)
class Event:
    salience: float = 0.5          # 0..1, decides what gets narrated first

@dataclass(frozen=True)
class Step:
    index: int
    phase: str                     # init, assign, update, epoch, split, done
    state: dict[str, Any]
    metrics: dict[str, float]
    events: tuple[Event, ...]

class Recorder:
    def __init__(self, enabled=True, every=1, max_steps=500): ...
    def step(self, phase, *, state, metrics=None, events=()): ...
    def freeze(self, meta) -> Trace: ...

@dataclass
class Trace:
    algorithm: str
    params: dict
    feature_names: list[str]
    target_names: list[str] | None
    steps: list[Step]
    # __len__, __getitem__, metric(name), to_frame(), save(), load()
```

```python
class GlassEstimator:
    def fit(self, X, y=None, *, record=True):
        X, y, names = validate(X, y)
        self.feature_names_in_ = names
        self.n_features_in_ = X.shape[1]
        rec = Recorder(enabled=record, **self._record_options())
        self._fit(X, y, rec)
        self.trace_ = rec.freeze(meta=self._meta(X, y))
        return self

    def explain(self, X=None, *, step=None, projection="auto",
                level="beginner", static=None): ...
```

Column names: `gl.set_config(output="auto")` returns pandas when pandas went in;
`model.summary()` gives a named DataFrame of coefficients / centroids / importances.

## Visualization

- **plotly** for drawing (in-place updates, smooth scrubbing), **ipywidgets** for
  controls. matplotlib only later (v0.3+) for PNG/GIF export.
- **ExplainView** (VBox): summary line; header ("Iteration 3 of 7 · update") +
  level toggle; Play + IntSlider + prev/next; main FigureWidget; small metric
  curve with current-step marker; HTML narration panel with warning box.
- **Static fallback** (`static=True` or no widget support): plotly figure with
  frames + built-in slider, narration as per-frame annotation. `.save("x.html")`.
- Views return a backend-neutral `Scene` (Points, Markers, Arrows, Lines, Region,
  Tree); `plotly_backend` converts it.

### High-dimensional data

- `projection="auto"`: raw axes for 1-2 features, else PCA.
- `projection=("age", "income")`: two named columns.
- `projection="pca"`: own numpy SVD; axis titles show variance kept ("PC1 (61%)").
- Boundaries in PCA space = slice through the mean of other directions; label it.
- Trees draw the tree itself; partition view only for a chosen feature pair.
- Regression with many features: predicted-vs-actual and residual plots.
- `max_points=2000` drawn (sampled per label); computation uses all rows.
- Never t-SNE/UMAP: they cannot place centroids or boundaries consistently.

## Narration

Templates filled from Events, ranked by salience, 2-4 sentences per step.

1. Collect step events + derived comparisons with the previous step.
2. Rank by salience, keep top 3; summarise the rest ("the other centroids barely moved").
3. Render each with its template at the chosen level (beginner / detailed / math).
4. Link effect to cause in the same step.
5. Append diagnostics in a warning box.
6. Gloss a technical term the first time it appears.

```python
@narrates(CentroidMoved)
def centroid_moved(e, ctx) -> str:
    where = ctx.direction(e.old, e.new)
    if ctx.level == "beginner":
        return (f"Centroid {e.cluster} moved {where} because {e.n_gained} points "
                f"joined it and {e.n_lost} left, so the average of its members changed.")
    ...
```

Whole-run summary example: "K-means settled in 7 iterations. 85% of the drop in
inertia happened in the first 2."

Diagnostics (rule-based, each with a fix): loss rose 3 epochs in a row (learning
rate too high); one feature's range 100x others (scale first); empty cluster; tree
at 100% train accuracy at high depth (overfitting); KNN tie (odd k); no change for
many iterations (tol / max_iter).

## Roadmap (8-10 h/week)

| Milestone | Weeks | Build | Gate |
| --- | --- | --- | --- |
| M0 Foundations | 1 | repo, uv, CI; base classes, Recorder, Step, Event, Trace | clone/get_params pass on a dummy estimator |
| M1 KMeans | 2-3 | Lloyd + k-means++, events, narrator, view, widget, notebook 01 | matches sklearn on fixed init; widget works in Lab, VS Code, Colab |
| M2 Algorithms | 4-6 | LinearRegression (normal + gd), LogisticRegression (gd), KNN, DecisionTree (CART) | parity suite green; check_estimator OK |
| M3 Projection | 7-8 | PCA, boundary slices, max_points, 3 levels, diagnostics, HTML export | 10-feature data renders < 1 s |
| M4 Docs | 9-10 | mkdocs-material, 5 notebooks, README GIF, add-an-algorithm guide | new user runs quickstart alone |
| M5 Release | 11 | trusted publishing via GitHub Actions, TestPyPI dry run | `pip install glasslearn` works in fresh Colab |

First upload: `0.1.0a1` once KMeans works (pre-release, claims the name with real code).

## Parity testing against scikit-learn

- KMeans: same explicit `init` array, `n_init=1`; compare centers, labels, inertia
  (rtol 1e-7), n_iter_. sklearn scales `tol` by mean feature variance and checks
  the squared centre shift.
- LogisticRegression: sklearn defaults to lbfgs with `C=1.0`; compare against
  `penalty=None` or matching C at convergence (~1e-4), not step by step.
- LinearRegression: coef_/intercept_ to 1e-10, including rank-deficient X.
- Trees: predictions and feature_importances_ on data without tied gains.
- KNN: kneighbors indices and distances exactly; distance ties compared as sets.
- API: `check_estimator`, `clone`, `Pipeline`, `GridSearchCV`.
- Hypothesis invariants: inertia never rises; leaves partition rows; probabilities sum to 1.

## Risks

| Risk | Mitigation |
| --- | --- |
| Scope creep | Tiers; ship 5 first |
| Wrong explanations | Text only from recorded numbers; golden tests; teacher review |
| Divergence from sklearn | Parity tests; document intentional differences |
| Speed (pure numpy, trees) | Learning tool positioning; vectorise; `record=False` |
| Recording memory | Store changes only; smallest dtypes; `max_steps` |
| Misleading 2D | Show variance kept; feature-pair option; label slices |
| Notebook fragmentation | Static fallback; CI smoke tests |
