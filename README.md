# glasslearn

*scikit-learn you can see through.*

glasslearn re-implements classic machine-learning algorithms in plain numpy,
behind the scikit-learn API. Every `fit` records what the algorithm does, step
by step, and `.explain()` turns that recording into a narrated, scrubbable
picture right inside your Jupyter or VS Code notebook cell.

It is built for learning: the numbers match scikit-learn, and every sentence of
the explanation comes from what was actually recorded, never from guesswork.

> Status: early alpha (0.1.0a2). `KMeans` is ready; more algorithms are on the way.

## Quickstart

```python
import glasslearn as gl
from glasslearn.cluster import KMeans

X = gl.datasets.make_blobs(n_samples=300, centers=4, random_state=0, as_frame=True)
km = KMeans(n_clusters=4, random_state=0).fit(X)
km.explain()  # narrated, scrubbable view in the notebook cell
```

Then, in new cells:

- `km.explain(level="math")` explains the same run with formulas (`"detailed"`
  is in between; `"beginner"` is the default).
- `km.explain(static=True).save("kmeans.html")` writes a standalone HTML file
  that opens offline.
- `km.cluster_centers_`, `km.labels_`, `km.n_iter_` and `km.inertia_` are the
  usual scikit-learn attributes.

In the notebook you get:

- a plot of the data with each centroid's move drawn as an arrow, and a faint
  trail of where it has been;
- a play button, a slider and step buttons to move through the run;
- a few sentences per step that say what changed and why ("Centroid 1 moved to
  the left by 0.31 because 1 point joined it and 15 points left, so the average of
  its points changed.");
- the inertia curve, with the current step marked;
- warnings with a suggested fix when something looks wrong, such as features on
  very different scales, an empty cluster or hitting `max_iter`.

Data with more than two features is drawn with PCA (the axis titles say how much
of the spread is kept), or with two features you choose:
`km.explain(projection=("age", "income"))`.

## Install

glasslearn needs Python 3.10 or newer. Install it from
[PyPI](https://pypi.org/project/glasslearn/):

```bash
pip install glasslearn
```

It depends on numpy, pandas, plotly, ipywidgets and anywidget, which pip
installs for you.

## Works like scikit-learn

Swap one import and keep the rest of your code:

```python
from glasslearn.cluster import KMeans  # instead of sklearn.cluster
```

- Same constructor parameters and fitted attributes (`cluster_centers_`,
  `labels_`, `inertia_`, `n_iter_`, `feature_names_in_`).
- Works with `clone`, `Pipeline`, `GridSearchCV` and `cross_val_score`, and
  passes scikit-learn's `check_estimator` (one documented exception).
- Tested against scikit-learn: same labels, same `n_iter_`, centroids and
  inertia equal to 1e-7 relative tolerance, including k-means++ seeding,
  restarts and empty-cluster handling.
- `fit(X, record=False)` gives the identical model without recording, for
  speed. The recording itself is plain data in `model.trace_`
  (`trace_.to_frame()`, `trace_.metric("inertia")`, `trace_.save(...)`).

Differences from scikit-learn are listed in each estimator's docstring (for
example, `KMeans.fit` takes no `sample_weight` yet).

## Algorithms

| Module | Estimator | Status |
| --- | --- | --- |
| `glasslearn.cluster` | `KMeans` | ready |
| `glasslearn.linear_model` | `LinearRegression`, `LogisticRegression` | planned |
| `glasslearn.neighbors` | `KNeighborsClassifier`, `KNeighborsRegressor` | planned |
| `glasslearn.tree` | `DecisionTreeClassifier`, `DecisionTreeRegressor` | planned |

The example notebook [`examples/01_kmeans.ipynb`](https://github.com/PriyanshuKumar-CSE/glasslearn/blob/main/examples/01_kmeans.ipynb)
walks through k-means, including what goes wrong with a bad start or unscaled
features.

## How it works

Each algorithm only *records*: its training loop reports steps and events
(`CentroidMoved`, `PointsReassigned`, `Converged`, ...) to a recorder. After
fitting, a narrator turns the events into sentences from tested templates, and
a view turns each step into a plot. Nothing is generated at runtime and no
network is used. The design is in [`docs/PLAN.md`](https://github.com/PriyanshuKumar-CSE/glasslearn/blob/main/docs/PLAN.md).

## Known issues

This is an alpha. Things that are still rough:

- **The live widget needs a notebook with ipywidgets 8**, such as JupyterLab 4,
  Notebook 7 or VS Code. In a plain Python script `explain()` displays nothing;
  use `km.explain().save("run.html")` instead. If the widget area stays blank,
  `km.explain(static=True)` shows a plain plotly figure with a slider.
- **Google Colab shows the static figure**, not the live widget: Colab does not
  reliably render plotly's `FigureWidget`. Use the ▶ button and the slider
  under the plot to step through training, and the narration above the plot.
- **VS Code may offer no kernel for a uv-created `.venv`.** uv 0.13 writes the
  Python version without its patch number (`version_info = 3.12`) to
  `.venv/pyvenv.cfg`, and VS Code then skips that environment. Either register
  it as a kernel with
  `uv run python -m ipykernel install --user --name glasslearn` (ipykernel must
  be installed in that venv), or put the full version in that file:
  `sed -i "s/^version_info = .*/version_info = $(.venv/bin/python -c 'import platform; print(platform.python_version())')/" .venv/pyvenv.cfg`,
  then reload the VS Code window.
- **Saved HTML files are large** (about 5.5 MB for the quickstart) because
  plotly.js is embedded so the file opens offline. They grow with the number of
  steps.
- **Plots always have a light background**, also in dark notebook themes.
- **`NotFittedError` is glasslearn's own class.** Like scikit-learn's, it is a
  `ValueError` and an `AttributeError`, but `except
  sklearn.exceptions.NotFittedError` will not catch it. Catch
  `glasslearn.NotFittedError` (or `ValueError`) instead. This is the one
  `check_estimator` check that fails.
- **`KMeans` gaps compared with scikit-learn:** no `sample_weight`, no sparse
  input, no `set_output`, and float32 input is computed in float64.

## Development

```bash
uv sync --all-extras          # install with dev tools
uv run pytest                 # tests, parity with scikit-learn, example notebooks
uv run pytest --update-golden # regenerate narration golden files (review the diff)
uv run ruff check . && uv run ruff format .
uv build                      # sdist and wheel into dist/
```

## License

BSD 3-Clause. See [LICENSE](https://github.com/PriyanshuKumar-CSE/glasslearn/blob/main/LICENSE).
