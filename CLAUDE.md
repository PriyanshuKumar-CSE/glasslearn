# Glasslearn

Open-source Python ML library for learners: classic algorithms implemented from
scratch in numpy behind the scikit-learn API. Every `fit` records a step-by-step
trace, and `.explain()` renders a narrated, scrubbable visual inside a Jupyter or
VS Code notebook cell. Tagline: *scikit-learn you can see through.*

Full design: `docs/PLAN.md`. Current task spec: `docs/prompts/`.

## Non-negotiables

- **Algorithms only record.** Algorithm code (`_fit`) writes `Step`s and `Event`s
  to a `Recorder`. It never formats text and never plots. Narration and views read
  the trace afterwards.
- **scikit-learn API parity.** Same import paths (`glasslearn.cluster.KMeans`),
  same constructor params, same fitted attributes (`coef_`, `labels_`, `n_iter_`).
  `__init__` only stores its args unchanged. Must pass
  `sklearn.utils.estimator_checks.check_estimator` and work with `clone`,
  `Pipeline` and `GridSearchCV`.
- **scikit-learn is test-only.** glasslearn never depends on sklearn and never
  imports it at module level or in normal code paths. The one exception is the
  `__sklearn_tags__` hooks in `base.py`: sklearn calls them and requires its own
  `Tags` objects, so they import those inside the method body (sklearn is always
  already loaded when they run). sklearn is the reference parity tests compare
  against.
- **No AI or network calls at runtime.** Explanations come only from recorded state.
- **Visuals live in the notebook cell.** No servers or web apps. Live mode =
  ipywidgets + plotly `FigureWidget`; fallback = plotly figure with frames.
- **Every number in a sentence comes from the trace.** Narration is deterministic
  and covered by golden-file tests.

## Stack and commands

- Python >= 3.10, src layout, hatchling, managed with **uv**.
- Runtime deps: numpy, plotly>=5.11, ipywidgets>=8, anywidget (plotly>=6 builds
  FigureWidget on it). pandas is optional (duck-typed).
- Dev deps: pytest, hypothesis, nbmake, ruff, scikit-learn, pandas.

```bash
uv sync --all-extras          # install
uv run pytest                 # all tests
uv run pytest tests/parity    # parity with scikit-learn only
uv run pytest --update-golden # regenerate narration golden files (review the diff!)
uv run ruff check . && uv run ruff format .
uv build                      # sdist + wheel into dist/
```

## Layout

```
src/glasslearn/
  base.py  _validation.py  _config.py
  trace/     recorder.py step.py events.py trace.py
  narrate/   registry.py format.py diagnostics.py glossary.py
  viz/       scene.py projection.py widget.py static.py backends/plotly_backend.py
  <module>/<algorithm>/  estimator.py events.py narrator.py view.py
  datasets/  metrics/
tests/  test_base.py test_trace.py parity/ narration/ test_widget_smoke.py
examples/  NN_<algorithm>.ipynb
docs/      PLAN.md prompts/
```

Each algorithm is one folder with exactly four files. Adding an algorithm = copy
an existing folder, then write `_fit`, its events, templates and view.

## Conventions

- Fitted attributes end in `_`; `trace_` is one of them. `fit(..., record=False)`
  must give identical results with an empty trace.
- Events are frozen dataclasses with a `salience` (0..1). Steps have a `phase`:
  `init`, `assign`, `update`, `epoch`, `split`, `done`.
- Snapshots stay small: store label arrays only when they change, in the smallest
  int dtype; one step per epoch, never per sample; top 5 candidate splits per node.
- Projection is fitted once on training X and reused for every step.
- Plotly updates happen in place inside `fig.batch_update()`; trace count and
  order stay fixed across steps.
- Type hints on public functions; numpy-style docstrings on public classes,
  including a "Differences from scikit-learn" section when any exist.
- Tests first for parity: before writing an algorithm, read its scikit-learn
  source in the installed package and mirror its semantics (tolerances, stopping
  rules, tie-breaking, n_iter_ meaning).

## Working style

- Plan first: list files and public signatures, then wait for approval before
  writing code.
- Work in small stages; run tests and ruff after each; commit after each stage
  with a clear message.
- Never add code, data or ideas from the author's employer projects.
- Keep README examples runnable; examples/ notebooks run in CI via nbmake.
