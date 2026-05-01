# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Overview
- URL: https://www.kaggle.com/competitions/playground-series-s6e5
- Task: binary classification - predict `pitnextlap` (0 or 1)
- Metric: `area under the ROC curve`

For each id in the test set, you must predict a probability for the pitnextlab target.

## Environment
- Package manager: uv
- Run scripts: `uv run python src/model.py`
- Add packages: `uv add <package>`
- Never use `pip install` or `python` directly

## Cross-Validation
- `KFold(n_splits=5)` - mandatory, no exceptions
- Never fit on full training set without CV; no test set peeking
- Report OOF area under the ROC curve as `mean ± std` across folds

## Structure

- `notebooks/` — Jupyter notebooks for exploration and experiments
- `src/` — reusable Python modules extracted from notebooks
- `data/` — raw and processed datasets (not committed)
- `submissions/` — Kaggle competition submission files
- `docs/` — documentation and notes
- `main.py` — minimal entry point

## Code Style
- Python only; functions small and single purpose
- Prefer explicit over clever; readability matters for iteration speed
- Do not overengineer - keep code as straightforward as possible
- No target leakage ever
- Always lowercase column names `df.columns = df.columns.str.lower()`
