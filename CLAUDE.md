# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python 3.13.2 with a local `.venv`. Activate before running anything:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

## Data Pipeline

Raw data lives in `data/raw/` and must not be modified. The processed file `data/processed/results_clean.parquet` is the output of notebook 01 and the input to notebook 02 — run them in order.

**Raw files:**
- `results.csv` — international match results from 1872 to present (includes future 2026 World Cup fixtures with `NaN` scores)
- `goalscorers.csv` — individual goalscorer events
- `shootouts.csv` — penalty shootout outcomes
- `former_names.csv` — historical country name mappings

**Cleaning decisions made in `01_EDA.ipynb`:**
- Drop rows with `NaN` scores (these are future fixtures, including 2026 WC games)
- Filter to matches from 2006 onward
- Drop teams with fewer than 30 total appearances

## Model Architecture (`02_dixon_coles.ipynb`)

Implements the **Dixon-Coles (1997)** bivariate Poisson model for match score prediction.

**Parameters per team:** `attack[i]` and `defense[i]` (log-scale ratings). Attack parameters are constrained to sum to zero (one is derived from the rest).

**Global parameters:**
- `home_adv` — log home advantage (set to 0 for neutral venues)
- `rho` — low-score correction factor, bounded to `[-0.1, 0.1]`
- `xi` — time-decay weight (`w = exp(-xi * days_ago)`); currently `xi = 0.0` (no decay)

**Expected goals for a match:**
```
lam = exp(home_adv + att[home] + def[away])   # home goals
mu  = exp(att[away] + def[home])              # away goals
```

**Dixon-Coles correction** adjusts the joint probability only for scorelines `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)` via `tau` factors.

**Optimization:** L-BFGS-B via `scipy.optimize.minimize` with an analytic gradient (`grad()` function). Converges on 18,989 matches across 226 teams.

**Prediction functions:**
- `expected_goals(home, away, neutral)` → `(lam, mu)`
- `score_matrix(home, away, neutral, max_goals=10)` → normalized score matrix
- `predict(home, away, neutral)` → dict with `home_win`, `draw`, `away_win` probabilities
