# World Cup Predictor

Match outcome predictions for international football using the Dixon-Coles (1997) bivariate Poisson model, applied to the 2026 FIFA World Cup.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Usage

Run the notebooks in order:

1. **`01_EDA.ipynb`** — cleans `data/raw/results.csv` and writes `data/processed/results_clean.parquet`
2. **`02_dixon_coles.ipynb`** — fits the model and produces win/draw/loss probabilities for any matchup

To predict a match after running notebook 02:

```python
p = predict("Brazil", "Argentina", neutral=True)
# {"home_win": 0.42, "draw": 0.27, "away_win": 0.31}
```

## Data

Raw CSVs in `data/raw/` cover international results from 1872 to present. The dataset includes 2026 World Cup fixtures (with no scores yet) which are filtered out before model training.

Training data: matches from 2006 onward, teams with at least 30 appearances (18,989 matches, 226 teams).

## Model

Dixon-Coles bivariate Poisson with per-team attack/defense ratings, a global home advantage term, and a low-score correction factor (rho). See `CLAUDE.md` for parameter details.
