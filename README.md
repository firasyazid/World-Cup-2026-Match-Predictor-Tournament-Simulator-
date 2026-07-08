# 🏆 World Cup 2026 Match Predictor & Tournament Simulator

A machine learning project that predicts match outcomes and simulates the entire **FIFA World Cup 2026** tournament thousands of times to estimate each team's probability of winning it all.

## Overview

This project builds an **XGBoost classifier** trained on ~20 years of international football results, then uses it to run a **Monte Carlo simulation** (10,000 iterations) of the full 2026 tournament bracket — group stage + knockouts — to produce win probabilities for all 48 qualified teams.

## How it works

1. **Data ingestion** — historical match results, Elo ratings, FIFA rankings, and the official WC 2026 fixtures/teams list are loaded and cleaned.
2. **Filtering** — only matches from major official competitions since 2006 are kept (World Cup, Euro, Copa América, AFCON, Nations Leagues, Gold Cup, Asian Cup, etc.), then restricted to matches between the 48 qualified nations.
3. **Team name normalization** — historical naming inconsistencies (e.g. *United States → USA*, *Turkey → Türkiye*) are reconciled with the official WC 2026 team list.
4. **Feature engineering** — for every match, rolling form stats are computed from each team's last 10 games (win rate, goals scored/conceded), combined with:
   - Elo rating (per-year snapshot) and Elo difference
   - FIFA ranking and ranking difference
   - Head-to-head win rate between the two teams
   - A recency weight (`1 / (1 + days_ago/365)`) so recent matches count more
5. **Modeling** — an `XGBClassifier` is trained to predict `home_win` / `draw` / `away_win`, using sample weights for recency.
6. **Tournament simulation** — using each team's current form/Elo/FIFA rank, the model plays out group stage standings, then simulates the full knockout bracket round by round (with random 50/50 tiebreaks on draws), repeated 10,000 times to estimate each team's odds of winning the tournament.

## Tech stack

- **Python** (Google Colab notebook)
- **pandas** — data wrangling
- **XGBoost** (`XGBClassifier`) — match outcome prediction
- **scikit-learn** — train/test split, label encoding, evaluation metrics
- **tqdm** — progress tracking for feature generation and simulations
- **joblib** — model persistence

## Data sources

| File | Description |
|---|---|
| `results.csv` | Historical international match results (Kaggle) |
| `elo_ratings_wc2026.csv` | Yearly Elo ratings per national team |
| `wc_2026_fixtures.csv` | Official World Cup 2026 group stage fixtures |
| `wc_2026_teams.csv` | The 48 qualified teams and their FIFA rankings |

> Note: raw CSV data files are not included in this repo — you'll need to source your own copies of the datasets above and place them in the project root before running the notebook.

## Model performance

The classifier achieves roughly **60% accuracy** on held-out match outcomes (home win / draw / away win) — a solid result given the inherent unpredictability of football and the three-way classification target.

## Sample output

```
🏆 TOP 10 FAVORIS WC 2026 :
1. Brazil: XX.X%
2. France: XX.X%
3. Argentina: XX.X%
...
```

(Run the notebook to generate the actual current probabilities.)

## Project structure

```
├── world_cup.ipynb    # Main notebook (Colab)
├── world_cup.py        # Exported script version
├── model.joblib         # Trained XGBoost model (generated)
├── label_encoder.joblib # Label encoder for result classes (generated)
└── feature_cols.joblib  # Feature column list used at inference (generated)
```

## Usage

1. Open `world_cup.ipynb` in Google Colab (or run `world_cup.py` locally with the required CSVs in place).
2. Upload the four data files when prompted.
3. Run all cells — the notebook will clean the data, engineer features, train the model, and run the 10,000-simulation tournament projection.
4. Trained model artifacts (`model.joblib`, `label_encoder.joblib`, `feature_cols.joblib`) are saved for reuse (e.g. in a FastAPI/Next.js app).

## Possible next steps

- Serve the trained model via a FastAPI endpoint for live predictions
- Build a Next.js front-end to visualize live tournament odds
- Add player-level / squad-strength features
- Experiment with ensemble models or calibrated probabilities

## License

MIT