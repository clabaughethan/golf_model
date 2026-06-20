# Golf Betting Model

XGBoost model for predicting PGA Tour tournament outcomes (win, top-5, top-10, top-20) with walk-forward validation.

## Setup

```
pip install -r requirements.txt
```

## Data Pipeline

Fetch all data (ESPN + PGA Tour GraphQL, no API keys needed):

```
python src/fetch_schedule.py
python src/fetch_results.py
python src/fetch_stats.py
python src/build_features.py
```

Raw data goes to `data/raw/`, processed features to `data/processed/`.

## Training

Walk-forward validation: train on seasons 1..N-1, test on season N.

```
python src/train.py                # train on all players (25th pctl imputation)
python src/train.py --stats-only   # train only on players with PGA Tour stats
```

Models saved to `models/`.

## Predictions

```
python src/predict_tournament.py                              # next tournament
python src/predict_tournament.py --event-id 401580333         # specific event
python src/predict_tournament.py --bankroll 2000 --odds-file odds.csv  # with bet sizing
```

Odds CSV format:
```csv
player_name,win_odds,top5_odds,top10_odds,top20_odds
Scottie Scheffler,5.0,2.5,1.8,1.5
```

## Evaluation

```
python src/evaluate.py
python src/importance.py
```

## Project Structure

```
src/
  fetch_schedule.py    ESPN tournament schedule
  fetch_results.py     Historical leaderboards (ESPN)
  fetch_stats.py       PGA Tour stats (20 stats, GraphQL)
  fetch_owgr.py        World rankings
  build_features.py    Feature engineering
  train.py             XGBoost walk-forward training
  evaluate.py          Backtest metrics
  predict_tournament.py  Live predictions
  importance.py        Feature importance analysis
  bankroll.py          Kelly criterion, bet sizing
  course_mapping.py    Event-to-course name mapping
  imputers.py          Custom sklearn imputers
```

## Model Performance

Walk-forward AUC (2020-2025):

| Target | AUC | Log Loss |
|--------|-----|----------|
| Win    | 0.945 | 0.030 |
| Top 5  | 0.973 | 0.071 |
| Top 10 | 0.963 | 0.122 |
| Top 20 | 0.945 | 0.208 |

## Notes

- Players without PGA Tour stats (Korn Ferry fill-ins, amateurs) are imputed at the 25th percentile
- Course history uses a hardcoded 376-event mapping (ESPN doesn't return course names)
- No odds data yet — bankroll module requires manual odds input via CSV
