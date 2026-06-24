# Golf Betting Model

XGBoost model for predicting PGA Tour tournament outcomes (win, top-5, top-10, top-20) with walk-forward validation and Platt scaling calibration.

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

Walk-forward validation: train on seasons 1..N-1, test on season N. Includes Platt scaling calibration fitted on walk-forward predictions.

```
python src/train.py                # train on all players (25th pctl imputation)
python src/train.py --stats-only   # train only on players with PGA Tour stats
```

Models saved to `models/`. Calibration parameters (A, B) saved in `models/xgb_meta.json`.

## Predictions

```
python src/predict_tournament.py                              # next tournament
python src/predict_tournament.py --event-id 401580333         # specific event
python src/predict_tournament.py --bankroll 2000 --odds-file odds.csv  # with bet sizing
```

### Odds Input

Sportsbook scraping is blocked by anti-bot protections. The weekly workflow is manual copy-paste from your browser:

1. Generate a blank template from the current field:
   ```
   python src/odds_template.py
   ```
2. Copy-paste odds from FanDuel/DraftKings into the template CSV
3. Or parse raw text from a sportsbook page:
   ```
   python src/parse_pasted_odds.py
   ```

Odds CSV format:
```csv
player_name,win_odds,top5_odds,top10_odds,top20_odds
Scottie Scheffler,5.0,2.5,1.8,1.5
```

## Evaluation & Analysis

```
python src/evaluate.py          # backtest metrics by season
python src/importance.py        # feature importance across targets
python src/calibration.py       # calibration analysis (raw vs Platt-scaled)
```

## Project Structure

```
src/
  fetch_schedule.py       ESPN tournament schedule
  fetch_results.py        Historical leaderboards (ESPN)
  fetch_stats.py          PGA Tour stats (20 stats, GraphQL)
  fetch_owgr.py           World rankings
  fetch_odds.py           The Odds API (majors only)
  build_features.py       Feature engineering (37 features)
  train.py                XGBoost walk-forward + Platt scaling
  imputers.py             PercentileImputer (25th pctl)
  evaluate.py             Backtest metrics
  predict_tournament.py   Live predictions with bankroll
  importance.py           Feature importance analysis
  bankroll.py             Kelly criterion, fractional Kelly, bet sizing
  calibration.py          Calibration analysis (raw vs calibrated)
  odds_template.py        Generate blank odds CSV from ESPN field
  parse_pasted_odds.py    Parse copy-pasted sportsbook text
  weekly_picks.py         End-to-end weekly workflow
  course_mapping.py       Event-to-course name mapping
```

## Model Performance

Walk-forward AUC (2020-2025):

| Target | AUC | Log Loss | Platt Gap |
|--------|-----|----------|-----------|
| Win    | 0.946 | 0.030 | -0.0% |
| Top 5  | 0.974 | 0.070 | -0.0% |
| Top 10 | 0.964 | 0.121 | -0.0% |
| Top 20 | 0.946 | 0.206 | +0.0% |

Platt Gap = mean predicted probability minus actual rate after calibration. Near zero means well-calibrated.

## Features

- **Rolling form** (12-event window): made cut rate, avg finish, top-5/10/20 rates, events played
- **Course history**: appearances, avg finish, best finish, made cut rate at specific course
- **Strokes gained**: total, T2G, OTT, approach, ARG, putting (6 stats)
- **Traditional stats**: driving distance/accuracy, GIR%, scoring avg, scrambling, birdie avg, etc. (14 stats)
- **World rank proxy**: rolling power ranking across all events
- **Tournament context**: field size, cut rate, no-cut indicator
- **Data quality**: has_stats flag (25th pctl imputation for missing players)

## Notes

- Players without PGA Tour stats (Korn Ferry fill-ins, amateurs) are imputed at the 25th percentile (not median) — no-stats players win at 0.21% vs 0.98% for stats players
- Korn Ferry stats are not available via PGA Tour GraphQL
- Course history uses a hardcoded 376-event mapping (ESPN doesn't return course names)
- Odds are not training features — model predicts outcomes, odds are compared post-hoc for value detection via edge = model_prob - implied_prob
- The Odds API only covers majors; weekly odds require manual input
- Bankroll management: fractional Kelly (1/4), max 3% per bet, 10% total exposure cap, 5% minimum edge
