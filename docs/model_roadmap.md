# Golf Model Roadmap

## Current State
4 separate binary classifiers: win, top5, top10, top20. Outputs are probabilities per market. Picks are filtered by x-factor (model_p / implied_p) per market.

## Goal
A single player quality rating that can be compared against any market's odds (win, top5, top10, top40, make cut, matchups) to find the best bet near even money.

---

## Phase 1: Retrain as Regression

Replace the 4 binary classifiers with a single XGBoost **regressor** predicting **normalized finish position** (`position_numeric / field_size`).

- Lower score = better finish
- One model, one rating per player per tournament
- Removes nested/redundant targets
- Simplifies feature engineering

## Phase 2: Rating → Market Probability Converter

Bucket historical finishes by model rating. For each rating bucket, compute empirical probability of:
- Winning
- Top 5
- Top 10
- Top 20
- Top 40
- Making the cut

This becomes a lookup table: *"players with this rating win X% of the time."*

For any new player, look up their rating bucket to get probabilities for all markets.

## Phase 3: Feed Expanded Odds

Extend odds input to include:
- Win
- Top 5
- Top 10
- Top 20
- Top 40
- Make Cut
- Matchups (future)

## Phase 4: Best Market Picker

For each player the model likes, scan all available markets:
1. Compute edge = model_p - implied_p for each market
2. Filter to markets with positive edge
3. Pick the market closest to +100 (even money)
4. Size bets using Kelly-proportional allocation

This naturally adapts to each player's profile:
- Favorites get bet on tighter markets (top 10, top 20)
- Long shots get bet on wider markets (top 40, make cut)
- Everyone gets a near-even-money bet if available

## Phase 5: Matchups

With a single rating per player, matchups become:
- `P(A beats B) = rating_z_A / (rating_z_A + rating_z_B)` (or logistic based on rating difference)
- Compare to matchup odds
- Same pipeline: find value, size by Kelly

## Key Benefits
- One model instead of four
- Naturally extends to any market (top 40, make cut, matchups)
- Player rating is interpretable and portable
- Picks are always near even money (user's style)
