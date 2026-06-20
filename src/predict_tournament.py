"""
Generate predictions for an upcoming PGA Tour tournament.

Usage:
  python predict_tournament.py                  # auto-detect next tournament
  python predict_tournament.py --event-id 401580333  # specific ESPN event
"""
import json
import sys
import joblib
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/golf/pga"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

from build_features import (
    normalize_name, compute_rolling_features, compute_course_history,
    compute_world_rank_proxy,
)
from course_mapping import get_course_name
from train import FEATURE_COLS
from bankroll import recommend_bets, print_bets


def polite_get(url, params=None):
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, params=params, timeout=30)
    resp.raise_for_status()
    return resp


def load_models():
    models = {}
    for name in ["win", "top5", "top10", "top20"]:
        path = MODEL_DIR / f"xgb_{name}_production.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    meta_path = MODEL_DIR / "xgb_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return models, meta


def get_next_tournament():
    """Find the next upcoming or current tournament."""
    resp = polite_get(f"{ESPN_BASE}/scoreboard")
    data = resp.json()
    events = data.get("events", [])
    if not events:
        return None, None
    ev = events[0]
    comp = ev.get("competitions", [{}])[0]
    return {
        "event_id": ev.get("id"),
        "event_name": ev.get("name"),
        "date": ev.get("date"),
    }, comp.get("competitors", [])


def get_leaderboard(event_id):
    """Get current leaderboard / field for a tournament."""
    resp = polite_get(f"{ESPN_BASE}/scoreboard", params={"event": event_id})
    data = resp.json()
    if not data.get("events"):
        return pd.DataFrame()

    players = []
    for ev in data["events"]:
        for comp in ev.get("competitions", []):
            for c in comp.get("competitors", []):
                athlete = c.get("athlete", {})
                players.append({
                    "player_id": athlete.get("id"),
                    "player_name": athlete.get("displayName"),
                })
    return pd.DataFrame(players)


def build_prediction_features(leaderboard, stats, results, world_rank_all):
    """Build feature matrix for the current tournament's field."""
    # Deduplicate and normalize results
    results = results.drop_duplicates(subset=["player_id", "event_id"], keep="first")
    results["season"] = pd.to_datetime(results["event_date"]).dt.year
    results["player_name_norm"] = results["player_name"].apply(normalize_name)

    # Normalize names
    leaderboard["player_name_norm"] = leaderboard["player_name"].apply(normalize_name)
    stats["player_name_norm"] = stats["player_name"].apply(normalize_name)

    # Ensure player_id types match
    leaderboard["player_id"] = pd.to_numeric(leaderboard["player_id"], errors="coerce").fillna(0).astype(int)
    results["player_id"] = pd.to_numeric(results["player_id"], errors="coerce").fillna(0).astype(int)

    # Start with leaderboard
    features = leaderboard[["player_id", "player_name", "player_name_norm"]].copy()

    # --- Stats ---
    latest_season = stats["season"].max() if "season" in stats.columns else 2025
    latest_stats = stats[stats["season"] == latest_season].copy()
    features = features.merge(
        latest_stats.drop(columns=["player_id", "player_name"], errors="ignore"),
        on="player_name_norm", how="left", suffixes=("", "_stat"),
    )

    # Flag whether stats were found
    features["has_stats"] = features["sg_total"].notna().astype(int)

    # --- Rolling features ---
    rolling = compute_rolling_features(results)
    if not rolling.empty:
        # Add player_name_norm to rolling for name-based merge
        id_to_name = results[["player_id", "player_name_norm"]].drop_duplicates("player_id")
        rolling = rolling.merge(id_to_name, on="player_id", how="left")
        # Get latest rolling for each player
        latest_rolling = rolling.sort_values("event_id").groupby("player_name_norm").last().reset_index()
        roll_cols = ["player_name_norm", "made_cut_l", "avg_finish_l", "top5_l", "top10_l", "top20_l", "events_played_l"]
        features = features.merge(latest_rolling[roll_cols], on="player_name_norm", how="left")

    for col in ["made_cut_l", "avg_finish_l", "top5_l", "top10_l", "top20_l", "events_played_l"]:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    # --- Course history ---
    for col in ["course_appearances", "course_avg_finish", "course_best_finish", "course_made_cut_rate"]:
        features[col] = np.nan

    # --- World rank proxy ---
    POWER_WINDOW = 20

    def pos_to_num(pos):
        if pd.isna(pos):
            return np.nan
        pos = str(pos).strip()
        if pos in ("CUT", "MC"):
            return 99.0
        if pos == "WD":
            return 200.0
        pos = pos.replace("T", "").replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
        try:
            return float(pos)
        except (ValueError, TypeError):
            return np.nan

    wr_data = []
    for pname in features["player_name_norm"]:
        player_res = results[results["player_name_norm"] == pname].sort_values("event_date")
        if len(player_res) < 3:
            wr_data.append({"player_name_norm": pname, "world_rank_proxy": np.nan})
            continue
        player_res = player_res.copy()
        player_res["pos_num"] = player_res["position_display"].apply(pos_to_num)
        recent = player_res.tail(POWER_WINDOW)
        power_score = -recent["pos_num"].mean() + 10 * (recent["pos_num"] <= 10).mean() + 5 * recent["made_cut"].mean()
        wr_data.append({"player_name_norm": pname, "power_score": power_score})

    wr_df = pd.DataFrame(wr_data)
    wr_df["world_rank_proxy"] = wr_df["power_score"].rank(ascending=False, method="min")
    features = features.merge(wr_df[["player_name_norm", "world_rank_proxy"]], on="player_name_norm", how="left")

    return features


def generate_picks(features, models, meta):
    """Generate predictions for each target."""
    feature_cols = meta.get("feature_cols", FEATURE_COLS)
    for col in feature_cols:
        if col not in features.columns:
            features[col] = np.nan

    X = features[feature_cols].values
    predictions = features[["player_id", "player_name"]].copy()

    for target_name, pipeline in models.items():
        try:
            probs = pipeline.predict_proba(X)[:, 1]
            predictions[f"p_{target_name}"] = probs
        except Exception as e:
            print(f"  Error predicting {target_name}: {e}")
            predictions[f"p_{target_name}"] = np.nan

    if "p_win" in predictions.columns:
        predictions = predictions.sort_values("p_win", ascending=False)

    return predictions


def print_picks(predictions, tournament_info):
    """Pretty-print the predictions."""
    print(f"\n{'='*70}")
    print(f"  {tournament_info.get('event_name', 'Unknown Tournament')}")
    print(f"  {tournament_info.get('date', '')[:10]}")
    print(f"{'='*70}")

    targets = ["p_win", "p_top5", "p_top10", "p_top20"]
    display = ["Win%", "Top5%", "Top10%", "Top20%"]

    print(f"\n{'Player':<25}", end="")
    for d in display:
        print(f"{d:>8}", end="")
    print()
    print("-" * 60)

    for _, row in predictions.head(30).iterrows():
        print(f"{row['player_name']:<25}", end="")
        for t in targets:
            val = row.get(t, np.nan)
            if pd.notna(val):
                print(f"{val:>7.1%}", end=" ")
            else:
                print(f"{'N/A':>8}", end=" ")
        print()


def save_predictions(predictions, tournament_info):
    """Save predictions to CSV."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    date_str = tournament_info.get("date", "")[:10]
    name = tournament_info.get("event_name", "unknown").replace(" ", "_").lower()
    out = PROC_DIR / f"predictions_{date_str}_{name}.csv"
    predictions.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    return out


def main(event_id=None, bankroll=None, odds_file=None):
    print("Loading models...")
    models, meta = load_models()
    if not models:
        print("ERROR: No models found. Run train.py first.")
        return

    print("Fetching tournament info...")
    if event_id:
        info = {"event_id": event_id, "event_name": "Unknown", "date": ""}
    else:
        info, _ = get_next_tournament()
        if not info:
            print("ERROR: Could not find upcoming tournament.")
            return

    print(f"  {info['event_name']} ({info.get('event_id')})")

    print("Fetching leaderboard...")
    leaderboard = get_leaderboard(info["event_id"])
    if leaderboard.empty:
        print("ERROR: Could not fetch leaderboard.")
        return
    print(f"  {len(leaderboard)} players")

    print("Loading stats and results...")
    stats_files = sorted(RAW_DIR.glob("stats_*.csv"), reverse=True)
    stats = pd.read_csv(stats_files[0]) if stats_files else pd.DataFrame()
    results = pd.read_csv(RAW_DIR / "results_all.csv")

    # Load or compute world rank proxy
    owgr_path = RAW_DIR / "owgr_proxy.csv"
    if owgr_path.exists():
        world_rank_all = pd.read_csv(owgr_path)
    else:
        print("Computing world rank proxy...")
        world_rank_all = compute_world_rank_proxy(results)

    print("Building features...")
    features = build_prediction_features(leaderboard, stats, results, world_rank_all)

    print("Generating predictions...")
    predictions = generate_picks(features, models, meta)

    print_picks(predictions, info)
    save_predictions(predictions, info)

    # Bankroll management
    if odds_file:
        odds_df = pd.read_csv(odds_file)
        if bankroll is None:
            bankroll = 1000
        bets = recommend_bets(predictions, odds_df, bankroll=bankroll)
        print_bets(bets, bankroll)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", type=str, default=None)
    parser.add_argument("--bankroll", type=float, default=None, help="Bankroll in dollars for bet sizing")
    parser.add_argument("--odds-file", type=str, default=None, help="CSV with columns: player_name, win_odds, top5_odds, top10_odds, top20_odds")
    args = parser.parse_args()
    main(args.event_id, args.bankroll, args.odds_file)
