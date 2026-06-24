"""
Weekly picks workflow: fetch field, paste odds, predict, bet recommendations.

Usage:
  python weekly_picks.py                       # interactive
  python weekly_picks.py --bankroll 1000       # set bankroll
  python weekly_picks.py --odds-file odds.txt  # skip pasting, use file
"""
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/golf/pga"


def get_tournament():
    """Get current tournament info and field."""
    resp = requests.get(f"{ESPN_BASE}/scoreboard", timeout=15)
    resp.raise_for_status()
    data = resp.json()
    events = data.get("events", [])
    if not events:
        return None, []

    ev = events[0]
    info = {
        "event_id": ev.get("id"),
        "event_name": ev.get("name", "Unknown"),
        "date": ev.get("date", "")[:10],
    }

    players = []
    for comp in ev.get("competitions", []):
        for c in comp.get("competitors", []):
            athlete = c.get("athlete", {})
            players.append({
                "player_name": athlete.get("displayName", ""),
                "player_id": athlete.get("id", ""),
            })

    return info, players


def parse_odds_from_text(text):
    """Parse player name + odds from pasted text."""
    import re
    results = []
    seen = set()

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        odds_matches = list(re.finditer(r'[+−-](\d{2,5})', line.replace(",", "")))
        if not odds_matches:
            continue

        odds_str = odds_matches[0].group().replace("−", "-")
        try:
            odds = int(odds_str)
        except ValueError:
            continue
        if abs(odds) < 100:
            continue

        name_part = line[:odds_matches[0].start()].strip()
        name_part = re.sub(r'[\t|•·]+', ' ', name_part).strip()
        name_part = re.sub(r'\s+', ' ', name_part)
        name_part = re.sub(r'^[\d.\s]+', '', name_part).strip()

        if len(name_part) < 3:
            continue

        key = name_part.lower()
        if key not in seen:
            seen.add(key)
            results.append({"player_name": name_part, "win_odds": odds})

    return results


def match_odds_to_field(odds_df, field_df):
    """Match odds to field players by normalized name."""
    def normalize(name):
        return str(name).lower().strip()

    odds_df = odds_df.copy()
    field_df = field_df.copy()
    odds_df["name_norm"] = odds_df["player_name"].apply(normalize)
    field_df["name_norm"] = field_df["player_name"].apply(normalize)

    merged = field_df.merge(odds_df[["name_norm", "win_odds"]], on="name_norm", how="left")
    matched = merged["win_odds"].notna().sum()
    print(f"  Matched {matched}/{len(field_df)} players")

    return merged


def run_predictions(features_path, model_dir):
    """Load models and generate predictions."""
    import joblib

    from train import FEATURE_COLS

    models = {}
    for name in ["win", "top5", "top10", "top20"]:
        path = model_dir / f"xgb_{name}_production.joblib"
        if path.exists():
            models[name] = joblib.load(path)

    if not models:
        print("ERROR: No models found. Run train.py first.")
        return None

    df = pd.read_csv(features_path)
    X = df[FEATURE_COLS].values

    predictions = df[["player_id", "player_name"]].copy()
    for target_name, pipeline in models.items():
        try:
            probs = pipeline.predict_proba(X)[:, 1]
            predictions[f"p_{target_name}"] = probs
        except Exception as e:
            print(f"  Error predicting {target_name}: {e}")

    if "p_win" in predictions.columns:
        predictions = predictions.sort_values("p_win", ascending=False)

    return predictions


def generate_bets(predictions, odds_df, bankroll, min_edge=0.02):
    """Generate betting recommendations."""
    from bankroll import find_value

    merged = predictions.merge(odds_df, on="player_name", how="inner")
    merged = merged.dropna(subset=["win_odds"])

    bets = []
    for _, row in merged.iterrows():
        model_prob = row.get("p_win", 0)
        odds = row["win_odds"]
        if odds <= 1 or model_prob <= 0:
            continue

        result = find_value(model_prob, odds, min_edge=min_edge)
        if result["recommended"]:
            kelly_bet = result["kelly_quarter"] * bankroll
            max_bet = bankroll * 0.05
            bet_amount = min(kelly_bet, max_bet)
            bets.append({
                "player": row["player_name"],
                "model_prob": model_prob,
                "odds": odds,
                "implied_prob": result["implied_prob"],
                "edge": result["edge"],
                "bet_amount": round(bet_amount, 2),
                "potential_profit": round(bet_amount * (odds - 1), 2),
            })

    bets.sort(key=lambda x: x["bet_amount"], reverse=True)
    return bets


def print_predictions(predictions, odds_df, event_info):
    """Pretty-print predictions with odds."""
    targets = ["p_win", "p_top5", "p_top10", "p_top20"]

    merged = predictions.merge(odds_df, on="player_name", how="left")

    print(f"\n{'='*80}")
    print(f"  {event_info['event_name']}  |  {event_info['date']}")
    print(f"{'='*80}")

    print(f"\n{'Player':<22} {'Win%':>6} {'Top5%':>6} {'Top10%':>7} {'Top20%':>7} {'Odds':>7}")
    print("-" * 65)

    for _, row in merged.head(30).iterrows():
        odds_str = ""
        if pd.notna(row.get("win_odds")):
            o = int(row["win_odds"])
            odds_str = f"+{o}" if o > 0 else str(o)

        win = f"{row['p_win']:.1%}" if pd.notna(row.get("p_win")) else "N/A"
        top5 = f"{row['p_top5']:.1%}" if pd.notna(row.get("p_top5")) else "N/A"
        top10 = f"{row['p_top10']:.1%}" if pd.notna(row.get("p_top10")) else "N/A"
        top20 = f"{row['p_top20']:.1%}" if pd.notna(row.get("p_top20")) else "N/A"

        print(f"{row['player_name']:<22} {win:>6} {top5:>6} {top10:>7} {top20:>7} {odds_str:>7}")


def print_bets(bets, bankroll):
    """Pretty-print betting recommendations."""
    if not bets:
        print("\nNo value bets found with current odds.")
        return

    print(f"\n{'='*80}")
    print(f"  BETTING RECOMMENDATIONS  |  Bankroll: ${bankroll:,.0f}")
    print(f"{'='*80}")

    print(f"\n{'Player':<22} {'Model%':>7} {'Odds':>7} {'Edge':>7} {'Bet':>9} {'Profit':>9}")
    print("-" * 70)

    total_bet = 0
    for b in bets:
        sign = "+" if b["odds"] > 0 else ""
        print(f"{b['player']:<22} {b['model_prob']:>6.1%} {sign}{b['odds']:<6} {b['edge']:>6.1%} ${b['bet_amount']:>8.2f} ${b['potential_profit']:>8.2f}")
        total_bet += b["bet_amount"]

    print("-" * 70)
    print(f"{'TOTAL':<22} {'':>7} {'':>7} {'':>7} ${total_bet:>8.2f}")
    print(f"{'% of bankroll':<22} {total_bet/bankroll*100:.1f}%")
    print(f"{'Number of bets':<22} {len(bets)}")


def main(bankroll=1000, odds_file=None):
    print("=" * 50)
    print("  WEEKLY PICKS")
    print("=" * 50)

    # 1. Get tournament
    print("\nFetching tournament...")
    event_info, field = get_tournament()
    if not event_info:
        print("ERROR: No active tournament found.")
        return

    print(f"  {event_info['event_name']} ({event_info['date']})")
    print(f"  {len(field)} players in field")

    # 2. Get odds
    if odds_file:
        print(f"\nReading odds from {odds_file}...")
        text = Path(odds_file).read_text(encoding="utf-8")
        odds = parse_odds_from_text(text)
    else:
        print("\nPaste the odds table from your sportsbook, then press Enter twice:")
        print("(Ctrl+V to paste, Enter twice when done)\n")

        lines = []
        empty_count = 0
        while True:
            try:
                line = input()
                lines.append(line)
                empty_count = 0 if line.strip() else empty_count + 1
                if empty_count >= 2:
                    break
            except EOFError:
                break

        text = "\n".join(lines)
        odds = parse_odds_from_text(text)

    if not odds:
        print("ERROR: No odds found. Make sure you copied the odds table correctly.")
        return

    odds_df = pd.DataFrame(odds)
    print(f"\nParsed {len(odds_df)} players with odds")

    # 3. Match odds to field
    print("\nMatching odds to field...")
    matched = match_odds_to_field(odds_df, pd.DataFrame(field))

    # 4. Save features for prediction
    # We need to build features for the current field
    # Use predict_tournament.py's feature builder
    from predict_tournament import build_prediction_features, generate_picks, load_models

    print("\nLoading models...")
    models, meta = load_models()
    if not models:
        print("ERROR: No models found. Run train.py first.")
        return

    print("Loading stats and results...")
    stats_files = sorted(DATA_DIR.glob("stats_*.csv"), reverse=True)
    stats = pd.read_csv(stats_files[0]) if stats_files else pd.DataFrame()
    results = pd.read_csv(DATA_DIR / "results_all.csv")

    from build_features import compute_world_rank_proxy
    owgr_path = DATA_DIR / "owgr_proxy.csv"
    if owgr_path.exists():
        world_rank_all = pd.read_csv(owgr_path)
    else:
        world_rank_all = compute_world_rank_proxy(results)

    print("Building features...")
    features = build_prediction_features(pd.DataFrame(field), stats, results, world_rank_all)

    print("Generating predictions...")
    predictions = generate_picks(features, models, meta)

    # 5. Print predictions with odds
    print_predictions(predictions, odds_df, event_info)

    # 6. Generate bet recommendations
    bets = generate_bets(predictions, odds_df, bankroll)
    print_bets(bets, bankroll)

    # 7. Save everything
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    date_str = event_info["date"]
    name = event_info["event_name"].replace(" ", "_").lower()

    pred_path = PROC_DIR / f"picks_{date_str}_{name}.csv"
    predictions.to_csv(pred_path, index=False)

    if bets:
        bets_df = pd.DataFrame(bets)
        bet_path = PROC_DIR / f"bets_{date_str}_{name}.csv"
        bets_df.to_csv(bet_path, index=False)

    print(f"\nSaved predictions to {pred_path}")
    return predictions, bets


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bankroll", type=float, default=1000)
    parser.add_argument("--odds-file", type=str, default=None)
    args = parser.parse_args()
    main(args.bankroll, args.odds_file)
