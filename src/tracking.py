"""
Track and score picks over time.

Saves each tournament's picks to a CSV, then scores them after results.
Running stats: ROI, hit rate, calibration by market.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PICKS_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "raw"


def american_to_decimal(odds):
    """Convert American odds to decimal odds."""
    if odds > 0:
        return 1 + odds / 100
    else:
        return 1 + 100 / abs(odds)


def save_picks(picks_df, tournament_name, event_id, date_str=None):
    """Save picks to CSV for later scoring."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    out = PICKS_DIR / f"picks_{date_str}_{tournament_name.replace(' ', '_').lower()}.csv"
    picks_df.to_csv(out, index=False)
    print(f"Saved {len(picks_df)} picks to {out}")
    return out


def score_picks(picks_path, results_df):
    """Score a saved picks file against actual results.
    
    Args:
        picks_path: path to picks CSV
        results_df: DataFrame with player_name, position_display (or pos_numeric), made_cut
    
    Returns:
        DataFrame with scored picks
    """
    picks = pd.read_csv(picks_path)

    def pos_to_numeric(pos):
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

    results_df = results_df.copy()
    results_df["pos_numeric"] = results_df["position_display"].apply(pos_to_numeric)

    # Merge picks with results
    scored = picks.merge(
        results_df[["player_name", "pos_numeric", "position_display"]],
        on="player_name", how="left"
    )

    # Determine if each pick won
    def check_win(row):
        market = row["market"]
        pos = row["pos_numeric"]
        if pd.isna(pos):
            return None  # no result yet
        if market == "Win":
            return pos == 1
        elif market == "Top 5":
            return pos <= 5
        elif market == "Top 10":
            return pos <= 10
        elif market == "Top 20":
            return pos <= 20
        return None

    scored["won"] = scored.apply(check_win, axis=1)

    # Use units column if present, otherwise bet_amount, otherwise 1 unit
    if "units" in scored.columns:
        scored["units"] = scored["units"].fillna(0)
    elif "bet_amount" in scored.columns:
        scored["units"] = scored["bet_amount"]
    else:
        scored["units"] = 1.0

    scored["profit_units"] = scored.apply(
        lambda r: r["units"] * (american_to_decimal(r["odds"]) - 1) if r["won"] else -r["units"]
        if pd.notna(r["won"]) else 0, axis=1
    )

    return scored


def print_scorecard(scored_df):
    """Print a scorecard for a tournament."""
    finished = scored_df[scored_df["won"].notna()].copy()
    if finished.empty:
        print("No finished picks yet.")
        return

    print(f"\n{'='*90}")
    print(f"  SCORECARD")
    print(f"{'='*90}")

    print(f"\n{'Player':<22} {'Market':<8} {'Model':>7} {'Odds':>7} {'Units':>6} {'Result':>8} {'P/L':>10}")
    print("-" * 75)

    for _, row in finished.iterrows():
        result = "WIN" if row["won"] else "LOSS"
        pnl = row["profit_units"]
        pnl_str = f"+{pnl:.2f}u" if pnl >= 0 else f"{pnl:.2f}u"
        print(f"{row['player_name']:<22} {row['market']:<8} {row['model_prob']:>6.1%} "
              f"{row['odds']:>7.0f} {row['units']:>5.2f} {result:>8} {pnl_str:>10}")

    total_pnl = finished["profit_units"].sum()
    n_wins = finished["won"].sum()
    n_bets = len(finished)
    total_staked = finished["units"].sum()

    print("-" * 75)
    pnl_str = f"+{total_pnl:.2f}u" if total_pnl >= 0 else f"{total_pnl:.2f}u"
    print(f"{'TOTAL':<22} {'':8} {'':7} {'':7} {total_staked:>5.2f} {n_wins:.0f}/{n_bets:.0f} {pnl_str:>10}")
    print(f"  Staked: {total_staked:.2f}u | ROI: {total_pnl/total_staked*100:+.1f}%")
    exp_rate = finished["model_prob"].apply(lambda x: float(str(x).rstrip('%')) / 100 if isinstance(x, str) else x).mean()
    print(f"  Win rate: {n_wins/n_bets:.1%} (expected: {exp_rate:.1%})")


def load_all_picks():
    """Load all historical picks for running stats."""
    dfs = []
    for p in sorted(PICKS_DIR.glob("picks_*.csv")):
        if "scored" in p.name:
            continue
        df = pd.read_csv(p)
        df["source_file"] = p.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def running_stats():
    """Compute running stats across all scored tournaments."""
    all_picks = load_all_picks()
    if all_picks.empty:
        print("No picks found.")
        return

    # Check if any have been scored (have 'won' column)
    if "won" not in all_picks.columns:
        print("No scored picks yet.")
        return

    finished = all_picks[all_picks["won"].notna()].copy()
    if finished.empty:
        print("No finished picks yet.")
        return

    print(f"\n{'='*70}")
    print(f"  RUNNING STATS")
    print(f"{'='*70}")

    # By market
    print(f"\n{'Market':<10} {'Bets':>6} {'Wins':>6} {'Win%':>7} {'Exp%':>7} {'P/L':>10} {'ROI':>8}")
    print("-" * 60)

    for market in ["Win", "Top 5", "Top 10", "Top 20"]:
        mkt = finished[finished["market"] == market]
        if mkt.empty:
            continue
        n = len(mkt)
        wins = mkt["won"].sum()
        win_rate = wins / n
        exp_rate = mkt["model_prob"].apply(lambda x: float(str(x).rstrip('%')) / 100).mean()
        pnl = mkt["profit"].sum()
        staked = mkt["bet_amount"].sum()
        roi = pnl / staked * 100 if staked > 0 else 0
        pnl_str = f"+${pnl:.0f}" if pnl >= 0 else f"-${abs(pnl):.0f}"
        print(f"{market:<10} {n:>6} {wins:>6.0f} {win_rate:>6.1%} {exp_rate:>6.1%} {pnl_str:>10} {roi:>+7.1f}%")

    # Overall
    total_n = len(finished)
    total_wins = finished["won"].sum()
    total_pnl = finished["profit"].sum()
    total_staked = finished["bet_amount"].sum()
    total_exp = finished["model_prob"].apply(lambda x: float(str(x).rstrip('%')) / 100).mean()
    total_roi = total_pnl / total_staked * 100 if total_staked > 0 else 0
    pnl_str = f"+${total_pnl:.0f}" if total_pnl >= 0 else f"-${abs(total_pnl):.0f}"
    print("-" * 60)
    print(f"{'TOTAL':<10} {total_n:>6} {total_wins:>6.0f} {total_wins/total_n:>6.1%} {total_exp:>6.1%} {pnl_str:>10} {total_roi:>+7.1f}%")


def cross_market_confidence(predictions, odds_df, min_markets=2, min_edge=0.02):
    """Score players by cross-market confidence.
    
    A player who the model likes in multiple markets is a more confident pick.
    
    Args:
        predictions: DataFrame with player_name, p_win, p_top5, p_top10, p_top20
        odds_df: DataFrame with player_name, win_odds, top5_odds, top10_odds, top20_odds
        min_markets: minimum markets with positive edge to qualify
        min_edge: minimum edge per market
    
    Returns:
        DataFrame with confidence scores
    """
    merged = predictions.merge(odds_df, on="player_name", how="inner")

    markets = [
        ("p_win", "win_odds", "Win"),
        ("p_top5", "top5_odds", "Top 5"),
        ("p_top10", "top10_odds", "Top 10"),
        ("p_top20", "top20_odds", "Top 20"),
    ]

    scores = []
    for _, row in merged.iterrows():
        edges = []
        market_bets = []
        for prob_col, odds_col, market_name in markets:
            if pd.isna(row.get(prob_col)) or pd.isna(row.get(odds_col)):
                continue
            if row[odds_col] <= 1:
                continue
            model_p = row[prob_col]
            implied = 100 / (row[odds_col] + 100)
            edge = model_p - implied
            if edge > min_edge:
                edges.append(edge)
                market_bets.append({
                    "market": market_name,
                    "model_prob": model_p,
                    "odds": row[odds_col],
                    "implied": implied,
                    "edge": edge,
                })

        if len(edges) >= min_markets:
            scores.append({
                "player_name": row["player_name"],
                "n_markets": len(edges),
                "total_edge": sum(edges),
                "avg_edge": np.mean(edges),
                "max_edge": max(edges),
                "markets": market_bets,
            })

    if not scores:
        return pd.DataFrame()

    df = pd.DataFrame(scores)
    # Rank by: number of markets first, then total edge
    df = df.sort_values(["n_markets", "total_edge"], ascending=[False, False])
    return df


def print_confidence_picks(conf_df, top_n=10):
    """Pretty-print cross-market confidence picks."""
    if conf_df.empty:
        print("No cross-market confidence picks found.")
        return

    print(f"\n{'='*90}")
    print(f"  CROSS-MARKET CONFIDENCE PICKS")
    print(f"{'='*90}")

    print(f"\n{'Player':<22} {'#Mkt':>5} {'Total Edge':>10} {'Avg Edge':>9} {'Markets':>30}")
    print("-" * 90)

    for _, row in conf_df.head(top_n).iterrows():
        mkt_str = " + ".join(
            f"{b['market']}({b['edge']:+.0%})" for b in row["markets"]
        )
        print(f"{row['player_name']:<22} {row['n_markets']:>5} {row['total_edge']:>9.1%} "
              f"{row['avg_edge']:>8.1%} {mkt_str:>30}")
