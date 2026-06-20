"""
Bankroll management for golf betting.

Implements:
  - Kelly Criterion bet sizing
  - Flat staking
  - Fractional Kelly (safer)
  - Bet recommendations based on model probabilities and market odds
"""
import pandas as pd
import numpy as np


def kelly_fraction(model_prob, odds_decimal):
    """Compute Kelly Criterion fraction for a single bet.
    
    Args:
        model_prob: Our estimated probability of the outcome (e.g., 0.05)
        odds_decimal: Decimal odds from the bookmaker (e.g., 20.0 for +1900)
    
    Returns:
        Kelly fraction of bankroll to bet (can be negative = don't bet)
    """
    if odds_decimal <= 1 or model_prob <= 0:
        return 0.0
    
    # Kelly formula: f* = (bp - q) / b
    # where b = odds - 1, p = model_prob, q = 1 - model_prob
    b = odds_decimal - 1
    p = model_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    return kelly


def fractional_kelly(model_prob, odds_decimal, fraction=0.25):
    """Fractional Kelly — bet a fraction of the full Kelly amount for safety.
    
    Standard is 1/4 Kelly for high-variance sports like golf.
    """
    full_kelly = kelly_fraction(model_prob, odds_decimal)
    return full_kelly * fraction


def flat_bet(bankroll, bet_size_pct=0.01):
    """Flat staking — bet a fixed percentage of bankroll per bet."""
    return bankroll * bet_size_pct


def implied_probability(odds_decimal):
    """Convert decimal odds to implied probability."""
    if odds_decimal <= 1:
        return 0.0
    return 1.0 / odds_decimal


def find_value(model_prob, odds_decimal, min_edge=0.02):
    """Find bets where our model disagrees with the market.
    
    Returns:
        edge: model_prob - implied_prob (positive = value bet)
        kelly: recommended Kelly fraction
        recommended: True if edge exceeds min_edge
    """
    imp_prob = implied_probability(odds_decimal)
    edge = model_prob - imp_prob
    kelly = kelly_fraction(model_prob, odds_decimal)
    
    return {
        "implied_prob": imp_prob,
        "model_prob": model_prob,
        "edge": edge,
        "kelly_full": kelly,
        "kelly_quarter": kelly * 0.25,
        "recommended": edge > min_edge and kelly > 0,
    }


def recommend_bets(predictions, odds_df, bankroll=1000, kelly_fraction_pct=0.25,
                   min_edge=0.02, max_bet_pct=0.05):
    """Generate betting recommendations for a tournament.
    
    Args:
        predictions: DataFrame with player_name, p_win, p_top5, p_top10, p_top20
        odds_df: DataFrame with player_name, win_odds, top5_odds, top10_odds, top20_odds
        bankroll: Total bankroll in dollars
        kelly_fraction_pct: Fraction of Kelly to use (0.25 = quarter Kelly)
        min_edge: Minimum edge to recommend a bet
        max_bet_pct: Maximum bet as % of bankroll (cap for safety)
    
    Returns:
        DataFrame with bet recommendations
    """
    # Merge predictions with odds
    merged = predictions.merge(odds_df, on="player_name", how="inner")
    
    if merged.empty:
        print("No matching players between predictions and odds.")
        return pd.DataFrame()
    
    bets = []
    targets = [
        ("p_win", "win_odds", "Win"),
        ("p_top5", "top5_odds", "Top 5"),
        ("p_top10", "top10_odds", "Top 10"),
        ("p_top20", "top20_odds", "Top 20"),
    ]
    
    for _, row in merged.iterrows():
        for prob_col, odds_col, bet_type in targets:
            if prob_col not in row or odds_col not in row:
                continue
            if pd.isna(row[prob_col]) or pd.isna(row[odds_col]):
                continue
            
            model_prob = row[prob_col]
            odds = row[odds_col]
            
            if odds <= 1:
                continue
            
            result = find_value(model_prob, odds, min_edge=min_edge)
            
            if result["recommended"]:
                kelly_bet = result["kelly_quarter"] * bankroll
                # Cap bet size
                max_bet = bankroll * max_bet_pct
                bet_amount = min(kelly_bet, max_bet)
                bet_amount = max(bet_amount, 0)
                
                bets.append({
                    "player": row["player_name"],
                    "bet_type": bet_type,
                    "model_prob": f"{model_prob:.1%}",
                    "odds": odds,
                    "implied_prob": f"{result['implied_prob']:.1%}",
                    "edge": f"{result['edge']:.1%}",
                    "kelly_full": f"{result['kelly_full']:.1%}",
                    "bet_amount": round(bet_amount, 2),
                    "potential_profit": round(bet_amount * (odds - 1), 2),
                })
    
    if not bets:
        print("No value bets found with current odds.")
        return pd.DataFrame()
    
    bets_df = pd.DataFrame(bets)
    bets_df = bets_df.sort_values("bet_amount", ascending=False)
    return bets_df


def print_bets(bets_df, bankroll):
    """Pretty-print betting recommendations."""
    if bets_df.empty:
        print("No bets to display.")
        return
    
    print(f"\n{'='*90}")
    print(f"  BETTING RECOMMENDATIONS  |  Bankroll: ${bankroll:,.0f}")
    print(f"{'='*90}")
    
    print(f"\n{'Player':<22} {'Type':<8} {'Model%':<8} {'Odds':<7} {'Edge':<8} {'Bet':<10} {'Profit':<10}")
    print("-" * 90)
    
    total_bet = 0
    for _, row in bets_df.iterrows():
        print(f"{row['player']:<22} {row['bet_type']:<8} {row['model_prob']:<8} "
              f"{row['odds']:<7.1f} {row['edge']:<8} ${row['bet_amount']:<9.2f} ${row['potential_profit']:<9.2f}")
        total_bet += row["bet_amount"]
    
    print("-" * 90)
    print(f"{'TOTAL BETS':<22} {'':8} {'':8} {'':7} {'':8} ${total_bet:<9.2f}")
    print(f"{'% of bankroll':<22} {total_bet/bankroll*100:.1f}%")
    print(f"{'Number of bets':<22} {len(bets_df)}")


def simulate_bets(historical_preds, historical_results, bankroll=1000, kelly_pct=0.25):
    """Simulate a Kelly staking strategy on historical data.
    
    Args:
        historical_preds: DataFrame with columns [event_id, player_name, p_win, odds]
        historical_results: DataFrame with columns [event_id, player_name, won]
        bankroll: Starting bankroll
        kelly_pct: Fraction of Kelly to use
    
    Returns:
        DataFrame with cumulative bankroll over time
    """
    merged = historical_preds.merge(
        historical_results[["event_id", "player_name", "won"]],
        on=["event_id", "player_name"],
        how="left",
    )
    merged["won"] = merged["won"].fillna(0)
    
    records = []
    current_bankroll = bankroll
    
    for event_id, event_bets in merged.groupby("event_id"):
        event_profit = 0
        
        for _, row in event_bets.iterrows():
            if pd.isna(row.get("p_win")) or pd.isna(row.get("odds")):
                continue
            
            kelly = fractional_kelly(row["p_win"], row["odds"], kelly_pct)
            if kelly <= 0:
                continue
            
            bet_amount = min(kelly * current_bankroll, current_bankroll * 0.05)
            
            if row["won"]:
                profit = bet_amount * (row["odds"] - 1)
            else:
                profit = -bet_amount
            
            event_profit += profit
        
        current_bankroll += event_profit
        records.append({
            "event_id": event_id,
            "profit": event_profit,
            "bankroll": current_bankroll,
        })
    
    return pd.DataFrame(records)


if __name__ == "__main__":
    # Demo with U.S. Open predictions
    print("Bankroll Management Module")
    print("=" * 40)
    print()
    print("Kelly Criterion Example:")
    print(f"  Model prob: 5.0%  |  Odds: 20.0  |  Kelly: {kelly_fraction(0.05, 20.0):.1%}")
    print(f"  Model prob: 5.0%  |  Odds: 15.0  |  Kelly: {kelly_fraction(0.05, 15.0):.1%}")
    print(f"  Model prob: 5.0%  |  Odds: 30.0  |  Kelly: {kelly_fraction(0.05, 30.0):.1%}")
    print()
    print("Value Bet Example:")
    result = find_value(0.05, 20.0, min_edge=0.02)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
