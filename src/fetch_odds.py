"""
Fetch current golf odds from The Odds API.

Supports: FanDuel, DraftKings, BetMGM, and other US books.
Free tier: 500 requests/month at https://the-odds-api.com

Usage:
  python fetch_odds.py                          # auto-detect golf event
  python fetch_odds.py --event "U.S. Open"      # specific event
  python fetch_odds.py --save                   # save to CSV for bankroll module
"""
import os
import sys
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.the-odds-api.com/v4/sports"

# The Odds API sport keys for golf
GOLF_SPORTS = [
    "golf_pga_championship",
    "golf_masters",
    "golf_us_open",
    "golf_the_open",
    "golf_presidents_cup",
    "golf_ryder_cup",
    "golf_pga_tour",
    "golf_pga_tour_championship",
]

# US books we care about
US_BOOKS = ["fanduel", "draftkings", "betmgm", "betrivers", "williamhill_us", "unibet"]


def get_api_key():
    """Get API key from environment or .env file."""
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()

    return None


def list_golf_sports(api_key):
    """List available golf sports/events."""
    resp = requests.get(f"{BASE_URL}/sports", params={"apiKey": api_key}, timeout=15)
    resp.raise_for_status()
    sports = resp.json()
    return [s for s in sports if "golf" in s["key"]]


def fetch_odds(api_key, sport_key, regions="us", markets="outrights", bookmakers=None):
    """Fetch odds for a specific sport/event.
    
    Args:
        api_key: The Odds API key
        sport_key: e.g., "golf_pga_championship"
        regions: "us" for US books
        markets: "outrights" for winner odds
        bookmakers: list of bookmaker keys, or None for all
    
    Returns:
        list of event dicts with odds
    """
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)

    resp = requests.get(f"{BASE_URL}/{sport_key}/odds", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def parse_odds_to_dataframe(events, min_books=1):
    """Parse API response into a clean DataFrame.
    
    Returns DataFrame with columns:
        player_name, bookmaker, odds_american, event_name, commence_time
    """
    rows = []
    for event in events:
        event_name = event.get("title", event.get("sport_title", ""))
        commence = event.get("commence_time", "")

        for book in event.get("bookmakers", []):
            book_key = book.get("key", "")
            book_name = book.get("title", book_key)

            for market in book.get("markets", []):
                if market.get("key") != "outrights":
                    continue

                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price", 0)

                    if name and price:
                        rows.append({
                            "player_name": name,
                            "bookmaker": book_name,
                            "bookmaker_key": book_key,
                            "odds_american": price,
                            "event_name": event_name,
                            "commence_time": commence,
                        })

    return pd.DataFrame(rows)


def aggregate_odds(df, method="best"):
    """Aggregate odds across bookmakers.
    
    Args:
        df: Raw odds DataFrame
        method: "best" takes best odds, "avg" averages them
    
    Returns DataFrame with columns:
        player_name, win_odds (best American odds)
    """
    if df.empty:
        return df

    # Convert American odds to decimal for comparison
    def american_to_decimal(american):
        if american > 0:
            return 1 + american / 100
        else:
            return 1 + 100 / abs(american)

    df = df.copy()
    df["decimal_odds"] = df["odds_american"].apply(american_to_decimal)

    if method == "best":
        # Best odds = highest decimal odds = best payout
        agg = df.groupby("player_name").agg(
            win_odds=("odds_american", "first"),  # placeholder
            decimal_best=("decimal_odds", "max"),
            n_books=("bookmaker", "nunique"),
        ).reset_index()

        # Get the American odds that correspond to the best decimal
        best_rows = []
        for _, row in agg.iterrows():
            mask = (df["player_name"] == row["player_name"]) & (df["decimal_odds"] == row["decimal_best"])
            best = df[mask].iloc[0]
            best_rows.append({
                "player_name": row["player_name"],
                "win_odds": best["odds_american"],
                "n_books": row["n_books"],
            })

        return pd.DataFrame(best_rows)

    elif method == "avg":
        agg = df.groupby("player_name").agg(
            win_odds=("odds_american", "mean"),
            n_books=("bookmaker", "nunique"),
        ).reset_index()
        agg["win_odds"] = agg["win_odds"].round(0).astype(int)
        return agg

    return df


def save_odds(df, event_name=""):
    """Save odds to CSV in format compatible with bankroll module."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    name = event_name.replace(" ", "_").lower() if event_name else "current"
    out_path = DATA_DIR / f"odds_{date_str}_{name}.csv"

    # Rename for bankroll module compatibility
    save_df = df[["player_name", "win_odds"]].copy()
    save_df.to_csv(out_path, index=False)
    print(f"Saved {len(save_df)} players to {out_path}")
    return out_path


def main(event_filter=None, save=False):
    api_key = get_api_key()
    if not api_key:
        print("ERROR: No API key found.")
        print("Get a free key at https://the-odds-api.com and set it:")
        print("  set ODDS_API_KEY=your_key")
        print("  or add ODDS_API_KEY=your_key to .env")
        return

    print("Listing available golf events...")
    sports = list_golf_sports(api_key)

    if not sports:
        print("No golf events found. Check your API key or sport availability.")
        return

    for s in sports:
        print(f"  {s['key']}: {s['title']} (active={s.get('active', False)})")

    # Fetch odds for each active golf event
    all_odds = []
    for sport in sports:
        if not sport.get("active", False):
            continue

        if event_filter and event_filter.lower() not in sport["title"].lower():
            continue

        print(f"\nFetching odds for {sport['title']}...")
        events = fetch_odds(api_key, sport["key"])

        if not events:
            print(f"  No events found for {sport['title']}")
            continue

        for event in events:
            event_name = event.get("title", "")
            print(f"  {event_name}: {len(event.get('bookmakers', []))} books")

            df = parse_odds_to_dataframe([event])
            all_odds.append(df)

    if not all_odds:
        print("\nNo odds fetched.")
        return

    combined = pd.concat(all_odds, ignore_index=True)
    print(f"\nTotal: {len(combined)} odds across {combined['bookmaker'].nunique()} books")

    # Aggregate best odds per player
    best = aggregate_odds(combined, method="best")
    best = best.sort_values("win_odds", ascending=False)

    print(f"\n{'Player':<25} {'Best Odds':>10} {'# Books':>8}")
    print("-" * 45)
    for _, row in best.head(30).iterrows():
        odds = row["win_odds"]
        sign = "+" if odds > 0 else ""
        print(f"{row['player_name']:<25} {sign}{odds:<9} {int(row['n_books']):>8}")

    if save:
        event_name = all_odds[0]["event_name"].iloc[0] if len(all_odds) == 1 else ""
        save_odds(best, event_name)

    return best


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=str, default=None, help="Filter by event name")
    parser.add_argument("--save", action="store_true", help="Save odds to CSV")
    args = parser.parse_args()
    main(args.event, args.save)
