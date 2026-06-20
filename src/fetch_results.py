"""
Fetch historical PGA Tour tournament results (full leaderboards) from ESPN API.
Each tournament gets one row per player with position, scores, and player info.
"""
import time
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/golf/pga"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]


def polite_get(url, params=None):
    headers = {"User-Agent": USER_AGENTS[0]}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp


def parse_leaderboard(data):
    rows = []
    for event in data.get("events", []):
        event_id = event.get("id")
        event_name = event.get("name")
        event_date = event.get("date")

        for comp in event.get("competitions", []):
            for c in comp.get("competitors", []):
                athlete = c.get("athlete", {})

                # Position: use 'order' field (1 = winner)
                order = c.get("order")

                # Round scores: linescores with 'period' are round-by-round
                linescores = c.get("linescores", [])
                round_scores = []
                for ls in linescores:
                    if ls.get("period"):
                        round_scores.append(ls.get("displayValue", ""))

                num_rounds = len(round_scores)

                # Total score: 'score' is a string like "-15" or "E"
                total_to_par = str(c.get("score", ""))

                # Detect CUT: players with < 4 rounds played
                # The Masters has a cut after round 2; regular events too
                is_cut = num_rounds < 4
                made_cut = not is_cut

                # Position display
                if is_cut:
                    pos_display = "CUT"
                    pos_rank = None
                else:
                    pos_display = str(order) if order else ""
                    pos_rank = order

                rows.append({
                    "event_id": event_id,
                    "event_name": event_name,
                    "event_date": event_date,
                    "player_id": c.get("id"),
                    "player_name": athlete.get("displayName"),
                    "country": athlete.get("flag", {}).get("alt", ""),
                    "position_display": pos_display,
                    "position_rank": pos_rank,
                    "total_to_par": total_to_par,
                    "total_strokes": None,
                    "round1": round_scores[0] if num_rounds > 0 else None,
                    "round2": round_scores[1] if num_rounds > 1 else None,
                    "round3": round_scores[2] if num_rounds > 2 else None,
                    "round4": round_scores[3] if num_rounds > 3 else None,
                    "made_cut": made_cut,
                })

    return pd.DataFrame(rows)


def fetch_tournament_results(event_date_str):
    """Fetch leaderboard for a tournament by its date (YYYY-MM-DD)."""
    dt = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
    start = dt.strftime("%Y%m%d")
    end = (dt + timedelta(days=7)).strftime("%Y%m%d")

    url = f"{ESPN_BASE}/scoreboard"
    resp = polite_get(url, params={"dates": f"{start}-{end}"})
    data = resp.json()

    if not data.get("events"):
        return pd.DataFrame()

    return parse_leaderboard(data)


def load_schedule(year):
    schedule_path = DATA_DIR / f"schedule_{year}.csv"
    if not schedule_path.exists():
        print(f"  Schedule not found for {year}. Run fetch_schedule.py first.")
        return pd.DataFrame()
    return pd.read_csv(schedule_path)


def fetch_season(year):
    """Fetch all tournament results for a given season."""
    schedule = load_schedule(year)
    if schedule.empty:
        return pd.DataFrame()

    all_results = []
    for _, row in schedule.iterrows():
        event_date = row["date"]
        event_name = row["event_name"]

        # Skip events with missing dates
        if pd.isna(event_date):
            print(f"  Skipping: {event_name} (no date)")
            continue

        print(f"  Fetching: {event_name} ({str(event_date)[:10]})...")

        try:
            df = fetch_tournament_results(event_date)
            if not df.empty:
                all_results.append(df)
        except Exception as e:
            print(f"    ERROR: {e}")

        time.sleep(1.5)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def main(start_year=2018, end_year=2025):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_seasons = []

    for year in range(start_year, end_year + 1):
        print(f"\n=== Season {year} ===")
        df = fetch_season(year)
        if not df.empty:
            out = DATA_DIR / f"results_{year}.csv"
            df.to_csv(out, index=False)
            all_seasons.append(df)
            n_tournaments = df["event_id"].nunique()
            n_players = len(df)
            print(f"  {n_tournaments} tournaments, {n_players} player-tournament rows")
        else:
            print(f"  No results found for {year}")

    if all_seasons:
        combined = pd.concat(all_seasons, ignore_index=True)
        combined.to_csv(DATA_DIR / "results_all.csv", index=False)
        print(f"\nTotal: {combined['event_id'].nunique()} tournaments, {len(combined)} rows")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()
    main(args.start, args.end)
