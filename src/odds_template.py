"""
Generate a blank odds CSV template for the current tournament field.

Usage:
  python odds_template.py                    # auto-detect tournament, generate template
  python odds_template.py --event-id 401580333  # specific event
"""
import sys
import requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/golf/pga"


def get_field(event_id=None):
    """Get the current tournament field from ESPN."""
    url = f"{ESPN_BASE}/scoreboard"
    if event_id:
        url += f"?event={event_id}"

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    events = data.get("events", [])
    if not events:
        return None, []

    ev = events[0]
    event_name = ev.get("name", "Unknown Tournament")
    event_date = ev.get("date", "")[:10]

    players = []
    for comp in ev.get("competitions", []):
        for c in comp.get("competitors", []):
            athlete = c.get("athlete", {})
            players.append({
                "player_name": athlete.get("displayName", ""),
                "player_id": athlete.get("id", ""),
            })

    return {"event_name": event_name, "date": event_date, "event_id": ev.get("id")}, players


def generate_template(event_info, players):
    """Generate a blank odds CSV template."""
    df = pd.DataFrame(players)
    df["win_odds"] = ""  # User fills in American odds (e.g., +340, -110)

    date_str = event_info["date"]
    name = event_info["event_name"].replace(" ", "_").lower()
    out_path = DATA_DIR / f"odds_{date_str}_{name}_template.csv"
    df.to_csv(out_path, index=False)

    return out_path, df


def main(event_id=None):
    print("Fetching tournament field...")
    event_info, players = get_field(event_id)

    if not event_info:
        print("ERROR: No active tournament found.")
        return

    print(f"  {event_info['event_name']} ({event_info['date']})")
    print(f"  {len(players)} players")

    out_path, df = generate_template(event_info, players)

    print(f"\nTemplate saved to: {out_path}")
    print(f"\nEdit the CSV and add 'win_odds' (American odds, e.g. +340):")
    print(f"\n{'Player':<25} {'Odds':>8}")
    print("-" * 35)
    for row in players[:15]:
        print(f"{row['player_name']:<25} {'':>8}")
    if len(players) > 15:
        print(f"... and {len(players) - 15} more")

    print(f"\nThen run:")
    print(f"  python predict_tournament.py --odds-file {out_path.name} --bankroll 1000")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", type=str, default=None)
    args = parser.parse_args()
    main(args.event_id)
