"""
Fetch PGA Tour tournament schedule from ESPN API.
Returns tournament IDs, dates, courses, and locations for each season.
"""
import time
import requests
import pandas as pd
from pathlib import Path

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


def fetch_schedule(year):
    url = f"{ESPN_BASE}/scoreboard"
    resp = polite_get(url, params={"dates": str(year)})
    data = resp.json()
    events = data.get("events", [])

    rows = []
    for ev in events:
        comp = ev.get("competitions", [{}])[0]
        loc = comp.get("venue", {})

        rows.append({
            "event_id": ev.get("id"),
            "event_name": ev.get("name"),
            "date": ev.get("date"),
            "year": year,
            "course_name": loc.get("fullName"),
            "city": loc.get("address", {}).get("city"),
            "state": loc.get("address", {}).get("state"),
            "country": loc.get("address", {}).get("country"),
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
            "par": comp.get("par"),
            "status": comp.get("status", {}).get("type", {}).get("state"),
        })

    return pd.DataFrame(rows)


def main(start_year=2018, end_year=2025):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    for year in range(start_year, end_year + 1):
        print(f"Fetching schedule for {year}...")
        df = fetch_schedule(year)
        out = DATA_DIR / f"schedule_{year}.csv"
        df.to_csv(out, index=False)
        all_dfs.append(df)
        print(f"  {len(df)} events")
        time.sleep(1.0)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(DATA_DIR / "schedule_all.csv", index=False)
    print(f"\nTotal: {len(combined)} events across {end_year - start_year + 1} seasons")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()
    main(args.start, args.end)
