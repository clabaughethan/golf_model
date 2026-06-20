"""
Fetch OWGR (Official World Golf Ranking) data.
Supports current rankings via JSON API and historical via ESPN rankings.
"""
import time
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OWGR_API = "https://apiweb.owgr.com/api/owgr/rankings/getRankings"
ESPN_RANKINGS = "https://sports.core.api.espn.com/v2/sports/golf/leagues/pga/rankings"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]


def polite_get(url, params=None):
    headers = {"User-Agent": USER_AGENTS[0]}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp


def fetch_owgr_current(max_rank=500):
    """Fetch current OWGR rankings from the JSON API."""
    all_players = []
    page = 1
    page_size = min(max_rank, 100)

    while len(all_players) < max_rank:
        resp = polite_get(OWGR_API, params={"pageNo": page, "pageSize": page_size})
        data = resp.json()
        rankings = data.get("rankingsList", [])

        if not rankings:
            break

        for r in rankings:
            player = r.get("player", {})
            country = player.get("country", {})
            all_players.append({
                "owgr_rank": r.get("rank"),
                "player_name": player.get("fullName", ""),
                "player_first": player.get("firstName", ""),
                "player_last": player.get("lastName", ""),
                "country_code": country.get("code2", ""),
                "avg_points": r.get("pointsAverage"),
                "total_points": r.get("pointsTotal"),
                "divisor": r.get("divisorApplied"),
                "last_week_rank": r.get("lastWeekRank"),
                "end_last_year_rank": r.get("endLastYearRank"),
                "week_end_date": r.get("weekEndDate"),
            })

        total_pages = data.get("totalNumberOfPages", 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.5)

    return pd.DataFrame(all_players)


def fetch_espn_rankings():
    """Fetch ESPN's own player rankings as a fallback / additional feature."""
    resp = polite_get(ESPN_RANKINGS)
    data = resp.json()
    rankings = data.get("rankings", [])

    rows = []
    for r in rankings:
        athlete = r.get("athlete", {})
        rows.append({
            "espn_rank": r.get("current").get("rank") if r.get("current") else None,
            "player_id": athlete.get("id"),
            "player_name": athlete.get("displayName", ""),
            "espn_points": r.get("current").get("rankingsPoints", {}).get("rankingsPointsTotal") if r.get("current") else None,
        })

    return pd.DataFrame(rows)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching current OWGR rankings...")
    owgr = fetch_owgr_current(max_rank=500)
    if not owgr.empty:
        owgr.to_csv(DATA_DIR / "owgr_current.csv", index=False)
        print(f"  {len(owgr)} players")

    print("Fetching ESPN rankings...")
    espn = fetch_espn_rankings()
    if not espn.empty:
        espn.to_csv(DATA_DIR / "espn_rankings.csv", index=False)
        print(f"  {len(espn)} players")


if __name__ == "__main__":
    main()
