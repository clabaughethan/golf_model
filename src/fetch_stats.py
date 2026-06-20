"""
Fetch PGA Tour season-level player stats via the public GraphQL API.
Covers strokes gained (SG), driving, approach, short game, and scoring stats.
"""
import time
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

ENDPOINT = "https://orchestrator.pgatour.com/graphql"
X_API_KEY = "da2-gsrx5bibzbb4njvhl7t37wqyl4"

SG_STATS = {
    "sg_total": ("02675", "Avg"),
    "sg_t2g": ("02674", "Avg"),
    "sg_ott": ("02567", "Avg"),
    "sg_app": ("02568", "Avg"),
    "sg_arg": ("02569", "Avg"),
    "sg_putt": ("02564", "Avg"),
}

TRADITIONAL_STATS = {
    "driving_distance": ("101", "Avg"),
    "driving_accuracy": ("102", "%"),
    "gir_pct": ("103", "%"),
    "scoring_average": ("120", "Avg"),
    "scrambling": ("130", "%"),
    "birdie_average": ("156", "Avg"),
    "putts_per_round": ("119", "Avg"),
    "par3_scoring": ("142", "Avg"),
    "par4_scoring": ("143", "Avg"),
    "par5_scoring": ("144", "Avg"),
    "total_driving": ("129", "Total"),
    "birde_or_better_pct": ("104", "Birdie Conversion"),
    "three_putt_avoidance": ("145", "%"),
    "sand_save_pct": ("111", "%"),
}

STAT_DETAILS_QUERY = """
query StatDetails($tourCode: TourCode!, $statId: String!, $year: Int) {
    statDetails(tourCode: $tourCode, statId: $statId, year: $year) {
        statTitle
        year
        tourAvg
        rows {
            ... on StatDetailsPlayer {
                playerId
                playerName
                rank
                stats {
                    statName
                    statValue
                }
            }
        }
    }
}
"""


def pga_request(query, variables=None):
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    headers = {
        "Content-Type": "application/json",
        "x-api-key": X_API_KEY,
    }
    resp = requests.post(ENDPOINT, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise Exception(f"PGA API error: {data['errors'][0]['message']}")
    return data["data"]


def clean_stat_value(val):
    """Clean stat value strings like '72.02%' or '1,190' to float."""
    if not val or val == "":
        return None
    s = str(val).strip()
    s = s.replace("%", "").replace(",", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def fetch_stat(stat_id, year, api_stat_name="Avg"):
    """Fetch a single stat for a single year. Returns dict of player_id -> stats."""
    data = pga_request(STAT_DETAILS_QUERY, {
        "tourCode": "R",
        "statId": stat_id,
        "year": year,
    })

    details = data.get("statDetails", {})
    if not details or not details.get("rows"):
        return {}

    players = {}
    for row in details["rows"]:
        pid = row.get("playerId")
        if not pid:
            continue

        stats = {s["statName"]: s["statValue"] for s in row.get("stats", [])}
        players[pid] = {
            "player_id": pid,
            "player_name": row.get("playerName", ""),
            "rank": row.get("rank"),
            "avg": stats.get(api_stat_name, ""),
            "measured_rounds": stats.get("Measured Rounds", ""),
        }

    return players


def fetch_season_stats(year):
    """Fetch all tracked stats for a season. Returns DataFrame."""
    all_players = {}

    # SG stats
    for stat_name, (stat_id, api_stat_name) in SG_STATS.items():
        print(f"  {stat_name}...")
        players = fetch_stat(stat_id, year, api_stat_name)
        for pid, pdata in players.items():
            if pid not in all_players:
                all_players[pid] = {
                    "player_id": pid,
                    "player_name": pdata["player_name"],
                    "season": year,
                }
            all_players[pid][stat_name] = clean_stat_value(pdata["avg"])
            all_players[pid][f"{stat_name}_rank"] = pdata["rank"]
        time.sleep(0.6)

    # Traditional stats
    for stat_name, (stat_id, api_stat_name) in TRADITIONAL_STATS.items():
        print(f"  {stat_name}...")
        players = fetch_stat(stat_id, year, api_stat_name)
        for pid, pdata in players.items():
            if pid not in all_players:
                all_players[pid] = {
                    "player_id": pid,
                    "player_name": pdata["player_name"],
                    "season": year,
                }
            all_players[pid][stat_name] = clean_stat_value(pdata["avg"])
            all_players[pid][f"{stat_name}_rank"] = pdata["rank"]
        time.sleep(0.6)

    df = pd.DataFrame(list(all_players.values()))
    return df


def main(start_year=2018, end_year=2025):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    for year in range(start_year, end_year + 1):
        print(f"\n=== Season {year} ===")
        df = fetch_season_stats(year)
        if not df.empty:
            out = DATA_DIR / f"stats_{year}.csv"
            df.to_csv(out, index=False)
            all_dfs.append(df)
            print(f"  {len(df)} players")
        else:
            print(f"  No stats found for {year}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(DATA_DIR / "stats_all.csv", index=False)
        print(f"\nTotal: {len(combined)} player-season rows")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()
    main(args.start, args.end)
