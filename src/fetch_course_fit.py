"""
Fetch DataGolf Course Fit data — skill importance weights for each
PGA Tour course.  Data is embedded server-side in the course-fit tool page.

Output:
  course_fit_weights.csv   — skill importance weights per course
  course_fit_players.csv   — player-level adjustments (current-week courses only)
"""
import re
import json
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
COURSE_FIT_URL = "https://datagolf.com/course-fit-tool"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

SKILL_AXES = [
    "Driving Distance",
    "Driving Accuracy",
    "Approach",
    "Around Green",
    "Putting",
]


def _extract_reload_data(html):
    """Extract and parse the reload_data JSON from the page."""
    m = re.search(r"var reload_data = JSON\.parse\('(.+?)'\);", html, re.DOTALL)
    if not m:
        raise ValueError("Could not find reload_data in course-fit page HTML")
    raw = m.group(1).replace("\\'", "'").replace('\\"', '"').replace('\\/', '/')
    return json.loads(raw)


def fetch_course_weights():
    """Fetch course skill importance weights for all courses.

    Returns:
        DataFrame with columns:
          course_name, course_num, driving_distance_weight,
          driving_accuracy_weight, approach_weight, around_green_weight,
          putting_weight
    """
    resp = requests.get(COURSE_FIT_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    data = _extract_reload_data(resp.text)

    rows = []
    for course_name, course_info in data.items():
        if course_name in ("players", "Avg PGA Tour Course", "Avg PGA Tour Course (Rel)"):
            continue
        if not isinstance(course_info, dict) or "coefs" not in course_info:
            continue

        course_num = course_info.get("course_num")
        coefs = {c["axis"]: c["value"] for c in course_info.get("coefs", [])}
        coefs_rel = {c["axis"]: c["value"] for c in course_info.get("coefs_rel", [])}

        row = {"course_name": course_name, "course_num": course_num}
        for axis in SKILL_AXES:
            safe_name = axis.lower().replace(" ", "_")
            row[f"{safe_name}_coef"] = coefs.get(axis)
            row[f"{safe_name}_weight"] = coefs_rel.get(axis)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def fetch_player_adjustments():
    """Fetch player-level course fit adjustments.

    These are only available for courses hosting a current-week event.
    Returns empty DataFrame if no current-week course.

    Returns:
        DataFrame with columns:
          course_name, dg_id, player_name, flag,
          driving_distance_adj, driving_accuracy_adj, approach_adj,
          around_green_adj, putting_adj, total_adj
    """
    resp = requests.get(COURSE_FIT_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    data = _extract_reload_data(resp.text)
    players = data.get("players", {})
    if not players:
        return pd.DataFrame()

    all_rows = []
    for course_name, player_list in players.items():
        if not isinstance(player_list, list):
            continue
        for p in player_list:
            all_rows.append({
                "course_name": course_name,
                "dg_id": p.get("dg_id"),
                "player_name": p.get("player_name"),
                "flag": p.get("flag"),
                "driving_distance_adj": p.get("dist"),
                "driving_distance_comp": p.get("distance_comp"),
                "driving_accuracy_adj": p.get("acc"),
                "driving_accuracy_comp": p.get("accuracy_comp"),
                "approach_adj": p.get("app"),
                "approach_comp": p.get("app_comp"),
                "around_green_adj": p.get("arg"),
                "around_green_comp": p.get("short_comp"),
                "putting_adj": p.get("putt"),
                "putting_comp": p.get("putt"),
                "total_adj": p.get("total_comp"),
            })

    return pd.DataFrame(all_rows)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    import argparse
    parser = argparse.ArgumentParser(description="Fetch DataGolf Course Fit data")
    parser.add_argument("--players", action="store_true",
                        help="Also fetch player-level adjustments (current-week courses only)")
    args = parser.parse_args()

    print("Fetching course fit weights...")
    weights = fetch_course_weights()
    if not weights.empty:
        out = DATA_DIR / "course_fit_weights.csv"
        weights.to_csv(out, index=False)
        print(f"  {len(weights)} courses -> {out}")

    if args.players or True:
        print("Fetching player-level adjustments...")
        players = fetch_player_adjustments()
        if not players.empty:
            out = DATA_DIR / "course_fit_players.csv"
            players.to_csv(out, index=False)
            print(f"  {len(players)} player-course rows -> {out}")
        else:
            print("  No player adjustments available (no current-week course)")


if __name__ == "__main__":
    main()
