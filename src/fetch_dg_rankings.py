"""
Fetch DataGolf Rankings — current + historical snapshots since 1983.
Data is embedded server-side in the HTML page, or fetched via API for
historical dates (YYYYMMDD format).

Usage:
  python src/fetch_dg_rankings.py                     # current snapshot
  python src/fetch_dg_rankings.py --date 20260105      # single historical date
  python src/fetch_dg_rankings.py --backfill           # all available dates
  python src/fetch_dg_rankings.py --backfill --max-dates 10  # test run
"""
import re
import json
import time
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "dg_rankings"
BASE_URL = "https://datagolf.com/datagolf-rankings"
API_URL = "https://datagolf.com/get-pro-rankings"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/json",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://datagolf.com/datagolf-rankings",
}


def _extract_reload_data(html):
    m = re.search(r"reload_data = JSON\.parse\('(.+?)'\)", html, re.DOTALL)
    if not m:
        raise ValueError("Could not find reload_data in page HTML")
    raw = m.group(1).replace("\\'", "'").replace('\\"', '"').replace('\\/', '/')
    return json.loads(raw)


def _parse_players(player_list):
    rows = []
    for p in player_list:
        row = {
            "dg_id": p.get("dg_id"),
            "dg_rank": p.get("dg_rank"),
            "dg_rank_change": p.get("dg_rank_change"),
            "dg_skill": p.get("dg_skill"),
            "dgp_rank": p.get("dgp_rank"),
            "dgp_rank_change": p.get("dgp_rank_change"),
            "first_name": p.get("first"),
            "last_name": p.get("last"),
            "flag": p.get("flag"),
            "tour": p.get("tour"),
            "sample_rounds": p.get("sample"),
            "days_since_last_round": p.get("days_since"),
            "is_amateur": bool(p.get("am", 0)),
        }
        for t in p.get("tours", []):
            row[f"tour_weight_{t['tour']}"] = t.get("weight")
        rows.append(row)
    return pd.DataFrame(rows)


def _fetch_available_dates():
    """Extract the exact list of available snapshot dates from the page."""
    resp = requests.get(BASE_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    dates = []
    for line in resp.text.split("\n"):
        if "date-option" in line and 'value=' in line:
            m = re.search(r'value="(\d+)"', line)
            if m:
                dates.append(m.group(1))
    return dates


def fetch_current():
    session = requests.Session()
    resp = session.get(BASE_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = _extract_reload_data(resp.text)
    df = _parse_players(
        data.get("data", {}).get("table_data", {}).get("data", [])
    )
    df["snapshot_date"] = data.get("current_date", "")
    return df


def fetch_snapshot(date_str):
    session = requests.Session()
    session.get(BASE_URL, headers=HEADERS, timeout=30)

    if re.match(r"^\d{8}$", date_str):
        payload = date_str
    else:
        payload = date_str

    resp = session.put(API_URL, data=json.dumps(payload), headers=HEADERS, timeout=30)
    if resp.status_code == 500:
        return pd.DataFrame()

    resp.raise_for_status()
    result = resp.json()
    df = _parse_players(result.get("data", []))
    df["snapshot_date"] = result.get("info", {}).get("current_date", date_str)
    return df


def backfill(max_dates=None, delay=0.7):
    """Fetch all available historical snapshots with checkpoint/resume.

    Each snapshot is saved as ``data/raw/dg_rankings/dg_rankings_YYYYMMDD.csv``.
    Already-fetched dates are skipped.  Every 50 dates the combined
    ``dg_rankings_all.csv`` is rebuilt.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dates = _fetch_available_dates()
    if max_dates:
        dates = dates[:max_dates]

    existing = {p.stem.replace("dg_rankings_", "") for p in DATA_DIR.glob("dg_rankings_2*.csv")}
    todo = [d for d in dates if d not in existing]

    print(f"Available: {len(dates)} snapshots")
    print(f"Already fetched: {len(dates) - len(todo)}")
    print(f"Remaining: {len(todo)}")

    if not todo:
        print("Nothing to fetch.")
        return _rebuild_combined()

    session = requests.Session()
    session.get(BASE_URL, headers=HEADERS, timeout=30)

    for i, d in enumerate(todo):
        print(f"  [{i+1}/{len(todo)}] {d}...", end=" ", flush=True)
        resp = session.put(
            API_URL, data=json.dumps(d), headers=HEADERS, timeout=30,
        )
        if resp.status_code != 200:
            print("no data")
            continue
        result = resp.json()
        df = _parse_players(result.get("data", []))
        df["snapshot_date"] = result.get("info", {}).get("current_date", d)
        df.to_csv(DATA_DIR / f"dg_rankings_{d}.csv", index=False)
        print(f"{len(df)} players")

        if (i + 1) % 50 == 0:
            _rebuild_combined()

        time.sleep(delay)

    _rebuild_combined()


def _rebuild_combined():
    """Rebuild the combined CSV from individual snapshot files."""
    files = sorted(DATA_DIR.glob("dg_rankings_2*.csv"))
    if not files:
        return pd.DataFrame()

    chunks = []
    for f in files:
        try:
            chunks.append(pd.read_csv(f))
        except Exception:
            continue

    if not chunks:
        return pd.DataFrame()

    combined = pd.concat(chunks, ignore_index=True)
    out = DATA_DIR / "dg_rankings_all.csv"
    combined.to_csv(out, index=False)
    print(f"  -> Combined: {len(combined)} rows, {combined['snapshot_date'].nunique()} dates")
    return combined


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    import argparse
    parser = argparse.ArgumentParser(description="Fetch DataGolf Rankings")
    parser.add_argument("--date", type=str, default=None,
                        help="Fetch a single date (YYYYMMDD)")
    parser.add_argument("--backfill", action="store_true",
                        help="Fetch all available historical snapshots")
    parser.add_argument("--max-dates", type=int, default=None,
                        help="Limit backfill (for testing)")
    parser.add_argument("--delay", type=float, default=0.7,
                        help="Delay between API requests (seconds)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild combined CSV from existing files")
    args = parser.parse_args()

    if args.rebuild:
        _rebuild_combined()
    elif args.date:
        print(f"Fetching DG Rankings for {args.date}...")
        df = fetch_snapshot(args.date)
        if not df.empty:
            out = DATA_DIR / f"dg_rankings_{args.date}.csv"
            df.to_csv(out, index=False)
            print(f"  {len(df)} players -> {out}")
        else:
            print("  No data returned")
    elif args.backfill:
        backfill(max_dates=args.max_dates, delay=args.delay)
    else:
        print("Fetching current DG Rankings...")
        df = fetch_current()
        if not df.empty:
            out = DATA_DIR / "dg_rankings_current.csv"
            df.to_csv(out, index=False)
            print(f"  {len(df)} players -> {out}")


if __name__ == "__main__":
    main()
