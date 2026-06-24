"""
Parse odds from pasted sportsbook text into a CSV.

How to use:
  1. Go to FanDuel/DraftKings golf page in your browser
  2. Select the odds table (Ctrl+A or drag to select)
  3. Copy (Ctrl+C)
  4. Run this script and paste when prompted, OR save to a .txt file:

  python parse_pasted_odds.py                  # paste interactively
  python parse_pasted_odds.py --file odds.txt  # read from file
  python parse_pasted_odds.py --save           # save output CSV

Output format matches predict_tournament.py --odds-file
"""
import sys
import re
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_odds(text):
    """Extract player name + odds pairs from pasted sportsbook text.
    
    Sportsbooks typically show:
      Player Name    +340
      Player Name    -110    +200    ...
    
    Or in tables:
      Player Name | +340 | ...
    
    We look for lines containing American odds (+/- followed by digits)
    and extract the player name from the same line.
    """
    results = []
    seen = set()

    # Split into lines
    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Find all American odds in this line
        # Pattern: +/- followed by 2-4 digits, possibly with commas
        odds_matches = list(re.finditer(r'[+−-](\d{2,5})', line.replace(",", "")))

        if not odds_matches:
            continue

        # The first odds is usually the outright/win odds
        odds_str = odds_matches[0].group()
        odds_str = odds_str.replace("−", "-")
        try:
            odds = int(odds_str)
        except ValueError:
            continue

        # Skip short odds (likely props, not outrights)
        if abs(odds) < 100:
            continue

        # Extract player name: everything before the first odds
        name_part = line[:odds_matches[0].start()].strip()

        # Clean up common separators and prefixes
        name_part = re.sub(r'[\t|•·]+', ' ', name_part).strip()
        name_part = re.sub(r'\s+', ' ', name_part)

        # Remove leading/trailing junk
        name_part = re.sub(r'^[\d.\s]+', '', name_part).strip()  # leading numbers
        name_part = re.sub(r'[+−-]\d+.*$', '', name_part).strip()  # trailing odds
        name_part = re.sub(r'\s*\d+%?\s*$', '', name_part).strip()  # trailing percentages

        # Skip if name is too short or looks like a header
        if len(name_part) < 3:
            continue
        if name_part.lower() in ("player", "name", "golfer", "outright", "winner", "bet"):
            continue
        if re.match(r'^[\d\s]+$', name_part):
            continue

        # Deduplicate (keep first/best odds)
        name_key = name_part.lower().strip()
        if name_key not in seen:
            seen.add(name_key)
            results.append({
                "player_name": name_part,
                "win_odds": odds,
            })

    return results


def parse_from_file(filepath):
    """Parse odds from a text file."""
    text = Path(filepath).read_text(encoding="utf-8")
    return extract_odds(text)


def parse_from_stdin():
    """Parse odds from interactive paste."""
    print("Paste the odds table from your browser, then press Enter twice:")
    print("(Ctrl+V to paste, then Enter twice when done)")
    print()

    lines = []
    empty_count = 0
    while True:
        try:
            line = input()
            lines.append(line)
            empty_count = 0 if line.strip() else empty_count + 1
            if empty_count >= 2:
                break
        except EOFError:
            break

    text = "\n".join(lines)
    return extract_odds(text)


def main(filepath=None, save=False):
    if filepath:
        print(f"Reading from {filepath}...")
        results = parse_from_file(filepath)
    else:
        results = parse_from_stdin()

    if not results:
        print("\nNo odds found. Make sure you copied the odds table correctly.")
        print("Tip: On FanDuel/DraftKings, select the outright/winner market table.")
        return

    # Sort by odds (longest first)
    results.sort(key=lambda x: x["win_odds"], reverse=True)

    print(f"\nFound {len(results)} players:\n")
    print(f"{'Player':<25} {'Odds':>8}")
    print("-" * 35)
    for r in results:
        sign = "+" if r["win_odds"] > 0 else ""
        print(f"{r['player_name']:<25} {sign}{r['win_odds']:<7}")

    if save:
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_path = DATA_DIR / f"odds_{date_str}_parsed.csv"
        df = pd.DataFrame(results)
        df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")
        print(f"Use with: python predict_tournament.py --odds-file {out_path.name} --bankroll 1000")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="Read from text file instead of stdin")
    parser.add_argument("--save", action="store_true", help="Save parsed odds to CSV")
    args = parser.parse_args()
    main(args.file, args.save)
