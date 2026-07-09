"""
Script to email picks after model run.

Usage:
  python send_picks_email.py

Requires env vars (set once):
  GMAIL_USER, GMAIL_APP_PASSWORD, NOTIFY_TO (optional)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
from notify import send_picks

# Find latest picks file
picks_dir = Path("data/processed")
picks_file = sorted(picks_dir.glob("picks_*.md"), reverse=True)

if not picks_file:
    print("No picks file found in data/processed/")
    sys.exit(1)

latest = picks_file[0]
picks_text = latest.read_text()

# Parse tournament info from filename
parts = latest.stem.split("_")
date = parts[1]
tournament = "_".join(parts[2:]).replace("_", " ").title()

send_picks(picks_text, tournament, date)
