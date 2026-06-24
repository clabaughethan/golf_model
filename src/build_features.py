"""
Build player × tournament feature matrix from raw data sources.

For each player in each tournament, assembles:
  - Season-level strokes gained and traditional stats
  - Rolling form (recent finishes, made cuts, top-10s)
  - Course history (past results at this specific course)
  - OWGR ranking
  - Field strength metrics
  - Labels: won, top5, top10, top20
"""
import re
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
from course_mapping import get_course_name
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Rolling window sizes
ROLLING_WINDOW = 12  # last N events


def normalize_name(name):
    """Normalize player name for matching across data sources."""
    if pd.isna(name):
        return ""
    name = unicodedata.normalize("NFKD", str(name))
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    return name


def load_raw(table):
    path = DATA_DIR / f"{table}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the corresponding fetch script first.")
    return pd.read_csv(path)


def load_all_results():
    dfs = []
    for p in sorted(DATA_DIR.glob("results_*.csv")):
        if p.name == "results_all.csv":
            continue
        year = int(p.stem.split("_")[1])
        df = pd.read_csv(p)
        df["season"] = year
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_all_stats():
    dfs = []
    for p in sorted(DATA_DIR.glob("stats_*.csv")):
        if p.name == "stats_all.csv":
            continue
        year = int(p.stem.split("_")[1])
        df = pd.read_csv(p)
        df["season"] = year
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def compute_rolling_features(results):
    """
    Compute rolling form features per player, ordered by event date.
    Must be called BEFORE joining to avoid look-ahead bias.
    """
    results = results.sort_values(["player_id", "event_date"]).copy()

    # Parse position into numeric (CUT = 100+, WD = 200)
    def pos_to_numeric(pos):
        if pd.isna(pos):
            return np.nan
        pos = str(pos).strip()
        if pos in ("CUT", "MC"):
            return 99.0
        if pos == "WD":
            return 200.0
        pos = pos.replace("T", "").replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
        try:
            return float(pos)
        except (ValueError, TypeError):
            return np.nan

    results["pos_numeric"] = results["position_display"].apply(pos_to_numeric)

    # Per-player rolling features
    roll_features = []
    for pid, grp in results.groupby("player_id"):
        grp = grp.sort_values("event_date")
        grp["made_cut_l"] = grp["made_cut"].rolling(ROLLING_WINDOW, min_periods=1).mean()
        grp["avg_finish_l"] = grp["pos_numeric"].rolling(ROLLING_WINDOW, min_periods=1).mean()
        grp["top5_l"] = (grp["pos_numeric"] <= 5).rolling(ROLLING_WINDOW, min_periods=1).mean()
        grp["top10_l"] = (grp["pos_numeric"] <= 10).rolling(ROLLING_WINDOW, min_periods=1).mean()
        grp["top20_l"] = (grp["pos_numeric"] <= 20).rolling(ROLLING_WINDOW, min_periods=1).mean()
        grp["events_played_l"] = range(1, len(grp) + 1)
        roll_features.append(grp[["player_id", "event_id", "season",
                                   "made_cut_l", "avg_finish_l", "top5_l", "top10_l", "top20_l",
                                   "events_played_l"]])

    return pd.concat(roll_features, ignore_index=True)


def compute_course_history(results):
    """Compute player's past results at each course (before the current event).
    
    Vectorized approach: group by player×course, use cumulative stats with shift.
    """
    results = results.sort_values("event_date").copy()

    def pos_to_numeric(pos):
        if pd.isna(pos):
            return np.nan
        pos = str(pos).strip()
        if pos in ("CUT", "MC"):
            return 99.0
        if pos == "WD":
            return 200.0
        pos = pos.replace("T", "").replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
        try:
            return float(pos)
        except (ValueError, TypeError):
            return np.nan

    results["pos_numeric"] = results["position_display"].apply(pos_to_numeric)

    # Drop rows without course_name
    has_course = results.dropna(subset=["course_name"]).copy()

    # Group by player × course, compute cumulative stats BEFORE current event
    grouped = has_course.groupby(["player_id", "course_name"], sort=False)

    # Count appearances (shifted so current event doesn't count itself)
    has_course["course_appearances"] = grouped.cumcount()  # 0-indexed, so this is "past appearances"

    # For avg/best/made_cut_rate, we need expanding stats shifted by 1
    has_course["course_sum_finish"] = grouped["pos_numeric"].cumsum().shift(1)
    has_course["course_count_valid"] = grouped["pos_numeric"].cumcount()  # number of past events with data
    has_course["course_best_min"] = grouped["pos_numeric"].cummin().shift(1)
    has_course["course_made_cut_sum"] = grouped["made_cut"].cumsum().shift(1)

    # Compute features
    has_course["course_avg_finish"] = has_course["course_sum_finish"] / has_course["course_count_valid"]
    has_course["course_best_finish"] = has_course["course_best_min"]
    has_course["course_made_cut_rate"] = has_course["course_made_cut_sum"] / has_course["course_count_valid"]

    # Clean up — first appearance at each course gets NaN for avg/best/rate
    has_course.loc[has_course["course_count_valid"] == 0, "course_avg_finish"] = np.nan
    has_course.loc[has_course["course_count_valid"] == 0, "course_best_finish"] = np.nan
    has_course.loc[has_course["course_count_valid"] == 0, "course_made_cut_rate"] = np.nan

    cols = ["player_id", "event_id", "course_appearances", "course_avg_finish",
            "course_best_finish", "course_made_cut_rate"]
    return has_course[cols]


def compute_world_rank_proxy(results):
    """Compute rolling power ranking as OWGR proxy.
    
    For each player at each event, rank them based on their rolling
    performance over the last 20 events across all tournaments.
    """
    POWER_WINDOW = 20

    def pos_to_numeric(pos):
        if pd.isna(pos):
            return np.nan
        pos = str(pos).strip()
        if pos in ("CUT", "MC"):
            return 99.0
        if pos == "WD":
            return 200.0
        pos = pos.replace("T", "").replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
        try:
            return float(pos)
        except (ValueError, TypeError):
            return np.nan

    results = results.sort_values("event_date").copy()
    results["pos_numeric"] = results["position_display"].apply(pos_to_numeric)

    player_events = []
    for pid, grp in results.groupby("player_id"):
        grp = grp.sort_values("event_date")
        grp["rolling_avg_finish"] = grp["pos_numeric"].rolling(POWER_WINDOW, min_periods=3).mean()
        grp["rolling_top10_rate"] = (grp["pos_numeric"] <= 10).rolling(POWER_WINDOW, min_periods=3).mean()
        grp["rolling_mc_rate"] = grp["made_cut"].rolling(POWER_WINDOW, min_periods=3).mean()
        player_events.append(grp[["player_id", "event_id", "rolling_avg_finish", "rolling_top10_rate", "rolling_mc_rate"]])

    power = pd.concat(player_events, ignore_index=True)
    power["power_score"] = -power["rolling_avg_finish"] + 10 * power["rolling_top10_rate"] + 5 * power["rolling_mc_rate"]

    event_ranks = power.dropna(subset=["power_score"]).copy()
    event_ranks["world_rank_proxy"] = event_ranks.groupby("event_id")["power_score"].rank(ascending=False, method="min")

    return event_ranks[["player_id", "event_id", "world_rank_proxy"]]


def build_field_strength(results):
    """Compute field strength as average OWGR rank of participants."""
    # Will be enhanced with actual OWGR data; for now use simple proxy
    field = results.groupby("event_id").agg(
        field_size=("player_id", "nunique"),
    ).reset_index()
    return field


def create_labels(results):
    """Create binary labels for win, top5, top10, top20."""
    def pos_to_numeric(pos):
        if pd.isna(pos):
            return np.nan
        pos = str(pos).strip()
        if pos in ("CUT", "MC", "WD"):
            return 200.0
        pos = pos.replace("T", "").replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
        try:
            return float(pos)
        except (ValueError, TypeError):
            return np.nan

    results["pos_numeric"] = results["position_display"].apply(pos_to_numeric)
    labels = results[["player_id", "event_id", "position_display", "pos_numeric", "made_cut"]].copy()
    labels["won"] = (labels["pos_numeric"] == 1).astype(int)
    labels["top5"] = (labels["pos_numeric"] <= 5).astype(int)
    labels["top10"] = (labels["pos_numeric"] <= 10).astype(int)
    labels["top20"] = (labels["pos_numeric"] <= 20).astype(int)
    return labels


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    results = load_all_results()
    stats = load_all_stats()

    if results.empty:
        raise SystemExit("No results data found. Run fetch_results.py first.")
    if stats.empty:
        raise SystemExit("No stats data found. Run fetch_stats.py first.")

    # Normalize player names for matching
    results["player_name_norm"] = results["player_name"].apply(normalize_name)
    stats["player_name_norm"] = stats["player_name"].apply(normalize_name)

    # Assign season from event_date
    results["season"] = pd.to_datetime(results["event_date"]).dt.year

    # Apply course name mapping
    results["course_name"] = results["event_name"].apply(get_course_name)
    n_mapped = results["course_name"].notna().sum()
    print(f"  Course names mapped: {n_mapped}/{len(results)} ({n_mapped/len(results)*100:.1f}%)")

    print(f"Results: {len(results)} rows, {results['event_id'].nunique()} tournaments")

    # Deduplicate (team events like Presidents Cup have multiple rows per player×event)
    before = len(results)
    results = results.drop_duplicates(subset=["player_id", "event_id"], keep="first")
    if len(results) < before:
        print(f"  Deduplicated: {before} -> {len(results)} rows ({before - len(results)} removed)")
    print(f"Stats: {len(stats)} player-season rows")

    # --- Rolling form features ---
    print("\nComputing rolling form features...")
    rolling = compute_rolling_features(results)

    # --- Course history ---
    print("Computing course history...")
    course_hist = compute_course_history(results)

    # --- World rank proxy ---
    print("Computing world rank proxy...")
    world_rank = compute_world_rank_proxy(results)

    # --- Labels ---
    print("Creating labels...")
    labels = create_labels(results)

    # --- Join stats to results ---
    # For each tournament, join the season's stats (use prior season for Jan-Feb events)
    stats_season = stats.drop(columns=["player_name", "player_name_norm", "player_id"], errors="ignore")

    # Merge on player_id and season
    base = results[["player_id", "event_id", "event_name", "event_date", "season",
                     "player_name", "player_name_norm", "country"]].copy()

    # --- Field size and cut format ---
    field_sizes = results.groupby("event_id").size().rename("field_size")
    cut_rates = results.groupby("event_id")["made_cut"].mean().rename("cut_rate")
    base = base.merge(field_sizes, on="event_id", how="left")
    base = base.merge(cut_rates, on="event_id", how="left")
    base["no_cut"] = (base["cut_rate"] > 0.8).astype(int)

    # Merge rolling features
    base = base.merge(rolling, on=["player_id", "event_id", "season"], how="left")

    # Merge course history
    base = base.merge(course_hist, on=["player_id", "event_id"], how="left")

    # Merge world rank proxy
    base = base.merge(world_rank, on=["player_id", "event_id"], how="left")

    # Merge season stats (on normalized name + season, not player_id)
    stats_for_merge = stats[["player_name_norm", "season"] + 
        [c for c in stats.columns if c not in ("player_id", "player_name", "player_name_norm", "season")]].copy()
    base = base.merge(stats_for_merge, on=["player_name_norm", "season"], how="left", suffixes=("", "_stat"))

    # Flag whether stats were found for this player (vs imputed)
    base["has_stats"] = base["sg_total"].notna().astype(int)

    # Merge labels
    base = base.merge(labels, on=["player_id", "event_id"], how="left")

    # --- Feature columns ---
    sg_cols = ["sg_total", "sg_t2g", "sg_ott", "sg_app", "sg_arg", "sg_putt"]
    trad_cols = [
        "driving_distance", "driving_accuracy", "gir_pct", "scoring_average", "scrambling",
        "birdie_average", "putts_per_round", "par3_scoring", "par4_scoring", "par5_scoring",
        "total_driving", "birde_or_better_pct",
        "three_putt_avoidance", "sand_save_pct",
    ]
    roll_cols = ["made_cut_l", "avg_finish_l", "top5_l", "top10_l", "top20_l", "events_played_l"]
    course_cols = ["course_appearances", "course_avg_finish", "course_best_finish", "course_made_cut_rate"]

    # Ensure numeric
    for col in sg_cols + trad_cols + roll_cols + course_cols:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    # --- Filter to events with at least some stats coverage ---
    stat_cols = sg_cols + trad_cols
    has_stats = base[stat_cols].notna().any(axis=1)
    print(f"\nRows with at least one stat: {has_stats.sum()} / {len(base)} ({has_stats.mean():.1%})")

    # --- Save ---
    base.to_csv(OUT_DIR / "features.csv", index=False)
    print(f"\nSaved features.csv: {len(base)} rows, {len(base.columns)} columns")
    print(f"  Tournaments: {base['event_id'].nunique()}")
    print(f"  Players: {base['player_id'].nunique()}")
    print(f"  Seasons: {sorted(base['season'].unique())}")

    # Feature coverage report
    print("\n--- Feature Coverage ---")
    for col in sg_cols + trad_cols + roll_cols + course_cols:
        if col in base.columns:
            pct = base[col].notna().mean()
            print(f"  {col}: {pct:.1%}")


if __name__ == "__main__":
    main()
