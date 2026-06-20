"""
Feature importance analysis for golf prediction models.
Uses XGBoost's built-in feature importance (gain) across all 4 targets.
"""
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import FEATURE_COLS, MODEL_DIR


def get_importance():
    """Extract feature importance from each production model."""
    targets = ["win", "top5", "top10", "top20"]
    results = {}

    for target in targets:
        model_path = MODEL_DIR / f"xgb_{target}_production.joblib"
        if not model_path.exists():
            print(f"Missing {model_path}")
            continue

        pipeline = joblib.load(model_path)
        # Get the XGBClassifier from the pipeline
        xgb = pipeline.named_steps["model"]

        # Get feature importance (gain)
        importance = xgb.feature_importances_
        feat_imp = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        results[target] = feat_imp

    return results


def print_report(results):
    """Print feature importance report."""
    # Aggregate across all targets
    all_features = set()
    for imp in results.values():
        all_features.update(imp["feature"].tolist())

    # Compute average rank and importance
    summary = {}
    for feat in all_features:
        ranks = []
        importances = []
        for target, imp_df in results.items():
            rank = imp_df[imp_df["feature"] == feat].index[0] + 1 if feat in imp_df["feature"].values else len(FEATURE_COLS)
            val = imp_df[imp_df["feature"] == feat]["importance"].values[0] if feat in imp_df["feature"].values else 0
            ranks.append(rank)
            importances.append(val)
        summary[feat] = {
            "avg_rank": np.mean(ranks),
            "avg_importance": np.mean(importances),
            "max_importance": max(importances),
            "ranks_by_target": {t: r for t, r in zip(results.keys(), ranks)},
        }

    # Sort by average importance
    sorted_feats = sorted(summary.items(), key=lambda x: x[1]["avg_importance"], reverse=True)

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE REPORT")
    print("=" * 80)
    print(f"\n{'Feature':<30} {'Avg Imp':>10} {'Max Imp':>10} {'Avg Rank':>10}  Ranks by Target")
    print("-" * 100)

    for feat, info in sorted_feats:
        ranks_str = "  ".join(f"{t[:4]}={info['ranks_by_target'][t]}" for t in results.keys())
        print(f"{feat:<30} {info['avg_importance']:>10.4f} {info['max_importance']:>10.4f} {info['avg_rank']:>10.1f}  {ranks_str}")

    # Group by category
    print("\n" + "=" * 80)
    print("IMPORTANCE BY CATEGORY")
    print("=" * 80)
    categories = {
        "Strokes Gained": ["sg_total", "sg_t2g", "sg_ott", "sg_app", "sg_arg", "sg_putt"],
        "Traditional Stats": ["driving_distance", "driving_accuracy", "gir_pct", "scoring_average",
                              "scrambling", "birdie_average", "putts_per_round", "par3_scoring",
                              "par4_scoring", "par5_scoring", "total_driving", "birde_or_better_pct",
                              "three_putt_avoidance", "sand_save_pct"],
        "Rolling Form": ["made_cut_l", "avg_finish_l", "top5_l", "top10_l", "top20_l", "events_played_l"],
        "Course History": ["course_appearances", "course_avg_finish", "course_best_finish", "course_made_cut_rate"],
        "World Rank": ["world_rank_proxy"],
    }

    for cat_name, cat_feats in categories.items():
        cat_imp = sum(summary[f]["avg_importance"] for f in cat_feats if f in summary)
        total_imp = sum(info["avg_importance"] for _, info in sorted_feats)
        pct = cat_imp / total_imp * 100 if total_imp > 0 else 0
        print(f"\n  {cat_name:<20} {cat_imp:>10.4f} ({pct:>5.1f}% of total)")
        for f in cat_feats:
            if f in summary:
                print(f"    {f:<28} {summary[f]['avg_importance']:>10.4f}")


if __name__ == "__main__":
    results = get_importance()
    if results:
        print_report(results)
    else:
        print("No models found. Run train.py first.")
