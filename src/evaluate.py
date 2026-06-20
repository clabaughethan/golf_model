"""
Evaluate model predictions with detailed backtest metrics.

Computes:
  - AUC, log loss, calibration by target
  - Top-N accuracy: did the winner appear in our top N predicted?
  - Predicted probability distribution for actual winners
  - By-season breakdown
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

TARGETS = {
    "won": "win",
    "top5": "top5",
    "top10": "top10",
    "top20": "top20",
}


def top_n_accuracy(preds_df, target_col, pred_col="pred_prob", n=10):
    """For each tournament, check if the actual positive was in the top N predicted."""
    hits = 0
    total = 0

    for event_id, grp in preds_df.groupby("event_id"):
        positive_rows = grp[grp[target_col] == 1]
        if len(positive_rows) == 0:
            continue

        # Get top N predicted probabilities
        top_n_players = grp.nlargest(n, pred_col)["player_id"].values

        for _, row in positive_rows.iterrows():
            total += 1
            if row["player_id"] in top_n_players:
                hits += 1

    return hits, total


def calibration_analysis(preds_df, target_col, pred_col="pred_prob", n_bins=10):
    """Check how well predicted probabilities match actual outcomes."""
    valid = preds_df.dropna(subset=[pred_col, target_col]).copy()
    if len(valid) == 0:
        return pd.DataFrame()

    valid["prob_bin"] = pd.qcut(valid[pred_col], n_bins, duplicates="drop")
    cal = valid.groupby("prob_bin").agg(
        mean_pred=(pred_col, "mean"),
        mean_actual=(target_col, "mean"),
        count=(target_col, "count"),
    ).reset_index()

    return cal


def evaluate_target(preds_df, target_col, name):
    """Full evaluation for a single target."""
    print(f"\n{'='*60}")
    print(f"  {name.upper()} ({target_col})")
    print(f"{'='*60}")

    valid = preds_df.dropna(subset=[target_col, "pred_prob"]).copy()
    if len(valid) == 0:
        print("  No valid predictions.")
        return

    y = valid[target_col].values
    probs = valid["pred_prob"].values

    # Basic metrics
    auc = roc_auc_score(y, probs) if y.sum() > 0 and y.sum() < len(y) else np.nan
    ll = log_loss(y, np.column_stack([1 - probs, probs]))
    brier = brier_score_loss(y, probs)
    base_rate = y.mean()

    print(f"  Samples: {len(valid)}")
    print(f"  Positives: {y.sum()} ({base_rate:.4f})")
    print(f"  AUC: {auc:.4f}")
    print(f"  Log Loss: {ll:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Base Rate: {base_rate:.4f}")

    # Top-N accuracy
    for n in [5, 10, 20]:
        hits, total = top_n_accuracy(valid, target_col, n=n)
        if total > 0:
            print(f"  Top-{n} accuracy: {hits}/{total} ({hits/total:.1%})")

    # Winner prediction probability distribution
    if target_col == "won":
        winners = valid[valid[target_col] == 1]["pred_prob"]
        non_winners = valid[valid[target_col] == 0]["pred_prob"]
        if len(winners) > 0:
            print(f"\n  Winner predicted probs: "
                  f"mean={winners.mean():.4f}, median={winners.median():.4f}, "
                  f"max={winners.max():.4f}, min={winners.min():.4f}")

    # Calibration
    cal = calibration_analysis(valid, target_col)
    if not cal.empty:
        print(f"\n  Calibration:")
        for _, row in cal.iterrows():
            print(f"    Pred {row['mean_pred']:.3f} -> Actual {row['mean_actual']:.3f} (n={row['count']})")

    # By-season breakdown
    if "season" in valid.columns:
        print(f"\n  By Season:")
        for season, grp in valid.groupby("season"):
            sy = grp[target_col].values
            sp = grp["pred_prob"].values
            s_auc = roc_auc_score(sy, sp) if sy.sum() > 0 and sy.sum() < len(sy) else np.nan
            print(f"    {season}: AUC={s_auc:.4f}, n={len(grp)}, pos={int(sy.sum())}")

    return {
        "target": name,
        "auc": auc,
        "log_loss": ll,
        "brier": brier,
        "n_samples": len(valid),
        "n_positives": int(y.sum()),
        "base_rate": base_rate,
    }


def main():
    print("Loading walk-forward predictions...")
    summary = []

    for target_col, name in TARGETS.items():
        path = DATA_DIR / f"predictions_{name}.csv"
        if not path.exists():
            print(f"  Skipping {name} - no predictions file. Run train.py first.")
            continue

        df = pd.read_csv(path)
        result = evaluate_target(df, target_col, name)
        if result:
            summary.append(result)

    # Summary table
    if summary:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
