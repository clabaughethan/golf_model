"""
Calibration analysis: are the model's probabilities accurate?

If the model says a player has a 30% chance of top-10, do those players
actually finish top-10 about 30% of the time?
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

TARGET_MAP = {
    "win": "won",
    "top5": "top5",
    "top10": "top10",
    "top20": "top20",
}


def platt_transform(p_raw, A, B):
    """Apply Platt scaling: sigmoid(A * logit(p) + B)."""
    eps = 1e-7
    p = np.clip(p_raw, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    z = A * logit_p + B
    return 1 / (1 + np.exp(-z))


def calibration_bucket(predicted, actual, bucket_edges):
    """Compute calibration by probability bucket."""
    results = []
    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        mask = (predicted >= lo) & (predicted < hi)
        if mask.sum() == 0:
            continue
        results.append({
            "bucket": f"{lo:.0%}-{hi:.0%}",
            "mean_predicted": predicted[mask].mean(),
            "actual_rate": actual[mask].mean(),
            "count": mask.sum(),
            "gap": predicted[mask].mean() - actual[mask].mean(),
        })
    return pd.DataFrame(results)


def print_calibration(df, target_name):
    """Pretty-print calibration table."""
    print(f"\n{'='*70}")
    print(f"  CALIBRATION: {target_name}")
    print(f"{'='*70}")
    print(f"{'Bucket':<12} {'Predicted':>10} {'Actual':>10} {'Gap':>8} {'N':>8}")
    print("-" * 55)
    for _, row in df.iterrows():
        gap_str = f"{row['gap']:+.1%}"
        print(f"{row['bucket']:<12} {row['mean_predicted']:>9.1%} {row['actual_rate']:>9.1%} {gap_str:>8} {int(row['count']):>8}")

    # Overall metrics
    total = df["count"].sum()
    avg_pred = (df["mean_predicted"] * df["count"]).sum() / total
    avg_actual = (df["actual_rate"] * df["count"]).sum() / total
    print("-" * 55)
    print(f"{'Overall':<12} {avg_pred:>9.1%} {avg_actual:>9.1%} {avg_pred - avg_actual:>+8.1%} {int(total):>8}")

    # Brier score decomposition
    print(f"\n  Overconfidence: model predicts {'HIGHER' if avg_pred > avg_actual else 'LOWER'} than reality")


def main():
    # Load calibration params
    meta_path = MODEL_DIR / "xgb_meta.json"
    cal_params = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        cal_params = meta.get("calibration", {})

    bucket_edges = np.arange(0, 1.05, 0.05)  # 5% buckets

    for target, actual_col in TARGET_MAP.items():
        filename = f"predictions_{target}.csv"
        path = DATA_DIR / filename
        if not path.exists():
            print(f"Missing {filename} — run train.py first")
            continue

        df = pd.read_csv(path)
        pred_col = "pred_prob"

        if pred_col not in df.columns or actual_col not in df.columns:
            print(f"Columns missing in {filename}: {pred_col}, {actual_col}")
            continue

        raw_probs = df[pred_col].values
        actual = df[actual_col].values

        # Raw calibration
        print(f"\n--- RAW (before Platt scaling) ---")
        cal_raw = calibration_bucket(raw_probs, actual, bucket_edges)
        print_calibration(cal_raw, target)

        # Calibrated calibration
        if target in cal_params:
            A = cal_params[target]["A"]
            B = cal_params[target]["B"]
            cal_probs = platt_transform(raw_probs, A, B)
            print(f"\n--- CALIBRATED (after Platt scaling) ---")
            cal_cal = calibration_bucket(cal_probs, actual, bucket_edges)
            print_calibration(cal_cal, target)

    # Also compare to market (use Travelers picks if available)
    picks_path = DATA_DIR / "picks_2026-06-25_travelers.csv"
    if picks_path.exists():
        print(f"\n{'='*70}")
        print(f"  MODEL vs MARKET (Travelers Championship)")
        print(f"{'='*70}")

        picks = pd.read_csv(picks_path)

        print(f"\n{'Player':<22} {'Model Win%':>10} {'Market%':>10} {'Gap':>8} {'Model Top10%':>13} {'Market Top10%':>14} {'Gap':>8}")
        print("-" * 95)

        for _, row in picks.sort_values("p_win", ascending=False).head(15).iterrows():
            model_win = row.get("p_win", 0)
            win_odds = row.get("win_odds")
            market_win = (1 / (1 + win_odds / 100)) if pd.notna(win_odds) and win_odds > 0 else (100 / (100 + abs(win_odds))) if pd.notna(win_odds) else None

            model_t10 = row.get("p_top10", 0)
            t10_odds = row.get("top10_odds")
            market_t10 = (1 / (1 + t10_odds / 100)) if pd.notna(t10_odds) and t10_odds > 0 else (100 / (100 + abs(t10_odds))) if pd.notna(t10_odds) else None

            mw = f"{model_win:.1%}" if pd.notna(model_win) else ""
            mk = f"{market_win:.1%}" if market_win else ""
            mg = f"{model_win - market_win:+.1%}" if market_win else ""
            tw = f"{model_t10:.1%}" if pd.notna(model_t10) else ""
            tk = f"{market_t10:.1%}" if market_t10 else ""
            tg = f"{model_t10 - market_t10:+.1%}" if market_t10 else ""

            print(f"{row['player_name']:<22} {mw:>10} {mk:>10} {mg:>8} {tw:>13} {tk:>14} {tg:>8}")


if __name__ == "__main__":
    main()
