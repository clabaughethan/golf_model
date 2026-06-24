"""
Walk-forward training for golf tournament predictions.

Trains XGBoost models to predict P(win), P(top5), P(top10), P(top20).
Uses walk-forward validation: train on seasons 1..N-1, test on season N.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imputers import PercentileImputer

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

FEATURE_COLS = [
    # Strokes gained
    "sg_total", "sg_t2g", "sg_ott", "sg_app", "sg_arg", "sg_putt",
    # Traditional
    "driving_distance", "driving_accuracy", "gir_pct", "scoring_average", "scrambling",
    "birdie_average", "putts_per_round", "par3_scoring", "par4_scoring", "par5_scoring",
    "total_driving", "birde_or_better_pct", "three_putt_avoidance", "sand_save_pct",
    # Rolling form
    "made_cut_l", "avg_finish_l", "top5_l", "top10_l", "top20_l", "events_played_l",
    # Course history
    "course_appearances", "course_avg_finish", "course_best_finish", "course_made_cut_rate",
    # World rank proxy
    "world_rank_proxy",
    # Tournament context
    "field_size", "cut_rate", "no_cut",
    # Data quality
    "has_stats",
]

TARGETS = {
    "won": "win",
    "top5": "top5",
    "top10": "top10",
    "top20": "top20",
}


def fit_platt_scaling(probs, actuals):
    """Fit Platt scaling (logistic recalibration) on walk-forward predictions.
    
    Returns (A, B) parameters for sigmoid(A * logit(p) + B).
    """
    eps = 1e-7
    clipped = np.clip(probs, eps, 1 - eps)
    logit_p = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(logit_p, actuals)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def platt_transform(p_raw, A, B):
    """Apply Platt scaling: sigmoid(A * logit(p) + B)."""
    eps = 1e-7
    p = np.clip(p_raw, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    z = A * logit_p + B
    return 1 / (1 + np.exp(-z))


def load_features():
    path = DATA_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run build_features.py first to create {path}")
    return pd.read_csv(path)


def walk_forward(df, target_col, min_train_seasons=2, stats_only=False):
    """
    Walk-forward validation: train on all prior seasons, test on current season.
    Returns list of (season, test_df_with_predictions) tuples.
    """
    if stats_only:
        df = df[df["has_stats"] == 1].copy()

    seasons = sorted(df["season"].unique())
    results = []

    for i, test_season in enumerate(seasons):
        if i < min_train_seasons:
            continue

        train_df = df[df["season"] < test_season].copy()
        test_df = df[df["season"] == test_season].copy()

        # Filter to rows with valid features and labels
        # Only require target + rolling form columns (stats handled by imputer)
        core_cols = [c for c in FEATURE_COLS if c in ("made_cut_l", "avg_finish_l", "top5_l", "top10_l", "top20_l", "events_played_l")]
        train_valid = train_df.dropna(subset=core_cols + [target_col])
        test_valid = test_df.dropna(subset=core_cols + [target_col])

        if len(train_valid) < 100 or len(test_valid) < 10:
            continue

        X_train = train_valid[FEATURE_COLS].values
        y_train = train_valid[target_col].values
        X_test = test_valid[FEATURE_COLS].values
        y_test = test_valid[target_col].values

        # Pipeline: impute -> scale -> xgb
        pipeline = Pipeline([
            ("imputer", PercentileImputer(percentile=25)),
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False,
            )),
        ])

        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]

        test_valid = test_valid.copy()
        test_valid["pred_prob"] = probs

        # Metrics
        auc = roc_auc_score(y_test, probs) if y_test.sum() > 0 and y_test.sum() < len(y_test) else np.nan
        ll = log_loss(y_test, np.column_stack([1 - probs, probs]))

        results.append({
            "season": test_season,
            "pipeline": pipeline,
            "test_df": test_valid,
            "auc": auc,
            "log_loss": ll,
            "n_test": len(test_valid),
            "n_positive": int(y_test.sum()),
        })

    return results


def train_production(df, target_col, stats_only=False):
    """Train final model on all available data."""
    if stats_only:
        df = df[df["has_stats"] == 1].copy()
    valid = df.dropna(subset=[c for c in FEATURE_COLS if c in ("made_cut_l", "avg_finish_l", "top5_l", "top10_l", "top20_l", "events_played_l")] + [target_col])
    X = valid[FEATURE_COLS].values
    y = valid[target_col].values

    pipeline = Pipeline([
        ("imputer", PercentileImputer(percentile=25)),
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )),
    ])

    pipeline.fit(X, y)
    return pipeline


def main(stats_only=False):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    df = load_features()
    if stats_only:
        df = df[df["has_stats"] == 1].copy()
        print(f"  Stats-only mode: {len(df)} rows (excluded players without PGA Tour stats)")
    else:
        print(f"  {len(df)} rows, seasons {sorted(df['season'].unique())}")

    all_results = {}
    calibration_params = {}

    for target_col, name in TARGETS.items():
        print(f"\n{'='*50}")
        print(f"Target: {name} ({target_col})")
        print(f"{'='*50}")

        # Walk-forward validation
        wf_results = walk_forward(df, target_col)

        if not wf_results:
            print("  Not enough data for walk-forward validation.")
            continue

        # Print results
        for r in wf_results:
            print(f"  Season {r['season']}: AUC={r['auc']:.4f}, "
                  f"LogLoss={r['log_loss']:.4f}, "
                  f"n={r['n_test']}, positives={r['n_positive']}")

        # Aggregate
        aucs = [r["auc"] for r in wf_results if not np.isnan(r["auc"])]
        lls = [r["log_loss"] for r in wf_results]
        print(f"\n  Mean AUC: {np.mean(aucs):.4f} (std: {np.std(aucs):.4f})")
        print(f"  Mean LogLoss: {np.mean(lls):.4f}")

        # Combine all test predictions
        all_test = pd.concat([r["test_df"] for r in wf_results], ignore_index=True)
        all_test.to_csv(DATA_DIR / f"predictions_{name}.csv", index=False)

        # Fit Platt scaling on walk-forward predictions
        raw_probs = all_test["pred_prob"].values
        actuals = all_test[target_col].values
        A, B = fit_platt_scaling(raw_probs, actuals)
        calibration_params[name] = {"A": A, "B": B}
        
        # Show calibration improvement
        cal_probs = platt_transform(raw_probs, A, B)
        gap_before = np.mean(raw_probs) - np.mean(actuals)
        gap_after = np.mean(cal_probs) - np.mean(actuals)
        print(f"  Platt scaling: A={A:.4f}, B={B:.4f}")
        print(f"  Mean predicted: {np.mean(raw_probs):.4%} -> {np.mean(cal_probs):.4%} (actual: {np.mean(actuals):.4%})")
        print(f"  Calibration gap: {gap_before:+.4%} -> {gap_after:+.4%}")

        # Train production model on all data
        print(f"\n  Training production model...")
        prod_pipeline = train_production(df, target_col, stats_only=stats_only)
        joblib.dump(prod_pipeline, MODEL_DIR / f"xgb_{name}_production.joblib")

        all_results[name] = {
            "auc": float(np.mean(aucs)) if aucs else None,
            "log_loss": float(np.mean(lls)) if lls else None,
            "n_seasons": len(wf_results),
        }

    # Save metadata
    meta = {
        "feature_cols": FEATURE_COLS,
        "targets": list(TARGETS.keys()),
        "results": all_results,
        "calibration": calibration_params,
    }
    with open(MODEL_DIR / "xgb_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved models to {MODEL_DIR}")
    for name in TARGETS.values():
        print(f"  xgb_{name}_production.joblib")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-only", action="store_true",
                        help="Train only on players with PGA Tour stats (no imputation)")
    args = parser.parse_args()
    main(stats_only=args.stats_only)
