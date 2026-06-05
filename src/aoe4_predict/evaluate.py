"""
Evaluation metrics: AUC, log loss, Brier score, calibration.
Also runs baseline comparisons and subgroup breakdowns.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, roc_auc_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Return dict of AUC, log loss, Brier score, accuracy."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

    single_class = len(np.unique(y_true)) < 2
    auc = float("nan") if single_class else roc_auc_score(y_true, y_pred)
    ll  = float("nan") if single_class else log_loss(y_true, y_pred)
    brier = float(np.mean((y_pred - y_true) ** 2))
    acc = float(np.mean((y_pred >= 0.5) == y_true))

    metrics = {
        "auc": float("nan") if np.isnan(auc) else round(auc, 4),
        "log_loss": float("nan") if np.isnan(ll) else round(ll, 4),
        "brier": round(brier, 4),
        "acc@0.5": round(acc, 4),
    }
    if label:
        metrics["label"] = label
    return metrics


def calibration_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Calibration table: predicted vs empirical win rate per probability bucket.
    A well-calibrated model has fraction_of_positives ≈ mean_predicted_value.
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")
    counts, edges = np.histogram(y_pred, bins=n_bins, range=(0, 1))
    return pd.DataFrame(
        {
            "predicted_wr": mean_pred.round(3),
            "empirical_wr": frac_pos.round(3),
            "gap": (frac_pos - mean_pred).round(3),
            "n": counts[: len(mean_pred)],
        }
    )


def _subgroup_eval(y_true, y_pred, mask, label) -> dict | None:
    n = mask.sum()
    if n < 50:
        return None
    return {"group": label, "n": int(n), **evaluate(y_true[mask], y_pred[mask])}


def evaluate_subgroups(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    target_col: str = "target",
) -> list[dict]:
    """Evaluate model on meaningful subgroups."""
    y = df[target_col].values.astype(float)
    results = []

    def add(mask, label):
        # Handle nullable pandas arrays before converting to numpy bool
        if hasattr(mask, "fillna"):
            mask = mask.fillna(False)
        mask = np.asarray(mask, dtype=bool)
        r = _subgroup_eval(y, y_pred, mask, label)
        if r:
            results.append(r)

    # History depth
    if "games_lifetime_a" in df.columns and "games_lifetime_b" in df.columns:
        min_hist = df[["games_lifetime_a", "games_lifetime_b"]].min(axis=1).values
        add(min_hist < 10, "low_history_player (<10 games)")
        add(min_hist >= 50, "high_history_player (≥50 games)")

    # MMR availability
    if "missing_mmr_a" in df.columns and "missing_mmr_b" in df.columns:
        add(df["missing_mmr_a"].values == 0, "mmr_available_a")
        add(df["missing_mmr_a"].values == 1, "mmr_missing_a")

    # MMR gap magnitude
    if "mmr_diff" in df.columns:
        gap = df["mmr_diff"].abs().values
        add(gap < 50, "small_mmr_gap (<50)")
        add(gap >= 200, "large_mmr_gap (≥200)")

    # Civ context
    if "civs_known" in df.columns:
        add(df["civs_known"].values == 1, "civs_known")
        add(df["civs_known"].values == 0, "civs_unknown")

    # Map context
    if "map_known" in df.columns:
        add(df["map_known"].values == 1, "map_known")

    return results


def print_report(
    label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame | None = None,
    show_calibration: bool = True,
    show_subgroups: bool = True,
) -> None:
    m = evaluate(y_true, y_pred, label)
    print(f"\n── {label} ──")
    print(f"  AUC:        {m['auc']:.4f}")
    print(f"  Log loss:   {m['log_loss']:.4f}")
    print(f"  Brier:      {m['brier']:.4f}")
    print(f"  Acc@0.5:    {m['acc@0.5']:.4f}")

    if show_calibration:
        cal = calibration_table(y_true, y_pred)
        print("  Calibration:")
        print("    " + cal.to_string(index=False).replace("\n", "\n    "))

    if show_subgroups and df is not None:
        subs = evaluate_subgroups(df, y_pred)
        if subs:
            print("  Subgroups:")
            for s in subs:
                print(f"    {s['group']:<35} n={s['n']:>7,}  AUC={s['auc']:.4f}  Brier={s['brier']:.4f}")


def compare_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_preds: np.ndarray,
    target_col: str = "target",
) -> None:
    """Fit and evaluate all three baselines plus the ML model on the test set."""
    from .baselines import CivMapBucketBaseline, ConstantBaseline, MMRLogisticBaseline

    y_test = test_df[target_col].values

    print("\n=== Baseline Comparison (test set) ===")
    for cls in [ConstantBaseline, MMRLogisticBaseline, CivMapBucketBaseline]:
        b = cls()
        b.fit(train_df, target_col)
        preds = b.predict_proba(test_df)
        print_report(b.name, y_test, preds, df=test_df, show_calibration=False, show_subgroups=False)

    print_report("LightGBM", y_test, model_preds, df=test_df, show_calibration=True, show_subgroups=True)
