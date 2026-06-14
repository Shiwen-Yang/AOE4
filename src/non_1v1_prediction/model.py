"""
LightGBM model for team-match outcome prediction + metrics + SHAP.

Temporal split by `started_at` (train | valid | test); team-swap augmentation makes the
Team-A/Team-B orientation carry no signal (swap _a<->_b, flip target, negate _diff).
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .config import MODEL_DIR, TEST_FRAC, VALID_FRAC
from .features import ALL_FEATURES, ALL_FEATURES_PREMADE, CATEGORICAL_FEATURES, NUMERIC_FEATURES

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting": "gbdt",
    "num_leaves": 127,
    "min_child_samples": 100,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "n_estimators": 1000,
    "verbose": -1,
    "random_state": 42,
}


def temporal_split(df: pd.DataFrame, valid_frac=VALID_FRAC, test_frac=TEST_FRAC):
    df = df.sort_values("started_at").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - valid_frac - test_frac))
    valid_end = int(n * (1 - test_frac))
    return df.iloc[:train_end], df.iloc[train_end:valid_end], df.iloc[valid_end:]


def make_team_swapped_rows(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """Swap Team A <-> Team B: exchange _a/_b columns, flip target, negate _diff columns."""
    swapped = df.copy()
    for col in df.columns:
        if col.endswith("_a"):
            other = col[:-2] + "_b"
            if other in df.columns:
                swapped[col] = df[other]
                swapped[other] = df[col]
    swapped[target_col] = 1 - df[target_col]
    for col in df.columns:
        if col.endswith("_diff"):
            swapped[col] = -df[col]
    return swapped


def augment_with_team_swaps(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    return pd.concat([df, make_team_swapped_rows(df, target_col)], ignore_index=True, copy=False)


def _prep(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    features = features or ALL_FEATURES_PREMADE
    X = df[features].copy()
    for c in features:
        if c in CATEGORICAL_FEATURES:
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def train_lgbm(train: pd.DataFrame, valid: pd.DataFrame, params: dict | None = None,
               features: list[str] | None = None):
    features = features or ALL_FEATURES_PREMADE
    cats = [c for c in features if c in CATEGORICAL_FEATURES]
    params = {**DEFAULT_PARAMS, **(params or {})}
    n_est = params.pop("n_estimators", 1000)

    ds_train = lgb.Dataset(_prep(train, features), label=train["target"].values,
                           categorical_feature=cats, free_raw_data=False)
    ds_valid = lgb.Dataset(_prep(valid, features), label=valid["target"].values,
                           categorical_feature=cats, reference=ds_train, free_raw_data=False)
    model = lgb.train(
        params, ds_train, num_boost_round=n_est, valid_sets=[ds_valid],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    return model


def predict(model, df: pd.DataFrame, features: list[str] | None = None) -> np.ndarray:
    return model.predict(_prep(df, features), num_iteration=model.best_iteration)


def compute_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    return {
        "n": int(len(y_true)),
        "auc": float(roc_auc_score(y_true, p)),
        "log_loss": float(log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_true, p)),
        "ece": expected_calibration_error(y_true, p),
        "base_rate": float(y_true.mean()),
    }


def calibration_table(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p)})
    df["bucket"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    g = df.groupby("bucket", observed=True).agg(
        n=("y", "size"), pred_mean=("p", "mean"), actual=("y", "mean")
    ).reset_index()
    return g


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Decile-binned, n-weighted mean |predicted − actual|."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.sum() / n * abs(p[m].mean() - y_true[m].mean())
    return float(ece)


def _subgroup_assignments(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Map each subgroup dimension to a per-row label Series (built on the test df)."""
    from .config import SKILL_BAND_LABELS, SKILL_BANDS

    out: dict[str, pd.Series] = {}

    skill = (df["mmr_mean_a"] + df["mmr_mean_b"]) / 2
    out["Skill band (match mean MMR)"] = pd.cut(
        skill, SKILL_BANDS, labels=SKILL_BAND_LABELS, include_lowest=True).astype(object)

    out["Team MMR gap"] = pd.cut(
        df["mmr_mean_diff"].abs(), [0, 25, 50, 100, 200, 1e9],
        labels=["0-25", "25-50", "50-100", "100-200", "200+"], include_lowest=True).astype(object)

    prem = np.where(df["both_premade"] == 1, "both premade",
                    np.where(df["premade_xor"] == 1, "one side premade", "neither premade"))
    out["Premade status"] = pd.Series(prem, index=df.index)

    top_maps = df["map"].value_counts().head(8).index
    out["Map"] = df["map"].where(df["map"].isin(top_maps), other="other")

    out["Server"] = df["server"].astype(object)

    smurf = (df["n_smurf_like_a"] > 0) | (df["n_smurf_like_b"] > 0)
    out["Has 1v1-smurf-like player"] = np.where(smurf, "yes", "no")

    cg = df[["carry_gap_a", "carry_gap_b"]].max(axis=1)
    cg_hi = cg.quantile(0.90)
    out["High carry-gap stack (top 10%)"] = np.where(cg >= cg_hi, "yes", "no")

    newp = (df["n_new_players_a"] > 0) | (df["n_new_players_b"] > 0)
    out["Has new/low-history player"] = np.where(newp, "yes", "no")
    return out


def compute_subgroup_metrics(df: pd.DataFrame, y_true: np.ndarray, p: np.ndarray,
                             min_n: int | None = None) -> pd.DataFrame:
    """
    Long-format per-subgroup metrics on the test set. Columns:
      dimension, subgroup, n, base_rate, auc, brier, ece, favored_winrate.
    Groups with n < min_n or a single outcome class are skipped (AUC undefined).
    """
    from .config import SUBGROUP_MIN_N

    min_n = SUBGROUP_MIN_N if min_n is None else min_n
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    # favored = higher mean-MMR team; its actual win rate (model-independent)
    gap = df["mmr_mean_diff"].to_numpy()
    fav_win = np.where(gap >= 0, y_true, 1 - y_true)

    rows = []
    for dim, labels in _subgroup_assignments(df).items():
        labels = pd.Series(labels, index=df.index)
        for val, mask in labels.groupby(labels):
            sel = labels == val
            idx = sel.to_numpy()
            n = int(idx.sum())
            if n < min_n or pd.isna(val):
                continue
            yv, pv = y_true[idx], p[idx]
            if len(np.unique(yv)) < 2:
                continue
            rows.append({
                "dimension": dim, "subgroup": str(val), "n": n,
                "base_rate": float(yv.mean()),
                "auc": float(roc_auc_score(yv, pv)),
                "brier": float(brier_score_loss(yv, pv)),
                "ece": expected_calibration_error(yv, pv),
                "favored_winrate": float(fav_win[idx].mean()),
            })
    return pd.DataFrame(rows)


def compute_shap(model, df: pd.DataFrame, max_rows: int = 20000,
                 features: list[str] | None = None) -> pd.Series:
    import shap

    sample = df.sample(min(len(df), max_rows), random_state=42)
    X = _prep(sample, features)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    imp = np.abs(sv).mean(axis=0)
    return pd.Series(imp, index=X.columns).sort_values(ascending=False)


def save_model(model, mode: str, seasons: list[int], metrics: dict,
               features: list[str] | None = None) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{mode}_s{'-'.join(str(s) for s in seasons)}"
    path = MODEL_DIR / f"lgbm_{tag}.txt"
    model.save_model(str(path), num_iteration=model.best_iteration)
    (MODEL_DIR / f"lgbm_{tag}_meta.json").write_text(json.dumps(
        {"mode": mode, "seasons": seasons,
         "features": features or ALL_FEATURES_PREMADE, "metrics": metrics},
        indent=2))
    return path
