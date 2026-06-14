"""LightGBM model, temporal split, persistence, and group-level metrics."""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb

from aoe4_predict.config import BASE_DIR

from .features import ALL_FEATURES, CONTEXT_FEATURES, prepare_X

MODEL_DIR = BASE_DIR / "models" / "civ_choice"
MODEL_PATH = MODEL_DIR / "lgbm_civ_choice.txt"
MODEL_META_PATH = MODEL_DIR / "lgbm_civ_choice_meta.json"

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "num_leaves": 63,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 50,
    "verbose": -1,
    "random_state": 42,
}


def temporal_split(
    df: pd.DataFrame,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Game-level temporal split: all candidate rows for the same player-match stay together."""
    player_games = (
        df[["game_id", "profile_id", "started_at"]]
        .drop_duplicates(["game_id", "profile_id"])
        .sort_values("started_at")
        .reset_index(drop=True)
    )
    n = len(player_games)
    train_end = int(n * (1 - valid_frac - test_frac))
    valid_end = int(n * (1 - test_frac))

    train_ids = set(map(tuple, player_games.iloc[:train_end][["game_id", "profile_id"]].values))
    valid_ids = set(map(tuple, player_games.iloc[train_end:valid_end][["game_id", "profile_id"]].values))

    df["_gp"] = list(zip(df["game_id"], df["profile_id"]))
    train_df = df[df["_gp"].isin(train_ids)].drop(columns="_gp").copy()
    valid_df = df[df["_gp"].isin(valid_ids)].drop(columns="_gp").copy()
    test_df = df[~df["_gp"].isin(train_ids | valid_ids)].drop(columns="_gp").copy()

    print(f"\n=== Temporal Split ===")
    print(f"  Train player-matches: {len(train_ids):>8,}  rows: {len(train_df):>10,}")
    print(f"  Valid player-matches: {len(valid_ids):>8,}  rows: {len(valid_df):>10,}")
    print(f"  Test  player-matches: {n - len(train_ids) - len(valid_ids):>8,}  rows: {len(test_df):>10,}")
    return train_df, valid_df, test_df


def normalize_predictions(
    df: pd.DataFrame,
    raw_pred: np.ndarray,
    method: str = "renorm",
) -> np.ndarray:
    """Normalize raw model predictions within each player-match group.

    method='renorm': P_i = raw_i / Σ raw_j  (probability renormalization)
    method='softmax': P_i = exp(raw_i) / Σ exp(raw_j)  (group softmax on raw scores)
    """
    tmp = df[["game_id", "profile_id"]].copy()
    tmp["_raw"] = raw_pred

    if method == "softmax":
        tmp["_raw"] = np.exp(np.clip(tmp["_raw"], -20, 20))

    tmp["_norm"] = tmp.groupby(["game_id", "profile_id"])["_raw"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else np.ones(len(x)) / len(x)
    )
    return tmp["_norm"].values


def normalize_raw_scores_with_temperature(
    df: pd.DataFrame,
    raw_scores: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Group-softmax LightGBM raw margins after temperature scaling.

    Temperature > 1 softens candidate probabilities; temperature < 1 sharpens
    them. Probabilities are normalized within each (game_id, profile_id) group.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    tmp = df[["game_id", "profile_id"]].copy()
    tmp["_scaled"] = np.asarray(raw_scores, dtype=float) / temperature
    group_cols = ["game_id", "profile_id"]
    group_max = tmp.groupby(group_cols)["_scaled"].transform("max")
    tmp["_exp"] = np.exp(np.clip(tmp["_scaled"] - group_max, -50, 50))
    group_sum = tmp.groupby(group_cols)["_exp"].transform("sum")
    tmp["_norm"] = np.where(group_sum > 0, tmp["_exp"] / group_sum, 0.0)
    return tmp["_norm"].values


def _temperature_nll(df: pd.DataFrame, raw_scores: np.ndarray, temperature: float) -> float:
    probs = normalize_raw_scores_with_temperature(df, raw_scores, temperature)
    chosen = df["target"].values == 1
    return float(-np.log(np.clip(probs[chosen], 1e-12, 1.0)).mean())


def fit_temperature(
    df: pd.DataFrame,
    raw_scores: np.ndarray,
    low: float = 0.05,
    high: float = 10.0,
    iterations: int = 40,
) -> tuple[float, float]:
    """Fit a single temperature on validation data by minimizing group NLL."""
    phi = (1 + 5 ** 0.5) / 2
    a, b = low, high
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc = _temperature_nll(df, raw_scores, c)
    fd = _temperature_nll(df, raw_scores, d)

    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / phi
            fc = _temperature_nll(df, raw_scores, c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / phi
            fd = _temperature_nll(df, raw_scores, d)

    temperature = (a + b) / 2
    return temperature, _temperature_nll(df, raw_scores, temperature)


def compute_group_metrics(
    df: pd.DataFrame,
    y_pred_norm: np.ndarray,
    eps: float = 1e-12,
) -> dict:
    """Compute per-player-match metrics then average.

    Returns Top-1/3/5 accuracy, multiclass log loss, Brier score,
    mean chosen-civ predicted probability.
    """
    tmp = df[["game_id", "profile_id", "target", "candidate_civ"]].copy()
    tmp["p"] = np.clip(y_pred_norm, eps, 1.0)

    results = []
    for (gid, pid), grp in tmp.groupby(["game_id", "profile_id"]):
        chosen_mask = grp["target"] == 1
        if chosen_mask.sum() != 1:
            continue  # skip malformed groups
        sorted_grp = grp.sort_values("p", ascending=False).reset_index(drop=True)
        chosen_civ = grp.loc[chosen_mask, "candidate_civ"].iloc[0]
        chosen_p = grp.loc[chosen_mask, "p"].iloc[0]
        rank_of_chosen = (sorted_grp["candidate_civ"] == chosen_civ).idxmax() + 1

        # Brier: Σ (p_i - y_i)^2 over all candidates in this group
        brier = float(((grp["p"] - grp["target"]) ** 2).sum())
        results.append({
            "top1": int(rank_of_chosen <= 1),
            "top3": int(rank_of_chosen <= 3),
            "top5": int(rank_of_chosen <= 5),
            "log_loss": -float(np.log(chosen_p)),
            "brier": brier,
            "chosen_p": float(chosen_p),
        })

    if not results:
        return {}

    r = pd.DataFrame(results)
    return {
        "n": len(r),
        "top1_acc": float(r["top1"].mean()),
        "top3_acc": float(r["top3"].mean()),
        "top5_acc": float(r["top5"].mean()),
        "log_loss": float(r["log_loss"].mean()),
        "brier": float(r["brier"].mean()),
        "mean_chosen_prob": float(r["chosen_p"].mean()),
    }


def compute_subgroup_metrics(
    df: pd.DataFrame,
    y_pred_norm: np.ndarray,
) -> dict[str, dict]:
    """Metrics for predefined player and civ-pool subgroups."""
    tmp = df.copy()
    tmp["_pred"] = y_pred_norm

    # Civ pool size (distinct civs played lifetime)
    pool_size = tmp.groupby(["game_id", "profile_id"])["candidate_is_in_pool_lifetime"].transform("sum")
    mmr = tmp["player_mmr"].fillna(tmp["player_rating"])

    subgroups = {
        "Specialist (1 civ)":       pool_size == 1,
        "2–3 civ players":          (pool_size >= 2) & (pool_size <= 3),
        "4–6 civ players":          (pool_size >= 4) & (pool_size <= 6),
        "7+ civ players":           pool_size >= 7,
        "Low MMR (< 1000)":         mmr < 1000,
        "Mid MMR (1000–1400)":      (mmr >= 1000) & (mmr < 1400),
        "High MMR (≥ 1400)":        mmr >= 1400,
        "games_lifetime < 20":      tmp["player_games_lifetime"] < 20,
        "games_lifetime ≥ 20":      tmp["player_games_lifetime"] >= 20,
        # Played-civ-only: candidate rows where player has used this civ before
        "Played-civ rows only":     tmp["cand_games_lifetime"] > 0,
    }

    results = {}
    for label, mask in subgroups.items():
        sub = tmp[mask]
        if len(sub) < 100:
            continue
        m = compute_group_metrics(sub, sub["_pred"].values)
        results[label] = m
    return results


def train_lgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    params: dict | None = None,
) -> lgb.LGBMClassifier:
    p = {**DEFAULT_PARAMS, **(params or {})}
    n_est = p.pop("n_estimators", 500)

    X_train = prepare_X(train_df)
    y_train = train_df["target"].values
    X_valid = prepare_X(valid_df)
    y_valid = valid_df["target"].values

    print(f"\n=== Training LightGBM ===")
    print(f"  Features: {len(ALL_FEATURES)}")
    print(f"  Train rows: {len(X_train):,}  |  Valid rows: {len(X_valid):,}")
    print(f"  Positive rate (train): {y_train.mean():.4f}")

    model = lgb.LGBMClassifier(n_estimators=n_est, **p)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        categorical_feature=CONTEXT_FEATURES,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


def save_model(
    model: lgb.LGBMClassifier | lgb.Booster,
    model_path: Path | str | None = None,
    meta_path: Path | str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the trained civ-choice LightGBM model and metadata."""
    model_path = Path(model_path) if model_path is not None else MODEL_PATH
    meta_path = Path(meta_path) if meta_path is not None else MODEL_META_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    booster = model.booster_ if hasattr(model, "booster_") else model
    booster.save_model(str(model_path))

    model_meta: dict[str, Any] = {
        "feature_cols": ALL_FEATURES,
        "cat_features": CONTEXT_FEATURES,
        "n_trees": booster.num_trees(),
    }
    if hasattr(model, "best_iteration_"):
        model_meta["best_iteration"] = model.best_iteration_
    if hasattr(model, "get_params"):
        model_meta["params"] = model.get_params()
    if meta:
        model_meta.update(meta)

    meta_path.write_text(json.dumps(model_meta, indent=2, default=str))
    return model_meta


def load_model(
    model_path: Path | str | None = None,
    meta_path: Path | str | None = None,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Load the persisted civ-choice LightGBM model and metadata."""
    model_path = Path(model_path) if model_path is not None else MODEL_PATH
    meta_path = Path(meta_path) if meta_path is not None else MODEL_META_PATH
    model = lgb.Booster(model_file=str(model_path))
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, meta


def predict_candidate_probs(
    model: lgb.LGBMClassifier | lgb.Booster,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return candidate-level positive-class probabilities for supported models."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def compute_shap(
    model: lgb.LGBMClassifier,
    df: pd.DataFrame,
    max_rows: int = 5000,
) -> tuple[np.ndarray, list[str]]:
    try:
        import shap
    except ImportError:
        print("  shap not installed — skipping SHAP analysis")
        return np.array([]), ALL_FEATURES

    X = prepare_X(df).sample(min(max_rows, len(df)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    # For binary classifiers, shap_values returns [class0, class1]; use class1
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    return mean_abs, ALL_FEATURES
