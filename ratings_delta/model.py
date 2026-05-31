"""LightGBM regression for rating delta prediction and shared metric utilities."""
import numpy as np
import pandas as pd
import lightgbm as lgb

NUMERIC_FEATURES = [
    "player_rating_before", "opponent_rating_before", "visible_rating_gap",
    "player_mmr_before", "opponent_mmr_before", "hidden_mmr_gap",
    "result",
    "games_lifetime_before", "games_this_season_before",
    "days_since_last_game", "current_streak",
    "recent_wr_10", "recent_wr_20",
    "missing_player_rating", "missing_opponent_rating",
    "missing_player_mmr", "missing_opponent_mmr",
]
CATEGORICAL_FEATURES = ["season", "patch", "map"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": ["rmse", "mae"],
    "num_leaves": 63,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "learning_rate": 0.05,
    "min_child_samples": 50,
    "n_estimators": 500,
    "verbose": -1,
    "random_state": 42,
}


def temporal_split(
    df: pd.DataFrame,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Game-level temporal split: both participant rows from the same game stay together."""
    games = (
        df[["game_id", "started_at"]]
        .drop_duplicates("game_id")
        .sort_values("started_at")
        .reset_index(drop=True)
    )
    n = len(games)
    train_end = int(n * (1 - valid_frac - test_frac))
    valid_end = int(n * (1 - test_frac))

    train_ids = set(games.iloc[:train_end]["game_id"])
    valid_ids = set(games.iloc[train_end:valid_end]["game_id"])
    test_ids = set(games.iloc[valid_end:]["game_id"])

    train_df = df[df["game_id"].isin(train_ids)].copy()
    valid_df = df[df["game_id"].isin(valid_ids)].copy()
    test_df = df[df["game_id"].isin(test_ids)].copy()

    print(f"\n=== Temporal Split ===")
    print(f"  Train games: {len(train_ids):>8,}  rows: {len(train_df):>10,}")
    print(f"  Valid games: {len(valid_ids):>8,}  rows: {len(valid_df):>10,}")
    print(f"  Test  games: {len(test_ids):>8,}  rows: {len(test_df):>10,}")

    return train_df, valid_df, test_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, R², mean signed error, median absolute error. Skips NaN predictions."""
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    yt = y_true[mask]
    yp = y_pred[mask]
    n = mask.sum()
    if n == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "mse_signed": float("nan"), "medae": float("nan")}
    err = yt - yp
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return {
        "n": int(n),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mse_signed": float(np.mean(err)),
        "medae": float(np.median(np.abs(err))),
    }


def compute_subgroup_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict]:
    """Metrics for each predefined subgroup."""
    df = df.copy()
    df["_y_true"] = y_true
    df["_y_pred"] = y_pred
    gap_abs = df["visible_rating_gap"].abs()

    subgroups: dict[str, pd.Series] = {
        "Winners":                           df["result"] == 1,
        "Losers":                            df["result"] == 0,
        "|rating_gap| ≤ 50":                 gap_abs <= 50,
        "|rating_gap| > 200":                gap_abs > 200,
        "games_this_season < 10":            df["games_this_season_before"] < 10,
        "games_this_season ≥ 10":            df["games_this_season_before"] >= 10,
        "Missing opponent rating":           df["missing_opponent_rating"] == 1,
        "Opponent rating present":           df["missing_opponent_rating"] == 0,
    }

    results = {}
    for label, mask in subgroups.items():
        sub = df[mask]
        results[label] = compute_metrics(
            sub["_y_true"].values.astype(float),
            sub["_y_pred"].values.astype(float),
        )
    return results


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals as category dtype; fill missing categoricals with 'missing'."""
    X = df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("missing").astype(str).astype("category")
    return X


def train_lgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str = "observed_rating_delta",
    params: dict | None = None,
) -> lgb.LGBMRegressor:
    """Train LightGBM regression model with early stopping."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    n_estimators = p.pop("n_estimators", 500)

    X_train = _prepare_X(train_df)
    y_train = train_df[target_col].values.astype(float)
    X_valid = _prepare_X(valid_df)
    y_valid = valid_df[target_col].values.astype(float)

    print(f"\n=== Training LightGBM ===")
    print(f"  Target: '{target_col}'")
    print(f"  Features: {len(ALL_FEATURES)} ({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical)")
    print(f"  Train rows: {len(X_train):,}  |  Valid rows: {len(X_valid):,}")

    model = lgb.LGBMRegressor(n_estimators=n_estimators, **p)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_names=["valid"],
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    val_pred = model.predict(X_valid)
    val_m = compute_metrics(y_valid, val_pred)
    print(f"  Valid → MAE={val_m['mae']:.3f}  RMSE={val_m['rmse']:.3f}  R²={val_m['r2']:.4f}")

    return model


def predict(model: lgb.LGBMRegressor, df: pd.DataFrame) -> np.ndarray:
    return model.predict(_prepare_X(df))


def compute_shap(
    model: lgb.LGBMRegressor,
    df: pd.DataFrame,
    max_rows: int = 5000,
) -> tuple[np.ndarray, list[str]]:
    """Compute mean |SHAP| per feature on a sample of rows."""
    try:
        import shap
    except ImportError:
        print("  shap not installed — skipping SHAP analysis")
        return np.array([]), ALL_FEATURES

    X = _prepare_X(df)
    sample = X.sample(min(max_rows, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    return mean_abs, ALL_FEATURES
