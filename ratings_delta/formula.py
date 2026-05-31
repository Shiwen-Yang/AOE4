"""Elo formula recovery for AOE4 visible rating updates."""
import numpy as np
import pandas as pd


def elo_expected(rating_a: np.ndarray, rating_b: np.ndarray, D: float = 400.0) -> np.ndarray:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / D))


def elo_delta(
    rating_a: np.ndarray,
    rating_b: np.ndarray,
    result: np.ndarray,
    K: float = 32.0,
    D: float = 400.0,
) -> np.ndarray:
    return K * (result - elo_expected(rating_a, rating_b, D))


def _clean_for_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with both ratings and the target non-null."""
    return df.dropna(subset=["player_rating_before", "opponent_rating_before", "observed_rating_delta"])


def check_determinism(df: pd.DataFrame) -> dict:
    """Group by (player_rating_before, opponent_rating_before, result) and check delta uniqueness."""
    clean = _clean_for_elo(df).copy()
    clean["player_rating_before"] = clean["player_rating_before"].astype(int)
    clean["opponent_rating_before"] = clean["opponent_rating_before"].astype(int)
    clean["observed_rating_delta"] = clean["observed_rating_delta"].astype(int)

    grp = clean.groupby(["player_rating_before", "opponent_rating_before", "result"])
    n_unique_inputs = grp.ngroups
    n_single_delta = (grp["observed_rating_delta"].nunique() == 1).sum()
    pct_single = n_single_delta / n_unique_inputs * 100

    # How many distinct delta values exist per input tuple on average?
    avg_distinct = grp["observed_rating_delta"].nunique().mean()

    result = {
        "n_unique_inputs": n_unique_inputs,
        "n_single_delta": n_single_delta,
        "pct_single_delta": pct_single,
        "avg_distinct_deltas": avg_distinct,
    }

    print("\n=== Determinism Check ===")
    print(f"  Unique (rating_a, rating_b, result) tuples: {n_unique_inputs:>10,}")
    print(f"  Tuples with exactly one delta value:        {n_single_delta:>10,}  ({pct_single:.1f}%)")
    print(f"  Avg distinct delta values per tuple:        {avg_distinct:>10.2f}")

    return result


def fit_elo_grid(df: pd.DataFrame) -> tuple[float, float, float]:
    """Grid search then fine search for best Elo K and D by MAE."""
    clean = _clean_for_elo(df)
    ra = clean["player_rating_before"].values.astype(float)
    rb = clean["opponent_rating_before"].values.astype(float)
    res = clean["result"].values.astype(float)
    obs = clean["observed_rating_delta"].values.astype(float)

    K_coarse = [16, 20, 24, 28, 32, 40, 50]
    D_coarse = [200, 300, 400, 500, 600]

    best_mae = np.inf
    best_K, best_D = 32.0, 400.0

    for K in K_coarse:
        for D in D_coarse:
            pred = K * (res - 1.0 / (1.0 + 10.0 ** ((rb - ra) / D)))
            mae = np.mean(np.abs(obs - pred))
            if mae < best_mae:
                best_mae = mae
                best_K, best_D = float(K), float(D)

    # Fine-grained search around the coarse best
    for K in np.arange(best_K - 8, best_K + 9, 1.0):
        for D in np.arange(best_D - 100, best_D + 101, 25.0):
            if K <= 0 or D <= 0:
                continue
            pred = K * (res - 1.0 / (1.0 + 10.0 ** ((rb - ra) / D)))
            mae = np.mean(np.abs(obs - pred))
            if mae < best_mae:
                best_mae = mae
                best_K, best_D = K, D

    # Report a sample comparison
    pred_best = elo_delta(ra, rb, res, best_K, best_D)
    rmse = np.sqrt(np.mean((obs - pred_best) ** 2))
    r2 = 1.0 - np.sum((obs - pred_best) ** 2) / np.sum((obs - obs.mean()) ** 2)

    print("\n=== Elo Formula Fit ===")
    print(f"  Best K = {best_K:.1f},  D = {best_D:.0f}")
    print(f"  MAE  = {best_mae:.3f} rating points")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  R²   = {r2:.4f}")

    # Sample comparison (20 rows)
    sample = clean.sample(min(20, len(clean)), random_state=0)
    pred_s = elo_delta(
        sample["player_rating_before"].values.astype(float),
        sample["opponent_rating_before"].values.astype(float),
        sample["result"].values.astype(float),
        best_K, best_D,
    )
    print("\n  Sample (rating_a, rating_b, result) → predicted vs actual:")
    print(f"  {'rating_a':>8}  {'rating_b':>8}  {'result':>6}  {'predicted':>10}  {'actual':>8}  {'residual':>9}")
    for _, row in sample.iterrows():
        p = elo_delta(
            np.array([row["player_rating_before"]]),
            np.array([row["opponent_rating_before"]]),
            np.array([row["result"]]),
            best_K, best_D,
        )[0]
        a = row["observed_rating_delta"]
        print(f"  {row['player_rating_before']:>8.0f}  {row['opponent_rating_before']:>8.0f}  "
              f"{row['result']:>6.0f}  {p:>10.2f}  {a:>8.0f}  {a - p:>9.2f}")

    return best_K, best_D, best_mae


def k_factor_segmentation(df: pd.DataFrame, D: float) -> dict:
    """Fit separate best-K for each games_this_season bucket by MAE."""
    clean = _clean_for_elo(df).dropna(subset=["games_this_season_before"])
    ra = clean["player_rating_before"].values.astype(float)
    rb = clean["opponent_rating_before"].values.astype(float)
    res = clean["result"].values.astype(float)
    obs = clean["observed_rating_delta"].values.astype(float)
    games = clean["games_this_season_before"].values

    buckets = [
        ("< 10 games (placement)",  games < 10),
        ("10–49 games",              (games >= 10) & (games < 50)),
        ("≥ 50 games (established)", games >= 50),
    ]

    K_range = np.arange(8.0, 65.0, 0.5)
    segments = {}

    print("\n=== K-Factor Segmentation ===")
    print(f"  {'Segment':<30}  {'N':>8}  {'Best K':>7}  {'MAE':>8}  {'R²':>7}")
    for label, mask in buckets:
        if mask.sum() < 10:
            continue
        best_k, best_mae_k = 32.0, np.inf
        for K in K_range:
            pred = K * (res[mask] - 1.0 / (1.0 + 10.0 ** ((rb[mask] - ra[mask]) / D)))
            mae = np.mean(np.abs(obs[mask] - pred))
            if mae < best_mae_k:
                best_mae_k = mae
                best_k = K
        pred_best = best_k * (res[mask] - 1.0 / (1.0 + 10.0 ** ((rb[mask] - ra[mask]) / D)))
        ss_res = np.sum((obs[mask] - pred_best) ** 2)
        ss_tot = np.sum((obs[mask] - obs[mask].mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  {label:<30}  {mask.sum():>8,}  {best_k:>7.1f}  {best_mae_k:>8.3f}  {r2:>7.4f}")
        segments[label] = {"K": best_k, "MAE": best_mae_k, "R2": r2, "n": int(mask.sum())}

    return segments


def compute_residuals(df: pd.DataFrame, K: float, D: float) -> pd.Series:
    """Return observed_rating_delta - Elo_predicted for each row (NaN where ratings missing)."""
    pred = elo_delta(
        df["player_rating_before"].values.astype(float),
        df["opponent_rating_before"].values.astype(float),
        df["result"].values.astype(float),
        K, D,
    )
    residuals = df["observed_rating_delta"].values.astype(float) - pred
    # Rows with missing ratings have NaN ratings → NaN in pred → NaN residual
    return pd.Series(residuals, index=df.index, name="elo_residual")


def analyze_residuals(df: pd.DataFrame, residuals: pd.Series) -> dict:
    """Compute correlations of Elo residuals with candidate variables."""
    analysis = df.copy()
    analysis["elo_residual"] = residuals

    valid = analysis.dropna(subset=["elo_residual"])

    print("\n=== Residual Analysis ===")
    print(f"  Residual mean:   {valid['elo_residual'].mean():>8.3f}")
    print(f"  Residual std:    {valid['elo_residual'].std():>8.3f}")
    print(f"  Residual median: {valid['elo_residual'].median():>8.3f}")

    correlates = {
        "visible_rating_gap": "visible_rating_gap",
        "games_this_season_before": "games_this_season_before",
        "games_lifetime_before": "games_lifetime_before",
        "player_mmr_before": "player_mmr_before",
        "hidden_mmr_gap": "hidden_mmr_gap",
        "current_streak": "current_streak",
        "recent_wr_10": "recent_wr_10",
    }

    corr_results = {}
    print("\n  Pearson correlation of residual with:")
    for label, col in correlates.items():
        if col in valid.columns:
            c = valid[["elo_residual", col]].dropna().corr().iloc[0, 1]
            print(f"    {label:<30}  {c:>7.4f}")
            corr_results[label] = c

    # Residual by season
    if "season" in valid.columns:
        print("\n  Mean residual by season:")
        for s, grp in valid.groupby("season"):
            print(f"    Season {s}:  mean={grp['elo_residual'].mean():>7.3f}  "
                  f"std={grp['elo_residual'].std():>7.3f}  n={len(grp):,}")

    return {
        "mean": valid["elo_residual"].mean(),
        "std": valid["elo_residual"].std(),
        "correlations": corr_results,
    }
