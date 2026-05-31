"""
Baseline models for comparison against LightGBM.

1. ConstantBaseline       — always predicts 0.5
2. MMRLogisticBaseline    — logistic regression on skill_diff only
3. CivMapBucketBaseline   — historical civ/map/matchup win rate with fallback hierarchy
4. EnhancedLogisticBaseline — logistic regression on skill features + civ/map priors
                              + caller-supplied extra features (e.g. top-N from LightGBM)
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ConstantBaseline:
    """Always predicts 0.5 (maximum-entropy baseline)."""

    name = "constant_0.5"

    def fit(self, df: pd.DataFrame, target_col: str = "target") -> "ConstantBaseline":
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), 0.5)


class MMRLogisticBaseline:
    """
    Logistic regression on skill_diff (MMR if available, else rating).
    Falls back to 0.5 when both are missing.
    """

    name = "mmr_logistic"

    def __init__(self):
        self._scaler = StandardScaler()
        self._model = LogisticRegression(max_iter=1000, C=1.0)
        self._global_mean_pred = 0.5

    def _skill_diff_array(self, df: pd.DataFrame) -> np.ndarray:
        diff = df["skill_diff"].copy()
        diff = diff.fillna(0.0)
        return diff.values.reshape(-1, 1)

    def fit(self, df: pd.DataFrame, target_col: str = "target") -> "MMRLogisticBaseline":
        X = self._skill_diff_array(df)
        y = df[target_col].values
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._skill_diff_array(df)
        X_scaled = self._scaler.transform(X)
        probs = self._model.predict_proba(X_scaled)[:, 1]
        # Where both players have no skill signal, revert to 0.5
        no_skill = (df["missing_skill_a"].values == 1) & (df["missing_skill_b"].values == 1)
        probs[no_skill] = 0.5
        return probs


class CivMapBucketBaseline:
    """
    Historical civ-matchup win rate with progressive fallback:
      civ_a + civ_b + map + patch  → prior_games ≥ min_n
      civ_a + civ_b + patch
      civ_a + civ_b  (all time in training data)
      civ_a global win rate
      0.5

    Only uses training data (no future leakage).
    """

    name = "civ_map_bucket"
    MIN_N = 20  # minimum games to trust a bucket

    def __init__(self):
        self._lookup: dict[tuple, tuple[float, int]] = {}  # key → (win_rate, games)
        self._global_wr = 0.5

    def fit(self, df: pd.DataFrame, target_col: str = "target") -> "CivMapBucketBaseline":
        t = df[target_col]

        def agg(group_cols):
            return (
                df.dropna(subset=group_cols)
                .groupby(group_cols)[target_col]
                .agg(["mean", "count"])
                .rename(columns={"mean": "wr", "count": "n"})
            )

        # Build lookup tables for each fallback level
        self._lookup4 = agg(["civ_a", "civ_b", "map", "patch"])
        self._lookup3 = agg(["civ_a", "civ_b", "patch"])
        self._lookup2 = agg(["civ_a", "civ_b"])
        self._lookup1 = agg(["civ_a"])
        self._global_wr = float(t.mean())
        return self

    def _get_wr(self, lookup, key) -> tuple[float, int] | None:
        try:
            row = lookup.loc[key]
            if row["n"] >= self.MIN_N:
                return float(row["wr"]), int(row["n"])
        except KeyError:
            pass
        return None

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), 0.5)
        for i, row in enumerate(df.itertuples(index=False)):
            civ_a = getattr(row, "civ_a", None)
            civ_b = getattr(row, "civ_b", None)
            map_ = getattr(row, "map", None)
            patch = getattr(row, "patch", None)

            # Level 4: civ + map + patch
            if civ_a and civ_b and map_ and patch:
                res = self._get_wr(self._lookup4, (civ_a, civ_b, map_, patch))
                if res:
                    out[i] = res[0]
                    continue

            # Level 3: civ + patch
            if civ_a and civ_b and patch:
                res = self._get_wr(self._lookup3, (civ_a, civ_b, patch))
                if res:
                    out[i] = res[0]
                    continue

            # Level 2: civ all-time
            if civ_a and civ_b:
                res = self._get_wr(self._lookup2, (civ_a, civ_b))
                if res:
                    out[i] = res[0]
                    continue

            # Level 1: civ_a global win rate
            if civ_a:
                res = self._get_wr(self._lookup1, (civ_a,))
                if res:
                    out[i] = res[0]
                    continue

            out[i] = self._global_wr
        return out


class EnhancedLogisticBaseline:
    """
    Logistic regression combining:
      - skill_diff (MMR → rating fallback)
      - prior_matchup_wr_a (civ matchup prior from previous seasons)
      - civ_wr_a, civ_wr_b (player-specific civ win rates, leakage-free)
      - map_wr_a, map_wr_b (player-specific map win rates, leakage-free)
      - overall_wr_a, overall_wr_b (lifetime win rates, leakage-free)
      - extra_features: caller-supplied list (e.g. top-N numeric LightGBM features)

    All features are already leakage-free (computed from window functions over prior games).
    Missing values are filled with 0 before scaling.
    """

    name = "enhanced_logistic"

    _BASE_FEATURES = [
        "skill_diff",
        "prior_matchup_wr_a",
        "civ_wr_a",
        "civ_wr_b",
        "map_wr_a",
        "map_wr_b",
        "overall_wr_a",
        "overall_wr_b",
        "wr_diff",
        "missing_skill_a",
        "missing_skill_b",
    ]

    def __init__(self, extra_features: list[str] | None = None):
        self.extra_features = [f for f in (extra_features or []) if f not in self._BASE_FEATURES]
        self._feature_cols: list[str] = []
        self._scaler = StandardScaler()
        self._model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")

    def _build_X(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in (self._BASE_FEATURES + self.extra_features) if c in df.columns]
        self._feature_cols = cols
        return df[cols].fillna(0.0).values.astype(float)

    def fit(self, df: pd.DataFrame, target_col: str = "target") -> "EnhancedLogisticBaseline":
        X = self._build_X(df)
        y = df[target_col].values
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self._feature_cols if c in df.columns]
        X = df[cols].fillna(0.0).values.astype(float)
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)[:, 1]

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": self._feature_cols, "coef": self._model.coef_[0]}
        ).sort_values("coef", key=abs, ascending=False)


class L1LogisticBaseline:
    """L1-penalized logistic regression on numeric features only. C tuned on validation Brier."""

    name = "L1 Logistic (numeric only)"

    def _numeric_cols(self, df: pd.DataFrame) -> list[str]:
        # Lazy import to avoid circular dependency at module load
        from .model import NUMERIC_FEATURES
        from .features_extra import ALL_EXTRA_FEATURES, P6_CATEGORICAL_FEATURES
        candidates = NUMERIC_FEATURES + [
            c for c in ALL_EXTRA_FEATURES if c not in P6_CATEGORICAL_FEATURES
        ]
        return [c for c in candidates if c in df.columns]

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        target_col: str = "target",
        max_train_rows: int = 500_000,
    ) -> "L1LogisticBaseline":
        cols = self._numeric_cols(train_df)
        self._cols = cols

        tr = train_df.sample(min(max_train_rows, len(train_df)), random_state=42) if len(train_df) > max_train_rows else train_df

        X_tr = tr[cols].to_numpy(dtype=float, na_value=np.nan)
        y_tr = tr[target_col].values
        X_va = valid_df[cols].to_numpy(dtype=float, na_value=np.nan)
        y_va = valid_df[target_col].values

        self._imp = SimpleImputer(strategy="median").fit(X_tr)
        X_tr = self._imp.transform(X_tr)
        X_va = self._imp.transform(X_va)

        self._scl = StandardScaler().fit(X_tr)
        X_tr = self._scl.transform(X_tr)
        X_va = self._scl.transform(X_va)

        best_C, best_brier, best_model = None, float("inf"), None
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            m = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=500)
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_va)[:, 1]
            b = float(np.mean((p - y_va) ** 2))
            if b < best_brier:
                best_brier, best_C, best_model = b, C, m

        self._model = best_model
        print(f"  L1 Logistic best C={best_C} (val Brier={best_brier:.4f}, {len(cols)} features)")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._scl.transform(self._imp.transform(df[self._cols].to_numpy(dtype=float, na_value=np.nan)))
        return self._model.predict_proba(X)[:, 1]
