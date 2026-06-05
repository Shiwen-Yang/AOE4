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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


class EnhancedL1LogisticBaseline:
    """
    L1-penalized logistic regression with optional OHE categoricals,
    compound categorical OHE (e.g. civ_a × map), polynomial pairwise
    interactions on a chosen numeric subset, and named domain interaction
    features. C is tuned on validation Brier score.
    """

    name = "Enhanced L1 Logistic"

    _DEFAULT_OHE_COLS = ["civ_a", "civ_b", "map", "patch", "season"]
    _DEFAULT_C_GRID   = [0.001, 0.01, 0.1, 1.0, 10.0]

    # (output_name, col1, col2) → product col1 * col2 after imputation.
    # "civ_wr_diff" and "map_wr_diff" are precomputed; the rest are raw columns.
    _DEFAULT_INTERACTIONS: list[tuple[str, str, str]] = [
        ("ix_skill_x_civwr_diff", "skill_diff",     "civ_wr_diff"),
        ("ix_skill_x_mapwr_diff", "skill_diff",      "map_wr_diff"),
        ("ix_skill_x_wr_diff",    "skill_diff",      "wr_diff"),
        ("ix_skill_x_matchup",    "skill_diff",      "prior_matchup_wr_a"),
        ("ix_civwr_x_matchup",    "civ_wr_diff",     "prior_matchup_wr_a"),
        ("ix_new_a_x_skill",      "is_new_player_a", "skill_diff"),
        ("ix_new_b_x_skill",      "is_new_player_b", "skill_diff"),
        ("ix_skill_x_form10",     "skill_diff",      "recent_wr_10_diff"),  # P3
        ("ix_skill_x_mmrslope",   "skill_diff",      "mmr_slope_10_diff"),  # P2
        ("ix_skill_x_h2h",        "skill_diff",      "h2h_wr_a"),           # P5
    ]

    def __init__(
        self,
        include_ohe: bool = True,
        include_interactions: bool = True,
        ohe_cols: list[str] | None = None,
        compound_ohe_pairs: list[tuple[str, str]] | None = None,
        interaction_specs: list[tuple[str, str, str]] | None = None,
        poly_features_on: list[str] | None = None,
        drop_features: list[str] | None = None,
        c_grid: list[float] | None = None,
        max_train_rows: int = 500_000,
    ):
        self.include_ohe          = include_ohe
        self.include_interactions = include_interactions
        self._ohe_cols_arg        = ohe_cols           # None → default + P6
        self.compound_ohe_pairs   = list(compound_ohe_pairs or [])
        self._interaction_specs   = interaction_specs  # None → _DEFAULT_INTERACTIONS
        self.poly_features_on     = list(poly_features_on or [])
        self._drop_features       = set(drop_features or [])
        self.c_grid               = list(c_grid or self._DEFAULT_C_GRID)
        self.max_train_rows       = max_train_rows
        self._compound_encoders: dict[tuple, OneHotEncoder] = {}

    # ── feature column helpers ─────────────────────────────────────────────────

    def _numeric_cols(self, df: pd.DataFrame) -> list[str]:
        from .model import NUMERIC_FEATURES
        from .features_extra import ALL_EXTRA_FEATURES, P6_CATEGORICAL_FEATURES
        candidates = NUMERIC_FEATURES + [
            c for c in ALL_EXTRA_FEATURES if c not in P6_CATEGORICAL_FEATURES
        ]
        return [c for c in candidates if c in df.columns and c not in self._drop_features]

    def _resolve_ohe_cols(self, df: pd.DataFrame) -> list[str]:
        from .features_extra import P6_CATEGORICAL_FEATURES
        base = self._ohe_cols_arg if self._ohe_cols_arg is not None else (
            self._DEFAULT_OHE_COLS + P6_CATEGORICAL_FEATURES
        )
        return [c for c in base if c in df.columns]

    # ── column dict (numeric names → imputed arrays + precomputed diffs) ───────

    @staticmethod
    def _make_col_dict(X_num: np.ndarray, num_cols: list[str]) -> dict[str, np.ndarray]:
        d: dict[str, np.ndarray] = {col: X_num[:, i] for i, col in enumerate(num_cols)}
        for a, b, out in [("civ_wr_a", "civ_wr_b", "civ_wr_diff"),
                          ("map_wr_a", "map_wr_b", "map_wr_diff")]:
            if a in d and b in d:
                d[out] = d[a] - d[b]
        return d

    # ── sub-block builders ─────────────────────────────────────────────────────

    def _numeric_block(self, df: pd.DataFrame, num_cols: list[str], fit: bool) -> np.ndarray:
        X = df[num_cols].to_numpy(dtype=float, na_value=np.nan)
        if fit:
            self._imp = SimpleImputer(strategy="median").fit(X)
        return self._imp.transform(X)

    def _ohe_block(
        self, df: pd.DataFrame, fit: bool
    ) -> tuple[np.ndarray | None, list[str]]:
        ohe_cols = self._resolve_ohe_cols(df) if fit else self._ohe_cols_fitted
        if not ohe_cols:
            if fit:
                self._ohe_cols_fitted: list[str] = []
            return None, []
        X_cat = df[ohe_cols].astype(str).fillna("__missing__")
        if fit:
            self._ohe_cols_fitted = ohe_cols
            self._ohe = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, drop="first"
            ).fit(X_cat)
        X_out = self._ohe.transform(
            X_cat.reindex(columns=self._ohe_cols_fitted, fill_value="__missing__")
        )
        return X_out, list(self._ohe.get_feature_names_out(self._ohe_cols_fitted))

    def _compound_block(
        self, df: pd.DataFrame, fit: bool
    ) -> tuple[np.ndarray | None, list[str]]:
        blocks: list[np.ndarray] = []
        names:  list[str]        = []
        for pair in self.compound_ohe_pairs:
            col_a, col_b = pair
            if col_a not in df.columns or col_b not in df.columns:
                continue
            compound = (df[col_a].astype(str) + "_x_" + df[col_b].astype(str)).fillna("__missing__")
            X_in = compound.values.reshape(-1, 1)
            if fit:
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
                enc.fit(X_in)
                self._compound_encoders[pair] = enc
            elif pair not in self._compound_encoders:
                continue
            enc = self._compound_encoders[pair]
            X_out = enc.transform(X_in)
            col_names = [f"{col_a}_x_{col_b}={v}" for v in enc.categories_[0][1:]]
            blocks.append(X_out)
            names.extend(col_names)
        if not blocks:
            return None, []
        return np.hstack(blocks), names

    def _poly_block(
        self, col_dict: dict[str, np.ndarray], fit: bool
    ) -> tuple[np.ndarray | None, list[str]]:
        if not self.poly_features_on:
            return None, []
        from sklearn.preprocessing import PolynomialFeatures
        if fit:
            cols_present = [c for c in self.poly_features_on if c in col_dict]
            if len(cols_present) < 2:
                self._poly_input_cols: list[str] = []
                return None, []
            self._poly_input_cols = cols_present
            X_in = np.column_stack([col_dict[c] for c in cols_present])
            self._poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            self._poly.fit(X_in)
        else:
            if not hasattr(self, "_poly") or not getattr(self, "_poly_input_cols", []):
                return None, []
            n = next(iter(col_dict.values())).shape[0]
            X_in = np.column_stack([
                col_dict.get(c, np.zeros(n)) for c in self._poly_input_cols
            ])
        X_out = self._poly.transform(X_in)
        names = [
            f"poly_{self._poly_input_cols[i]}_x_{self._poly_input_cols[j]}"
            for i in range(len(self._poly_input_cols))
            for j in range(i + 1, len(self._poly_input_cols))
        ]
        return X_out, names

    def _interaction_block(
        self, col_dict: dict[str, np.ndarray]
    ) -> tuple[np.ndarray | None, list[str]]:
        specs = self._interaction_specs if self._interaction_specs is not None else self._DEFAULT_INTERACTIONS
        blocks: list[np.ndarray] = []
        names:  list[str]        = []
        for (out_name, col1, col2) in specs:
            if col1 in col_dict and col2 in col_dict:
                blocks.append((col_dict[col1] * col_dict[col2]).reshape(-1, 1))
                names.append(out_name)
        if not blocks:
            return None, []
        return np.hstack(blocks), names

    # ── main feature assembly ──────────────────────────────────────────────────

    def _build_X(
        self, df: pd.DataFrame, fit: bool = False
    ) -> tuple[np.ndarray, list[str]]:
        num_cols = self._numeric_cols(df) if fit else self._num_cols_fitted
        if fit:
            self._num_cols_fitted = num_cols

        all_blocks: list[np.ndarray] = []
        all_names:  list[str]        = []

        X_num = self._numeric_block(df, num_cols, fit=fit)
        all_blocks.append(X_num)
        all_names.extend(num_cols)

        col_dict = self._make_col_dict(X_num, num_cols)

        if self.include_ohe:
            X_ohe, ohe_names = self._ohe_block(df, fit=fit)
            if X_ohe is not None:
                all_blocks.append(X_ohe)
                all_names.extend(ohe_names)

        if self.compound_ohe_pairs:
            X_comp, comp_names = self._compound_block(df, fit=fit)
            if X_comp is not None:
                all_blocks.append(X_comp)
                all_names.extend(comp_names)

        if self.poly_features_on:
            X_poly, poly_names = self._poly_block(col_dict, fit=fit)
            if X_poly is not None:
                all_blocks.append(X_poly)
                all_names.extend(poly_names)

        if self.include_interactions:
            X_ix, ix_names = self._interaction_block(col_dict)
            if X_ix is not None:
                all_blocks.append(X_ix)
                all_names.extend(ix_names)

        X_full = np.hstack(all_blocks)

        if fit:
            self._scl = StandardScaler().fit(X_full)
            self._all_feature_names = all_names

        return self._scl.transform(X_full), all_names

    # ── public API ─────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        target_col: str = "target",
    ) -> "EnhancedL1LogisticBaseline":
        tr = (
            train_df.sample(min(self.max_train_rows, len(train_df)), random_state=42)
            if len(train_df) > self.max_train_rows else train_df
        )

        X_tr, _ = self._build_X(tr, fit=True)
        y_tr = tr[target_col].values
        X_va, _ = self._build_X(valid_df, fit=False)
        y_va = valid_df[target_col].values

        best_C, best_brier, best_model = None, float("inf"), None
        for C in self.c_grid:
            m = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=1000, tol=1e-4)
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_va)[:, 1]
            b = float(np.mean((p - y_va) ** 2))
            if b < best_brier:
                best_brier, best_C, best_model = b, C, m

        self._model        = best_model
        self.best_C        = best_C
        self.best_val_brier = best_brier
        n_nz = int(np.sum(best_model.coef_[0] != 0))
        print(
            f"  best C={best_C} | val Brier={best_brier:.4f} | "
            f"{len(self._all_feature_names)} features | {n_nz} nonzero"
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self._build_X(df, fit=False)
        return self._model.predict_proba(X)[:, 1]

    def coef_table(self) -> pd.DataFrame:
        coefs = self._model.coef_[0]
        return (
            pd.DataFrame({
                "feature":  self._all_feature_names,
                "coef":     coefs,
                "abs_coef": np.abs(coefs),
            })
            .sort_values("abs_coef", ascending=False)
            .reset_index(drop=True)
        )

    def zeroed_features(self) -> list[str]:
        coefs = self._model.coef_[0]
        return [f for f, c in zip(self._all_feature_names, coefs) if c == 0.0]

    def selected_features(self) -> list[str]:
        coefs = self._model.coef_[0]
        return [f for f, c in zip(self._all_feature_names, coefs) if c != 0.0]

    def feature_breakdown(self) -> dict[str, int]:
        """Return feature count by block type."""
        num_set = set(self._num_cols_fitted)
        counts: dict[str, int] = {"numeric": 0, "ohe": 0, "compound_ohe": 0,
                                   "poly": 0, "interaction": 0}
        for f in self._all_feature_names:
            if f in num_set:
                counts["numeric"] += 1
            elif f.startswith("poly_"):
                counts["poly"] += 1
            elif f.startswith("ix_"):
                counts["interaction"] += 1
            elif any(f.startswith(f"{a}_x_{b}=") for a, b in self.compound_ohe_pairs):
                counts["compound_ohe"] += 1
            else:
                counts["ohe"] += 1
        return counts

    def ohe_summary(self, top_n: int = 5) -> dict[str, list[tuple[str, float]]]:
        """Top non-zero OHE entries per categorical column group."""
        ct = self.coef_table()
        ct = ct[ct["coef"] != 0].copy()
        result: dict[str, list[tuple[str, float]]] = {}
        prefixes = list(getattr(self, "_ohe_cols_fitted", []))
        for a, b in self.compound_ohe_pairs:
            prefixes.append(f"{a}_x_{b}")
        for prefix in prefixes:
            mask = ct["feature"].str.startswith(prefix + "_") | ct["feature"].str.startswith(prefix + "=")
            sub = ct[mask].head(top_n)
            if not sub.empty:
                result[prefix] = list(zip(sub["feature"].tolist(), sub["coef"].tolist()))
        return result
