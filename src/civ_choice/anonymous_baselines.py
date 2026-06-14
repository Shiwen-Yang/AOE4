"""Baselines for anonymous opponent civ-choice prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rates_from_train(df: pd.DataFrame, group_cols: list[str]) -> dict[tuple, float]:
    chosen = df[df["target"] == 1]
    counts = chosen.groupby(group_cols + ["candidate_civ"], observed=True).size()
    totals = chosen.groupby(group_cols, observed=True).size()
    rates: dict[tuple, float] = {}
    for key, count in counts.items():
        if not isinstance(key, tuple):
            key = (key,)
        group_key = key[:-1]
        total_key = group_key[0] if len(group_key) == 1 else group_key
        rates[key] = float(count / totals.loc[total_key])
    return rates


class AnonymousGlobalPickRateBaseline:
    """P(civ) from training data."""

    def __init__(self) -> None:
        self._rates: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "AnonymousGlobalPickRateBaseline":
        counts = df[df["target"] == 1]["candidate_civ"].value_counts(normalize=True)
        self._rates = counts.to_dict()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self._rates.get(c, 1e-6) for c in df["candidate_civ"]])


class MMRTierPickRateBaseline:
    """P(civ | mmr_bucket), with global fallback."""

    def __init__(self) -> None:
        self._global = AnonymousGlobalPickRateBaseline()
        self._rates: dict[tuple, float] = {}

    def fit(self, df: pd.DataFrame) -> "MMRTierPickRateBaseline":
        self._global.fit(df)
        self._rates = _rates_from_train(df, ["mmr_bucket"])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        global_raw = self._global.predict_proba(df)
        raw = []
        for fallback, row in zip(global_raw, df.itertuples(index=False)):
            raw.append(self._rates.get((row.mmr_bucket, row.candidate_civ), fallback))
        return np.asarray(raw)


class MapPatchPickRateBaseline:
    """P(civ | map, patch), fallback to P(civ | map), then global."""

    def __init__(self) -> None:
        self._global = AnonymousGlobalPickRateBaseline()
        self._map_rates: dict[tuple, float] = {}
        self._map_patch_rates: dict[tuple, float] = {}

    def fit(self, df: pd.DataFrame) -> "MapPatchPickRateBaseline":
        self._global.fit(df)
        self._map_rates = _rates_from_train(df, ["map"])
        self._map_patch_rates = _rates_from_train(df, ["map", "patch"])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        global_raw = self._global.predict_proba(df)
        raw = []
        for fallback, row in zip(global_raw, df.itertuples(index=False)):
            key = (row.map, row.patch, row.candidate_civ)
            map_key = (row.map, row.candidate_civ)
            raw.append(self._map_patch_rates.get(key, self._map_rates.get(map_key, fallback)))
        return np.asarray(raw)


class RecentOpponentMetaBaseline:
    """Use known user's recent opponent civ distribution when available."""

    def __init__(self, column: str = "cand_user_recent_opp_pr_30") -> None:
        self.column = column
        self._mmr = MMRTierPickRateBaseline()

    def fit(self, df: pd.DataFrame) -> "RecentOpponentMetaBaseline":
        self._mmr.fit(df)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        fallback = self._mmr.predict_proba(df)
        if self.column not in df.columns:
            return fallback
        raw = df[self.column].astype(float).fillna(0.0).to_numpy()
        return np.where(raw > 0, raw, fallback)
