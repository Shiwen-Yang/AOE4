"""Five civ-choice baselines, all returning probability distributions."""
import numpy as np
import pandas as pd


def _normalize(d: dict) -> dict:
    total = sum(d.values())
    if total == 0:
        n = len(d)
        return {k: 1.0 / n for k in d}
    return {k: v / total for k, v in d.items()}


class GlobalPickRateBaseline:
    """P(civ) = global pick share in training data."""

    def __init__(self):
        self._rates: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "GlobalPickRateBaseline":
        counts = df[df["target"] == 1]["candidate_civ"].value_counts()
        self._rates = _normalize(counts.to_dict())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability for each candidate row."""
        return np.array([self._rates.get(c, 1e-6) for c in df["candidate_civ"]])

    def __repr__(self) -> str:
        return "GlobalPickRateBaseline()"


class LifetimeFreqBaseline:
    """P(civ | player) = smoothed lifetime pick share; fallback to global."""

    def __init__(self, smooth: float = 2.0):
        self._smooth = smooth
        self._global: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "LifetimeFreqBaseline":
        counts = df[df["target"] == 1]["candidate_civ"].value_counts(normalize=True)
        self._global = counts.to_dict()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        g_rate = np.array([self._global.get(c, 1e-6) for c in df["candidate_civ"]])
        # Smooth player lifetime share toward global
        lt_share = df["cand_pick_share_lifetime"].fillna(0).values
        preds = (lt_share + self._smooth * g_rate) / (1 + self._smooth)
        return preds

    def __repr__(self) -> str:
        return f"LifetimeFreqBaseline(smooth={self._smooth})"


class Recent30dBaseline:
    """P(civ | player) = 30d pick share; fallback: lifetime → global."""

    def __init__(self, smooth: float = 2.0):
        self._smooth = smooth
        self._global: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "Recent30dBaseline":
        counts = df[df["target"] == 1]["candidate_civ"].value_counts(normalize=True)
        self._global = counts.to_dict()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        g_rate = np.array([self._global.get(c, 1e-6) for c in df["candidate_civ"]])
        share_30 = df["cand_pick_share_30d"].fillna(0).values
        share_lt = df["cand_pick_share_lifetime"].fillna(0).values
        # Prefer 30d if player has 30d history, else fall back
        has_30d = df["player_games_30d"].fillna(0).values > 0
        share = np.where(has_30d, share_30, share_lt)
        preds = (share + self._smooth * g_rate) / (1 + self._smooth)
        return preds

    def __repr__(self) -> str:
        return f"Recent30dBaseline(smooth={self._smooth})"


class MapSpecificBaseline:
    """P(civ | player, map) = map pick share; fallback chain: map → 30d → lifetime → global."""

    def __init__(self, smooth: float = 2.0):
        self._smooth = smooth
        self._global: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "MapSpecificBaseline":
        counts = df[df["target"] == 1]["candidate_civ"].value_counts(normalize=True)
        self._global = counts.to_dict()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        g_rate = np.array([self._global.get(c, 1e-6) for c in df["candidate_civ"]])
        share_map = df["cand_pick_share_this_map"].fillna(0).values
        share_30 = df["cand_pick_share_30d"].fillna(0).values
        share_lt = df["cand_pick_share_lifetime"].fillna(0).values
        has_map = df["player_games_this_map"].fillna(0).values >= 3
        has_30d = df["player_games_30d"].fillna(0).values > 0
        share = np.where(has_map, share_map, np.where(has_30d, share_30, share_lt))
        preds = (share + self._smooth * g_rate) / (1 + self._smooth)
        return preds

    def __repr__(self) -> str:
        return f"MapSpecificBaseline(smooth={self._smooth})"


class LastCivBaseline:
    """Strong baseline: 0.60 on last civ, 0.40 over recent/lifetime shares.

    For Top-1 hard-label evaluation: predict prev_civ.
    For log-loss: smoothed distribution with mass on last civ.
    """

    LAST_CIV_WEIGHT = 0.60

    def __init__(self):
        self._global: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "LastCivBaseline":
        counts = df[df["target"] == 1]["candidate_civ"].value_counts(normalize=True)
        self._global = counts.to_dict()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        is_last = df["candidate_is_last_civ"].fillna(0).values.astype(float)
        g_rate = np.array([self._global.get(c, 1e-6) for c in df["candidate_civ"]])
        share_lt = df["cand_pick_share_lifetime"].fillna(0).values
        # 60% mass on last civ candidate, 40% split by lifetime share
        return self.LAST_CIV_WEIGHT * is_last + (1 - self.LAST_CIV_WEIGHT) * share_lt

    def __repr__(self) -> str:
        return f"LastCivBaseline(last_civ_weight={self.LAST_CIV_WEIGHT})"


def normalize_within_group(df: pd.DataFrame, raw_pred: np.ndarray) -> np.ndarray:
    """Normalize raw predictions within each (game_id, profile_id) group."""
    df = df.copy()
    df["_raw"] = raw_pred
    df["_norm"] = df.groupby(["game_id", "profile_id"])["_raw"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else x / len(x)
    )
    return df["_norm"].values
