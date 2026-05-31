"""Simple baseline models for rating delta prediction."""
import numpy as np
import pandas as pd

from .formula import elo_delta


class ResultOnlyBaseline:
    """Predict mean rating delta conditioned on win/loss."""

    def __init__(self):
        self._mean_win: float = 0.0
        self._mean_loss: float = 0.0

    def fit(self, df: pd.DataFrame) -> "ResultOnlyBaseline":
        self._mean_win = df.loc[df["result"] == 1, "observed_rating_delta"].mean()
        self._mean_loss = df.loc[df["result"] == 0, "observed_rating_delta"].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.where(df["result"].values == 1, self._mean_win, self._mean_loss)

    def __repr__(self) -> str:
        return f"ResultOnlyBaseline(win={self._mean_win:.2f}, loss={self._mean_loss:.2f})"


class EloBaseline:
    """Elo formula with fixed K and D."""

    def __init__(self, K: float = 32.0, D: float = 400.0):
        self.K = K
        self.D = D

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return elo_delta(
            df["player_rating_before"].values.astype(float),
            df["opponent_rating_before"].values.astype(float),
            df["result"].values.astype(float),
            self.K,
            self.D,
        )

    def __repr__(self) -> str:
        return f"EloBaseline(K={self.K}, D={self.D})"


class ResultRatingBucketBaseline:
    """Predict mean delta per (rating-gap bucket × result)."""

    BINS = [-np.inf, -300, -200, -100, -50, 50, 100, 200, 300, np.inf]
    LABELS = ["<-300", "-300:-200", "-200:-100", "-100:-50",
              "-50:50", "50:100", "100:200", "200:300", ">300"]

    def __init__(self):
        self._table: dict[tuple, float] = {}
        self._fallback: dict[int, float] = {}

    def _bucket(self, gap: np.ndarray) -> np.ndarray:
        return pd.cut(
            gap, bins=self.BINS, labels=self.LABELS, right=False
        ).astype(str)

    def fit(self, df: pd.DataFrame) -> "ResultRatingBucketBaseline":
        clean = df.dropna(subset=["visible_rating_gap", "observed_rating_delta"])
        clean = clean.copy()
        clean["_bucket"] = self._bucket(clean["visible_rating_gap"].values)
        for (bucket, result), grp in clean.groupby(["_bucket", "result"]):
            self._table[(str(bucket), int(result))] = grp["observed_rating_delta"].mean()
        # Fallback: result-only mean
        for result in [0, 1]:
            sub = clean[clean["result"] == result]["observed_rating_delta"]
            self._fallback[int(result)] = sub.mean() if len(sub) else 0.0
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        buckets = self._bucket(df["visible_rating_gap"].fillna(0).values)
        results = df["result"].values.astype(int)
        return np.array([
            self._table.get((str(b), int(r)), self._fallback.get(int(r), 0.0))
            for b, r in zip(buckets, results)
        ])

    def __repr__(self) -> str:
        return f"ResultRatingBucketBaseline({len(self._table)} cells)"
