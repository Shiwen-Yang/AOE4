"""
Baselines that frame the matchmaking-quality question.

- ConstantBaseline: predicts the base rate (≈0.5 after team-swap symmetry). The floor.
- MMRMeanDiffLogistic: a one-feature logistic on the Team A-vs-B mean-MMR gap. This is
  the crux of the investigation — if the raw skill gap alone predicts outcomes well, the
  matchmaker is handing out decided games. LightGBM's lift over this measures how much
  *additional* (non-mean-MMR) structure exists (carry stacks, 1v1 smurfs, etc.).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import MMR_DIFF_FEATURE


class ConstantBaseline:
    def fit(self, df: pd.DataFrame, target_col: str = "target"):
        self.p_ = float(df[target_col].mean())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.p_)


class MMRMeanDiffLogistic:
    def __init__(self, feature: str = MMR_DIFF_FEATURE):
        self.feature = feature
        self.model = LogisticRegression()

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        return df[[self.feature]].fillna(0.0).to_numpy()

    def fit(self, df: pd.DataFrame, target_col: str = "target"):
        self.model.fit(self._x(df), df[target_col].values)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._x(df))[:, 1]
