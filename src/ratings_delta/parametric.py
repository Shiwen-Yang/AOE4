"""Nested parametric models P0–P4 for AOE4 visible rating delta prediction.

Implements the nested sequence from the improvement plan:
  P0 — K(y − p_MMR), single K and D
  P1 — β₀ + K_win*(1−p) for wins; β₀ + K_loss*(−p) for losses
  P2 — P1 + per-experience-bucket K_win_j/K_loss_j + piecewise b(g) + γI(g_o<10)
  P3 — P2 + additive missing-MMR indicator corrections
  P4 — P3 + linear rating-minus-MMR reconciliation terms h(d_p) + h(d_o)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .formula import elo_expected

# scipy is only needed at fit time; importing lazily keeps prediction-only
# consumers (e.g. the backend serving a saved P3 model) free of the dependency.


def _lstsq(X, y):
    from scipy.linalg import lstsq
    return lstsq(X, y)


def _curve_fit(*args, **kwargs):
    from scipy.optimize import curve_fit
    return curve_fit(*args, **kwargs)


# ── shared helpers ────────────────────────────────────────────────────────────

def _ra_rb(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """MMR with visible-rating fallback for both players."""
    def _f(c): return df[c].to_numpy(dtype=float, na_value=np.nan)
    ra_m, rb_m = _f("player_mmr_before"), _f("opponent_mmr_before")
    ra_r, rb_r = _f("player_rating_before"), _f("opponent_rating_before")
    return np.where(np.isnan(ra_m), ra_r, ra_m), np.where(np.isnan(rb_m), rb_r, rb_m)


def _obs(df: pd.DataFrame) -> np.ndarray:
    return df["observed_rating_delta"].to_numpy(dtype=float, na_value=np.nan)


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """Standard evaluation metrics, ignoring NaN rows."""
    v = ~np.isnan(obs) & ~np.isnan(pred)
    r = obs[v] - pred[v]
    ss_r = float(np.sum(r ** 2))
    ss_t = float(np.sum((obs[v] - obs[v].mean()) ** 2))
    return dict(
        mae=float(np.mean(np.abs(r))),
        median_ae=float(np.median(np.abs(r))),
        rmse=float(np.sqrt(np.mean(r ** 2))),
        mean_signed=float(np.mean(r)),
        r2=float(1 - ss_r / ss_t) if ss_t > 0 else float("nan"),
        n=int(v.sum()),
    )


# ── piecewise intercept helpers ───────────────────────────────────────────────

def _seg_mean(g: np.ndarray, r: np.ndarray, lo: float, hi):
    """Per-game mean residual and sqrt-n weights for segment [lo, hi)."""
    mask = (g >= lo) if hi is None else (g >= lo) & (g < hi)
    if mask.sum() == 0:
        return np.array([]), np.array([]), np.array([])
    agg = (pd.DataFrame({"g": g[mask].astype(int), "r": r[mask]})
           .groupby("g")["r"].agg(["mean", "count"]).reset_index())
    agg.columns = ["g", "mean", "n"]
    return (agg["g"].values.astype(float),
            agg["mean"].values.astype(float),
            np.sqrt(agg["n"].values.astype(float)))


def _fit_linear(x, y, w):
    return np.polyfit(x, y, 1, w=w) if len(x) >= 2 else np.array([0.0, float(np.average(y, weights=w))])


def _fit_exp(x, y, w):
    """Fit A·exp(−λ·(n − x[0])) + C; returns (popt, x0) or (None, x0)."""
    x0 = float(x[0])
    C0 = float(np.average(y[-max(1, len(y) // 5):], weights=w[-max(1, len(y) // 5):]))
    A0 = float(y[0]) - C0

    def f(n, A, lam, C):
        return A * np.exp(-lam * (n - x0)) + C

    try:
        popt, _ = _curve_fit(f, x, y, p0=[A0, 0.05, C0],
                            sigma=1.0 / (w + 1e-6),
                            bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
                            maxfev=8000)
        return popt, x0
    except Exception:
        return None, x0


def _intercept_eval(n: np.ndarray, pieces: list) -> np.ndarray:
    """Evaluate piecewise intercept at each value in n."""
    out = np.zeros(len(n))
    for s, e, kind, params in pieces:
        mask = (n >= s) if e is None else (n >= s) & (n < e)
        if not mask.any():
            continue
        if kind == "linear":
            out[mask] = np.polyval(params, n[mask])
        elif kind == "constant":
            out[mask] = float(params)
        else:  # exp
            popt, x0 = params
            A, lam, C = popt
            out[mask] = A * np.exp(-lam * (n[mask] - x0)) + C
    neg = n < 0
    if neg.any() and pieces:
        s0, _, k0, p0 = pieces[0]
        if k0 == "linear":
            out[neg] = float(np.polyval(p0, 0.0))
        elif k0 == "constant":
            out[neg] = float(p0)
        else:
            popt, x0 = p0
            A, lam, C = popt
            out[neg] = float(A * np.exp(-lam * (0.0 - x0)) + C)
    return out


def fit_piecewise_b(g: np.ndarray, resid: np.ndarray,
                    changepoints: list[int] | None = None) -> list:
    """Fit piecewise intercept b(g) on per-game mean residuals."""
    if changepoints is None:
        changepoints = [10, 49, 50]

    pieces = []

    # [0, 10): single linear through per-game means (n=0..9)
    x, y, w = _seg_mean(g, resid, 0, 10)
    if len(x) == 0:
        pieces.append((0, 10, "constant", 0.0))
    else:
        pieces.append((0, 10, "linear", _fit_linear(x, y, w)))

    # [10, 49): test exponential
    x, y, w = _seg_mean(g, resid, 10, 49)
    if len(x) >= 4:
        popt, x0 = _fit_exp(x, y, w)
        pieces.append((10, 49, "exp", (popt, x0)) if popt is not None
                      else (10, 49, "linear", _fit_linear(x, y, w)))
    elif len(x) > 0:
        pieces.append((10, 49, "linear", _fit_linear(x, y, w)))

    # [49, 50): constant
    x, y, w = _seg_mean(g, resid, 49, 50)
    if len(x) > 0:
        pieces.append((49, 50, "constant", float(np.average(y, weights=w))))

    # [50, ∞): test exponential
    x, y, w = _seg_mean(g, resid, 50, None)
    if len(x) >= 4:
        popt, x0 = _fit_exp(x, y, w)
        pieces.append((50, None, "exp", (popt, x0)) if popt is not None
                      else (50, None, "linear", _fit_linear(x, y, w)))
    elif len(x) > 0:
        pieces.append((50, None, "linear", _fit_linear(x, y, w)))

    return pieces


def _pieces_to_list(pieces: list) -> list:
    out = []
    for s, e, kind, params in pieces:
        if kind == "constant":
            out.append({"s": s, "e": e, "kind": kind, "params": float(params)})
        elif kind == "linear":
            out.append({"s": s, "e": e, "kind": kind, "params": list(map(float, params))})
        else:  # exp
            popt, x0 = params
            A, lam, C = popt
            out.append({"s": s, "e": e, "kind": kind,
                        "params": {"A": float(A), "lam": float(lam),
                                   "C": float(C), "x0": float(x0)}})
    return out


def _list_to_pieces(lst: list) -> list:
    pieces = []
    for d in lst:
        s, e, kind = d["s"], d["e"], d["kind"]
        if kind == "constant":
            pieces.append((s, e, kind, float(d["params"])))
        elif kind == "linear":
            pieces.append((s, e, kind, np.array(d["params"])))
        else:
            p = d["params"]
            popt = np.array([p["A"], p["lam"], p["C"]])
            pieces.append((s, e, kind, (popt, float(p["x0"]))))
    return pieces


def describe_pieces(pieces: list) -> str:
    lines = []
    for s, e, kind, params in pieces:
        e_str = str(e) if e is not None else "∞"
        if kind == "constant":
            lines.append(f"  [{s:>3}, {e_str:>4}): {kind:<12}  b = {float(params):+.4f}")
        elif kind == "linear":
            c = params
            lines.append(f"  [{s:>3}, {e_str:>4}): {kind:<12}  b = {c[0]:+.5f}·n {c[1]:+.4f}")
        else:
            popt, x0 = params
            A, lam, C = popt
            lines.append(f"  [{s:>3}, {e_str:>4}): exponential   b = {A:+.4f}·exp(−{lam:.5f}·(n−{x0:.0f})) {C:+.4f}")
    return "\n".join(lines)


# ── P0 ────────────────────────────────────────────────────────────────────────

class P0Model:
    """P0: K(y − p_MMR), single K and D, no intercept."""

    name = "P0 — Elo baseline"

    def __init__(self):
        self.K: float = 46.0
        self.D: float = 475.0

    def fit(self, df: pd.DataFrame) -> "P0Model":
        ra, rb = _ra_rb(df)
        obs = _obs(df)
        res = df["result"].values.astype(float)
        mask = ~np.isnan(ra) & ~np.isnan(rb) & ~np.isnan(obs)
        ra, rb, res, obs = ra[mask], rb[mask], res[mask], obs[mask]

        best_k, best_d, best_mae = 46.0, 475.0, np.inf
        for K in np.arange(8.0, 70.0, 1.0):
            for D in np.arange(200.0, 1200.0, 25.0):
                pred = K * (res - elo_expected(ra, rb, D))
                mae = float(np.mean(np.abs(obs - pred)))
                if mae < best_mae:
                    best_mae = mae; best_k, best_d = K, D
        self.K, self.D = best_k, best_d
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ra, rb = _ra_rb(df)
        return self.K * (df["result"].values.astype(float) - elo_expected(ra, rb, self.D))

    def residuals(self, df: pd.DataFrame) -> np.ndarray:
        return _obs(df) - self.predict(df)

    def metrics(self, df: pd.DataFrame) -> dict:
        return metrics(_obs(df), self.predict(df))

    def n_params(self) -> int:
        return 2  # K, D

    def __repr__(self) -> str:
        return f"P0(K={self.K:.1f}, D={self.D:.0f})"


# ── P1 ────────────────────────────────────────────────────────────────────────

class P1Model:
    """P1: β₀ + K_win*(1−p) for wins; β₀ + K_loss*(−p) for losses.

    Grid search over D; OLS fit for K_win, K_loss, β₀ at each D.
    """

    name = "P1 — Global intercept + asymmetric K"

    def __init__(self):
        self.K_win: float = 47.0
        self.K_loss: float = 47.0
        self.beta0: float = 0.0
        self.D: float = 475.0

    def _design(self, ra, rb, res, D):
        p = elo_expected(ra, rb, D)
        return np.column_stack([
            (1 - p) * (res == 1),   # K_win
            (-p) * (res == 0),       # K_loss
            np.ones(len(ra)),        # β₀
        ])

    def fit(self, df: pd.DataFrame) -> "P1Model":
        ra, rb = _ra_rb(df)
        obs = _obs(df)
        res = df["result"].values.astype(float)
        mask = ~np.isnan(ra) & ~np.isnan(rb) & ~np.isnan(obs)
        ra, rb, res, obs = ra[mask], rb[mask], res[mask], obs[mask]

        best_d, best_mae = 475.0, np.inf
        best_params = np.array([47.0, 47.0, 0.0])
        for D in np.arange(200.0, 1200.0, 25.0):
            X = self._design(ra, rb, res, D)
            params, _, _, _ = _lstsq(X, obs)
            mae = float(np.mean(np.abs(obs - X @ params)))
            if mae < best_mae:
                best_mae = mae; best_d = D; best_params = params

        self.D = best_d
        self.K_win, self.K_loss, self.beta0 = best_params
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ra, rb = _ra_rb(df)
        res = df["result"].values.astype(float)
        p = elo_expected(ra, rb, self.D)
        return np.where(
            res == 1,
            self.beta0 + self.K_win * (1 - p),
            self.beta0 + self.K_loss * (-p),
        )

    def residuals(self, df): return _obs(df) - self.predict(df)
    def metrics(self, df): return metrics(_obs(df), self.predict(df))
    def n_params(self): return 4  # K_win, K_loss, β₀, D

    def __repr__(self):
        return (f"P1(β₀={self.beta0:+.3f}, K_win={self.K_win:.1f}, "
                f"K_loss={self.K_loss:.1f}, D={self.D:.0f})")


# ── P2 ────────────────────────────────────────────────────────────────────────

class P2Model:
    """P2: per-experience-bucket K_win_j / K_loss_j + β₀ + b(g_p) + γI(g_o<10).

    Three experience buckets: <10, 10–49, ≥50 games this season.
    Piecewise intercept b(g_p) fitted on residuals after K/D correction.
    Opponent-placement offset γ fitted last.
    """

    name = "P2 — Placement + piecewise intercept + opp offset"
    _THRESHOLDS = (10, 50)

    def __init__(self):
        self.K_win: list[float] = [47.0, 47.0, 47.0]
        self.K_loss: list[float] = [47.0, 47.0, 47.0]
        self.beta0: float = 0.0
        self.D: float = 475.0
        self.pieces: list = []
        self.gamma: float = 0.0
        self.changepoints: list[int] = [10, 49, 50]

    def _bucket_masks(self, games: np.ndarray) -> list[np.ndarray]:
        t0, t1 = self._THRESHOLDS
        return [games < t0, (games >= t0) & (games < t1), games >= t1]

    def fit(self, df: pd.DataFrame) -> "P2Model":
        ra, rb = _ra_rb(df)
        obs = _obs(df)
        res = df["result"].values.astype(float)
        games = df["games_this_season_before"].to_numpy(dtype=float, na_value=np.nan)
        mask = ~np.isnan(ra) & ~np.isnan(rb) & ~np.isnan(obs) & ~np.isnan(games)
        ra, rb, res, obs, games = ra[mask], rb[mask], res[mask], obs[mask], games[mask]
        bmasks = self._bucket_masks(games)

        # Grid over D, OLS fit for K_win_j, K_loss_j, β₀
        best_d, best_mae = 475.0, np.inf
        best_params = None
        for D in np.arange(200.0, 1200.0, 25.0):
            p = elo_expected(ra, rb, D)
            cols = []
            for bm in bmasks:
                cols.append((1 - p) * bm * (res == 1))  # K_win_j
                cols.append((-p) * bm * (res == 0))      # K_loss_j
            X = np.column_stack(cols + [np.ones(len(ra))])
            params, _, _, _ = _lstsq(X, obs)
            mae = float(np.mean(np.abs(obs - X @ params)))
            if mae < best_mae:
                best_mae = mae; best_d = D; best_params = params

        self.D = best_d
        self.K_win  = [float(best_params[j * 2])     for j in range(3)]
        self.K_loss = [float(best_params[j * 2 + 1]) for j in range(3)]
        self.beta0  = float(best_params[-1])

        self._fit_bg_and_gamma(df, mask)
        return self

    def _fit_bg_and_gamma(self, df: pd.DataFrame, mask: np.ndarray | None = None) -> None:
        """Fit piecewise b(g) and γ on df (or the masked subset of df).

        Can be called standalone after D/K are already set — used by fit_on_full()
        to re-estimate the intercept curve on more data without repeating the grid search.

        b(g) is fitted only on MMR-complete rows (both player and opponent have MMR)
        so the piecewise intercept is not contaminated by the missing-MMR alpha effect.
        γ uses the full mask — it is a within-group comparison robust to this.
        """
        ra, rb  = _ra_rb(df)
        obs     = _obs(df)
        res     = df["result"].values.astype(float)
        games   = df["games_this_season_before"].to_numpy(dtype=float, na_value=np.nan)

        if mask is None:
            mask = ~np.isnan(ra) & ~np.isnan(rb) & ~np.isnan(obs) & ~np.isnan(games)

        # For the placement range (n < 10), filter to player-MMR-present rows.
        # The missing-player-MMR fraction shifts from ~18% in S10 (training)
        # to ~50% in S11 (test), so the full-mask b(g) absorbs an S10-specific
        # average that doesn't generalize.  At n >= 10, missing-MMR fraction is
        # stable at ~0.2% across seasons — no filtering needed there.
        mmr_p = df["player_mmr_before"].to_numpy(dtype=float, na_value=np.nan) if "player_mmr_before" in df.columns else np.full(len(df), np.nan)
        games_all = df["games_this_season_before"].to_numpy(dtype=float, na_value=np.nan)
        bg_mask = mask & ((games_all >= 10) | (~np.isnan(mmr_p)))

        ra_bg, rb_bg = ra[bg_mask], rb[bg_mask]
        res_bg, obs_bg, games_bg = res[bg_mask], obs[bg_mask], games[bg_mask]
        bmasks_bg = self._bucket_masks(games_bg)

        p_bg = elo_expected(ra_bg, rb_bg, self.D)
        kd_pred_bg = np.empty(len(ra_bg))
        for bm, kw, kl in zip(bmasks_bg, self.K_win, self.K_loss):
            kd_pred_bg[bm] = np.where(
                res_bg[bm] == 1,
                self.beta0 + kw * (1 - p_bg[bm]),
                self.beta0 + kl * (-p_bg[bm]),
            )
        raw_resid_bg = obs_bg - kd_pred_bg

        self.pieces = fit_piecewise_b(games_bg.astype(int), raw_resid_bg, self.changepoints)

        # Full-mask versions for gamma (comparison of n_opp buckets, not affected by missingness)
        ra, rb, res, obs, games = ra[mask], rb[mask], res[mask], obs[mask], games[mask]
        bmasks = self._bucket_masks(games)

        p = elo_expected(ra, rb, self.D)
        kd_pred = np.empty(len(ra))
        for bm, kw, kl in zip(bmasks, self.K_win, self.K_loss):
            kd_pred[bm] = np.where(
                res[bm] == 1,
                self.beta0 + kw * (1 - p[bm]),
                self.beta0 + kl * (-p[bm]),
            )
        raw_resid = obs - kd_pred

        self.gamma = 0.0
        if "opponent_games_this_season_before" in df.columns:
            n_opp_full = df["opponent_games_this_season_before"].to_numpy(dtype=float, na_value=np.nan)
            n_opp = n_opp_full[mask]
            b_pred = _intercept_eval(games, self.pieces)
            resid2 = raw_resid - b_pred
            has_opp  = ~np.isnan(n_opp)
            opp_place = has_opp & (n_opp < self._THRESHOLDS[0])
            opp_estab = has_opp & (n_opp >= self._THRESHOLDS[0])
            if opp_place.sum() >= 100 and opp_estab.sum() >= 100:
                self.gamma = float(np.mean(resid2[opp_place]) - np.mean(resid2[opp_estab]))

    def fit_on_full(self, full_df: pd.DataFrame) -> "P2Model":
        """Re-fit K/β₀ (OLS, D fixed) + b(g) + γ on full_df.

        Assumes self.D is already set (e.g., via fit() on a subsample).
        Use this to get stable b(g) asymptotes without repeating the D grid search.
        """
        ra, rb  = _ra_rb(full_df)
        obs     = _obs(full_df)
        res     = full_df["result"].values.astype(float)
        games   = full_df["games_this_season_before"].to_numpy(dtype=float, na_value=np.nan)
        mask    = ~np.isnan(ra) & ~np.isnan(rb) & ~np.isnan(obs) & ~np.isnan(games)
        ra, rb, res, obs, games = ra[mask], rb[mask], res[mask], obs[mask], games[mask]

        p      = elo_expected(ra, rb, self.D)
        bmasks = self._bucket_masks(games)
        cols   = []
        for bm in bmasks:
            cols.append((1 - p) * bm * (res == 1))
            cols.append((-p)    * bm * (res == 0))
        X = np.column_stack(cols + [np.ones(len(ra))])
        params, _, _, _ = _lstsq(X, obs)

        self.K_win  = [float(params[j * 2])     for j in range(3)]
        self.K_loss = [float(params[j * 2 + 1]) for j in range(3)]
        self.beta0  = float(params[-1])

        self._fit_bg_and_gamma(full_df, mask)
        return self

    def _kd_term(self, ra, rb, res, games) -> np.ndarray:
        p = elo_expected(ra, rb, self.D)
        out = np.empty(len(ra))
        for bm, kw, kl in zip(self._bucket_masks(games), self.K_win, self.K_loss):
            out[bm] = np.where(
                res[bm] == 1,
                self.beta0 + kw * (1 - p[bm]),
                self.beta0 + kl * (-p[bm]),
            )
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ra, rb = _ra_rb(df)
        res = df["result"].values.astype(float)
        games = df["games_this_season_before"].to_numpy(dtype=float, na_value=-1.0)
        out = self._kd_term(ra, rb, res, games)
        out += _intercept_eval(games, self.pieces)
        if self.gamma != 0.0 and "opponent_games_this_season_before" in df.columns:
            n_opp = df["opponent_games_this_season_before"].to_numpy(dtype=float, na_value=np.nan)
            out[~np.isnan(n_opp) & (n_opp < self._THRESHOLDS[0])] += self.gamma
        return out

    def residuals(self, df): return _obs(df) - self.predict(df)
    def metrics(self, df): return metrics(_obs(df), self.predict(df))

    def n_params(self) -> int:
        piece_params = sum(2 if k == "linear" else 3 if k == "exp" else 1
                          for _, _, k, _ in self.pieces)
        return 3 * 2 + 1 + 1 + piece_params + (1 if self.gamma != 0 else 0)

    def report(self) -> str:
        lines = [repr(self)]
        lines.append(f"  Piecewise intercept b(g_p):")
        lines.append(describe_pieces(self.pieces))
        if self.gamma != 0:
            lines.append(f"  opp_placement_offset γ = {self.gamma:+.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model": "P2",
            "D": self.D,
            "beta0": self.beta0,
            "K_win": self.K_win,
            "K_loss": self.K_loss,
            "gamma": self.gamma,
            "changepoints": self.changepoints,
            "pieces": _pieces_to_list(self.pieces),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "P2Model":
        m = cls()
        m.D           = d["D"]
        m.beta0       = d["beta0"]
        m.K_win       = d["K_win"]
        m.K_loss      = d["K_loss"]
        m.gamma       = d["gamma"]
        m.changepoints = d["changepoints"]
        m.pieces      = _list_to_pieces(d["pieces"])
        return m

    def __repr__(self) -> str:
        kw = ", ".join(f"{k:.1f}" for k in self.K_win)
        kl = ", ".join(f"{k:.1f}" for k in self.K_loss)
        return (f"P2(β₀={self.beta0:+.3f}, D={self.D:.0f}, "
                f"K_win=[{kw}], K_loss=[{kl}], γ={self.gamma:+.3f})")


# ── P3 ────────────────────────────────────────────────────────────────────────

class P3Model:
    """P3: P2 + additive indicator corrections for missing-MMR rows.

    First test: simple additive indicators (α_p, α_o).
    Can be extended to separate regime branches if validation supports it.
    """

    name = "P3 — P2 + missing-MMR indicators"

    def __init__(self):
        self.p2 = P2Model()
        self.alpha_missing_p: float = 0.0
        self.alpha_missing_o: float = 0.0

    def fit(self, df: pd.DataFrame) -> "P3Model":
        self.p2.fit(df)
        self._fit_alphas(df)
        return self

    def fit_two_stage(self, sub_df: pd.DataFrame, full_df: pd.DataFrame) -> "P3Model":
        """Two-stage fit: D grid search on sub_df, everything else on full_df.

        The grid search over D is the only slow step (41 D values × OLS on sub_df).
        K/β₀, b(g), γ, and the missing-MMR alphas are all re-estimated on full_df
        using the D found on sub_df, giving stable b(g) asymptotes.
        """
        self.p2.fit(sub_df)               # grid-search D on subsample
        self.p2.fit_on_full(full_df)      # re-fit K/β₀/b(g)/γ on full data
        self._fit_alphas(full_df)
        return self

    def _fit_alphas(self, df: pd.DataFrame) -> None:
        resid     = self.p2.residuals(df)
        obs       = _obs(df)
        missing_p = df["player_mmr_before"].isna().values.astype(float)
        missing_o = df["opponent_mmr_before"].isna().values.astype(float)
        valid     = ~np.isnan(obs) & ~np.isnan(resid)

        X = np.column_stack([missing_p[valid], missing_o[valid]])
        y = resid[valid]
        if X[:, 0].sum() >= 50 and X[:, 1].sum() >= 50:
            params, _, _, _ = _lstsq(X, y)
            self.alpha_missing_p = float(params[0])
            self.alpha_missing_o = float(params[1])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.p2.predict(df)
        missing_p = df["player_mmr_before"].isna().values.astype(float)
        missing_o = df["opponent_mmr_before"].isna().values.astype(float)
        return pred + self.alpha_missing_p * missing_p + self.alpha_missing_o * missing_o

    def residuals(self, df): return _obs(df) - self.predict(df)
    def metrics(self, df): return metrics(_obs(df), self.predict(df))
    def n_params(self): return self.p2.n_params() + 2

    def to_dict(self) -> dict:
        return {
            "model": "P3",
            "alpha_missing_p": self.alpha_missing_p,
            "alpha_missing_o": self.alpha_missing_o,
            "p2": self.p2.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "P3Model":
        m = cls()
        m.alpha_missing_p = d["alpha_missing_p"]
        m.alpha_missing_o = d["alpha_missing_o"]
        m.p2 = P2Model.from_dict(d["p2"])
        return m

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "P3Model":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self):
        return (f"P3(α_missing_p={self.alpha_missing_p:+.3f}, "
                f"α_missing_o={self.alpha_missing_o:+.3f}  [{repr(self.p2)}])")


# ── P4 ────────────────────────────────────────────────────────────────────────

class P4Model:
    """P4: P3 + linear reconciliation h1(d_p) + h2(d_o).

    d_p = R_p − MMR_p,  d_o = R_o − MMR_o.
    Both discrepancies default to 0 when either signal is missing,
    so P3's missing-MMR corrections still apply to those rows.
    After fitting, replot residuals vs absolute MMR to see how much slope remains.
    """

    name = "P4 — P3 + rating-minus-MMR reconciliation"

    def __init__(self):
        self.p3 = P3Model()
        self.beta_dp: float = 0.0
        self.beta_do: float = 0.0
        self.beta0_p4: float = 0.0

    def _discrepancies(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        def _f(c): return df[c].to_numpy(dtype=float, na_value=np.nan)
        mmr_p, mmr_o = _f("player_mmr_before"), _f("opponent_mmr_before")
        rat_p, rat_o = _f("player_rating_before"), _f("opponent_rating_before")
        both_p = ~np.isnan(mmr_p) & ~np.isnan(rat_p)
        both_o = ~np.isnan(mmr_o) & ~np.isnan(rat_o)
        d_p = np.where(both_p, rat_p - mmr_p, 0.0)
        d_o = np.where(both_o, rat_o - mmr_o, 0.0)
        return d_p, d_o

    def fit(self, df: pd.DataFrame) -> "P4Model":
        """Fit the full nested chain (P3 + reconciliation) on df."""
        self.p3.fit(df)
        return self.fit_reconciliation(df)

    def fit_reconciliation(self, df: pd.DataFrame) -> "P4Model":
        """Fit only the P4 reconciliation terms given an already-fitted self.p3.

        Call this after fitting self.p3 on a subsample to fit β_dp / β_do / β₀
        on a (potentially larger) dataset without re-running the P3 grid search.
        """
        resid = self.p3.residuals(df)
        obs   = _obs(df)
        d_p, d_o = self._discrepancies(df)
        valid = ~np.isnan(obs) & ~np.isnan(resid)

        X = np.column_stack([d_p[valid], d_o[valid], np.ones(valid.sum())])
        y = resid[valid]
        params, _, _, _ = _lstsq(X, y)
        self.beta_dp   = float(params[0])
        self.beta_do   = float(params[1])
        self.beta0_p4  = float(params[2])

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.p3.predict(df)
        d_p, d_o = self._discrepancies(df)
        return pred + self.beta_dp * d_p + self.beta_do * d_o + self.beta0_p4

    def residuals(self, df): return _obs(df) - self.predict(df)
    def metrics(self, df): return metrics(_obs(df), self.predict(df))
    def n_params(self): return self.p3.n_params() + 3

    def __repr__(self):
        return (f"P4(β₀={self.beta0_p4:+.4f}, β_dp={self.beta_dp:+.5f}, β_do={self.beta_do:+.5f}  "
                f"[{repr(self.p3)}])")


# ── P3b ───────────────────────────────────────────────────────────────────────

class P3bModel:
    """P3b: same as P2 but with a 4th 'bucket' for missing-player-MMR rows.

    When player_mmr_before is missing, the player's rating signal is visible
    rating only, which has different scaling than MMR.  Using a separate
    K_win_m / K_loss_m for those rows (instead of the game-count bucket K)
    is the minimal extension that tests whether the K regime differs.

    Missing-opponent-MMR rows are NOT given a separate K — the opponent's
    MMR only enters through p = elo_expected(ra, rb, D), and that path is
    already falling back to opponent's visible rating.  Opponent missingness
    is captured by the γ offset inherited from P2 plus P3's α_o indicator
    (which can be stacked on top as P3→P3b if needed; here we keep P3b
    standalone for a clean comparison).
    """

    name = "P3b — P2 + separate K for missing-player-MMR"
    _THRESHOLDS = (10, 50)

    def __init__(self):
        # K for non-missing rows (3 game-count buckets)
        self.K_win:  list[float] = [47.0, 47.0, 47.0]
        self.K_loss: list[float] = [47.0, 47.0, 47.0]
        # K for missing-player-MMR rows
        self.K_win_m:  float = 47.0
        self.K_loss_m: float = 47.0
        self.beta0: float = 0.0
        self.D: float = 475.0
        self.pieces: list = []
        self.gamma: float = 0.0
        self.changepoints: list[int] = [10, 49, 50]

    def _masks(self, games: np.ndarray, missing_p: np.ndarray):
        """Return 4 bucket masks: [b0, b1, b2, missing]."""
        t0, t1 = self._THRESHOLDS
        present = ~missing_p
        return [
            present & (games < t0),
            present & (games >= t0) & (games < t1),
            present & (games >= t1),
            missing_p,
        ]

    def fit(self, df: pd.DataFrame) -> "P3bModel":
        ra, rb  = _ra_rb(df)
        obs     = _obs(df)
        res     = df["result"].values.astype(float)
        games   = df["games_this_season_before"].to_numpy(dtype=float, na_value=np.nan)
        miss_p  = df["player_mmr_before"].isna().values

        mask = ~np.isnan(ra) & ~np.isnan(rb) & ~np.isnan(obs) & ~np.isnan(games)
        ra, rb, res, obs, games, miss_p = (
            ra[mask], rb[mask], res[mask], obs[mask], games[mask], miss_p[mask]
        )
        bmasks = self._masks(games, miss_p)

        # Grid over D, OLS for all K params + β₀
        best_d, best_mae = 475.0, np.inf
        best_params = None
        for D in np.arange(200.0, 1200.0, 25.0):
            p = elo_expected(ra, rb, D)
            cols = []
            for bm in bmasks:
                cols.append((1 - p) * bm * (res == 1))  # K_win_j
                cols.append((-p)    * bm * (res == 0))   # K_loss_j
            X = np.column_stack(cols + [np.ones(len(ra))])
            params, _, _, _ = _lstsq(X, obs)
            mae = float(np.mean(np.abs(obs - X @ params)))
            if mae < best_mae:
                best_mae = mae; best_d = D; best_params = params

        self.D = best_d
        # params layout: [K_win_0, K_loss_0, K_win_1, K_loss_1, K_win_2, K_loss_2,
        #                  K_win_m, K_loss_m, β₀]
        self.K_win  = [float(best_params[j * 2])     for j in range(3)]
        self.K_loss = [float(best_params[j * 2 + 1]) for j in range(3)]
        self.K_win_m  = float(best_params[6])
        self.K_loss_m = float(best_params[7])
        self.beta0    = float(best_params[8])

        # Compute K/D prediction → residual for piecewise b(g) fit
        p = elo_expected(ra, rb, self.D)
        kd_pred = np.empty(len(ra))
        for bm, kw, kl in zip(bmasks[:3], self.K_win, self.K_loss):
            kd_pred[bm] = np.where(
                res[bm] == 1,
                self.beta0 + kw * (1 - p[bm]),
                self.beta0 + kl * (-p[bm]),
            )
        bm_m = bmasks[3]
        kd_pred[bm_m] = np.where(
            res[bm_m] == 1,
            self.beta0 + self.K_win_m  * (1 - p[bm_m]),
            self.beta0 + self.K_loss_m * (-p[bm_m]),
        )
        raw_resid = obs - kd_pred

        # Fit piecewise b(g) on non-missing rows only (missing rows don't need
        # season-game-count intercept as accurately; b(g) still applied to them)
        self.pieces = fit_piecewise_b(games.astype(int), raw_resid, self.changepoints)

        # Opponent-placement offset γ
        self.gamma = 0.0
        if "opponent_games_this_season_before" in df.columns:
            n_opp_full = df["opponent_games_this_season_before"].to_numpy(dtype=float, na_value=np.nan)
            n_opp = n_opp_full[mask]
            b_pred = _intercept_eval(games, self.pieces)
            resid2 = raw_resid - b_pred
            has_opp  = ~np.isnan(n_opp)
            opp_place = has_opp & (n_opp < self._THRESHOLDS[0])
            opp_estab = has_opp & (n_opp >= self._THRESHOLDS[0])
            if opp_place.sum() >= 100 and opp_estab.sum() >= 100:
                self.gamma = float(np.mean(resid2[opp_place]) - np.mean(resid2[opp_estab]))

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ra, rb  = _ra_rb(df)
        res     = df["result"].values.astype(float)
        games   = df["games_this_season_before"].to_numpy(dtype=float, na_value=-1.0)
        miss_p  = df["player_mmr_before"].isna().values
        bmasks  = self._masks(games, miss_p)
        p       = elo_expected(ra, rb, self.D)

        out = np.empty(len(ra))
        for bm, kw, kl in zip(bmasks[:3], self.K_win, self.K_loss):
            out[bm] = np.where(
                res[bm] == 1,
                self.beta0 + kw * (1 - p[bm]),
                self.beta0 + kl * (-p[bm]),
            )
        bm_m = bmasks[3]
        out[bm_m] = np.where(
            res[bm_m] == 1,
            self.beta0 + self.K_win_m  * (1 - p[bm_m]),
            self.beta0 + self.K_loss_m * (-p[bm_m]),
        )
        out += _intercept_eval(games, self.pieces)
        if self.gamma != 0.0 and "opponent_games_this_season_before" in df.columns:
            n_opp = df["opponent_games_this_season_before"].to_numpy(dtype=float, na_value=np.nan)
            out[~np.isnan(n_opp) & (n_opp < self._THRESHOLDS[0])] += self.gamma
        return out

    def residuals(self, df): return _obs(df) - self.predict(df)
    def metrics(self, df): return metrics(_obs(df), self.predict(df))

    def n_params(self) -> int:
        piece_params = sum(2 if k == "linear" else 3 if k == "exp" else 1
                          for _, _, k, _ in self.pieces)
        return 3 * 2 + 2 + 1 + piece_params + (1 if self.gamma != 0 else 0)

    def __repr__(self) -> str:
        kw = ", ".join(f"{k:.1f}" for k in self.K_win)
        kl = ", ".join(f"{k:.1f}" for k in self.K_loss)
        return (f"P3b(β₀={self.beta0:+.3f}, D={self.D:.0f}, "
                f"K_win=[{kw}], K_loss=[{kl}], "
                f"K_win_m={self.K_win_m:.1f}, K_loss_m={self.K_loss_m:.1f}, "
                f"γ={self.gamma:+.3f})")
