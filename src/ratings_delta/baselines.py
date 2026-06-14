"""Simple baseline models for rating delta prediction."""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .formula import elo_delta, elo_expected


def _effective_ratings(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ra, rb, valid_mask) using MMR where available, rating as fallback.

    valid_mask is True for rows where at least one signal exists for both players.
    """
    def _to_f(col): return df[col].to_numpy(dtype=float, na_value=np.nan)

    ra_mmr = _to_f("player_mmr_before")
    rb_mmr = _to_f("opponent_mmr_before")
    ra_rat = _to_f("player_rating_before")
    rb_rat = _to_f("opponent_rating_before")

    ra = np.where(np.isnan(ra_mmr), ra_rat, ra_mmr)
    rb = np.where(np.isnan(rb_mmr), rb_rat, rb_mmr)
    valid = ~np.isnan(ra) & ~np.isnan(rb)
    return ra, rb, valid


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


class DynamicKEloBaseline:
    """Elo formula with per-experience-bucket (K, D) fit on training data.

    Three buckets by games_this_season_before: <10 (placement), 10-49, >=50 (established).
    Both K and D are fit jointly by MAE grid search per bucket.
    Set use_mmr=True to use MMR as the skill signal (falling back to visible rating).
    """

    _THRESHOLDS = (10, 50)
    _BUCKET_LABELS = ["<10 games", "10–49 games", "≥50 games"]

    def __init__(self, use_mmr: bool = False):
        self.use_mmr = use_mmr
        self.K_segments: list[float] = []
        self.D_segments: list[float] = []
        self.bias_segments: list[float] = []

    def _bucket_mask(self, games: np.ndarray) -> list[np.ndarray]:
        return [
            games < self._THRESHOLDS[0],
            (games >= self._THRESHOLDS[0]) & (games < self._THRESHOLDS[1]),
            games >= self._THRESHOLDS[1],
        ]

    def _get_ratings(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.use_mmr:
            ra, rb, valid = _effective_ratings(df)
            obs = df["observed_rating_delta"].values.astype(float)
            games = df["games_this_season_before"].values
            mask = valid & ~np.isnan(obs) & ~np.isnan(games.astype(float))
            return ra[mask], rb[mask], mask
        else:
            needed = ["player_rating_before", "opponent_rating_before",
                      "observed_rating_delta", "games_this_season_before"]
            clean = df.dropna(subset=needed)
            full_mask = df.index.isin(clean.index)
            return (clean["player_rating_before"].values.astype(float),
                    clean["opponent_rating_before"].values.astype(float),
                    full_mask)

    def fit(self, df: pd.DataFrame) -> "DynamicKEloBaseline":
        if self.use_mmr:
            ra_all, rb_all, valid = _effective_ratings(df)
            needed_mask = (valid &
                           ~np.isnan(df["observed_rating_delta"].to_numpy(dtype=float, na_value=np.nan)) &
                           ~np.isnan(df["games_this_season_before"].to_numpy(dtype=float, na_value=np.nan)))
            clean = df[needed_mask].copy()
            ra = ra_all[needed_mask]
            rb = rb_all[needed_mask]
        else:
            clean = df.dropna(subset=[
                "player_rating_before", "opponent_rating_before",
                "observed_rating_delta", "games_this_season_before",
            ])
            ra = clean["player_rating_before"].values.astype(float)
            rb = clean["opponent_rating_before"].values.astype(float)
        games = clean["games_this_season_before"].values
        res = clean["result"].values.astype(float)
        obs = clean["observed_rating_delta"].values.astype(float)

        K_range = np.arange(8.0, 70.0, 1.0)
        D_range = np.arange(200.0, 1200.0, 25.0)

        self.K_segments, self.D_segments = [], []
        for mask in self._bucket_mask(games):
            if mask.sum() < 10:
                self.K_segments.append(46.0)
                self.D_segments.append(675.0)
                continue
            best_k, best_d, best_mae = 46.0, 675.0, np.inf
            for K in K_range:
                for D in D_range:
                    pred = K * (res[mask] - elo_expected(ra[mask], rb[mask], D))
                    mae = np.mean(np.abs(obs[mask] - pred))
                    if mae < best_mae:
                        best_mae = mae
                        best_k, best_d = K, D
            self.K_segments.append(best_k)
            self.D_segments.append(best_d)

        # Fit per-segment intercept as mean residual after (K, D) correction
        self.bias_segments = []
        for mask, K, D in zip(self._bucket_mask(games), self.K_segments, self.D_segments):
            if mask.sum() < 10:
                self.bias_segments.append(0.0)
                continue
            pred = K * (res[mask] - elo_expected(ra[mask], rb[mask], D))
            self.bias_segments.append(float(np.mean(obs[mask] - pred)))

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        games = df["games_this_season_before"].fillna(-1).values
        if self.use_mmr:
            ra, rb, _ = _effective_ratings(df)
        else:
            ra = df["player_rating_before"].values.astype(float)
            rb = df["opponent_rating_before"].values.astype(float)
        res = df["result"].values.astype(float)
        masks = self._bucket_mask(games)
        out = np.empty(len(df))
        for mask, K, D, bias in zip(masks, self.K_segments, self.D_segments, self.bias_segments):
            out[mask] = K * (res[mask] - elo_expected(ra[mask], rb[mask], D)) + bias
        return out

    def residuals(self, df: pd.DataFrame) -> np.ndarray:
        return df["observed_rating_delta"].values.astype(float) - self.predict(df)

    def __repr__(self) -> str:
        if not self.K_segments:
            return "DynamicKEloBaseline(unfitted)"
        mmr_str = ",mmr" if self.use_mmr else ""
        seg_str = ", ".join(
            f"{lbl}:(K={k:.1f},D={d:.0f},b={b:+.2f})"
            for lbl, k, d, b in zip(self._BUCKET_LABELS, self.K_segments, self.D_segments, self.bias_segments)
        )
        return f"DynamicKEloBaseline{mmr_str}({seg_str})"


# ---------------------------------------------------------------------------
# Helpers for PiecewiseInterceptEloBaseline
# ---------------------------------------------------------------------------

def _seg_curve(g_all, r_all, start, end):
    """Per-game mean residual and sqrt-n weights for one segment."""
    mask = (g_all >= start) if end is None else (g_all >= start) & (g_all < end)
    agg = (pd.DataFrame({"g": g_all[mask], "r": r_all[mask]})
           .groupby("g")["r"].agg(["mean", "count"]).reset_index())
    agg.columns = ["g", "mean", "n"]
    x = agg["g"].values.astype(float)
    y = agg["mean"].values.astype(float)
    w = np.sqrt(agg["n"].values.astype(float))
    return x, y, w


def _fit_linear_seg(x, y, w):
    if len(x) < 2:
        c = float(np.average(y, weights=w))
        return np.array([0.0, c])
    return np.polyfit(x, y, 1, w=w)


def _fit_exp_seg(x, y, w):
    """Fit A·exp(−λ·(n − x[0])) + C; return (popt, x0) or (None, x0)."""
    x0 = float(x[0])
    C0 = float(np.average(y[-max(1, len(y) // 5):], weights=w[-max(1, len(y) // 5):]))
    A0 = float(y[0]) - C0
    lam0 = 0.05

    def f(n, A, lam, C):
        return A * np.exp(-lam * (n - x0)) + C

    try:
        popt, _ = curve_fit(
            f, x, y, p0=[A0, lam0, C0],
            sigma=1.0 / (w + 1e-6),
            bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=8000,
        )
        return popt, x0
    except Exception:
        return None, x0


def _r2(y, y_pred, w):
    y_mean = np.average(y, weights=w)
    ss_res = float(np.sum(w * (y - y_pred) ** 2))
    ss_tot = float(np.sum(w * (y - y_mean) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _aic(y, y_pred, w, k):
    ss_res = float(np.sum(w * (y - y_pred) ** 2))
    n = len(y)
    return n * np.log(ss_res / n + 1e-12) + 2 * k


def test_exp_vs_linear(x, y, w, label=""):
    """Fit exponential and linear to segment; print comparison; return winner ('exp'/'linear')."""
    lin_c = _fit_linear_seg(x, y, w)
    lin_pred = np.polyval(lin_c, x)
    lin_r2 = _r2(y, lin_pred, w)
    lin_aic = _aic(y, lin_pred, w, k=2)

    popt, x0 = _fit_exp_seg(x, y, w)
    if popt is not None:
        def f(n, A, lam, C): return A * np.exp(-lam * (n - x0)) + C
        exp_pred = f(x, *popt)
        exp_r2 = _r2(y, exp_pred, w)
        exp_aic = _aic(y, exp_pred, w, k=3)
        winner = "exp" if exp_aic < lin_aic else "linear"
    else:
        exp_r2, exp_aic, winner = float("nan"), float("nan"), "linear"
        popt = None

    if label:
        print(f"\n  --- Exponential test: {label} ---")
        print(f"    Linear:      R²={lin_r2:.4f}  AIC={lin_aic:.2f}  "
              f"b(n) = {lin_c[0]:+.5f}·n {lin_c[1]:+.4f}")
        if popt is not None:
            print(f"    Exponential: R²={exp_r2:.4f}  AIC={exp_aic:.2f}  "
                  f"b(n) = {popt[0]:+.4f}·exp(−{popt[1]:.5f}·(n−{x0:.0f})) {popt[2]:+.4f}")
        else:
            print("    Exponential: fit failed")
        print(f"    → Winner: {winner}  (lower AIC wins)")

    return winner, lin_c, popt, x0


class PiecewiseInterceptEloBaseline:
    """Elo baseline with piecewise intercept b(n).

    Proposal 1 (use_mmr=True): use MMR as skill signal with rating fallback.
    Proposal 2 (use_mmr_tiers=True): fit K/D on a 3×3 grid of
        (experience bucket) × (avg-MMR tier) — 9 cells total.
    Proposal 3 (use_mmr_tiers=True): intercept uses min(n, n_opp) as the
        effective experience, capturing both players' placement status.

    Changepoints are loaded from tools/changepoints.json if not supplied.
    """

    _KD_THRESHOLDS = (10, 50)
    _MMR_TIERS     = (900, 1300)
    _KD_LABELS     = ["<10 games", "10–49 games", "≥50 games"]
    _MMR_LABELS    = ["low MMR", "mid MMR", "high MMR"]

    def __init__(self, changepoints: list | None = None,
                 kd_baseline: "DynamicKEloBaseline | None" = None,
                 use_mmr: bool = False,
                 use_mmr_tiers: bool = False):
        self._user_changepoints = changepoints
        self._kd_baseline = kd_baseline   # used only when use_mmr_tiers=False
        self.use_mmr = use_mmr
        self.use_mmr_tiers = use_mmr_tiers
        # 3-bucket storage (backward compat / use_mmr_tiers=False)
        self.K_segments: list[float] = []
        self.D_segments: list[float] = []
        # 9-cell storage: KD_grid[exp_i][mmr_j] = (K, D)
        self.KD_grid: list[list] = []
        self.pieces: list[tuple] = []
        self.changepoints: list[int] = []
        # Proposal 3: additive offset when opponent is in placement (<10 games)
        self.opp_placement_offset: float = 0.0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_changepoints(self) -> list[int]:
        if self._user_changepoints is not None:
            return sorted(int(c) for c in self._user_changepoints)
        import json
        from pathlib import Path
        cp_path = Path(__file__).parent.parent.parent / "tools" / "changepoints.json"
        if cp_path.exists():
            data = json.loads(cp_path.read_text())
            return sorted(int(x) for x in data["changepoints"])
        return [2, 3, 4, 9, 10, 49, 50]

    def _exp_masks(self, games: np.ndarray) -> list[np.ndarray]:
        t0, t1 = self._KD_THRESHOLDS
        return [games < t0, (games >= t0) & (games < t1), games >= t1]

    def _mmr_masks(self, avg_mmr: np.ndarray) -> list[np.ndarray]:
        m0, m1 = self._MMR_TIERS
        nan = np.isnan(avg_mmr)
        return [
            (avg_mmr < m0) | nan,                         # low  (NaN → low tier)
            (avg_mmr >= m0) & (avg_mmr < m1) & ~nan,
            (avg_mmr >= m1) & ~nan,
        ]

    def _cell_masks(self, games: np.ndarray, avg_mmr: np.ndarray):
        """Yield (mask, exp_i, mmr_j) for all 9 (exp × mmr) cells."""
        for i, em in enumerate(self._exp_masks(games)):
            for j, mm in enumerate(self._mmr_masks(avg_mmr)):
                yield em & mm, i, j

    def _get_skill(self, df: pd.DataFrame):
        """Return (ra, rb) using MMR-with-fallback when use_mmr or use_mmr_tiers."""
        if self.use_mmr or self.use_mmr_tiers:
            ra, rb, _ = _effective_ratings(df)
        else:
            ra = df["player_rating_before"].values.astype(float)
            rb = df["opponent_rating_before"].values.astype(float)
        return ra, rb

    def _effective_n(self, df: pd.DataFrame) -> np.ndarray:
        """Player's own season game count for intercept lookup."""
        return df["games_this_season_before"].to_numpy(dtype=float, na_value=-1.0)

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "PiecewiseInterceptEloBaseline":
        cps = self._load_changepoints()
        self.changepoints = cps

        # Build clean subset
        ra_all, rb_all, valid = _effective_ratings(df)
        obs_arr  = df["observed_rating_delta"].to_numpy(dtype=float, na_value=np.nan)
        games_arr = df["games_this_season_before"].to_numpy(dtype=float, na_value=np.nan)
        needed = valid & ~np.isnan(obs_arr) & ~np.isnan(games_arr)

        if not (self.use_mmr or self.use_mmr_tiers):
            rat_valid = (~np.isnan(df["player_rating_before"].to_numpy(dtype=float, na_value=np.nan)) &
                         ~np.isnan(df["opponent_rating_before"].to_numpy(dtype=float, na_value=np.nan)))
            needed = needed & rat_valid

        clean   = df[needed].copy()
        ra      = ra_all[needed]
        rb      = rb_all[needed]
        games   = games_arr[needed]
        res     = clean["result"].values.astype(float)
        obs     = obs_arr[needed]
        avg_mmr = (ra + rb) / 2

        K_range = np.arange(8.0, 70.0, 1.0)
        D_range = np.arange(200.0, 1200.0, 25.0)

        # ── 1a. 9-cell K/D grid (Proposals 2) ────────────────────────────────
        if self.use_mmr_tiers:
            self.KD_grid = [[None] * 3 for _ in range(3)]
            for cell, i, j in self._cell_masks(games, avg_mmr):
                if cell.sum() < 10:
                    self.KD_grid[i][j] = (46.0, 475.0)
                    continue
                best_k, best_d, best_mae = 46.0, 475.0, np.inf
                for K in K_range:
                    for D in D_range:
                        pred = K * (res[cell] - elo_expected(ra[cell], rb[cell], D))
                        mae  = np.mean(np.abs(obs[cell] - pred))
                        if mae < best_mae:
                            best_mae = mae; best_k, best_d = K, D
                self.KD_grid[i][j] = (best_k, best_d)
            # For backward-compat / report: per-exp averages
            self.K_segments = [float(np.mean([self.KD_grid[i][j][0] for j in range(3)])) for i in range(3)]
            self.D_segments = [float(np.mean([self.KD_grid[i][j][1] for j in range(3)])) for i in range(3)]

        # ── 1b. 3-bucket K/D (original path) ─────────────────────────────────
        else:
            if self._kd_baseline is not None:
                self.K_segments = list(self._kd_baseline.K_segments)
                self.D_segments = list(self._kd_baseline.D_segments)
            else:
                self.K_segments, self.D_segments = [], []
                for mask in self._exp_masks(games):
                    if mask.sum() < 10:
                        self.K_segments.append(46.0); self.D_segments.append(675.0)
                        continue
                    best_k, best_d, best_mae = 46.0, 675.0, np.inf
                    for K in K_range:
                        for D in D_range:
                            pred = K * (res[mask] - elo_expected(ra[mask], rb[mask], D))
                            mae  = np.mean(np.abs(obs[mask] - pred))
                            if mae < best_mae:
                                best_mae = mae; best_k, best_d = K, D
                    self.K_segments.append(best_k); self.D_segments.append(best_d)

        # ── 2. Per-game mean residual after K/D correction ────────────────────
        pred_all = np.empty(len(clean))
        if self.use_mmr_tiers:
            for cell, i, j in self._cell_masks(games, avg_mmr):
                K, D = self.KD_grid[i][j]
                pred_all[cell] = K * (res[cell] - elo_expected(ra[cell], rb[cell], D))
        else:
            for mask, K, D in zip(self._exp_masks(games), self.K_segments, self.D_segments):
                pred_all[mask] = K * (res[mask] - elo_expected(ra[mask], rb[mask], D))
        raw_resid = obs - pred_all

        # ── 3. Effective n = player's own experience ──────────────────────────
        eff_n = games.astype(int)

        # ── 4. Fit piecewise intercept ────────────────────────────────────────
        self.pieces = []
        cps_low = sorted(c for c in cps if 0 < c <= 10)
        boundaries_low = [0] + cps_low
        if boundaries_low[-1] < 10:
            boundaries_low.append(10)
        for s, e in zip(boundaries_low[:-1], boundaries_low[1:]):
            x, y, w = _seg_curve(eff_n, raw_resid, s, e)
            self.pieces.append((s, e, "linear", _fit_linear_seg(x, y, w)))

        x, y, w = _seg_curve(eff_n, raw_resid, 10, 49)
        winner, lin_c, exp_popt, x0 = test_exp_vs_linear(x, y, w, label="[10, 49)")
        self.pieces.append((10, 49, "exp", (exp_popt, x0)) if winner == "exp" and exp_popt is not None
                           else (10, 49, "linear", lin_c))

        x, y, w = _seg_curve(eff_n, raw_resid, 49, 50)
        self.pieces.append((49, 50, "linear", _fit_linear_seg(x, y, w)))

        x, y, w = _seg_curve(eff_n, raw_resid, 50, None)
        winner, lin_c, exp_popt, x0 = test_exp_vs_linear(x, y, w, label="[50, ∞)")
        self.pieces.append((50, None, "exp", (exp_popt, x0)) if winner == "exp" and exp_popt is not None
                           else (50, None, "linear", lin_c))

        # ── 5. Proposal 3: additive offset when opponent is in placement ──────
        self.opp_placement_offset = 0.0
        if "opponent_games_this_season_before" in clean.columns:
            n_opp = clean["opponent_games_this_season_before"].to_numpy(dtype=float, na_value=np.nan)
            b_pred = self._intercept(eff_n.astype(float))
            resid2 = raw_resid - b_pred
            has_opp = ~np.isnan(n_opp)
            opp_place = has_opp & (n_opp < self._KD_THRESHOLDS[0])
            opp_estab = has_opp & (n_opp >= self._KD_THRESHOLDS[0])
            if opp_place.sum() >= 100 and opp_estab.sum() >= 100:
                self.opp_placement_offset = float(
                    np.mean(resid2[opp_place]) - np.mean(resid2[opp_estab])
                )

        return self

    # ── predict / residuals ──────────────────────────────────────────────────

    def _intercept(self, n: np.ndarray) -> np.ndarray:
        out = np.zeros(len(n))
        for start, end, kind, params in self.pieces:
            mask = (n >= start) if end is None else (n >= start) & (n < end)
            if not mask.any():
                continue
            if kind == "linear":
                out[mask] = np.polyval(params, n[mask])
            else:
                popt, x0 = params
                A, lam, C = popt
                out[mask] = A * np.exp(-lam * (n[mask] - x0)) + C
        neg = n < 0
        if neg.any() and self.pieces:
            s0, _, k0, p0 = self.pieces[0]
            out[neg] = np.polyval(p0, 0.0) if k0 == "linear" else (
                lambda A, lam, C: A * np.exp(-lam * (0.0 - p0[1])) + C)(*p0[0])
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ra, rb   = self._get_skill(df)
        games    = df["games_this_season_before"].to_numpy(dtype=float, na_value=-1.0)
        avg_mmr  = (ra + rb) / 2
        res      = df["result"].values.astype(float)
        out      = np.empty(len(df))

        if self.use_mmr_tiers and self.KD_grid:
            for cell, i, j in self._cell_masks(games, avg_mmr):
                K, D = self.KD_grid[i][j]
                out[cell] = K * (res[cell] - elo_expected(ra[cell], rb[cell], D))
        else:
            for mask, K, D in zip(self._exp_masks(games), self.K_segments, self.D_segments):
                out[mask] = K * (res[mask] - elo_expected(ra[mask], rb[mask], D))

        n_player = self._effective_n(df)
        out += self._intercept(n_player)
        if self.opp_placement_offset != 0.0 and "opponent_games_this_season_before" in df.columns:
            n_opp = df["opponent_games_this_season_before"].to_numpy(dtype=float, na_value=np.nan)
            opp_place = ~np.isnan(n_opp) & (n_opp < self._KD_THRESHOLDS[0])
            out[opp_place] += self.opp_placement_offset
        return out

    def residuals(self, df: pd.DataFrame) -> np.ndarray:
        return df["observed_rating_delta"].values.astype(float) - self.predict(df)

    # ── report ───────────────────────────────────────────────────────────────

    def report(self) -> str:
        lines = [f"PiecewiseInterceptEloBaseline — changepoints: {self.changepoints}"]
        if self.use_mmr_tiers and self.KD_grid:
            lines.append(f"  K/D grid (exp × avg_mmr)  "
                         f"{'  '.join(f'{lbl:>10}' for lbl in self._MMR_LABELS)}")
            for i, exp_lbl in enumerate(self._KD_LABELS):
                cells = "  ".join(f"K={self.KD_grid[i][j][0]:.0f},D={self.KD_grid[i][j][1]:.0f}"
                                  for j in range(3))
                lines.append(f"    {exp_lbl:<15}  {cells}")
        else:
            lines.append("  KD buckets: " + ", ".join(
                f"{lbl}(K={k:.1f},D={d:.0f})"
                for lbl, k, d in zip(self._KD_LABELS, self.K_segments, self.D_segments)
            ))
        if self.opp_placement_offset != 0.0:
            lines.append(f"  opp_placement_offset = {self.opp_placement_offset:+.4f}  (added when opp games < {self._KD_THRESHOLDS[0]})")
        lines.append(f"  Intercept b(n):")
        for start, end, kind, params in self.pieces:
            end_str = str(end) if end is not None else "∞"
            if kind == "linear":
                c = params
                fn   = "linear" if len(c) == 2 and abs(float(c[0])) > 1e-10 else "constant"
                expr = f"{c[0]:+.4f}·n {c[1]:+.4f}" if fn == "linear" else f"{c[-1]:+.4f}"
            else:
                popt, x0 = params; A, lam, C = popt
                fn   = "exponential"
                expr = f"{A:+.4f}·exp(−{lam:.5f}·(n−{x0:.0f})) {C:+.4f}"
            lines.append(f"    [{start:>3}, {end_str:>4}): {fn:<12}  b = {expr}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"PiecewiseInterceptEloBaseline("
                f"mmr_tiers={self.use_mmr_tiers}, changepoints={self.changepoints}, "
                f"pieces={len(self.pieces)})")
