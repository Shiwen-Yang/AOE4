import numpy as np
import pandas as pd

MIN_NONRANDOM_GAMES = 50
MIN_GAMES_PER_MAP = 10
MIN_ELIGIBLE_MAPS = 2
ALPHA_OVERALL = 10
ALPHA_MAP = 5
ALPHA_META = 10


def hellinger_rows(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Vectorized Hellinger distance between corresponding rows of P and Q."""
    return np.sqrt(0.5 * np.sum(
        (np.sqrt(np.clip(P, 0, None)) - np.sqrt(np.clip(Q, 0, None))) ** 2,
        axis=1,
    ))


def build_player_features(df: pd.DataFrame):
    """
    Build V1 clustering inputs and diagnostic features.

    Returns
    -------
    X_civ_sqrt : ndarray (n_v1_eligible, n_civs)
        Hellinger-transformed smoothed civ pick vectors for V1-eligible players.
    X_behavior : ndarray (n_v1_eligible, 3)
        [map_dependence_score, off_meta_score, random_civ_rate] for V1-eligible players.
    player_features : DataFrame indexed by profile_id (all base-eligible)
        Diagnostic + behavior + MMR + season metadata. Contains v1_eligible flag.
    civs : list[str]
    smoothed : DataFrame (base-eligible profile_id × civs)
        Smoothed civ distributions (pre-sqrt), used for cluster summaries.
    """
    df_picks = df[~df["civilization_randomized"]].copy()

    # ── Base eligibility ──────────────────────────────────────────────────────
    player_total = df.groupby("profile_id").agg(
        n_games=("game_id", "count"),
        random_civ_rate=("civilization_randomized", "mean"),
    )
    player_nonrandom = df_picks.groupby("profile_id").agg(
        n_nonrandom_games=("game_id", "count")
    )
    player_meta = player_total.join(player_nonrandom, how="left")
    player_meta["n_nonrandom_games"] = player_meta["n_nonrandom_games"].fillna(0).astype(int)

    base_eligible = player_meta.query("n_nonrandom_games >= @MIN_NONRANDOM_GAMES").index
    df_picks_base = df_picks[df_picks["profile_id"].isin(base_eligible)]
    df_base = df[df["profile_id"].isin(base_eligible)]

    # ── Overall civ distribution (base-eligible) ──────────────────────────────
    global_prior = df_picks_base["civilization"].value_counts(normalize=True).sort_index()
    civs = global_prior.index.tolist()

    counts = (
        df_picks_base.groupby(["profile_id", "civilization"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=civs, fill_value=0)
    )
    n_raw = counts.sum(axis=1)
    prior_counts = pd.Series(ALPHA_OVERALL * global_prior, index=civs)
    smoothed = counts.add(prior_counts, axis=1).div(n_raw + ALPHA_OVERALL, axis=0)

    # ── Diagnostic features from raw counts ───────────────────────────────────
    p_raw = counts.div(n_raw, axis=0)
    entropy_raw = -(p_raw.clip(lower=1e-12) * np.log(p_raw.clip(lower=1e-12))).sum(axis=1)
    entropy_normalized = entropy_raw / np.log(len(civs))
    effective_pool = np.exp(entropy_raw)
    sorted_shares = np.sort(p_raw.values, axis=1)[:, ::-1]
    diagnostics = pd.DataFrame({
        "top_civ": p_raw.idxmax(axis=1),
        "top_civ_share": pd.Series(sorted_shares[:, 0], index=counts.index),
        "top_3_civ_share": pd.Series(sorted_shares[:, :3].sum(axis=1), index=counts.index),
        "civ_entropy": entropy_normalized,
        "effective_civ_pool_size": effective_pool,
    })

    # ── MMR features ──────────────────────────────────────────────────────────
    mmr_agg = df_base.groupby("profile_id").agg(
        mean_mmr=("mmr", "mean"),
        n_mmr_obs=("mmr", "count"),
        n_games_all=("game_id", "count"),
    )
    mmr_agg["mmr_missing_rate"] = 1 - mmr_agg["n_mmr_obs"] / mmr_agg["n_games_all"]
    latest_mmr = (
        df_base.dropna(subset=["mmr"])
        .sort_values(["profile_id", "started_at"])
        .groupby("profile_id")
        .tail(1)[["profile_id", "mmr"]]
        .set_index("profile_id")
        .rename(columns={"mmr": "latest_mmr"})
    )

    # ── Season metadata ────────────────────────────────────────────────────────
    season_meta = _season_metadata(df_base)

    # ── V1 eligibility: 2+ maps with 10+ non-random games ─────────────────────
    games_per_player_map = df_picks_base.groupby(["profile_id", "map"]).size()
    eligible_map_count = (
        games_per_player_map[games_per_player_map >= MIN_GAMES_PER_MAP]
        .groupby("profile_id").size()
    )
    v1_eligible = (
        eligible_map_count[eligible_map_count >= MIN_ELIGIBLE_MAPS]
        .index.intersection(base_eligible)
    )

    # ── Map dependence score (V1-eligible only) ────────────────────────────────
    map_dep = _compute_map_dependence(
        df_picks_base, v1_eligible, smoothed, civs, games_per_player_map
    )

    # ── Off-meta score (all base-eligible) ────────────────────────────────────
    off_meta = _compute_off_meta_score(df_base, df_picks, smoothed, civs)

    # ── Assemble player_features ──────────────────────────────────────────────
    player_features = (
        player_meta.loc[base_eligible]
        .join(diagnostics)
        .join(mmr_agg.drop(columns=["n_games_all"]))
        .join(latest_mmr)
        .join(season_meta)
        .join(map_dep.rename("map_dependence_score"))
        .join(off_meta.rename("off_meta_score"))
    )
    player_features["v1_eligible"] = player_features.index.isin(v1_eligible)
    player_features["map_dependence_available"] = player_features["v1_eligible"]

    # ── Build clustering matrices (V1-eligible only) ───────────────────────────
    v1_idx = player_features[player_features["v1_eligible"]].index
    X_civ_sqrt = np.sqrt(smoothed.loc[v1_idx].values)
    X_behavior = (
        player_features.loc[v1_idx, ["map_dependence_score", "off_meta_score", "random_civ_rate"]]
        .fillna(0)
        .values
    )

    return X_civ_sqrt, X_behavior, player_features, civs, smoothed


def _compute_map_dependence(
    df_picks_base: pd.DataFrame,
    v1_eligible,
    smoothed: pd.DataFrame,
    civs: list,
    games_per_player_map: pd.Series,
) -> pd.Series:
    """
    Weighted average Hellinger distance between each player's map-specific civ
    distribution (smoothed toward their own overall distribution) and their overall
    civ distribution. Vectorized over all (player, map) pairs simultaneously.
    """
    eligible_pairs = (
        games_per_player_map[
            games_per_player_map.index.get_level_values("profile_id").isin(v1_eligible)
            & (games_per_player_map >= MIN_GAMES_PER_MAP)
        ]
        .reset_index(name="n_map")
    )

    if eligible_pairs.empty:
        return pd.Series(dtype=float)

    # Raw (player, map) civ counts
    map_civ = (
        df_picks_base[df_picks_base["profile_id"].isin(v1_eligible)]
        .groupby(["profile_id", "map", "civilization"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=civs, fill_value=0)
    )

    player_ids = eligible_pairs["profile_id"].values
    map_names = eligible_pairs["map"].values
    n_map = eligible_pairs["n_map"].values

    # Look up raw counts and overall distributions for each (player, map) pair
    pair_idx = pd.MultiIndex.from_arrays([player_ids, map_names])
    raw = map_civ.reindex(pair_idx, fill_value=0).values          # (n_pairs, n_civs)
    overall = smoothed.reindex(player_ids).values                  # (n_pairs, n_civs)

    # Smooth toward player's own overall distribution
    smoothed_map = (raw + ALPHA_MAP * overall) / (n_map[:, None] + ALPHA_MAP)

    eligible_pairs["hellinger"] = hellinger_rows(smoothed_map, overall)

    # Weighted average per player
    total_per_player = eligible_pairs.groupby("profile_id")["n_map"].sum().rename("total")
    eligible_pairs = eligible_pairs.join(total_per_player, on="profile_id")
    eligible_pairs["weighted_h"] = eligible_pairs["hellinger"] * eligible_pairs["n_map"] / eligible_pairs["total"]

    return eligible_pairs.groupby("profile_id")["weighted_h"].sum()


def _compute_off_meta_score(
    df_base: pd.DataFrame,
    df_picks_all: pd.DataFrame,
    smoothed: pd.DataFrame,
    civs: list,
) -> pd.Series:
    """
    Hellinger distance between each player's overall civ distribution and their
    expected meta distribution (weighted average of P(civ | map, season) over
    that player's actual map-season game counts). Fully vectorized.
    """
    # Global P(civ | map, season) smoothed toward global civ prior
    global_prior = (
        df_picks_all["civilization"].value_counts(normalize=True).reindex(civs, fill_value=0)
    )
    prior_counts = pd.Series(ALPHA_META * global_prior, index=civs)

    ms_civ = (
        df_picks_all.groupby(["map", "season", "civilization"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=civs, fill_value=0)
    )
    ms_total = df_picks_all.groupby(["map", "season"]).size()
    meta = ms_civ.add(prior_counts, axis=1).div(ms_total + ALPHA_META, axis=0)

    # Player map-season weights (all games, not just non-random)
    player_ms = (
        df_base.groupby(["profile_id", "map", "season"])
        .size()
        .reset_index(name="n")
    )
    player_ms["ms_key"] = list(zip(player_ms["map"], player_ms["season"]))

    # Keep only (map, season) pairs that exist in meta
    valid_keys = set(meta.index)
    player_ms = player_ms[player_ms["ms_key"].isin(valid_keys)]

    if player_ms.empty:
        return pd.Series(dtype=float)

    # Pivot to player × map_season weight matrix
    W = (
        player_ms.set_index(["profile_id", "ms_key"])["n"]
        .unstack(fill_value=0)
    )
    W = W.div(W.sum(axis=1), axis=0)  # row-normalize to weights

    valid_cols = [c for c in W.columns if c in meta.index]
    W = W[valid_cols]
    M = meta.loc[valid_cols].values  # (n_valid_ms, n_civs)

    expected = W.values @ M          # (n_players, n_civs)
    player_dist = smoothed.reindex(W.index).values

    scores = hellinger_rows(player_dist, expected)
    return pd.Series(scores, index=W.index)


def _season_metadata(df: pd.DataFrame) -> pd.DataFrame:
    season_counts = df.groupby(["profile_id", "season"]).size().reset_index(name="n")
    dominant_season = (
        season_counts.loc[
            season_counts.groupby("profile_id")["n"].idxmax(),
            ["profile_id", "season"],
        ]
        .set_index("profile_id")
        .rename(columns={"season": "dominant_season"})
    )
    agg = df.groupby("profile_id").agg(
        first_game_date=("started_at", "min"),
        last_game_date=("started_at", "max"),
        n_seasons=("season", "nunique"),
    )
    season_range = df.groupby("profile_id")["season"].agg(first_season="min", last_season="max")
    season_range["season_span"] = season_range["last_season"] - season_range["first_season"]
    return agg.join(season_range).join(dominant_season)
