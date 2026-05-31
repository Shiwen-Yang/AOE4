"""
Prediction entry point.

Given two player IDs and optional context (map, civ_a, civ_b),
returns a structured prediction dict.
"""
from typing import Any

from .db import get_conn
from .features import get_inference_features
from .model import load_model, predict_from_features


def predict_match(
    player_a_id: int,
    player_b_id: int,
    civ_a: str | None = None,
    civ_b: str | None = None,
    map_name: str | None = None,
    conn=None,
    db_path=None,
    model=None,
    meta: dict | None = None,
) -> dict[str, Any]:
    """
    Returns:
        {
            player_a_id, player_b_id,
            context_level: "id_only" | "map_known" | "civ_known" | "full_context"
            win_prob_a: float,  # P(player A wins)
            features: dict,     # raw feature values used
            warnings: list[str],
            model_meta: dict,
        }
    """
    own_conn = conn is None
    if own_conn:
        conn = get_conn(db_path, read_only=True)

    if model is None:
        model, meta = load_model()

    feat = get_inference_features(
        player_a_id=player_a_id,
        player_b_id=player_b_id,
        civ_a=civ_a,
        civ_b=civ_b,
        map_name=map_name,
        conn=conn,
    )

    feature_cols = meta["feature_cols"]
    win_prob_a = predict_from_features(model, feat, feature_cols)

    # Determine context level
    if civ_a and civ_b and map_name:
        context = "full_context"
    elif civ_a and civ_b:
        context = "civ_known"
    elif map_name:
        context = "map_known"
    else:
        context = "id_only"

    # Reliability warnings
    warnings: list[str] = []
    if feat["games_lifetime_a"] < 10:
        warnings.append(f"Player A has only {feat['games_lifetime_a']} prior RM 1v1 games — prediction less reliable.")
    if feat["games_lifetime_b"] < 10:
        warnings.append(f"Player B has only {feat['games_lifetime_b']} prior RM 1v1 games — prediction less reliable.")
    if feat["missing_mmr_a"] and not feat["missing_rating_a"]:
        warnings.append("MMR missing for Player A — using visible rating as fallback.")
    if feat["missing_mmr_b"] and not feat["missing_rating_b"]:
        warnings.append("MMR missing for Player B — using visible rating as fallback.")
    if feat["missing_skill_a"]:
        warnings.append("No MMR or rating found for Player A — skill signal absent.")
    if feat["missing_skill_b"]:
        warnings.append("No MMR or rating found for Player B — skill signal absent.")
    if context == "id_only":
        warnings.append("No civ or map context provided — prediction is less specific.")
    if feat.get("prior_matchup_games", 0) < 20 and civ_a and civ_b:
        warnings.append(
            f"Limited historical data for {civ_a} vs {civ_b} matchup "
            f"({int(feat.get('prior_matchup_games', 0))} prior games) — prediction reverts toward global prior."
        )

    if own_conn:
        conn.close()

    return {
        "player_a_id": player_a_id,
        "player_b_id": player_b_id,
        "civ_a": civ_a,
        "civ_b": civ_b,
        "map_name": map_name,
        "context_level": context,
        "win_prob_a": round(win_prob_a, 4),
        "win_prob_b": round(1 - win_prob_a, 4),
        "features": feat,
        "warnings": warnings,
        "model_meta": {
            "patch": feat.get("patch"),
            "season": feat.get("season"),
            "n_trees": meta.get("n_trees"),
            "valid_auc": meta.get("metrics", {}).get("valid", {}).get("auc"),
        },
    }
