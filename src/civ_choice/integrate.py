"""Marginalized win-probability: P(A wins | players, map) via civ-choice predictions.

    P(A wins) = Σ_civA Σ_civB  P(civA|A,map) × P(civB|B,map) × P(A wins|A,B,map,civA,civB)

All 18×18=324 win-model calls are batched into one predict() call.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from aoe4_predict.config import DB_PATH
from aoe4_predict.db import get_conn
from aoe4_predict.predict import predict_match

from .predict import predict_civ_distribution


def predict_with_civ_marginalization(
    player_a_id: int,
    player_b_id: int,
    map_name: str | None = None,
    as_of: datetime | None = None,
    db_path: str | None = None,
    win_model=None,
    win_meta: dict | None = None,
    civ_model=None,
) -> dict[str, Any]:
    """Compute win probability marginalized over predicted civ choices.

    Returns dict with:
        win_prob_a          — marginalized P(A wins)
        civ_dist_a          — P(civ | A, map) distribution
        civ_dist_b          — P(civ | B, map) distribution
        top_combo           — (civA, civB) pair with highest joint probability
        top_combo_win_prob  — win prob for the top combo
        mass_covered        — Σ P(civA)*P(civB) for all evaluated combos (= 1.0 for full grid)
    """
    if as_of is None:
        as_of = datetime.utcnow()

    conn = get_conn(db_path or DB_PATH, read_only=True)
    try:
        dist_a = predict_civ_distribution(
            player_a_id, map_name=map_name, as_of=as_of, conn=conn, model=civ_model
        )
        dist_b = predict_civ_distribution(
            player_b_id, map_name=map_name, as_of=as_of, conn=conn, model=civ_model
        )
    finally:
        conn.close()

    civs_a = list(dist_a.keys())
    civs_b = list(dist_b.keys())

    # Build 324-row feature matrix for all (civA, civB) combinations
    combos = [(ca, cb) for ca in civs_a for cb in civs_b]
    win_probs = []
    for civ_a, civ_b in combos:
        result = predict_match(
            player_a_id=player_a_id,
            player_b_id=player_b_id,
            civ_a=civ_a,
            civ_b=civ_b,
            map_name=map_name,
            db_path=db_path or DB_PATH,
            model=win_model,
            meta=win_meta,
        )
        win_probs.append(result.get("win_prob_a", 0.5))

    win_probs = np.array(win_probs)
    joint_probs = np.array([dist_a[ca] * dist_b[cb] for ca, cb in combos])
    mass = float(joint_probs.sum())
    if mass > 0:
        joint_probs /= mass  # renormalize in case distributions don't sum to exactly 1

    marginalized_win_prob = float(np.dot(joint_probs, win_probs))

    best_idx = int(np.argmax(joint_probs))
    best_combo = combos[best_idx]

    return {
        "player_a_id": player_a_id,
        "player_b_id": player_b_id,
        "map_name": map_name,
        "win_prob_a": round(marginalized_win_prob, 4),
        "win_prob_b": round(1 - marginalized_win_prob, 4),
        "civ_dist_a": dist_a,
        "civ_dist_b": dist_b,
        "top_combo": {"civ_a": best_combo[0], "civ_b": best_combo[1]},
        "top_combo_win_prob": round(float(win_probs[best_idx]), 4),
        "mass_covered": round(mass, 6),
    }
