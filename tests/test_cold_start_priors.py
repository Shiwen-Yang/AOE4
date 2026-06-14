from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoe4_predict.config import COLD_START_BASE_SKILL, COLD_START_OPPONENT_SKILL_GAP
from aoe4_predict.features import _apply_cold_start_skill_priors
from aoe4_predict.model import make_slot_swapped_rows


def _feat(skill_a, skill_b):
    return {
        "skill_a": skill_a,
        "skill_b": skill_b,
        "mmr_a": skill_a,
        "mmr_b": skill_b,
        "rating_a": None,
        "rating_b": None,
        "skill_diff": (skill_a or 0) - (skill_b or 0),
        "mmr_diff": (skill_a or 0) - (skill_b or 0),
        "rating_diff": 0,
    }


def test_known_players_keep_local_skills():
    feat = _apply_cold_start_skill_priors(_feat(1400, 1200))

    assert feat["skill_a"] == 1400
    assert feat["skill_b"] == 1200
    assert feat["skill_diff"] == 200
    assert feat["cold_start_prior_applied"] == 0
    assert feat["imputations"] == []
    assert feat["feature_sources"] == {
        "skill_a": "local_history",
        "skill_b": "local_history",
    }


def test_unknown_player_b_uses_opponent_aware_prior():
    feat = _apply_cold_start_skill_priors(_feat(1600, None))

    expected_b = 1600 - COLD_START_OPPONENT_SKILL_GAP
    assert feat["skill_b"] == expected_b
    assert feat["skill_diff"] == COLD_START_OPPONENT_SKILL_GAP
    assert feat["mmr_diff"] == COLD_START_OPPONENT_SKILL_GAP
    assert feat["rating_diff"] == COLD_START_OPPONENT_SKILL_GAP
    assert feat["cold_start_prior_applied"] == 1
    assert feat["feature_sources"]["skill_b"] == "cold_start_prior"
    assert feat["imputations"][0]["method"] == "opponent_aware"


def test_unknown_player_a_uses_opponent_aware_prior():
    feat = _apply_cold_start_skill_priors(_feat(None, 1600))

    expected_a = 1600 - COLD_START_OPPONENT_SKILL_GAP
    assert feat["skill_a"] == expected_a
    assert feat["skill_diff"] == -COLD_START_OPPONENT_SKILL_GAP
    assert feat["mmr_diff"] == -COLD_START_OPPONENT_SKILL_GAP
    assert feat["rating_diff"] == -COLD_START_OPPONENT_SKILL_GAP
    assert feat["feature_sources"]["skill_a"] == "cold_start_prior"
    assert feat["imputations"][0]["player"] == "Player A"


def test_both_unknown_players_use_global_baseline():
    feat = _apply_cold_start_skill_priors(_feat(None, None))

    assert feat["skill_a"] == COLD_START_BASE_SKILL
    assert feat["skill_b"] == COLD_START_BASE_SKILL
    assert feat["skill_diff"] == 0
    assert feat["mmr_diff"] == 0
    assert feat["rating_diff"] == 0
    assert len(feat["imputations"]) == 2
    assert {item["method"] for item in feat["imputations"]} == {"global_baseline"}


def test_make_slot_swapped_rows_inverts_directional_features():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "target": 1,
                "profile_id_a": 10,
                "profile_id_b": 20,
                "civ_a": "english",
                "civ_b": "french",
                "mmr_a": 1400,
                "mmr_b": 1200,
                "mmr_diff": 200,
                "rating_diff": 150,
                "skill_diff": 175,
                "games_diff": 12,
                "wr_diff": 0.2,
                "prior_matchup_games": 30,
                "prior_matchup_wins": 20,
                "prior_matchup_wr_a": 0.625,
            }
        ]
    )

    swapped = make_slot_swapped_rows(df)

    assert swapped.loc[0, "target"] == 0
    assert swapped.loc[0, "profile_id_a"] == 20
    assert swapped.loc[0, "profile_id_b"] == 10
    assert swapped.loc[0, "civ_a"] == "french"
    assert swapped.loc[0, "civ_b"] == "english"
    assert swapped.loc[0, "mmr_a"] == 1200
    assert swapped.loc[0, "mmr_b"] == 1400
    assert swapped.loc[0, "mmr_diff"] == -200
    assert swapped.loc[0, "rating_diff"] == -150
    assert swapped.loc[0, "skill_diff"] == -175
    assert swapped.loc[0, "games_diff"] == -12
    assert swapped.loc[0, "wr_diff"] == -0.2
    assert swapped.loc[0, "prior_matchup_wins"] == 10
    assert swapped.loc[0, "prior_matchup_wr_a"] == 0.375
