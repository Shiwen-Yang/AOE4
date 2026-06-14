"""
Terminal-formatted prediction output.
"""
from typing import Any


def _bar(prob: float, width: int = 20) -> str:
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_player_line(label: str, feat: dict, suffix: str) -> str:
    mmr = feat.get(f"mmr_{suffix}")
    rating = feat.get(f"rating_{suffix}")
    games = feat.get(f"games_lifetime_{suffix}", 0)
    wr = feat.get(f"overall_wr_{suffix}", 0.5)

    skill_str = ""
    if feat.get("feature_sources", {}).get(f"skill_{suffix}") == "cold_start_prior":
        skill_str = f"Imputed skill {int(feat.get(f'skill_{suffix}'))} (cold-start prior)"
    elif mmr is not None:
        skill_str = f"MMR {int(mmr)}"
    elif rating is not None:
        skill_str = f"Rating {int(rating)} (MMR unavailable)"
    else:
        skill_str = "No skill data"

    return f"  {label}: {skill_str}  |  {games} RM 1v1 games  |  {wr*100:.1f}% lifetime WR"


def _fmt_civ_line(label: str, feat: dict, suffix: str, civ: str | None) -> str:
    if not civ:
        return f"  {label} civ: not specified"
    games = feat.get(f"civ_games_{suffix}", 0)
    wr = feat.get(f"civ_wr_{suffix}", 0.5)
    return f"  {label} ({civ}): {games} prior games with this civ  |  {wr*100:.1f}% WR"


def _fmt_map_line(feat: dict, map_name: str | None) -> list[str]:
    if not map_name:
        return []
    ga = feat.get("map_games_a", 0)
    gb = feat.get("map_games_b", 0)
    wa = feat.get("map_wr_a", 0.5)
    wb = feat.get("map_wr_b", 0.5)
    return [
        f"Map ({map_name}):",
        f"  Player A: {ga} games on this map  |  {wa*100:.1f}% WR",
        f"  Player B: {gb} games on this map  |  {wb*100:.1f}% WR",
    ]


def format_prediction(pred: dict[str, Any]) -> str:
    feat = pred["features"]
    pa = pred["win_prob_a"]
    pb = pred["win_prob_b"]
    civ_a = pred.get("civ_a")
    civ_b = pred.get("civ_b")
    map_name = pred.get("map_name")
    ctx = pred["context_level"].replace("_", " ").title()

    lines = [
        "",
        "=" * 54,
        "  AOE4 RM 1v1 Match Prediction",
        "=" * 54,
        f"  Player A:  {pred['player_a_id']}",
        f"  Player B:  {pred['player_b_id']}",
        f"  Context:   {ctx}",
    ]
    if feat.get("patch"):
        lines.append(f"  Patch:     {feat['patch']}")
    if feat.get("season"):
        lines.append(f"  Season:    {feat['season']}")
    lines.append("")

    # Win probability bar
    civ_a_label = f" ({civ_a})" if civ_a else ""
    civ_b_label = f" ({civ_b})" if civ_b else ""
    lines += [
        "Win Probability",
        f"  Player A{civ_a_label}: {pa*100:.1f}%  {_bar(pa)}",
        f"  Player B{civ_b_label}: {pb*100:.1f}%  {_bar(pb)}",
        "",
    ]

    # Player stats
    lines += [
        "Player History",
        _fmt_player_line("Player A", feat, "a"),
        _fmt_player_line("Player B", feat, "b"),
        "",
    ]

    # Civ stats (if known)
    if civ_a or civ_b:
        lines += [
            "Civilization Stats (prior games)",
            _fmt_civ_line("Player A", feat, "a", civ_a),
            _fmt_civ_line("Player B", feat, "b", civ_b),
            "",
        ]

    # Matchup prior (if civs known)
    if civ_a and civ_b:
        prior_games = feat.get("prior_matchup_games", 0)
        prior_wr = feat.get("prior_matchup_wr_a", 0.5)
        lines += [
            "Civ Matchup (historical prior, previous seasons)",
            f"  {civ_a} vs {civ_b}: {prior_wr*100:.1f}% WR for Player A  (n={int(prior_games):,} games)",
            "",
        ]

    # Map stats
    map_lines = _fmt_map_line(feat, map_name)
    if map_lines:
        lines += map_lines + [""]

    # Main factors
    lines.append("Main Factors")
    skill_a = feat.get("skill_a")
    skill_b = feat.get("skill_b")
    if skill_a is not None and skill_b is not None:
        diff = int(skill_a - skill_b)
        favor = "Player A" if diff > 0 else "Player B"
        lines.append(f"  Skill difference: {abs(diff):+d} in favor of {favor} (MMR or rating)")
    wr_diff = feat.get("wr_diff", 0.0)
    if abs(wr_diff) > 0.02:
        favor = "Player A" if wr_diff > 0 else "Player B"
        lines.append(f"  Win-rate edge: {abs(wr_diff)*100:.1f}pp in favor of {favor}")
    if feat.get("prior_matchup_games", 0) >= 20 and civ_a and civ_b:
        lines.append(
            f"  {civ_a} vs {civ_b} prior: {feat.get('prior_matchup_wr_a', 0.5)*100:.1f}% WR for {civ_a} "
            f"({int(feat.get('prior_matchup_games', 0)):,} games)"
        )

    # Model info
    m = pred.get("model_meta", {})
    if m.get("valid_auc"):
        lines += ["", f"  Model valid AUC: {m['valid_auc']:.4f}  ({m.get('n_trees', '?')} trees)"]

    # Warnings
    if pred["warnings"]:
        lines += ["", "Reliability Warnings"]
        for w in pred["warnings"]:
            lines.append(f"  ⚠  {w}")

    lines.append("=" * 54)
    return "\n".join(lines)
