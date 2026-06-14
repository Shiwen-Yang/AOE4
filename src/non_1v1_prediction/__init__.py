"""
non_1v1_prediction — pre-game outcome prediction for AOE4 ranked TEAM matches (4v4).

Investigation goal: measure how *predictable* team matches are before they start,
as a proxy for matchmaking quality. A well-matched game should be close to a coin
flip; high predictability (especially from raw skill gaps) means matchmaking is
handing out decided games.

Sibling of `aoe4_predict` (1v1). Uses a SEPARATE database (`aoe4_team.duckdb`) so the
1v1 `aoe4.duckdb` is never modified; the 1v1 DB is only ATTACHed read-only to enrich
team players with their 1v1-ladder skill.
"""

__version__ = "0.1.0"
