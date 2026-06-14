# Outcome Prediction Hydration API Notes

The current outcome predictor does not call external APIs during inference.
These endpoints are candidates for a later background hydration/cache layer.

## AoE4World Player Profile

Endpoint:

```text
https://aoe4world.com/api/v0/players/{profile_id}
```

Useful fields observed:

- `profile_id`, `name`, `country`
- `modes.rm_solo.rating`
- `modes.rm_solo.rank`, `rank_level`
- `modes.rm_solo.games_count`, `wins_count`, `losses_count`
- `modes.rm_solo.last_game_at`
- `modes.rm_solo.rating_history`

Best use:

- Fill current visible rating and coarse win-rate priors for a player absent
  from the local DuckDB.
- Mark values as external/profile-level, not as locally hydrated match history.

## AoE4World Recent Games

Endpoint:

```text
https://aoe4world.com/api/v0/players/{profile_id}/games?mode=rm_solo&page=1
```

Useful fields observed:

- `game_id`, `started_at`, `duration`, `map`, `kind`, `season`, `patch`
- per-player `profile_id`, `result`, `civilization`,
  `civilization_randomized`, `rating`, `rating_diff`, `mmr`, `mmr_diff`

Best use:

- Background hydration of recent `games` and `participants` rows.
- Refresh stale players without blocking the prediction request path.

## Official World's Edge Recent Match History

Endpoint:

```text
https://aoe-api.worldsedgelink.com/community/leaderboard/getRecentMatchHistory?title=age4&profile_ids=[{profile_id}]
```

This endpoint responds successfully, but the payload is lower-level and less
directly aligned with the local `games` and `participants` schema. Treat it as a
fallback candidate if AoE4World is unavailable or insufficient.
