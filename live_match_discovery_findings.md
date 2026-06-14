# Live Match Discovery Findings

This note summarizes the investigation into discovering recent or ongoing AoE4
matches for civ-pick prediction. The immediate product goal is to identify the
current match and opponent quickly enough to run the civ-choice model during the
roughly 30-second civilization-picking stage.

## Goal

Given a player profile id, discover the current live match id and opponent
profile id without relying on AoE4World's `games/last` endpoint.

Required live data:

- `match_id`
- player and opponent `profile_id`
- player and opponent `statgroup_id`
- selected or candidate civ/race ids if exposed
- map
- server region

## Tested Endpoints

### `game/advertisement/getAdvertisements`

Status: useful.

This authenticated Relic endpoint accepts known `match_ids` and returns live
advertisement rows. We confirmed that it can return ranked automatch games while
they are still live.

Example finding:

- Known live match: `237615448`
- Target profile: `20128941`
- Returned players:
  - `8390197`
  - `20128941`
- Returned statgroup ids and civ/race ids.
- Returned server information in the payload.

Important limitation: it requires candidate `match_id`s. It is not directly a
`profile_id -> match_id` lookup.

Operational use:

1. Estimate current `match_id`.
2. Scan a range around the estimate in chunks.
3. Parse returned advertisements.
4. Filter rows by target `profile_id`.

Scripts:

- `tools/relic_match_id_scanner.py`

Known batch behavior:

- Batch size around `41` ids worked.
- A batch around `101` ids returned HTTP 400 in one test.
- Large fast scans can hit HTTP 429 rate limits.

### `game/advertisement/findAdvertisements`

Status: not useful for ranked automatch discovery.

We tested this endpoint against a known live ranked match. The sweep covered:

- `doc_crc` and `patch_crc`
- match type ids `0`, `1`, `20`, `21`, `22`, `23`
- GET and POST
- single profile/statgroup/race filter
- repeated profile/statgroup plus civ/race array filter

It returned zero matches for the known live automatch case.

LibreMatch docs also note that only public games are found by
`findAdvertisements`, which matches the observed behavior.

Script:

- `tools/relic_find_advertisements_probe.py`

Conclusion: keep the probe for reference, but do not build the product path on
this endpoint.

### `game/advertisement/findObservableAdvertisements`

Status: limited.

This endpoint finds observable/spectatable games. It can return match ids and
server region data for visible observer-browser games.

It did not solve arbitrary ranked civ-pick discovery in our tests. It is useful
only if the target match is observable through the observer system.

### `game/automatch2/polling`

Status: promising for the authenticated user's own queue.

LibreMatch captures show that this is the endpoint the game client uses while
polling automatch state. Once a match is found, the documented response can
include:

- `match_id`
- self profile id
- opponent profile id
- statgroup ids
- race/civ ids
- team ids
- map name
- server IP and ports
- server region
- auth token

Likely scope:

- Works for the authenticated user's own queue/match.
- Does not work for arbitrary target profile ids.

Open question:

Can our helper call this endpoint successfully while the authenticated account is
actively queueing, or does it require exact game-client queue state such as
`partySessionID`, selected race ids, veto maps, relay pings, and active search
state?

This is the decisive future test. If it works, it is the cleanest product path
for a local app used by the player who is queueing.

### `community/leaderboard/getRecentMatchHistory`

Status: useful for completed matches, not ongoing matches.

This public endpoint returns completed recent match history. We tested it while
players were live and it did not expose the ongoing match.

Use it for historical feature updates, not live match discovery.

### `game/Leaderboard/getRecentMatchHistory`

Status: useful for completed matches, not proven useful for ongoing matches.

This is the authenticated Relic version of recent match history. It is still a
history endpoint and should not be relied on for civ-pick discovery.

### `community/leaderboard/getMatchHistory`

Status: useful after match ids are known.

Given match ids, this returns richer completed-match details. It is not a
discovery endpoint.

### `community/leaderboard/getReplayFiles`

Status: post-match only.

Given match ids, this returns replay file URLs. It is useful after completion,
not during civ-pick.

## Match ID Time Model

We tested whether `match_id` increases predictably with time.

Data source:

- `aoe4.duckdb`
- `6,740,423` distinct games
- time range: `2024-03-19` to `2026-04-22`
- id range: `118,750,992` to `229,896,380`

Findings:

- Overall drift is about `101 match_ids / minute`.
- Raw rows are not strictly monotonic by `started_at`.
- Aggregated trend is effectively monotonic:
  - median id by `1min`: only `16` violations over `801,411` minute buckets
  - median id by `5min`: `0` violations
  - median id by `1h`: `0` violations
  - median id by `1d`: `0` violations

With a fresh prior 24h sample of roughly `10k` games, simulated prediction
accuracy was:

- next `30s`: median error about `7.6k` ids, p95 about `14.5k`
- next `1m`: median error about `7.6k` ids, p95 about `14.5k`
- next `5m`: median error about `7.8k` ids, p95 about `14.4k`
- next `1h`: median error about `8.6k` ids, p95 about `14.3k`
- next `6h`: median error about `10.8k` ids, p95 about `23.3k`

Conclusion:

A fresh 24h sample can estimate current `match_id` well enough to seed a scan.
It should usually land within `+-10k` ids, and a `+-15k` scan radius should
catch most cases. A safer fallback radius is `+-25k`.

However, a cold scan of `+-15k` ids can require hundreds of Relic calls if using
batch size `41`, so the production system should keep a warm live cache instead
of starting from zero during civ-pick.

Script:

- `tools/match_id_time_regression.py`

## Does Match ID Encode Server Region?

Finding: no evidence that server region is encoded in `match_id`.

Checks performed:

- Major server regions span nearly the full match-id range.
- `match_id % 16` is roughly uniform for each top server.
- Server ordering by median match id varies by hour.
- Adjacent match ids are interleaved across server regions.

Conclusion:

`match_id` is best treated as a global-ish increasing id, not a structured id
with embedded server information. Server region should be read from the
advertisement payload returned by Relic.

## Recommended Future Architecture

For the local-player product path:

1. Authenticate the user through the Relic/Steam helper.
2. Test `game/automatch2/polling` while the authenticated account is queueing.
3. If polling exposes current match id and opponent profile id, use it as the
   primary live discovery path.
4. Use `getAdvertisements([match_id])` to enrich the match payload.
5. Fetch/build opponent historical features.
6. Run civ-choice prediction during civ-pick.

For global or arbitrary-profile discovery:

1. Maintain a background scanner using `getAdvertisements`.
2. Keep a rolling high-water `match_id` estimate.
3. Scan around the estimate continuously in small chunks.
4. Upsert live advertisements into a local cache:
   - `match_id`
   - `seen_at`
   - `profile_ids`
   - `statgroup_ids`
   - race/civ ids
   - teams
   - map
   - server region
5. Query the local cache by `profile_id` during civ-pick.
6. Expand scan radius only if the cache misses.

## Next Experiments

1. Queue on the authenticated account and capture/call
   `game/automatch2/polling`.
2. Verify whether the polling response exposes:
   - `match_id`
   - opponent `profile_id`
   - opponent `statgroup_id`
   - map
   - server region
3. Determine whether standalone polling works with our proxy or whether it needs
   live game-client parameters.
4. Build a small warm-cache scanner around `getAdvertisements` with:
   - chunk size around `41`
   - rate-limit backoff
   - center-out scan order
   - local persistence of live advertisements
5. Once the DB is updated with fresher matches, rerun:
   - `tools/match_id_time_regression.py`
   - the scanner radius tests

