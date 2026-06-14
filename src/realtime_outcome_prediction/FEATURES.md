# Realtime Outcome Prediction — Feature Reference

**Dataset**: `data/realtime_outcome_prediction/features/v1/snapshots.parquet`  
**Total columns**: 362 (350 model features + 12 meta/label columns)  
**Snapshot checkpoints**: every 5 minutes up to observed game end

---

## Row structure

Each row is one **snapshot** for one match at a specific minute mark. A 25-minute game produces 5 rows (5, 10, 15, 20, 25 min).

### Meta columns (excluded from training)

| Column | Description |
|--------|-------------|
| `replay_id` | Game ID (may differ from `base_replay_id` for swapped rows) |
| `base_replay_id` | Original game ID |
| `snapshot_minute` | Checkpoint in minutes (5, 10, 15, …) |
| `snapshot_time_s` | Checkpoint in seconds |
| `snapshot_phase` | `early` (≤10 min) / `mid` (11–20 min) / `late` (>20 min) |
| `latest_event_time_s` | Timestamp of the last event seen before this checkpoint |
| `match_duration_observed_s` | Total observed game duration in seconds |
| `snapshots_in_match` | Number of snapshot rows this match contributes |
| `split` | `train` / `valid` / `test` |
| `is_swapped` | `True` for slot-swapped augmentation rows (train only) |
| `target` | 1 if slot1 wins, 0 if slot2 wins |
| `row_weight` | `1 / snapshots_in_match` (equalises match influence) |

---

## Feature columns

All features are **cumulative from game-start up to the snapshot minute** (not deltas between snapshots). Every base metric is expanded into up to 5 views:

| Prefix | Meaning |
|--------|---------|
| `slot1_*` | Player 1's value |
| `slot2_*` | Player 2's value |
| `sum_*` | slot1 + slot2 (combined game total) |
| `diff_*` | slot1 − slot2 (advantage/deficit) |
| `share_slot1_*` | slot1 / (slot1 + slot2) (relative share, omitted when sum = 0) |

Most count-based metrics also have a `_per_min` rate variant (cumulative / elapsed minutes).

---

### 1. Event volume

Raw event count (all intents combined).

- `{prefix}_events` — total events
- `{prefix}_events_per_min`

---

### 2. Intent types

Counts of each high-level intent issued by a player.

| Base metric | What it counts |
|-------------|----------------|
| `intent_building_count` | Building placement orders |
| `intent_unit_production_count` | Unit train/queue orders |
| `intent_technology_research_count` | Technology research orders |
| `intent_unit_action_count` | Unit move/attack/patrol orders |
| `intent_villager_action_count` | Villager task assignments |
| `intent_cancel_count` | Any cancel order |

Each has `slot1_`, `slot2_`, `sum_`, `diff_`, `share_slot1_`, and `_per_min` variants.

---

### 3. Buildings

Cumulative resource cost and count of all buildings placed up to this checkpoint. Costs come from the AoE4 World metadata catalogue; buildings with unknown IDs are counted separately.

| Base metric | What it measures |
|-------------|-----------------|
| `building_known_metadata_count` | Count of identified buildings placed |
| `building_unknown_metadata_count` | Count of unidentified buildings placed |
| `building_known_cost_food` | Total food cost of identified buildings |
| `building_known_cost_wood` | Total wood cost |
| `building_known_cost_gold` | Total gold cost |
| `building_known_cost_stone` | Total stone cost |
| `building_known_cost_time` | Total build-time cost (seconds) |
| `building_known_cost_population` | Total population cost |
| `building_known_cost_total_resources` | food + wood + gold + stone combined |

---

### 4. Units

Cumulative resource cost and count of all units trained.

| Base metric | What it measures |
|-------------|-----------------|
| `unit_known_metadata_count` | Count of identified units trained |
| `unit_unknown_metadata_count` | Count of unidentified units trained |
| `unit_known_cost_food` | Total food cost |
| `unit_known_cost_wood` | Total wood cost |
| `unit_known_cost_gold` | Total gold cost |
| `unit_known_cost_stone` | Total stone cost |
| `unit_known_cost_time` | Total train-time (seconds) |
| `unit_known_cost_population` | Total population cost |
| `unit_known_cost_total_resources` | food + wood + gold + stone combined |
| `unit_action_stop_or_cancel_action_count` | Stop/cancel orders issued to units |

---

### 5. Technology

Cumulative resource cost and count of all technologies researched.

| Base metric | What it measures |
|-------------|-----------------|
| `technology_known_metadata_count` | Count of identified techs researched |
| `technology_unknown_metadata_count` | Count of unidentified techs researched |
| `technology_known_cost_food` | Total food cost |
| `technology_known_cost_wood` | Total wood cost |
| `technology_known_cost_gold` | Total gold cost |
| `technology_known_cost_stone` | Total stone cost |
| `technology_known_cost_time` | Total research time (seconds) |
| `technology_known_cost_population` | Total population cost |
| `technology_known_cost_total_resources` | food + wood + gold + stone combined |

---

### 6. Cancellations

Counts of cancelled production queue items, broken down by what was cancelled.

| Base metric | What it counts |
|-------------|----------------|
| `cancel_item_kind_unit_production_count` | Unit productions cancelled |
| `cancel_item_kind_technology_research_count` | Tech researches cancelled |
| `cancel_item_kind_unknown_count` | Unknown queue item cancellations |
| `cancel_kind_foundation_count` | Building foundations cancelled (before completion) |
| `cancel_kind_queue_item_count` | Generic queue item cancellations |

---

### 7. Production queue

| Base metric | What it measures |
|-------------|-----------------|
| `queued_count` | Total items ever added to the production queue |

---

### 8. Producer links

Attribution of unit productions to the building that produced them. Unresolved counts indicate how many productions couldn't be matched to a specific building (sparse replay data or ambiguous state).

| Base metric | What it measures |
|-------------|-----------------|
| `producer_link_linked_observed_building_count` | Productions linked to a tracked building |
| `producer_link_linked_starting_town_center_count` | Productions linked to the starting TC |
| `producer_link_unresolved_ambiguous_order_count` | Unresolved: multiple compatible buildings |
| `producer_link_unresolved_missing_catalog_entry_count` | Unresolved: unit not in metadata |
| `producer_link_unresolved_no_compatible_building_count` | Unresolved: no matching building found |

---

### 9. Villagers

Worker-specific activity signals.

| Base metric | What it measures |
|-------------|-----------------|
| `villager_action_gather_or_interact_assignment_count` | Gather/interact task assignments |
| `villager_resource_unknown_count` | Interactions with unidentified resources |
| `villager_selected_unit_count_sum` | Villager selection events (proxy for micro attention) |

---

## What is NOT currently included

- **Time-delta features**: changes *between* snapshots (e.g. units trained in the last 5 min). Only cumulative totals and cumulative/time rates are present.
- **Per-entity (pbgid) features**: individual unit/building/tech type breakdowns. Removed due to extreme sparsity (~32K columns across 7K games).
- **Pre-game features**: player MMR, season win rates, civ pick history (from `aoe4_predict`). 93% of replay players exist in the S10+S11 dataset and could be joined in.
- **Map / civilisation**: not extracted from replays yet.
