import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import PHASES, SNAPSHOT_MINUTES

RESOURCE_KEYS = ("food", "wood", "gold", "stone", "time", "population")


def snapshot_phase(minute: int) -> str:
    for name, start, end in PHASES:
        if minute >= start and (end is None or minute <= end):
            return name
    return "unknown"


def checkpoint_minutes(max_time_s: float, interval_minutes: int = SNAPSHOT_MINUTES) -> list[int]:
    max_minute = int(max_time_s // 60)
    return list(range(interval_minutes, max_minute + 1, interval_minutes))


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    events.sort(key=lambda row: (float(row.get("time_s") or 0), int(row.get("command_index") or 0)))
    return events


def _safe_token(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value).lower().replace(" ", "_").replace("/", "_")


def _inc(counter: Counter, key: str, amount: float = 1.0) -> None:
    counter[key] += amount


def _add_cost_features(counter: Counter, prefix: str, pbgid: Any, pbgid_index: dict[int, dict]) -> None:
    try:
        pbgid_int = int(pbgid)
    except (TypeError, ValueError):
        _inc(counter, f"{prefix}_unknown_metadata_count")
        return

    meta = pbgid_index.get(pbgid_int)
    if not meta:
        _inc(counter, f"{prefix}_unknown_metadata_count")
        return

    costs = meta.get("costs") or {}
    for resource in RESOURCE_KEYS:
        _inc(counter, f"{prefix}_known_cost_{resource}", float(costs.get(resource, 0.0)))
    _inc(counter, f"{prefix}_known_cost_total_resources", float(meta.get("total_resources") or 0.0))
    _inc(counter, f"{prefix}_known_metadata_count")


def event_features(event: dict[str, Any], pbgid_index: dict[int, dict]) -> dict[str, float]:
    category = _safe_token(event.get("intent_category"))
    intent = event.get("intent") or {}
    feats: Counter = Counter()

    _inc(feats, "events")
    _inc(feats, f"intent_{category}_count")
    if intent.get("is_queued"):
        _inc(feats, "queued_count")

    if category == "unit_production":
        pbgid = intent.get("unit_pbgid")
        _inc(feats, f"unit_pbgid_{_safe_token(pbgid)}_count")
        _add_cost_features(feats, "unit", pbgid, pbgid_index)
        _inc(feats, f"producer_link_{_safe_token(intent.get('producer_building_link_status'))}_count")
    elif category == "building":
        pbgid = intent.get("pbgid")
        _inc(feats, f"building_pbgid_{_safe_token(pbgid)}_count")
        _add_cost_features(feats, "building", pbgid, pbgid_index)
    elif category == "technology_research":
        pbgid = intent.get("technology_pbgid")
        _inc(feats, f"technology_pbgid_{_safe_token(pbgid)}_count")
        _add_cost_features(feats, "technology", pbgid, pbgid_index)
    elif category == "villager_action":
        _inc(feats, f"villager_action_{_safe_token(intent.get('action_kind'))}_count")
        _inc(feats, f"villager_resource_{_safe_token(intent.get('resource_kind'))}_count")
        count = intent.get("selected_unit_count")
        if isinstance(count, (int, float)):
            _inc(feats, "villager_selected_unit_count_sum", float(count))
    elif category == "cancel":
        _inc(feats, f"cancel_kind_{_safe_token(intent.get('cancel_kind'))}_count")
        _inc(feats, f"cancel_item_kind_{_safe_token(intent.get('canceled_item_kind'))}_count")
        pbgid = intent.get("canceled_pbgid")
        if pbgid is not None:
            _inc(feats, f"cancel_pbgid_{_safe_token(pbgid)}_count")
    elif category == "unit_action":
        _inc(feats, f"unit_action_{_safe_token(intent.get('action_kind'))}_count")

    return dict(feats)


def _slot_prefixed(slot_totals: dict[int, Counter]) -> dict[str, float]:
    row: dict[str, float] = {}
    keys = set(slot_totals[1]) | set(slot_totals[2])
    for key in keys:
        v1 = float(slot_totals[1].get(key, 0.0))
        v2 = float(slot_totals[2].get(key, 0.0))
        row[f"slot1_{key}"] = v1
        row[f"slot2_{key}"] = v2
        row[f"sum_{key}"] = v1 + v2
        row[f"diff_{key}"] = v1 - v2
        denom = v1 + v2
        if denom > 0:
            row[f"share_slot1_{key}"] = v1 / denom
    return row


_AGE_TIERS = (2, 3, 4)
_AGE_CONTENT_TYPES = ("unit", "tech", "building")
_NAN = float("nan")


def _add_rate_features(row: dict[str, float], minute: int) -> None:
    if minute <= 0:
        return
    for key, value in list(row.items()):
        if key.startswith("share_") or key.startswith("delta_"):
            continue
        if key.endswith("_count") or key.endswith("_events") or key in ("slot1_events", "slot2_events", "sum_events"):
            row[f"{key}_per_min"] = float(value) / minute


def _track_age_events(
    event: dict[str, Any],
    time_s: float,
    slot: int,
    pbgid_index: dict[int, dict],
    first_times: dict[int, dict[str, float]],
) -> None:
    """Record first-occurrence times for age-progression events per slot."""
    category = _safe_token(event.get("intent_category"))
    intent = event.get("intent") or {}
    times = first_times[slot]

    def _meta(pbgid_val):
        try:
            return pbgid_index.get(int(pbgid_val))
        except (TypeError, ValueError):
            return None

    if category == "building":
        meta = _meta(intent.get("pbgid"))
        if meta:
            if meta.get("is_age_up_building"):
                tier = meta.get("ageup_to_tier")
                if tier is not None and f"ageup{tier}_commit_s" not in times:
                    times[f"ageup{tier}_commit_s"] = time_s
            age = meta.get("age")
            if age and age >= 2 and not meta.get("is_age_up_building"):
                key = f"first_age{age}_building_s"
                if key not in times:
                    times[key] = time_s

    elif category == "technology_research":
        meta = _meta(intent.get("technology_pbgid"))
        if meta:
            if meta.get("is_age_up_tech"):
                age = meta.get("age")
                if age is not None and f"ageup{age}_commit_s" not in times:
                    times[f"ageup{age}_commit_s"] = time_s
            else:
                age = meta.get("age")
                if age and age >= 2:
                    key = f"first_age{age}_tech_s"
                    if key not in times:
                        times[key] = time_s

    elif category == "unit_production":
        meta = _meta(intent.get("unit_pbgid"))
        if meta:
            age = meta.get("age")
            if age and age >= 2:
                key = f"first_age{age}_unit_s"
                if key not in times:
                    times[key] = time_s


def _add_age_snapshot_features(
    row: dict[str, Any],
    first_times: dict[int, dict[str, float]],
) -> None:
    """Expand first_times into slot-prefixed row features plus cross-slot diffs."""
    for slot in (1, 2):
        prefix = f"slot{slot}_"
        times = first_times[slot]
        for tier in _AGE_TIERS:
            row[f"{prefix}ageup{tier}_commit_s"] = float(times[f"ageup{tier}_commit_s"]) if f"ageup{tier}_commit_s" in times else _NAN
            for ct in _AGE_CONTENT_TYPES:
                row[f"{prefix}first_age{tier}_{ct}_s"] = float(times[f"first_age{tier}_{ct}_s"]) if f"first_age{tier}_{ct}_s" in times else _NAN
        for tier in (2, 3):
            commit = times.get(f"ageup{tier}_commit_s")
            first_unit = times.get(f"first_age{tier}_unit_s")
            row[f"{prefix}age{tier}_commit_to_unit_s"] = (float(first_unit) - float(commit)) if (commit is not None and first_unit is not None) else _NAN

    for tier in _AGE_TIERS:
        c1 = first_times[1].get(f"ageup{tier}_commit_s")
        c2 = first_times[2].get(f"ageup{tier}_commit_s")
        row[f"diff_ageup{tier}_commit_s"] = (float(c1) - float(c2)) if (c1 is not None and c2 is not None) else _NAN
        for ct in _AGE_CONTENT_TYPES:
            v1 = first_times[1].get(f"first_age{tier}_{ct}_s")
            v2 = first_times[2].get(f"first_age{tier}_{ct}_s")
            row[f"diff_first_age{tier}_{ct}_s"] = (float(v1) - float(v2)) if (v1 is not None and v2 is not None) else _NAN


def _add_fraction_features(row: dict[str, float]) -> None:
    """Add within-player resource composition fractions (F3)."""
    for slot in ("slot1", "slot2"):
        for cat in ("unit", "building", "technology"):
            total = row.get(f"{slot}_{cat}_known_cost_total_resources") or 0.0
            denom = max(total, 1e-6)
            for resource in ("food", "wood", "gold", "stone"):
                row[f"{slot}_{cat}_{resource}_frac"] = (row.get(f"{slot}_{cat}_known_cost_{resource}") or 0.0) / denom
        unit_t = row.get(f"{slot}_unit_known_cost_total_resources") or 0.0
        bldg_t = row.get(f"{slot}_building_known_cost_total_resources") or 0.0
        tech_t = row.get(f"{slot}_technology_known_cost_total_resources") or 0.0
        all_t = max(unit_t + bldg_t + tech_t, 1e-6)
        row[f"{slot}_unit_spend_frac"] = unit_t / all_t
        row[f"{slot}_building_spend_frac"] = bldg_t / all_t
        row[f"{slot}_tech_spend_frac"] = tech_t / all_t

    for suffix in (
        "unit_food_frac", "unit_wood_frac", "unit_gold_frac", "unit_stone_frac",
        "building_food_frac", "building_wood_frac", "building_gold_frac", "building_stone_frac",
        "technology_food_frac", "technology_wood_frac", "technology_gold_frac", "technology_stone_frac",
        "unit_spend_frac", "building_spend_frac", "tech_spend_frac",
    ):
        row[f"diff_{suffix}"] = (row.get(f"slot1_{suffix}") or 0.0) - (row.get(f"slot2_{suffix}") or 0.0)


def _add_cancel_features(row: dict[str, float]) -> None:
    """Add cancellation efficiency rates (F4)."""
    for slot in ("slot1", "slot2"):
        u_int = row.get(f"{slot}_intent_unit_production_count") or 0.0
        u_can = row.get(f"{slot}_cancel_item_kind_unit_production_count") or 0.0
        t_int = row.get(f"{slot}_intent_technology_research_count") or 0.0
        t_can = row.get(f"{slot}_cancel_item_kind_technology_research_count") or 0.0
        b_int = row.get(f"{slot}_intent_building_count") or 0.0
        b_can = row.get(f"{slot}_cancel_kind_foundation_count") or 0.0
        row[f"{slot}_unit_cancel_rate"] = u_can / max(u_int, 1.0)
        row[f"{slot}_tech_cancel_rate"] = t_can / max(t_int, 1.0)
        row[f"{slot}_building_cancel_rate"] = b_can / max(b_int, 1.0)
        row[f"{slot}_net_unit_production"] = u_int - u_can

    for suffix in ("unit_cancel_rate", "tech_cancel_rate", "building_cancel_rate", "net_unit_production"):
        row[f"diff_{suffix}"] = (row.get(f"slot1_{suffix}") or 0.0) - (row.get(f"slot2_{suffix}") or 0.0)


def build_match_snapshots(
    replay_id: int,
    events: list[dict[str, Any]],
    pbgid_index: dict[int, dict],
    target: int | None = None,
    split: str | None = None,
    include_swapped: bool = False,
    include_delta: bool = False,
    include_age_features: bool = False,
    include_fractions: bool = False,
    include_cancel_eff: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not events:
        return [], {"replay_id": replay_id, "snapshot_count": 0, "duration_s": 0.0}

    max_time_s = max(float(event.get("time_s") or 0.0) for event in events)
    minutes = checkpoint_minutes(max_time_s)
    slot_totals: dict[int, Counter] = defaultdict(Counter)
    prev_totals: dict[int, Counter] = {1: Counter(), 2: Counter()}
    first_times: dict[int, dict[str, float]] = {1: {}, 2: {}}
    rows: list[dict[str, Any]] = []
    event_idx = 0
    latest_event_time_s = 0.0

    for minute in minutes:
        cutoff = minute * 60
        while event_idx < len(events) and float(events[event_idx].get("time_s") or 0.0) <= cutoff:
            event = events[event_idx]
            latest_event_time_s = float(event.get("time_s") or 0.0)
            slot = int(event.get("player_slot") or 0)
            if slot in (1, 2):
                slot_totals[slot].update(event_features(event, pbgid_index))
                if include_age_features:
                    _track_age_events(event, latest_event_time_s, slot, pbgid_index, first_times)
            event_idx += 1

        row = {
            "replay_id": replay_id,
            "base_replay_id": replay_id,
            "snapshot_minute": minute,
            "snapshot_time_s": cutoff,
            "snapshot_phase": snapshot_phase(minute),
            "latest_event_time_s": latest_event_time_s,
            "match_duration_observed_s": max_time_s,
            "split": split,
            "is_swapped": False,
        }
        if target is not None:
            row["target"] = int(target)

        row.update(_slot_prefixed(slot_totals))
        # Rate features only on cumulative columns (skip delta_ prefix keys added below)
        _add_rate_features(row, minute)

        if include_delta:
            delta_totals = {
                s: Counter({k: max(0, slot_totals[s].get(k, 0) - prev_totals[s].get(k, 0)) for k in slot_totals[s]})
                for s in (1, 2)
            }
            delta_prefixed = _slot_prefixed(delta_totals)
            row.update({f"delta_{k}": v for k, v in delta_prefixed.items()})
            prev_totals = {s: Counter(slot_totals[s]) for s in (1, 2)}

        if include_age_features:
            _add_age_snapshot_features(row, first_times)

        if include_fractions:
            _add_fraction_features(row)

        if include_cancel_eff:
            _add_cancel_features(row)

        rows.append(row)

    snapshot_count = len(rows)
    if snapshot_count:
        for row in rows:
            row["snapshots_in_match"] = snapshot_count
            row["row_weight"] = 1.0 / snapshot_count

    if include_swapped:
        swapped = [swap_snapshot_row(row) for row in rows]
        for row in rows + swapped:
            row["row_weight"] = 1.0 / (2 * snapshot_count) if snapshot_count else 0.0
        rows.extend(swapped)

    return rows, {"replay_id": replay_id, "snapshot_count": snapshot_count, "duration_s": max_time_s}


def swap_snapshot_row(row: dict[str, Any]) -> dict[str, Any]:
    swapped = dict(row)
    swapped["is_swapped"] = True
    if "target" in swapped and swapped["target"] is not None:
        swapped["target"] = 1 - int(swapped["target"])
    if "pregame_slot1_win_prob" in swapped and swapped["pregame_slot1_win_prob"] is not None:
        swapped["pregame_slot1_win_prob"] = 1.0 - float(swapped["pregame_slot1_win_prob"])

    for key, value in list(row.items()):
        if key.startswith("slot1_"):
            suffix = key[len("slot1_"):]
            other = "slot2_" + suffix
            if other in row:
                swapped[key] = row[other]
                swapped[other] = value
        elif key.startswith("delta_slot1_"):
            suffix = key[len("delta_slot1_"):]
            other = "delta_slot2_" + suffix
            if other in row:
                swapped[key] = row[other]
                swapped[other] = value
        elif key.startswith("delta_diff_"):
            swapped[key] = -float(value)
        elif key.startswith("diff_"):
            swapped[key] = -float(value)
        elif key.startswith("delta_share_slot1_"):
            swapped[key] = None if value is None else 1.0 - float(value)
        elif key.startswith("share_slot1_"):
            swapped[key] = None if value is None else 1.0 - float(value)
    return swapped


def feature_manifest(columns: list[str]) -> dict[str, Any]:
    transforms = {}
    for col in columns:
        if col.startswith("delta_slot1_"):
            transforms[col] = "swap_with_delta_slot2"
        elif col.startswith("delta_slot2_"):
            transforms[col] = "swap_with_delta_slot1"
        elif col.startswith("delta_diff_"):
            transforms[col] = "negate"
        elif col.startswith("delta_share_slot1_"):
            transforms[col] = "one_minus"
        elif col.startswith("delta_"):
            transforms[col] = "unchanged"
        elif col.startswith("slot1_"):
            transforms[col] = "swap_with_slot2"
        elif col.startswith("slot2_"):
            transforms[col] = "swap_with_slot1"
        elif col.startswith("diff_"):
            transforms[col] = "negate"
        elif col.startswith("sum_"):
            transforms[col] = "unchanged"
        elif col.startswith("share_slot1_"):
            transforms[col] = "one_minus"
    return {
        "version": "v1",
        "snapshot_interval_minutes": SNAPSHOT_MINUTES,
        "cross_slot_transform_rules": transforms,
        "columns": columns,
    }
