"""Building-placement heatmap features for the realtime outcome prediction model.

Generates per-player 5-channel 64×64 grids from replay building intent events.
Channel layout:
  0  Eco            (TC, Farms, Mills, Mining/Lumber Camps, Docks, Tech buildings, eco-type landmarks)
  1  Military prod  (Barracks, Stable, Archery Range, Siege Workshop, etc., military-type landmarks)
  2  Defensive      (Tower, Wall, Keep, Castle, Fort, Outpost, Gate, Barbican, etc.)
  3  Landmarks      (all is_age_up_building structures — victory-condition layer)
  4  Valid-map mask (1 inside map bounds in rotated frame; 0 in padding)

Canonical frame: translate so the midpoint between the two TCs is at grid center,
then rotate so the TC1→TC2 vector points along the +X axis. Slot 2's grid is
then flipped horizontally so both players always have own-TC on the left.
After this rotation, cells outside the actual map boundary are masked in ch 4.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    Dataset = object  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Map geometry
# ---------------------------------------------------------------------------

MAP_X_MIN, MAP_X_MAX = -450.0, 450.0
MAP_Y_MIN, MAP_Y_MAX = -480.0, 450.0
GRID_SIZE = 64
N_CHANNELS = 5
WORLD_HALF = 650.0  # half-side of the grid's world-unit extent; covers full rotated map

# ---------------------------------------------------------------------------
# Building classification
# ---------------------------------------------------------------------------

_DEFENSIVE_TERMS: frozenset[str] = frozenset({
    "keep",
    "tower",
    "wall",
    "palisade",
    "barbican",
    "castle",
    "fort",
    "outpost",
    "gate",
})

_MILITARY_PRODUCTION_TERMS: frozenset[str] = frozenset({
    "archery range",
    "barracks",
    "foreign engineering company",
    "mercenary house",
    "military school",
    "prayer tent",
    "siege workshop",
    "stable",
    "war academy",
})

# Tech/research and all unrecognized buildings fall through to eco (ch 0).


def building_channel_v2(name: str) -> int:
    """Map a building name to channel 0 (eco), 1 (military prod), or 2 (defensive).

    Priority: defensive > military production > eco (default).
    TC, Tech/Research, and everything else → eco.
    Used for both ordinary buildings and as the primary-channel classifier for landmarks.
    """
    n = name.lower()
    if any(t in n for t in _DEFENSIVE_TERMS):
        return 2
    if any(t in n for t in _MILITARY_PRODUCTION_TERMS):
        return 1
    return 0  # eco: TC, farm, mill, blacksmith, university, mosque, dock, etc.


# ---------------------------------------------------------------------------
# TC-position detection
# ---------------------------------------------------------------------------


def find_tc_position(
    events: list[dict[str, Any]],
    slot: int,
    pbgid_index: dict[int, dict],
) -> tuple[float, float]:
    """Estimate the starting TC position for a player slot.

    Tries the earliest TC placement command first; falls back to the centroid of
    all buildings placed by that slot in the first 300 s.  Returns (0.0, 0.0) if
    no building events are found at all.
    """
    # Pass 1: look for earliest TC placement
    for event in events:
        if event.get("player_slot") != slot:
            continue
        if event.get("intent_category") != "building":
            continue
        time_s = float(event.get("time_s") or 0.0)
        if time_s > 600.0:
            break
        intent = event.get("intent") or {}
        pbgid = intent.get("pbgid")
        if pbgid is None:
            continue
        info = pbgid_index.get(int(pbgid))
        if info is None:
            continue
        if "town center" in (info.get("name") or "").lower():
            x = intent.get("position_x")
            y = intent.get("position_y")
            if x is not None and y is not None:
                return float(x), float(y)

    # Pass 2: centroid of all buildings in first 300 s
    xs: list[float] = []
    ys: list[float] = []
    for event in events:
        if event.get("player_slot") != slot:
            continue
        if event.get("intent_category") != "building":
            continue
        time_s = float(event.get("time_s") or 0.0)
        if time_s > 300.0:
            break
        intent = event.get("intent") or {}
        x = intent.get("position_x")
        y = intent.get("position_y")
        if x is not None and y is not None:
            xs.append(float(x))
            ys.append(float(y))

    if xs:
        return float(np.mean(xs)), float(np.mean(ys))
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Canonical transform
# ---------------------------------------------------------------------------


def compute_canonical_transform(
    tc1_xy: tuple[float, float],
    tc2_xy: tuple[float, float],
) -> tuple[float, float, float]:
    """Compute (midpoint_x, midpoint_y, angle) for the canonical frame.

    The canonical frame translates by the midpoint between the two TCs and
    rotates so that the TC1→TC2 vector points along the positive X axis.
    Rotating world coordinates by −angle puts TC1 to the left and TC2 to the
    right at the same Y level (same horizontal line).
    """
    mx = (tc1_xy[0] + tc2_xy[0]) / 2.0
    my = (tc1_xy[1] + tc2_xy[1]) / 2.0
    dx = tc2_xy[0] - tc1_xy[0]
    dy = tc2_xy[1] - tc1_xy[1]
    angle = math.atan2(dy, dx)
    return mx, my, angle


def _rotated_to_cell(
    dx_r: float,
    dy_r: float,
    grid_size: int,
) -> tuple[int, int]:
    """Map a rotated-frame offset (dx_r, dy_r) to a (row, col) grid cell.

    The grid spans [−WORLD_HALF, +WORLD_HALF] in both rotated X and Y.
    Row 0 is at the top (positive rotated Y), col 0 is at the left (negative rotated X).
    """
    col = int((dx_r + WORLD_HALF) / (2.0 * WORLD_HALF) * grid_size)
    row = int((WORLD_HALF - dy_r) / (2.0 * WORLD_HALF) * grid_size)
    col = max(0, min(grid_size - 1, col))
    row = max(0, min(grid_size - 1, row))
    return row, col


def compute_map_mask(
    mx: float,
    my: float,
    angle: float,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """Compute the valid-map mask for the canonical frame.

    Returns a (grid_size, grid_size) float32 array: 1.0 where the grid cell's
    absolute map coordinates fall within the actual map bounds, 0.0 in padding.
    """
    # Cell centers in rotated frame (dx_r, dy_r)
    cols = np.arange(grid_size, dtype=np.float64) + 0.5
    rows = np.arange(grid_size, dtype=np.float64) + 0.5
    dx_r = cols / grid_size * (2.0 * WORLD_HALF) - WORLD_HALF    # (G,)
    dy_r = WORLD_HALF - rows / grid_size * (2.0 * WORLD_HALF)     # (G,)

    dx_r_grid, dy_r_grid = np.meshgrid(dx_r, dy_r)  # (G, G) each

    # Rotate back by +angle to get absolute offsets from midpoint
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx_abs = dx_r_grid * cos_a - dy_r_grid * sin_a
    dy_abs = dx_r_grid * sin_a + dy_r_grid * cos_a

    abs_x = mx + dx_abs
    abs_y = my + dy_abs

    mask = (
        (abs_x >= MAP_X_MIN) & (abs_x <= MAP_X_MAX) &
        (abs_y >= MAP_Y_MIN) & (abs_y <= MAP_Y_MAX)
    ).astype(np.float32)
    return mask


# ---------------------------------------------------------------------------
# Heatmap builder
# ---------------------------------------------------------------------------


def build_player_heatmap(
    events: list[dict[str, Any]],
    slot: int,
    pbgid_index: dict[int, dict],
    cutoff_s: float,
    mx: float,
    my: float,
    angle: float,
    map_mask: np.ndarray,
    is_slot2: bool,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """Build a 5-channel building placement heatmap for one player.

    Args:
        events:      Sorted building intent events for the full match.
        slot:        Player slot (1 or 2).
        pbgid_index: pbgid → metadata dict from build_pbgid_index().
        cutoff_s:    Include only events with time_s ≤ cutoff_s.
        mx, my:      Midpoint between the two TCs (world coordinates).
        angle:       Rotation angle so TC1→TC2 points along +X (see compute_canonical_transform).
        map_mask:    Precomputed valid-map mask, shape (grid_size, grid_size).
        is_slot2:    If True, mirror the grid horizontally so own TC appears on the left.
        grid_size:   Edge length of the square grid (default 64).

    Returns:
        Float32 ndarray of shape (5, grid_size, grid_size).
        Channels 0–3: log(1 + deduplicated placement count).
        Channel 4: valid-map mask (1 = inside map bounds in canonical frame).
    """
    cos_neg = math.cos(-angle)
    sin_neg = math.sin(-angle)

    raw_counts = np.zeros((N_CHANNELS, grid_size, grid_size), dtype=np.float32)

    # Dedup: (ch, row, col) → last time_s seen
    last_placed: dict[tuple[int, int, int], float] = {}

    for event in events:
        if event.get("player_slot") != slot:
            continue
        if event.get("intent_category") != "building":
            continue
        time_s = float(event.get("time_s") or 0.0)
        if time_s > cutoff_s:
            break

        intent = event.get("intent") or {}
        pbgid = intent.get("pbgid")
        if pbgid is None:
            continue
        x = intent.get("position_x")
        y = intent.get("position_y")
        if x is None or y is None:
            continue
        x, y = float(x), float(y)

        info = pbgid_index.get(int(pbgid))
        if info is None:
            continue
        name = info.get("name") or ""
        is_landmark = bool(info.get("is_age_up_building")) and "town center" not in name.lower()

        # Primary channel classification
        ch = building_channel_v2(name)
        # Landmark primary: if the landmark name has a military term → ch 1, else eco → ch 0
        # (Defensive landmarks are treated as eco per design: landmarks are eco or military only)
        if is_landmark and ch == 2:
            ch = 0

        # Canonical-frame transform
        dx = x - mx
        dy = y - my
        dx_r = dx * cos_neg - dy * sin_neg
        dy_r = dx * sin_neg + dy * cos_neg
        row, col = _rotated_to_cell(dx_r, dy_r, grid_size)

        # Deduplication: same (ch, cell) within 10 s → skip
        key = (ch, row, col)
        prev = last_placed.get(key)
        if prev is not None and (time_s - prev) < 10.0:
            continue
        last_placed[key] = time_s

        raw_counts[ch, row, col] += 1.0

        # Also stamp ch 3 (landmarks layer) without dedup against primary channel
        if is_landmark:
            key3 = (3, row, col)
            prev3 = last_placed.get(key3)
            if prev3 is None or (time_s - prev3) >= 10.0:
                last_placed[key3] = time_s
                raw_counts[3, row, col] += 1.0

    grid = np.log1p(raw_counts)

    # Channel 4: valid-map mask (precomputed, binary)
    grid[4] = map_mask

    # Own-TC on left: for slot 2 in canonical frame, TC2 is on the right → flip
    if is_slot2:
        grid = grid[:, :, ::-1].copy()

    return grid


# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------


def precompute_heatmaps(
    snapshots_path: Path,
    parsed_dir: Path,
    pbgid_index: dict[int, dict],
    cache_dir: Path,
    grid_size: int = GRID_SIZE,
    force: bool = False,
) -> dict[str, int]:
    """Precompute and cache building heatmaps for every (replay_id, minute) snapshot.

    Cache files: cache_dir/{replay_id}_{minute}.npy — shape (2, 5, grid_size, grid_size).
    Index 0 = slot 1 heatmap, index 1 = slot 2 heatmap (both in own-TC-left orientation).

    Returns a summary dict with counts of processed/skipped/errors.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas required for precompute_heatmaps") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(snapshots_path, columns=["replay_id", "snapshot_minute"])
    pairs = df.drop_duplicates().to_dict("records")

    counts = {"processed": 0, "skipped": 0, "missing_replay": 0, "error": 0}

    replay_to_minutes: dict[int, list[int]] = {}
    for row in pairs:
        rid = int(row["replay_id"])
        replay_to_minutes.setdefault(rid, []).append(int(row["snapshot_minute"]))

    for replay_id, minutes in replay_to_minutes.items():
        replay_path = parsed_dir / f"replay_{replay_id}.intent_timeline.jsonl"
        if not replay_path.exists():
            counts["missing_replay"] += len(minutes)
            continue

        if not force:
            all_cached = all(
                (cache_dir / f"{replay_id}_{m}.npy").exists() for m in minutes
            )
            if all_cached:
                counts["skipped"] += len(minutes)
                continue

        try:
            events = _read_events(replay_path)
        except Exception:
            counts["error"] += len(minutes)
            continue

        # Compute canonical transform once per replay
        tc1_xy = find_tc_position(events, 1, pbgid_index)
        tc2_xy = find_tc_position(events, 2, pbgid_index)
        mx, my, angle = compute_canonical_transform(tc1_xy, tc2_xy)
        map_mask = compute_map_mask(mx, my, angle, grid_size)

        for minute in sorted(minutes):
            cache_path = cache_dir / f"{replay_id}_{minute}.npy"
            if not force and cache_path.exists():
                counts["skipped"] += 1
                continue

            try:
                cutoff_s = float(minute) * 60.0
                h1 = build_player_heatmap(
                    events, 1, pbgid_index, cutoff_s, mx, my, angle, map_mask,
                    is_slot2=False, grid_size=grid_size,
                )
                h2 = build_player_heatmap(
                    events, 2, pbgid_index, cutoff_s, mx, my, angle, map_mask,
                    is_slot2=True, grid_size=grid_size,
                )
                combined = np.stack([h1, h2], axis=0)  # (2, 5, G, G)
                np.save(str(cache_path), combined)
                counts["processed"] += 1
            except Exception:
                counts["error"] += 1

    return counts


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    events.sort(key=lambda e: (float(e.get("time_s") or 0.0), int(e.get("command_index") or 0)))
    return events


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class BuildingHeatmapDataset(Dataset):
    """One sample per (replay_id, snapshot_minute) row.

    Loads precomputed heatmaps from cache files.  Samples where the cache file
    is missing are silently dropped at construction time.

    Args:
        records:   List of dicts with keys: replay_id, snapshot_minute, target.
        cache_dir: Directory containing {replay_id}_{minute}.npy files.
        in_memory: If True (default), preload all arrays into RAM at init.

    Returns per __getitem__:
        h1        (5, G, G) float32 tensor — slot 1 heatmap
        h2        (5, G, G) float32 tensor — slot 2 heatmap
        target    scalar float32 tensor (1 if slot 1 wins, 0 otherwise)
        minute    scalar int32 tensor
    """

    def __init__(self, records: list[dict], cache_dir: Path, in_memory: bool = True) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for BuildingHeatmapDataset")
        import torch
        self.cache_dir = cache_dir
        self.records = [
            r for r in records
            if (cache_dir / f"{r['replay_id']}_{r['snapshot_minute']}.npy").exists()
        ]
        self._in_memory = in_memory
        if in_memory and self.records:
            print(f"  Preloading {len(self.records)} heatmaps into RAM...", flush=True)
            arrays = []
            for r in self.records:
                path = cache_dir / f"{r['replay_id']}_{r['snapshot_minute']}.npy"
                arrays.append(np.load(str(path)))
            self._data = torch.from_numpy(np.stack(arrays, axis=0))
            print(f"  Loaded {self._data.shape}  {self._data.nbytes / 1e9:.2f} GB", flush=True)
        else:
            self._data = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        import torch
        r = self.records[idx]
        if self._data is not None:
            combined = self._data[idx]
            h1 = combined[0]
            h2 = combined[1]
        else:
            path = self.cache_dir / f"{r['replay_id']}_{r['snapshot_minute']}.npy"
            combined = np.load(str(path))
            h1 = torch.from_numpy(combined[0].copy())
            h2 = torch.from_numpy(combined[1].copy())
        target = torch.tensor(float(r["target"]), dtype=torch.float32)
        minute = torch.tensor(int(r["snapshot_minute"]), dtype=torch.int32)
        return h1, h2, target, minute


def make_records_from_snapshots(snapshots_path: Path, split: str | None = None) -> list[dict]:
    """Load snapshot parquet and return a list of record dicts for BuildingHeatmapDataset."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas required") from exc

    cols = ["replay_id", "snapshot_minute", "target", "split"]
    df = pd.read_parquet(snapshots_path, columns=cols)
    if split is not None:
        df = df[df["split"] == split]
    df = df.drop_duplicates(subset=["replay_id", "snapshot_minute"])
    return df.to_dict("records")
