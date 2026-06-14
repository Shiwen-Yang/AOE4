import re
from dataclasses import dataclass
from pathlib import Path


REPLAY_RE = re.compile(r"^replay_(\d+)\.intent_timeline\.jsonl$")


@dataclass(frozen=True)
class ReplayFile:
    replay_id: int
    path: Path


def parse_replay_id(path: Path) -> int:
    match = REPLAY_RE.match(path.name)
    if not match:
        raise ValueError(f"Not a parsed replay timeline path: {path}")
    return int(match.group(1))


def discover_replays(parsed_dir: Path) -> list[ReplayFile]:
    files: list[ReplayFile] = []
    for path in parsed_dir.glob("replay_*.intent_timeline.jsonl"):
        try:
            files.append(ReplayFile(parse_replay_id(path), path))
        except ValueError:
            continue
    return sorted(files, key=lambda r: r.replay_id)


def sample_across_id_range(replays: list[ReplayFile], limit: int) -> list[ReplayFile]:
    """Return a deterministic sample spread across the sorted replay ID range."""
    if limit <= 0 or limit >= len(replays):
        return list(replays)
    if limit == 1:
        return [replays[0]]

    last = len(replays) - 1
    indices = sorted({round(i * last / (limit - 1)) for i in range(limit)})

    # Rounding can theoretically collapse adjacent picks. Fill deterministically
    # from left to right to keep the exact requested size.
    if len(indices) < limit:
        seen = set(indices)
        for idx in range(len(replays)):
            if idx not in seen:
                indices.append(idx)
                seen.add(idx)
                if len(indices) == limit:
                    break
        indices.sort()

    return [replays[i] for i in indices[:limit]]


def assign_match_splits(
    replay_ids: list[int],
    train_frac: float,
    valid_frac: float,
) -> dict[int, str]:
    """Assign chronological splits by replay ID after sampling."""
    ordered = sorted(replay_ids)
    n = len(ordered)
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))

    splits: dict[int, str] = {}
    for i, replay_id in enumerate(ordered):
        if i < train_end:
            split = "train"
        elif i < valid_end:
            split = "valid"
        else:
            split = "test"
        splits[replay_id] = split
    return splits
