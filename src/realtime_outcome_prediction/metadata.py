import json
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .config import AOE4_WORLD_BASE_URL, AOE4_WORLD_REPO_DIR, AOE4_WORLD_REPO_URL


METADATA_ENDPOINTS = {
    "units": "/units/all.json",
    "buildings": "/buildings/all.json",
    "technologies": "/technologies/all.json",
}


def _fetch_json(url: str) -> object:
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def load_or_fetch_aoe4world(cache_dir: Path, refresh: bool = False) -> dict:
    """Load cached AoE4 World data, fetching current metadata if needed.

    This URL fallback is kept for offline tests and old caches. Production
    dataset builds should use load_or_update_aoe4world_repo().
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "aoe4world_metadata.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())

    payload = {
        "source": AOE4_WORLD_BASE_URL,
        "fetched_at_unix": int(time.time()),
        "fetched_at_readable": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoints": {},
        "items": {},
    }
    try:
        for name, endpoint in METADATA_ENDPOINTS.items():
            url = AOE4_WORLD_BASE_URL + endpoint
            payload["endpoints"][name] = url
            payload["items"][name] = _fetch_json(url)
    except (URLError, TimeoutError, OSError) as exc:
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        raise RuntimeError(
            "Could not fetch AoE4 World metadata and no cache exists. "
            "Re-run with network access or provide a cache file."
        ) from exc

    cache_path.write_text(json.dumps(payload, indent=2))
    return payload


def _git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )


def _checkout_revision(repo_dir: Path) -> str | None:
    result = _git(["rev-parse", "HEAD"], cwd=repo_dir)
    return result.stdout.strip() if result.returncode == 0 else None


def _load_repo_json(repo_dir: Path) -> dict:
    payload = {
        "source": AOE4_WORLD_REPO_URL,
        "repo_dir": str(repo_dir),
        "revision": _checkout_revision(repo_dir),
        "fetched_at_unix": int(time.time()),
        "fetched_at_readable": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "items": {},
    }
    for name, rel in {
        "units": "units/all.json",
        "buildings": "buildings/all.json",
        "technologies": "technologies/all.json",
    }.items():
        path = repo_dir / rel
        if path.exists():
            payload["items"][name] = json.loads(path.read_text())
    return payload


def load_or_update_aoe4world_repo(
    cache_dir: Path,
    repo_dir: Path | None = None,
    update: bool = True,
) -> dict:
    """Maintain a local aoe4world/data git checkout and load metadata from it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = repo_dir or AOE4_WORLD_REPO_DIR
    cache_path = cache_dir / "aoe4world_metadata.json"

    update_error: str | None = None
    if repo_dir.exists():
        if update:
            result = _git(["pull", "--ff-only"], cwd=repo_dir)
            if result.returncode != 0:
                update_error = (result.stderr or result.stdout or "git pull failed").strip()
    elif update:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        result = _git(["clone", "--depth", "1", AOE4_WORLD_REPO_URL, str(repo_dir)])
        if result.returncode != 0:
            update_error = (result.stderr or result.stdout or "git clone failed").strip()

    if repo_dir.exists():
        payload = _load_repo_json(repo_dir)
        payload["update_attempted"] = update
        payload["update_error"] = update_error
        cache_path.write_text(json.dumps(payload, indent=2))
        return payload

    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        payload["update_attempted"] = update
        payload["update_error"] = update_error
        payload["loaded_from_stale_cache"] = True
        return payload

    raise RuntimeError(
        "Could not update AoE4 World metadata from git and no cache exists. "
        f"Error: {update_error or 'unknown'}"
    )


def _iter_items(value: object):
    if isinstance(value, list):
        yield from value
    elif isinstance(value, dict):
        if "data" in value:
            yield from _iter_items(value["data"])
        else:
            for child in value.values():
                if isinstance(child, dict) and "pbgid" in child:
                    yield child
                elif isinstance(child, list):
                    yield from _iter_items(child)


def _costs(item: dict) -> dict[str, float]:
    raw = item.get("costs") or item.get("cost") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key in ("food", "wood", "gold", "stone", "vizier", "olive_oil", "time", "population"):
        val = raw.get(key)
        if isinstance(val, (int, float)):
            out[key] = float(val)
    return out


_LANDMARK_CLASSES: frozenset[str] = frozenset({
    "age1_landmark1", "age1_landmark2",
    "age2_landmark1", "age2_landmark2",
    "age3_landmark1", "age3_landmark2",
})
_AGEUP_CLASS_TO_TIER: dict[str, int] = {
    "age1_landmark1": 2, "age1_landmark2": 2,
    "age2_landmark1": 3, "age2_landmark2": 3,
    "age3_landmark1": 4, "age3_landmark2": 4,
}


def build_pbgid_index(metadata: dict) -> dict[int, dict]:
    """Flatten AoE4 World metadata to pbgid -> compact cost/type info."""
    index: dict[int, dict] = {}
    for kind, value in metadata.get("items", {}).items():
        for item in _iter_items(value):
            if not isinstance(item, dict) or item.get("pbgid") is None:
                continue
            try:
                pbgid = int(item["pbgid"])
            except (TypeError, ValueError):
                continue
            costs = _costs(item)
            total_resources = sum(costs.get(k, 0.0) for k in ("food", "wood", "gold", "stone"))
            raw_classes = item.get("classes") or []
            class_set = set(raw_classes)
            index[pbgid] = {
                "kind": kind,
                "name": item.get("name") or item.get("id") or str(pbgid),
                "age": item.get("age"),
                "costs": costs,
                "total_resources": total_resources,
                "is_age_up_building": bool(class_set & _LANDMARK_CLASSES),
                "ageup_to_tier": next((v for k, v in _AGEUP_CLASS_TO_TIER.items() if k in class_set), None),
                "is_age_up_tech": "age_up_upgrade" in class_set,
            }
    return index
