"""
Download AOE4 ranked TEAM game dumps from https://aoe4world.com/dumps.

The dump links are signed Google Cloud Storage URLs that EXPIRE, so they cannot be
hardcoded — we scrape the dumps page at run time. Each signed URL carries the real
filename in its `response-content-disposition` query parameter, e.g.
    ...&response-content-disposition=attachment; filename="games_rm_4v4_s9.json.gz"...
which is how we map a (mode, season) to its current signed link.
"""
import html
import re
import shutil
import urllib.request
from pathlib import Path

from .config import DUMPS_URL, TEAM_DATA_DIR, TEAM_SEASONS

_UA = "Mozilla/5.0 (compatible; aoe4-non1v1-research/0.1)"
_HREF_RE = re.compile(r'href=["\']([^"\']*storage\.googleapis\.com[^"\']*)["\']')
_FILENAME_RE = re.compile(r'filename%3D%22([^%]+\.json\.gz)%22')


def _scrape_links() -> dict[str, str]:
    """Return {dump_filename: signed_url} for every dump on the page."""
    req = urllib.request.Request(DUMPS_URL, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        page = resp.read().decode("utf-8", "replace")

    links: dict[str, str] = {}
    for raw in _HREF_RE.findall(page):
        url = html.unescape(raw)  # &amp; -> &
        m = _FILENAME_RE.search(url)
        if m:
            links[m.group(1)] = url
    return links


def _download_one(url: str, dest: Path) -> int:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.replace(dest)
    return dest.stat().st_size


def download_dumps(
    mode: str = "rm_4v4",
    seasons: list[int] | None = None,
    dest_dir: Path | None = None,
    force: bool = False,
) -> list[Path]:
    """
    Download `games_<mode>_s<season>.json.gz` for the requested seasons.

    Skips files already present locally unless `force=True`. Returns the list of
    local paths now available.
    """
    seasons = seasons or TEAM_SEASONS
    dest_dir = Path(dest_dir or TEAM_DATA_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    wanted = {s: f"games_{mode}_s{s}.json.gz" for s in seasons}
    have = []
    need = {}
    for s, fname in wanted.items():
        dest = dest_dir / fname
        if dest.exists() and not force:
            print(f"  {fname}: already present ({dest.stat().st_size/1e6:.1f} MB), skipping.")
            have.append(dest)
        else:
            need[fname] = dest

    if not need:
        return have

    print(f"  Scraping {DUMPS_URL} for signed links ...")
    links = _scrape_links()

    for fname, dest in need.items():
        url = links.get(fname)
        if not url:
            print(f"  WARNING: {fname} not found on dumps page — skipping.")
            continue
        print(f"  Downloading {fname} ...", end=" ", flush=True)
        size = _download_one(url, dest)
        print(f"{size/1e6:.1f} MB")
        have.append(dest)

    return have
