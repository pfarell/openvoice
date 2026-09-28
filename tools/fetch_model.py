"""Explicit, opt-in model downloader for OpenVoice.

The application itself never downloads a model: ``transcribe.load()`` opens its
model with ``local_files_only=True``, so a model that is not already cached is a
hard error at runtime rather than a silent network call. This script is the
**only** network operation in the project, and it is deliberately manual - run
it once, on purpose, when you want a model on disk:

    python tools/fetch_model.py --list
    python tools/fetch_model.py --model small
    python tools/fetch_model.py --model Systran/faster-whisper-large-v3

Aliases map to the official faster-whisper (CTranslate2) HuggingFace
repositories; any other value containing ``/`` is treated as a repo id and
handed to the Hub unchanged. Downloads go through
``huggingface_hub.snapshot_download``. There are no paid APIs, accounts or
subscriptions involved.

Cost: one Hub round trip per requested repo; sizing a cached repo is O(files)
and downloads are O(model bytes) on disk.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

#: Friendly name -> HuggingFace repository id.
ALIASES: dict[str, str] = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}


def hub_cache_dir() -> Path:
    """The HuggingFace hub cache directory.

    Uses ``huggingface_hub``'s own constant when it is importable so the scan and
    the download agree, then falls back to ``HF_HUB_CACHE``, ``HF_HOME`` and the
    platform default in that order.
    """
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:
        pass
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_cache_name(repo_id: str) -> str:
    """The directory name HuggingFace uses for ``repo_id`` in the hub cache."""
    return "models--" + repo_id.replace("/", "--")


def cached_path(repo_id: str, cache_dir: Path | None = None) -> Path | None:
    """The cache directory for ``repo_id``, or ``None`` when it is not present."""
    path = (cache_dir or hub_cache_dir()) / repo_cache_name(repo_id)
    return path if path.is_dir() else None


def is_cached(repo_id: str, cache_dir: Path | None = None) -> bool:
    return cached_path(repo_id, cache_dir) is not None


def dir_size(path: Path) -> int:
    """Total size in bytes of every file under ``path``. O(files)."""
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def human_bytes(value: int) -> str:
    """Render a byte count with a binary unit, e.g. ``1.4 GiB``."""
    size = float(value)
    unit = "B"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            break
        size /= 1024.0
    if unit == "B":
        return f"{int(size)} B"
    return f"{size:.1f} {unit}"


def resolve_repo(value: str) -> str | None:
    """Map an alias to a repo id, or pass through a repo id.

    Returns ``None`` for a bare value that is neither a known alias nor a repo id
    (a repo id always contains a ``/``).
    """
    if value in ALIASES:
        return ALIASES[value]
    if "/" in value:
        return value
    return None


def print_list(cache_dir: Path) -> None:
    print(f"HuggingFace hub cache: {cache_dir}")
    print()
    for alias, repo in ALIASES.items():
        status = "cached" if is_cached(repo, cache_dir) else "absent"
        print(f"  {alias:<16} {repo:<48} {status}")
    print()
    print("Pass any other HuggingFace repo id containing '/' to fetch it directly.")


def fetch(repo_id: str, cache_dir: Path) -> int:
    """Download ``repo_id`` into ``cache_dir``. Returns a process exit code."""
    existing = cached_path(repo_id, cache_dir)
    if existing is not None:
        total = dir_size(existing)
        print(
            f"{repo_id} is already cached at {existing} "
            f"({human_bytes(total)}); nothing to do."
        )
        return 0

    print(f"Downloading {repo_id} ...")
    print(
        "This is OpenVoice's only network operation; it is deliberate and "
        "manual. The app itself runs offline with local_files_only=True."
    )
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        print(f"[fetch_model] huggingface_hub is unavailable: {exc}", file=sys.stderr)
        return 1

    try:
        destination = snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir))
    except Exception as exc:
        print(f"[fetch_model] download failed for {repo_id}: {exc}", file=sys.stderr)
        return 1

    location = cached_path(repo_id, cache_dir) or Path(str(destination))
    total = dir_size(location)
    print()
    print(f"repo:        {repo_id}")
    print(f"destination: {destination}")
    print(f"on disk:     {human_bytes(total)} ({total} bytes)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_model.py",
        description=(
            "Explicitly download a faster-whisper model into the local "
            "HuggingFace cache. OpenVoice itself never downloads models."
        ),
    )
    parser.add_argument(
        "--model",
        "-m",
        help="alias or HuggingFace repo id to download (e.g. small, base)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="show the known aliases and whether each is cached or absent",
    )
    args = parser.parse_args(argv)

    cache_dir = hub_cache_dir()

    if args.list:
        print_list(cache_dir)
        return 0

    if not args.model:
        parser.print_help()
        return 2

    repo_id = resolve_repo(args.model)
    if repo_id is None:
        print(f"[fetch_model] unknown alias {args.model!r}", file=sys.stderr)
        print("valid aliases:", file=sys.stderr)
        for alias, repo in ALIASES.items():
            print(f"  {alias:<16} {repo}", file=sys.stderr)
        print(
            "...or pass any HuggingFace repo id containing '/'.", file=sys.stderr
        )
        return 2

    return fetch(repo_id, cache_dir)


if __name__ == "__main__":
    raise SystemExit(main())
