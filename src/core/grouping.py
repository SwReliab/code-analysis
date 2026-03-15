from __future__ import annotations

from pathlib import PurePosixPath
from typing import Tuple


def _normalize_rel_path(rel_path: str) -> str:
    # Normalize to POSIX separators regardless of host OS.
    return str(PurePosixPath(str(rel_path).replace("\\", "/")))


def get_module_key(rel_path: str, depth: int) -> Tuple[str, str]:
    """
    Return (group_key, group_type) where group_type is "file" or "folder".

     Rules:
        1) Root-level file (e.g. "main.py")
            -> key is file path, type="file"
        2) Paths under at least one directory (e.g. "core/embed.py")
            -> key is first `depth` directories, type="folder"
    """
    normalized = _normalize_rel_path(rel_path).lstrip("./")
    parts = [p for p in normalized.split("/") if p]

    if not parts:
        return ("", "file")

    if len(parts) == 1:
        return (parts[0], "file")

    if depth <= 0:
        raise ValueError(f"depth must be >= 1 (got {depth!r})")

    dir_parts = parts[:-1]
    key = "/".join(dir_parts[:min(depth, len(dir_parts))])
    return (key, "folder")