from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .week_rle import module_week_rle_rows, write_rle_csv


def extract_folders_from_embeddings(jsonl_path: Path) -> List[str]:
    folders = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            folder = obj.get("folder")
            if folder:
                folders.add(str(folder))
    return sorted(folders)


def folder_hash(folder: str) -> str:
    return hashlib.sha1(folder.encode("utf-8")).hexdigest()


def load_existing_map(map_json: Path) -> Dict[str, str]:
    if not map_json.exists():
        return {}

    payload = json.loads(map_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "folder_to_hash" in payload and isinstance(payload["folder_to_hash"], dict):
        return dict(payload["folder_to_hash"])
    if isinstance(payload, dict):
        out: Dict[str, str] = {}
        for k, v in payload.items():
            if isinstance(k, str) and isinstance(v, str):
                out[k] = v
        return out
    return {}


def save_map(out_dir: Path, source_jsonl: str, folder_to_hash: Dict[str, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    map_json = out_dir / "folder_hash_map.json"
    map_jsonl = out_dir / "folder_hash_map.jsonl"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_jsonl": source_jsonl,
        "folder_to_hash": folder_to_hash,
        "hash_to_folder": {h: folder for folder, h in folder_to_hash.items()},
    }
    map_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with map_jsonl.open("w", encoding="utf-8") as f:
        for folder, h in folder_to_hash.items():
            f.write(json.dumps({"folder": folder, "hash": h}, ensure_ascii=False) + "\n")


def execute_faultdata(
    embeddings: Path,
    commits_csv: Path,
    out_dir: Path,
    start: str,
    until: str,
) -> int:
    if not embeddings.exists():
        raise FileNotFoundError(f"embeddings jsonl not found: {embeddings}")
    if not commits_csv.exists():
        raise FileNotFoundError(f"commits csv not found: {commits_csv}")

    folders = extract_folders_from_embeddings(embeddings)
    out_dir.mkdir(parents=True, exist_ok=True)

    map_json = out_dir / "folder_hash_map.json"
    folder_to_hash = load_existing_map(map_json)

    for folder in folders:
        folder_to_hash.setdefault(folder, folder_hash(folder))

    rev: Dict[str, str] = {}
    for folder, h in folder_to_hash.items():
        if h in rev and rev[h] != folder:
            raise RuntimeError(f"Hash collision detected: {h} maps to both '{rev[h]}' and '{folder}'")
        rev[h] = folder

    for i, folder in enumerate(folders, start=1):
        h = folder_to_hash[folder]
        output_path = out_dir / f"{h}.csv"
        print(f"[faultdata] ({i}/{len(folders)}) {folder} -> {output_path}")

        rows = module_week_rle_rows(
            in_path=commits_csv,
            module=folder,
            start=start,
            until=until,
        )
        write_rle_csv(rows, output_path)

    save_map(out_dir=out_dir, source_jsonl=str(embeddings), folder_to_hash=folder_to_hash)
    return 0
