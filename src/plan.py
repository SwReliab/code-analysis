from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import AppCfg


def _norm(p: str) -> str:
    return str(Path(p))


def _cfg_dict(cfg: AppCfg) -> Dict[str, Any]:
    return {
        "module": asdict(cfg.module),
        "outputs": asdict(cfg.outputs),
        "rle": asdict(cfg.rle),
        "embedding": asdict(cfg.embedding),
        "lizard": asdict(cfg.lizard),
        "module_key_rule": {
            "root_rel_base": _norm(cfg.module.root_src),
            "subdir_rel_base": _norm(cfg.module.git_subdir),
            "depth": cfg.module.depth,
            "key_format": "join(rel_path_parts[:depth])",
        },
    }


def build_step(cfg: AppCfg, command: str) -> Dict[str, Any]:
    m = cfg.module
    out = cfg.outputs
    lz = cfg.lizard

    if command == "embed":
        return {
            "command": "embed",
            "reads": [_norm(m.root_src)],
            "writes": [_norm(out.embeddings_jsonl)],
            "notes": ["Compute module groups from root_src using module.depth", "Generate embeddings per group and write JSONL"],
        }

    if command == "commits":
        return {
            "command": "commits",
            "reads": [_norm(m.git_root)],
            "writes": [_norm(out.commits_by_module_week_csv)],
            "notes": ["Scan git history under git_root/git_subdir and aggregate by module.depth and week"],
        }

    if command == "lizard":
        lizard_root = lz.root or m.root_src
        return {
            "command": "lizard",
            "reads": [_norm(lizard_root)],
            "writes": [_norm(lz.csv), _norm(out.module_metrics_csv)],
            "notes": ["Run lizard scan -> write lizard.csv", "Aggregate lizard.csv by module.depth -> write module_metrics.csv"],
        }

    if command == "faultdata":
        return {
            "command": "faultdata",
            "reads": [_norm(out.commits_by_module_week_csv), _norm(out.embeddings_jsonl)],
            "writes": [_norm(out.faultdata_dir)],
            "notes": ["Build faultdata time series per module key", "Use rle.start/until window"],
        }

    raise ValueError(f"Unknown command: {command}")


def build_plan(cfg: AppCfg, command: str) -> Dict[str, Any]:
    if command == "run":
        steps = [
            build_step(cfg, "embed"),
            build_step(cfg, "commits"),
            build_step(cfg, "lizard"),
            build_step(cfg, "faultdata"),
        ]
        return {"cfg": _cfg_dict(cfg), "command": "run", "steps": steps}

    return {"cfg": _cfg_dict(cfg), "command": command, "steps": [build_step(cfg, command)]}


# -----------------------
# Check / validation
# -----------------------

def _parse_week_id(s: str) -> Optional[Tuple[int, int]]:
    if not isinstance(s, str):
        return None
    try:
        year_str, w_str = s.split("-W", 1)
        year = int(year_str)
        week = int(w_str)
        if year < 0 or not (1 <= week <= 53):
            return None
        return (year, week)
    except Exception:
        return None


def _week_leq(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[0] or (a[0] == b[0] and a[1] <= b[1])


def check_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    cfg = plan.get("cfg", {})
    module = (cfg.get("module") or {})
    rle = (cfg.get("rle") or {})

    errors: List[str] = []
    warnings: List[str] = []

    depth = module.get("depth")
    if not isinstance(depth, int) or depth < 1:
        errors.append(f"module.depth must be an integer >= 1 (got {depth!r})")

    start = rle.get("start")
    until = rle.get("until")
    ps = _parse_week_id(start) if start is not None else None
    pu = _parse_week_id(until) if until is not None else None
    if ps is None:
        errors.append(f"rle.start must look like 'YYYY-Www' (got {start!r})")
    if pu is None:
        errors.append(f"rle.until must look like 'YYYY-Www' (got {until!r})")
    if ps is not None and pu is not None and not _week_leq(ps, pu):
        errors.append(f"rle.start must be <= rle.until (got {start!r} > {until!r})")

    steps = plan.get("steps") or []
    if not isinstance(steps, list) or not steps:
        errors.append("plan.steps must be a non-empty list")
        return {"ok": False, "errors": errors, "warnings": warnings}

    produced: set[str] = set()

    for i, st in enumerate(steps):
        cmd = st.get("command", f"step[{i}]")
        reads = st.get("reads") or []
        writes = st.get("writes") or []

        for rp in reads:
            rp_s = str(rp)
            p = Path(rp_s)
            if p.exists() or (rp_s in produced):
                continue
            errors.append(f"[{cmd}] read path not found: {rp_s}")

        for wp in writes:
            wp_s = str(wp)
            p = Path(wp_s)
            parent = p.parent if wp_s not in (".", "") else Path(".")
            if not parent.exists():
                errors.append(f"[{cmd}] write parent dir not found: {parent} (for {wp_s})")
            produced.add(wp_s)

    ok = len(errors) == 0
    return {"ok": ok, "errors": errors, "warnings": warnings}