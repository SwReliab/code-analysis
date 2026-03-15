from __future__ import annotations

import csv
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .grouping import get_module_key


def execute_lizard_scan(
    root: Path,
    out_csv: Path,
    languages: Optional[List[str]] = None,
    extra_args: Optional[List[str]] = None,
) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["lizard", "--csv"]
    for lang in (languages or []):
        cmd.extend(["-l", lang])
    cmd.extend(list(extra_args or []))
    cmd.append(str(root))

    print(f"[lizard] scan: {' '.join(cmd)}")
    with out_csv.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)

    if p.returncode != 0:
        raise RuntimeError(f"lizard scan failed (rc={p.returncode}). stderr:\n{p.stderr}")

    print(f"[lizard] wrote: {out_csv}")
    return 0


@dataclass
class Agg:
    group_key: str
    group_type: str

    functions: int = 0
    files: Set[str] = field(default_factory=set)

    sum_nloc: int = 0
    sum_ccn: int = 0
    sum_token: int = 0
    sum_params: int = 0

    max_nloc: int = 0
    max_ccn: int = 0
    max_token: int = 0
    max_params: int = 0

    def add(self, file_path: str, nloc: int, ccn: int, token: int, params: int) -> None:
        self.functions += 1
        self.files.add(file_path)

        self.sum_nloc += nloc
        self.sum_ccn += ccn
        self.sum_token += token
        self.sum_params += params

        self.max_nloc = max(self.max_nloc, nloc)
        self.max_ccn = max(self.max_ccn, ccn)
        self.max_token = max(self.max_token, token)
        self.max_params = max(self.max_params, params)

    @property
    def file_count(self) -> int:
        return len(self.files)

    def avg(self, x: int) -> float:
        return (x / self.functions) if self.functions else 0.0


def safe_int(s: str) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return 0


def looks_like_header(row: List[str]) -> bool:
    return any(re.search(r"[A-Za-z]", c or "") for c in row)


def normalize_relpath(path: str, root: str) -> str:
    root_abs = os.path.abspath(root)
    p = (path or "").strip().strip('"').strip()
    if not p:
        return ""

    p_posix = p.replace("\\", "/")

    root_tail = "/".join(os.path.normpath(root).replace("\\", "/").split("/")[-2:])
    if p_posix == root_tail:
        p_posix = ""
    elif p_posix.startswith(root_tail + "/"):
        p_posix = p_posix[len(root_tail) + 1 :]

    if os.path.isabs(p_posix):
        cand_list = [p_posix]
    else:
        cand_list = [
            os.path.abspath(p_posix),
            os.path.abspath(os.path.join(root_abs, p_posix)),
        ]

    cand = cand_list[0]
    for c in cand_list:
        if os.path.exists(c):
            cand = c
            break

    try:
        rel = os.path.relpath(cand, root_abs).replace("\\", "/")
    except ValueError:
        rel = cand.replace("\\", "/")

    return rel.lstrip("./")


def parse_lizard_row_noheader(r: List[str]) -> Tuple[int, int, int, int, str]:
    nloc = safe_int(r[0])
    ccn = safe_int(r[1])
    token = safe_int(r[2])
    params = safe_int(r[4])
    file_path = r[6] if len(r) > 6 else ""
    return nloc, ccn, token, params, file_path


def aggregate_lizard_csv(
    csv_path: Path,
    root: Path,
    out_path: Path,
    group_depth: int,
    include: Optional[str] = None,
    exclude: Optional[str] = None,
) -> int:
    include_re = re.compile(include) if include else None
    exclude_re = re.compile(exclude) if exclude else None

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f) if r]

    if not rows:
        raise RuntimeError("Input CSV is empty.")

    start_idx = 1 if looks_like_header(rows[0]) else 0
    agg: Dict[str, Agg] = {}

    for r in rows[start_idx:]:
        if len(r) < 7:
            continue

        nloc, ccn, token, params, file_col = parse_lizard_row_noheader(r)
        rel_file = normalize_relpath(file_col, str(root))
        if not rel_file:
            continue

        if include_re and not include_re.search(rel_file):
            continue
        if exclude_re and exclude_re.search(rel_file):
            continue

        gkey, gtype = get_module_key(rel_file, depth=group_depth)

        a = agg.get(gkey)
        if a is None:
            a = Agg(group_key=gkey, group_type=gtype)
            agg[gkey] = a

        a.add(rel_file, nloc=nloc, ccn=ccn, token=token, params=params)
        if a.group_type != gtype and "folder" in (a.group_type, gtype):
            a.group_type = "folder"

    aggs = sorted(agg.values(), key=lambda a: (a.sum_ccn, a.functions, a.group_key), reverse=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "group_key", "group_type",
            "files", "functions",
            "sum_nloc", "sum_ccn", "sum_token",
            "avg_nloc", "avg_ccn", "avg_token",
            "max_nloc", "max_ccn", "max_token",
            "avg_params", "max_params",
        ])
        for a in aggs:
            w.writerow([
                a.group_key, a.group_type,
                a.file_count, a.functions,
                a.sum_nloc, a.sum_ccn, a.sum_token,
                f"{a.avg(a.sum_nloc):.3f}",
                f"{a.avg(a.sum_ccn):.3f}",
                f"{a.avg(a.sum_token):.3f}",
                a.max_nloc, a.max_ccn, a.max_token,
                f"{a.avg(a.sum_params):.3f}", a.max_params,
            ])

    print(f"[lizard] aggregated: {out_path} (groups={len(aggs)})")
    return 0
