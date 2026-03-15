from __future__ import annotations

import csv
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .grouping import get_module_key
from .week_rle import IsoWeek, iter_weeks


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def run_git(repo: Path, args: List[str]) -> str:
    cmd = ["git", "-C", str(repo)] + args
    try:
        cp = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return cp.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"git command failed: {' '.join(cmd)}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}\n"
        ) from e


def parse_iso_datetime(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def iso_week_bucket(dt: datetime) -> str:
    iso_year, iso_week, _ = dt.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def norm_path(p: str) -> str:
    return p.replace("\\", "/").lstrip("./")


def path_is_under_root(path: str, root_rel: str) -> bool:
    if root_rel == "":
        return True
    root_rel = root_rel.rstrip("/") + "/"
    return path.startswith(root_rel)


def parse_date_yyyy_mm_dd(s: str) -> datetime:
    d = datetime.strptime(s, "%Y-%m-%d")
    return d.replace(tzinfo=timezone.utc)


def within_date_range(dt: datetime, since: Optional[datetime], until: Optional[datetime]) -> bool:
    if since is not None and dt < since:
        return False
    if until is not None and dt > until:
        return False
    return True


@dataclass
class CommitEntry:
    dt: datetime
    subject: str
    files: List[str]


RE_COMMIT_HEADER = re.compile(r"^@@@ (?P<iso>.+?)\t(?P<subj>.*)$")


def iter_commits(repo: Path, since: Optional[str], until: Optional[str]) -> Iterable[CommitEntry]:
    args = [
        "log",
        "--date=iso-strict",
        "--pretty=format:@@@ %ad\t%s",
        "--name-only",
    ]
    if since:
        args.append(f"--since={since}")
    if until:
        args.append(f"--until={until}")

    out = run_git(repo, args)
    lines = out.splitlines()

    cur_dt: Optional[datetime] = None
    cur_subj: str = ""
    cur_files: List[str] = []

    def flush():
        nonlocal cur_dt, cur_subj, cur_files
        if cur_dt is not None:
            yield CommitEntry(dt=cur_dt, subject=cur_subj, files=cur_files)
        cur_dt = None
        cur_subj = ""
        cur_files = []

    for line in lines:
        m = RE_COMMIT_HEADER.match(line)
        if m:
            yield from flush()
            cur_dt = parse_iso_datetime(m.group("iso"))
            cur_subj = m.group("subj").strip()
            continue

        line = line.strip()
        if not line:
            continue

        cur_files.append(norm_path(line))

    yield from flush()


def count_commits_by_module_week(
    repo: Path,
    root_rel: str,
    depth: int,
    since: Optional[str],
    until: Optional[str],
    msg_keywords: Optional[List[str]],
) -> Dict[Tuple[str, str], int]:
    root_rel = norm_path(root_rel).rstrip("/")
    keywords = [k.lower() for k in (msg_keywords or []) if k.strip()]

    since_dt: Optional[datetime] = None
    until_dt: Optional[datetime] = None
    if since:
        since_dt = parse_date_yyyy_mm_dd(since)
    if until:
        until_dt = parse_date_yyyy_mm_dd(until).replace(hour=23, minute=59, second=59)

    counts: Dict[Tuple[str, str], int] = {}

    for c in iter_commits(repo=repo, since=since, until=until):
        if keywords:
            subj_l = c.subject.lower()
            if not any(k in subj_l for k in keywords):
                continue

        if not within_date_range(c.dt.astimezone(timezone.utc), since_dt, until_dt):
            continue

        modules_touched: Set[str] = set()
        for f in c.files:
            if not path_is_under_root(f, root_rel):
                continue

            rel = f[len(root_rel):].lstrip("/") if root_rel else f
            mod, _ = get_module_key(rel, depth=depth)
            modules_touched.add(mod)

        if not modules_touched:
            continue

        bucket = iso_week_bucket(c.dt)
        for mod in modules_touched:
            counts[(bucket, mod)] = counts.get((bucket, mod), 0) + 1

    return counts


def write_counts_csv(rows: List[Tuple[str, str, int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "module", "count"])
        for b, m, c in rows:
            w.writerow([b, m, c])


def modules_from_tree(scan_root: Path, depth: int) -> List[str]:
    modules: Set[str] = set()
    if not scan_root.exists():
        return []

    for p in scan_root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(scan_root).as_posix()
        mod, _ = get_module_key(rel, depth=depth)
        if mod:
            modules.add(mod)

    return sorted(modules)


def generate_test_counts_rows(
    modules: List[str],
    start_week: str,
    until_week: str,
    seed: int,
    max_count: int,
) -> List[Tuple[str, str, int]]:
    if max_count < 1:
        raise ValueError(f"max_count must be >= 1 (got {max_count!r})")

    start = IsoWeek.parse(start_week)
    until = IsoWeek.parse(until_week)
    if until.monday() < start.monday():
        raise ValueError(f"until_week must be >= start_week (got {until_week!r} < {start_week!r})")

    rng = random.Random(seed)
    rows: List[Tuple[str, str, int]] = []

    for iw in iter_weeks(start, until):
        bucket = iw.to_bucket()
        for mod in modules:
            # Sparse synthetic activity: ~40% of module-weeks get commits.
            if rng.random() < 0.6:
                continue
            rows.append((bucket, mod, rng.randint(1, max_count)))

    return rows


def run_commit_counts(
    repo: Path,
    root_rel: str,
    depth: int,
    out_path: Path,
    since: Optional[str] = None,
    until: Optional[str] = None,
    msg_keywords: Optional[List[str]] = None,
    test_mode: bool = False,
    test_seed: int = 42,
    test_max_count: int = 5,
    test_start_week: Optional[str] = None,
    test_until_week: Optional[str] = None,
) -> int:
    if depth < 1:
        raise ValueError("depth must be >= 1")

    if test_mode:
        scan_root = (repo / root_rel).resolve() if root_rel else repo.resolve()
        modules = modules_from_tree(scan_root=scan_root, depth=depth)
        rows = generate_test_counts_rows(
            modules=modules,
            start_week=test_start_week or "2025-W01",
            until_week=test_until_week or "2025-W52",
            seed=test_seed,
            max_count=test_max_count,
        )
        rows.sort(key=lambda x: (x[0], x[1]))
        write_counts_csv(rows, out_path)
        eprint(f"[commits] test mode wrote: {out_path} (rows={len(rows)}, modules={len(modules)})")
        return 0

    counts = count_commits_by_module_week(
        repo=repo,
        root_rel=root_rel,
        depth=depth,
        since=since,
        until=until,
        msg_keywords=msg_keywords,
    )
    rows = sorted([(b, m, c) for (b, m), c in counts.items()], key=lambda x: (x[0], x[1]))
    write_counts_csv(rows, out_path)
    eprint(f"[commits] wrote: {out_path} (rows={len(rows)})")
    return 0
