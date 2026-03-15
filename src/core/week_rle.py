from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class IsoWeek:
    year: int
    week: int

    @staticmethod
    def parse(s: str) -> "IsoWeek":
        s = s.strip()
        try:
            y_str, w_str = s.split("-W", 1)
            y = int(y_str)
            w = int(w_str)
            if not (1 <= w <= 53):
                raise ValueError
            _ = date.fromisocalendar(y, w, 1)
            return IsoWeek(y, w)
        except Exception as e:
            raise ValueError(f"Invalid ISO week: {s!r} (expected like '2013-W08')") from e

    def to_bucket(self) -> str:
        return f"{self.year}-W{self.week:02d}"

    def monday(self) -> date:
        return date.fromisocalendar(self.year, self.week, 1)

    @staticmethod
    def from_monday(d: date) -> "IsoWeek":
        y, w, _ = d.isocalendar()
        return IsoWeek(int(y), int(w))


def iter_weeks(start: IsoWeek, end: IsoWeek) -> Iterable[IsoWeek]:
    cur = start.monday()
    last = end.monday()
    step = timedelta(days=7)
    while cur <= last:
        yield IsoWeek.from_monday(cur)
        cur += step


def rle_encode(values: List[int]) -> List[Tuple[int, int]]:
    if not values:
        return []
    out: List[Tuple[int, int]] = []
    cur_v = values[0]
    span = 1
    for v in values[1:]:
        if v == cur_v:
            span += 1
        else:
            out.append((span, cur_v))
            cur_v = v
            span = 1
    out.append((span, cur_v))
    return out


def module_week_rle_rows(in_path: Path, module: str, start: str, until: str) -> List[Tuple[int, int]]:
    start_w = IsoWeek.parse(start)
    end_w = IsoWeek.parse(until)
    if end_w.monday() < start_w.monday():
        raise ValueError(f"until must be >= start (got {until} < {start})")

    agg: Dict[str, int] = {}
    with in_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or not {"bucket", "module", "count"}.issubset(set(r.fieldnames)):
            raise ValueError("CSV header must include: bucket,module,count")

        for row in r:
            b = (row.get("bucket") or "").strip()
            m = (row.get("module") or "").strip()
            c = (row.get("count") or "").strip()
            if not b or not m or m != module:
                continue
            try:
                iw = IsoWeek.parse(b)
                val = int(float(c)) if c else 0
            except Exception:
                continue
            agg[iw.to_bucket()] = agg.get(iw.to_bucket(), 0) + val

    series: List[int] = []
    for iw in iter_weeks(start_w, end_w):
        series.append(agg.get(iw.to_bucket(), 0))

    return rle_encode(series)


def write_rle_csv(rows: List[Tuple[int, int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["week", "counts"])
        for span, val in rows:
            w.writerow([span, val])
