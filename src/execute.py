from __future__ import annotations

from pathlib import Path

from .config import AppCfg
from .core.commit_counts import run_commit_counts
from .core.embed import run_embed
from .core.faultdata import execute_faultdata as run_faultdata
from .core.lizard import aggregate_lizard_csv, execute_lizard_scan as run_lizard_scan


# -----------------------
# Lizard scan
# -----------------------

def execute_lizard_scan(cfg: AppCfg) -> int:
    lz = cfg.lizard
    m = cfg.module
    return run_lizard_scan(
        root=Path(lz.root or m.root_src),
        out_csv=Path(lz.csv),
        languages=lz.languages,
        extra_args=lz.extra_args,
    )


def execute_lizard_agg(cfg: AppCfg) -> int:
    """
    Aggregate lizard CSV to module metrics.
    """
    m = cfg.module
    out = cfg.outputs
    lz = cfg.lizard
    return aggregate_lizard_csv(
        csv_path=Path(lz.csv),
        root=Path(m.root_src),
        out_path=Path(out.module_metrics_csv),
        group_depth=m.depth,
    )


# -----------------------
# Execute steps
# -----------------------

def execute_embed(cfg: AppCfg, test_mode: bool = False, test_dim: int = 256) -> int:
    m = cfg.module
    out = cfg.outputs

    model = cfg.embedding.model
    weight_mode = cfg.embedding.weight_mode
    return run_embed(
        root=Path(m.root_src),
        out_path=Path(out.embeddings_jsonl),
        group_depth=m.depth,
        model=str(model),
        weight_mode=str(weight_mode),
        test_mode=test_mode,
        test_dim=test_dim,
    )


def execute_commits(
    cfg: AppCfg,
    test_mode: bool = False,
    test_seed: int = 42,
    test_max_count: int = 5,
) -> int:
    m = cfg.module
    out = cfg.outputs
    rle = cfg.rle
    return run_commit_counts(
        repo=Path(m.git_root),
        root_rel=m.git_subdir,
        depth=m.depth,
        out_path=Path(out.commits_by_module_week_csv),
        test_mode=test_mode,
        test_seed=test_seed,
        test_max_count=test_max_count,
        test_start_week=rle.start,
        test_until_week=rle.until,
    )


def execute_lizard(cfg: AppCfg) -> int:
    """
    Run scan -> agg
    """
    rc = execute_lizard_scan(cfg)
    if rc != 0:
        return rc
    return execute_lizard_agg(cfg)


def execute_faultdata(cfg: AppCfg) -> int:
    out = cfg.outputs
    rle = cfg.rle

    embeddings = Path(out.embeddings_jsonl)
    commits_csv = Path(out.commits_by_module_week_csv)
    out_dir = Path(out.faultdata_dir)
    return run_faultdata(
        embeddings=embeddings,
        commits_csv=commits_csv,
        out_dir=out_dir,
        start=rle.start,
        until=rle.until,
    )


def execute_run(cfg: AppCfg) -> int:
    steps = [
        ("embed", lambda: execute_embed(cfg)),
        ("commits", lambda: execute_commits(cfg)),
        ("lizard", lambda: execute_lizard(cfg)),
        ("faultdata", lambda: execute_faultdata(cfg)),
    ]

    for name, fn in steps:
        print(f"[run] start: {name}")
        rc = fn()
        if rc != 0:
            print(f"[run] FAILED: {name} (rc={rc})")
            return rc
        print(f"[run] done: {name}")

    print("[run] all done")
    return 0