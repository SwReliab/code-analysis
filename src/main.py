from __future__ import annotations

import argparse
import json
from typing import Any

from .config import load_config
from .plan import build_plan, check_plan

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", required=True, help="Path to analysis.yaml/.json")
    p.add_argument("--dry-run", action="store_true", help="Print plan only; do not execute")
    p.add_argument("--format", choices=["yaml", "json"], default="yaml")
    p.add_argument("--check", action="store_true", help="Validate plan & filesystem inputs; no execution")
    p.add_argument("--execute", action="store_true", help="Actually execute. Without this flag, execution is blocked.")


def _cmd(name: str, sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(name)
    _add_common(p)
    return p


def _dump(obj: Any, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        return
    if yaml is None:
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        return
    print(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="code-analysis")
    sub = ap.add_subparsers(dest="command", required=True)

    p_embed = _cmd("embed", sub)
    p_embed.add_argument("--test", action="store_true", help="Use deterministic random vectors instead of OpenAI API")
    p_embed.add_argument("--test-dim", type=int, default=256, help="Embedding dimension used with --test")
    p_commits = _cmd("commits", sub)
    p_commits.add_argument("--test", action="store_true", help="Generate synthetic commit counts instead of reading git history")
    p_commits.add_argument("--test-seed", type=int, default=42, help="Random seed used with --test")
    p_commits.add_argument("--test-max-count", type=int, default=5, help="Max per-module commit count used with --test")
    _cmd("lizard", sub)
    _cmd("faultdata", sub)
    _cmd("run", sub)

    ns = ap.parse_args(argv)
    cfg = load_config(ns.config)
    plan = build_plan(cfg, ns.command)

    if ns.check:
        result = check_plan(plan)
        out = {"plan": plan, "check": result} if ns.dry_run else {"check": result}
        _dump(out, ns.format)
        return 0 if result.get("ok") else 2

    if ns.dry_run:
        _dump(plan, ns.format)
        return 0

    if not ns.execute:
        raise SystemExit("Execution is blocked. Add --execute (or use --dry-run / --check).")

    # Always validate before execution.
    result = check_plan(plan)
    if not result.get("ok"):
        _dump({"check": result}, ns.format)
        return 2

    from .execute import (
        execute_embed,
        execute_commits,
        execute_lizard,
        execute_faultdata,
        execute_run,
    )

    if ns.command == "embed":
        return execute_embed(cfg, test_mode=bool(getattr(ns, "test", False)), test_dim=int(getattr(ns, "test_dim", 256)))
    if ns.command == "commits":
        return execute_commits(
            cfg,
            test_mode=bool(getattr(ns, "test", False)),
            test_seed=int(getattr(ns, "test_seed", 42)),
            test_max_count=int(getattr(ns, "test_max_count", 5)),
        )
    if ns.command == "lizard":
        return execute_lizard(cfg)
    if ns.command == "faultdata":
        return execute_faultdata(cfg)
    if ns.command == "run":
        return execute_run(cfg)

    raise SystemExit(f"Unknown command: {ns.command}")


if __name__ == "__main__":
    raise SystemExit(main())