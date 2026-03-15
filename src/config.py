from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None


@dataclass(frozen=True)
class ModuleCfg:
    root_src: str
    git_root: str
    git_subdir: str = "src"
    depth: int = 2


@dataclass(frozen=True)
class OutputCfg:
    embeddings_jsonl: str = "group_embeddings.jsonl"
    commits_by_module_week_csv: str = "commits_by_module_week.csv"
    module_metrics_csv: str = "module_metrics.csv"
    faultdata_dir: str = "faultdata"


@dataclass(frozen=True)
class RleCfg:
    start: str
    until: str


@dataclass(frozen=True)
class EmbeddingCfg:
    model: str = "text-embedding-3-small"
    weight_mode: str = "loc"


@dataclass(frozen=True)
class LizardCfg:
    csv: str = "lizard.csv"
    root: Optional[str] = None               # None -> use module.root_src
    languages: List[str] = None              # e.g. ["c","cpp"]
    extra_args: List[str] = None             # e.g. ["--exclude", "build"]

    def norm(self) -> "LizardCfg":
        return LizardCfg(
            csv=self.csv,
            root=self.root,
            languages=list(self.languages or []),
            extra_args=list(self.extra_args or []),
        )


@dataclass(frozen=True)
class AppCfg:
    module: ModuleCfg
    outputs: OutputCfg
    rle: RleCfg
    embedding: EmbeddingCfg
    lizard: LizardCfg


def _load_mapping(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("YAML config requires pyyaml. Install with: pip install pyyaml")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("YAML root must be a mapping/object")
        return data
    if path.suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object")
        return data
    raise ValueError(f"Unsupported config extension: {path.suffix} (use .yaml/.yml/.json)")


def load_config(config_path: str) -> AppCfg:
    p = Path(config_path).expanduser().resolve()
    d = _load_mapping(p)

    if "module" not in d:
        raise ValueError("config missing required key: module")
    if "rle" not in d:
        raise ValueError("config missing required key: rle")

    module = ModuleCfg(**d["module"])
    outputs = OutputCfg(**d.get("outputs", {}))
    rle = RleCfg(**d["rle"])
    embedding_raw = d.get("embedding", {})
    if not isinstance(embedding_raw, dict):
        raise ValueError("embedding must be a mapping/object")
    embedding = EmbeddingCfg(**embedding_raw)

    lizard_raw = d.get("lizard", {}) or {}
    if not isinstance(lizard_raw, dict):
        raise ValueError("lizard must be a mapping/object if provided")
    lizard = LizardCfg(**lizard_raw).norm()

    return AppCfg(module=module, outputs=outputs, rle=rle, embedding=embedding, lizard=lizard)