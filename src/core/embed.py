from __future__ import annotations

import fnmatch
import json
import sys
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from .grouping import get_module_key

load_dotenv()

DEFAULT_INCLUDE_GLOBS = [
    "*.c", "*.cc", "*.cpp", "*.cxx",
    "*.h", "*.hpp", "*.hxx",
    "*.py", "*.rs", "*.go", "*.java",
    "*.js", "*.ts", "*.tsx",
    "*.swift", "*.kt",
    "*.m", "*.mm",
    "*.cs",
    "*.php",
    "*.rb",
    "*.scala",
    "*.sh",
]

DEFAULT_EXCLUDE_DIRS = [
    ".git", ".hg", ".svn",
    "node_modules",
    "dist", "build", "out",
    "__pycache__",
    ".venv", "venv",
    ".idea", ".vscode",
    "third_party", "3rdparty", "vendor",
    "external",
    "generated", "gen",
]

DEFAULT_EXCLUDE_GLOBS = [
    "*.min.js",
    "*.map",
    "*.lock",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp",
    "*.pdf",
    "*.zip", "*.tar", "*.gz", "*.7z",
]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def match_any_glob(path: str, globs: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in globs)


def should_skip_path(p: Path, root: Path, exclude_dirs: List[str], exclude_globs: List[str]) -> bool:
    parts = p.relative_to(root).parts
    if any(part in exclude_dirs for part in parts):
        return True

    rel_str = str(p.relative_to(root)).replace("\\", "/")
    if match_any_glob(rel_str, exclude_globs) or match_any_glob(p.name, exclude_globs):
        return True

    return False


def is_included_file(p: Path, include_globs: List[str]) -> bool:
    return match_any_glob(p.name, include_globs)


def read_text_file(p: Path, max_bytes: Optional[int] = None) -> str:
    data = p.read_bytes()
    if max_bytes is not None and len(data) > max_bytes:
        data = data[:max_bytes]
    for enc in ("utf-8", "utf-8-sig", "cp932", "latin-1"):
        try:
            return data.decode(enc, errors="replace")
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def split_into_chunks(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    if chunk_chars <= 0:
        return [text]
    if overlap_chars < 0:
        overlap_chars = 0
    chunks: List[str] = []
    n = len(text)
    i = 0
    while i < n:
        j = min(i + chunk_chars, n)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap_chars)
    return chunks


def count_loc(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def deterministic_random_unit_vector(rel_path: str, dim: int) -> np.ndarray:
    if dim <= 0:
        raise ValueError(f"test_dim must be >= 1 (got {dim!r})")
    seed_src = hashlib.sha1(rel_path.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_src, byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return l2_normalize(v)


@dataclass
class FileEmbeddingResult:
    rel_path: str
    group_key: str
    group_type: str
    loc: int
    n_chars: int
    embedding: np.ndarray


class OpenAIEmbedder:
    def __init__(self, model: str, batch_size: int = 32, max_retries: int = 6, sleep_base: float = 1.0):
        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.sleep_base = sleep_base

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            out.extend(self._embed_batch_with_retry(batch))
        return out

    def _embed_batch_with_retry(self, batch: List[str]) -> List[np.ndarray]:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.embeddings.create(model=self.model, input=batch)
                data_sorted = sorted(resp.data, key=lambda d: d.index)
                return [np.array(d.embedding, dtype=np.float32) for d in data_sorted]
            except Exception as e:
                last_err = e
                sleep_s = self.sleep_base * (2 ** attempt)
                eprint(f"[warn] embedding batch failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(sleep_s)
        raise RuntimeError(f"OpenAI embeddings failed after retries: {last_err}") from last_err


def compute_file_embedding(
    embedder: OpenAIEmbedder,
    file_path: Path,
    root: Path,
    chunk_chars: int,
    overlap_chars: int,
    max_file_bytes: Optional[int],
    group_depth: int,
) -> Optional[FileEmbeddingResult]:
    rel_path = str(file_path.relative_to(root)).replace("\\", "/")

    try:
        text = read_text_file(file_path, max_bytes=max_file_bytes)
    except Exception as e:
        eprint(f"[skip] failed reading: {rel_path} ({e})")
        return None

    text = text.strip("\ufeff")
    if not text.strip():
        return None

    loc = count_loc(text)
    n_chars = len(text)
    if n_chars == 0:
        return None

    chunks = split_into_chunks(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        return None

    chunk_embs = embedder.embed_texts(chunks)
    weights = np.array([max(1, len(c)) for c in chunks], dtype=np.float32)
    mat = np.stack(chunk_embs, axis=0)
    file_emb = np.average(mat, axis=0, weights=weights)
    file_emb = l2_normalize(file_emb)

    gkey, gtype = get_module_key(rel_path, depth=group_depth)

    return FileEmbeddingResult(
        rel_path=rel_path,
        group_key=gkey,
        group_type=gtype,
        loc=loc,
        n_chars=n_chars,
        embedding=file_emb,
    )


def compute_file_embedding_test(
    file_path: Path,
    root: Path,
    group_depth: int,
    max_file_bytes: Optional[int],
    test_dim: int,
) -> Optional[FileEmbeddingResult]:
    rel_path = str(file_path.relative_to(root)).replace("\\", "/")

    try:
        text = read_text_file(file_path, max_bytes=max_file_bytes)
    except Exception as e:
        eprint(f"[skip] failed reading: {rel_path} ({e})")
        return None

    text = text.strip("\ufeff")
    if not text.strip():
        return None

    loc = count_loc(text)
    n_chars = len(text)
    if n_chars == 0:
        return None

    gkey, gtype = get_module_key(rel_path, depth=group_depth)
    file_emb = deterministic_random_unit_vector(rel_path=rel_path, dim=test_dim)

    return FileEmbeddingResult(
        rel_path=rel_path,
        group_key=gkey,
        group_type=gtype,
        loc=loc,
        n_chars=n_chars,
        embedding=file_emb,
    )


def aggregate_group_embeddings(files: List[FileEmbeddingResult], weight_mode: str = "loc") -> Dict[str, Dict]:
    by_group: Dict[str, List[FileEmbeddingResult]] = {}
    for f in files:
        by_group.setdefault(f.group_key, []).append(f)

    group_records: Dict[str, Dict] = {}

    for group_key, flist in by_group.items():
        mat = np.stack([f.embedding for f in flist], axis=0)

        if weight_mode == "loc":
            w = np.array([max(1, f.loc) for f in flist], dtype=np.float32)
        elif weight_mode == "chars":
            w = np.array([max(1, f.n_chars) for f in flist], dtype=np.float32)
        elif weight_mode == "uniform":
            w = np.ones((len(flist),), dtype=np.float32)
        else:
            raise ValueError("weight_mode must be one of: loc, chars, uniform")

        group_emb = np.average(mat, axis=0, weights=w)
        group_emb = l2_normalize(group_emb)

        total_loc = int(sum(f.loc for f in flist))
        total_chars = int(sum(f.n_chars for f in flist))
        group_type = "folder" if any(f.group_type == "folder" for f in flist) else "file"

        group_records[group_key] = {
            "folder": group_key,
            "group_type": group_type,
            "n_files": len(flist),
            "total_loc": total_loc,
            "total_chars": total_chars,
            "weight_mode": weight_mode,
            "embedding": group_emb.astype(float).tolist(),
            "files": [f.rel_path for f in flist],
        }

    return group_records


def walk_source_files(root: Path, include_globs: List[str], exclude_dirs: List[str], exclude_globs: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if should_skip_path(p, root, exclude_dirs, exclude_globs):
            continue
        if not is_included_file(p, include_globs):
            continue
        out.append(p)
    return out


def run_embed(
    root: Path,
    out_path: Path,
    group_depth: int,
    model: str,
    weight_mode: str,
    include_globs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    exclude_globs: Optional[List[str]] = None,
    chunk_chars: int = 6000,
    overlap_chars: int = 300,
    max_file_bytes: int = 2_000_000,
    batch_size: int = 32,
    limit_files: int = 0,
    test_mode: bool = False,
    test_dim: int = 256,
) -> int:
    include_globs = include_globs or list(DEFAULT_INCLUDE_GLOBS)
    exclude_dirs = exclude_dirs or list(DEFAULT_EXCLUDE_DIRS)
    exclude_globs = exclude_globs or list(DEFAULT_EXCLUDE_GLOBS)

    max_file_bytes_opt = None if max_file_bytes == 0 else max_file_bytes

    files = walk_source_files(
        root=root,
        include_globs=include_globs,
        exclude_dirs=exclude_dirs,
        exclude_globs=exclude_globs,
    )
    files.sort()

    if limit_files and limit_files > 0:
        files = files[:limit_files]

    eprint(f"[embed] files to process: {len(files)}")

    embedder: Optional[OpenAIEmbedder] = None
    if not test_mode:
        embedder = OpenAIEmbedder(model=model, batch_size=batch_size)
    else:
        eprint(f"[embed] test mode enabled: deterministic random vectors (dim={test_dim})")

    file_results: List[FileEmbeddingResult] = []
    for idx, fp in enumerate(files, start=1):
        eprint(f"[embed] ({idx}/{len(files)}) {fp.relative_to(root)}")
        if test_mode:
            r = compute_file_embedding_test(
                file_path=fp,
                root=root,
                group_depth=group_depth,
                max_file_bytes=max_file_bytes_opt,
                test_dim=test_dim,
            )
        else:
            assert embedder is not None
            r = compute_file_embedding(
                embedder=embedder,
                file_path=fp,
                root=root,
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
                max_file_bytes=max_file_bytes_opt,
                group_depth=group_depth,
            )
        if r is not None:
            file_results.append(r)

    group_records = aggregate_group_embeddings(file_results, weight_mode=weight_mode)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, rec in sorted(group_records.items(), key=lambda x: x[0]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    eprint(f"[embed] wrote: {out_path} (groups={len(group_records)})")
    return 0
