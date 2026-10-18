#!/usr/bin/env python3
"""Download public BORO inputs without overwriting existing files.

Examples
--------
Download all supported resources into the repository's default locations::

    python scripts/download_data.py

Download selected BEIR collections only::

    python scripts/download_data.py --resource beir --datasets scifact fiqa

Preview actions without touching the filesystem::

    python scripts/download_data.py --resource all --dry-run

The downloader deliberately does not redistribute raw data. It records the
source URL and prints a SHA-256 digest after each completed download; the
release manifest can capture those digests in a later step.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEIR_ROOT = PROJECT_ROOT / "data" / "raw" / "ir"
DEFAULT_E5_ROOT = PROJECT_ROOT / "data" / "pretrained" / "e5"

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
BEIR_DATASETS = (
    "msmarco",
    "arguana",
    "dbpedia-entity",
    "scidocs",
    "scifact",
    "fiqa",
    "nfcorpus",
)

E5_REPOSITORY = "intfloat/e5-small-v2"
E5_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path, dry_run: bool) -> bool:
    """Download one URL atomically, returning whether a file was written."""
    if destination.exists():
        print(f"[skip] {destination} already exists")
        return False
    if dry_run:
        print(f"[dry-run] {url} -> {destination}")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=str(destination.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        request = Request(url, headers={"User-Agent": "BORO-data-downloader/1.0"})
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"[downloaded] {destination} sha256={sha256_file(destination)}")
    return True


def _safe_extract_zip(archive: Path, destination: Path, dry_run: bool) -> None:
    """Extract a ZIP without path traversal or overwriting existing files."""
    if dry_run:
        print(f"[dry-run] extract {archive} -> {destination}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member outside destination: {member.filename}")
        skipped = 0
        extracted = 0
        for member in members:
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member, "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted += 1
        if skipped:
            print(f"[skip] {skipped} existing files while extracting {archive.name}")
        if extracted:
            print(f"[extracted] {extracted} files from {archive.name}")


def _dataset_ready(dataset_dir: Path) -> bool:
    qrels_dir = dataset_dir / "qrels"
    return (
        (dataset_dir / "corpus.jsonl").is_file()
        and (dataset_dir / "queries.jsonl").is_file()
        and qrels_dir.is_dir()
        and any(qrels_dir.glob("*.tsv"))
    )


def download_beir(
    datasets: Iterable[str], data_root: Path, dry_run: bool = False
) -> None:
    data_root = data_root.expanduser()
    for name in datasets:
        dataset_dir = data_root / name
        archive = data_root / f"{name}.zip"
        if _dataset_ready(dataset_dir):
            print(f"[skip] BEIR {name} already contains corpus.jsonl and queries.jsonl")
            continue
        url = f"{BEIR_BASE_URL}/{quote(name)}.zip"
        _download_file(url, archive, dry_run=dry_run)
        if not dry_run and not archive.is_file():
            raise FileNotFoundError(f"Download did not create archive: {archive}")
        if archive.is_file():
            try:
                _safe_extract_zip(archive, data_root, dry_run=dry_run)
            except zipfile.BadZipFile as exc:
                raise RuntimeError(f"Invalid BEIR archive: {archive}") from exc
        if not dry_run and not _dataset_ready(dataset_dir):
            raise FileNotFoundError(
                f"Extracted {archive}, but {dataset_dir} does not contain corpus.jsonl and queries.jsonl"
            )
        if not dry_run:
            print(f"[ready] BEIR {name}: {dataset_dir}")


def download_e5(
    model_root: Path, revision: str = "main", dry_run: bool = False
) -> None:
    model_root = model_root.expanduser()
    encoded_revision = quote(revision, safe="")
    for filename in E5_FILES:
        encoded_file = quote(filename, safe="")
        url = (
            f"https://huggingface.co/{E5_REPOSITORY}/resolve/"
            f"{encoded_revision}/{encoded_file}?download=true"
        )
        _download_file(url, model_root / filename, dry_run=dry_run)
    if not dry_run:
        missing = [name for name in E5_FILES if not (model_root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"E5 model is incomplete; missing: {', '.join(missing)}")
        print(f"[ready] E5 {E5_REPOSITORY}@{revision}: {model_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resource",
        choices=("all", "beir", "e5"),
        default="all",
        help="Resource family to download (default: all).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=BEIR_DATASETS,
        default=list(BEIR_DATASETS),
        help="BEIR collections to download when --resource is all or beir.",
    )
    parser.add_argument(
        "--beir-root",
        type=Path,
        default=DEFAULT_BEIR_ROOT,
        help=f"BEIR data directory (default: {DEFAULT_BEIR_ROOT}).",
    )
    parser.add_argument(
        "--e5-root",
        type=Path,
        default=DEFAULT_E5_ROOT,
        help=f"E5 model directory (default: {DEFAULT_E5_ROOT}).",
    )
    parser.add_argument(
        "--e5-revision",
        default="main",
        help="Hugging Face revision for intfloat/e5-small-v2 (default: main).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads and extraction without changing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resource in {"all", "beir"}:
        download_beir(args.datasets, args.beir_root, dry_run=args.dry_run)
    if args.resource in {"all", "e5"}:
        download_e5(args.e5_root, revision=args.e5_revision, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
