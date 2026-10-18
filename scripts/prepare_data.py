#!/usr/bin/env python3
"""Prepare BORO's derived inputs while preserving existing files.

The command is intentionally idempotent: an existing output is skipped rather
than overwritten. Remove an output directory yourself if you want to rebuild a
specific artifact.

Examples
--------
Prepare all available inputs (missing BEIR data may be downloaded)::

    python scripts/prepare_data.py

Prepare only controlled datasets and ADE split files::

    python scripts/prepare_data.py --task controlled --seeds 7103 7121
    python scripts/prepare_data.py --task ade --seeds 20260417

Preview all actions without writing files or downloading data::

    python scripts/prepare_data.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from boro.ade.train import (  # noqa: E402
    DEFAULT_SEEDS as ADE_SEEDS,
    build_clean_protocol,
    save_protocol_splits,
)
from boro.controlled.data import (  # noqa: E402
    MixtureConfig,
    generate_dataset,
    save_dataset,
)
from download_data import BEIR_DATASETS, download_beir  # noqa: E402

CONTROLLED_SEEDS = (
    7103,
    7121,
    7151,
    7177,
    7207,
    7237,
    7253,
    7283,
    7307,
    7331,
    7351,
    7387,
    7411,
    7433,
    7457,
    7481,
    7507,
    7529,
    7559,
    7583,
)

DEFAULT_ADE_ROOT = PROJECT_ROOT / "data" / "raw" / "ade"
DEFAULT_ADE_OUTPUT = PROJECT_ROOT / "data" / "processed" / "ade"
DEFAULT_CONTROLLED_CONFIG = PROJECT_ROOT / "experiments" / "controlled" / "config.json"
DEFAULT_CONTROLLED_OUTPUT = PROJECT_ROOT / "data" / "processed" / "controlled"
DEFAULT_PRSM_OUTPUT = PROJECT_ROOT / "data" / "processed" / "prsm"
DEFAULT_PRSM_MODELS = PROJECT_ROOT / "data" / "pretrained" / "prsm"
DEFAULT_BEIR_ROOT = PROJECT_ROOT / "data" / "raw" / "ir"


def _parse_seed_list(values: Iterable[str], default: tuple[int, ...]) -> list[int]:
    values = list(values)
    return list(default) if not values else [int(value) for value in values]


def prepare_ade(
    seeds: Iterable[int], raw_root: Path, output_root: Path, dry_run: bool
) -> None:
    raw_root = raw_root.expanduser()
    output_root = output_root.expanduser()
    required = (raw_root / "train.jsonl", raw_root / "test.jsonl")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"ADE raw files are missing: {', '.join(missing)}")
    for seed in seeds:
        target = output_root / f"seed_{seed}"
        expected = [target / "train.jsonl", target / "val.jsonl", target / "test.jsonl", target / "metadata.json"]
        if all(path.is_file() for path in expected):
            print(f"[skip] ADE seed {seed}: {target}")
            continue
        if dry_run:
            print(f"[dry-run] prepare ADE seed {seed}: {target}")
            continue
        protocol = build_clean_protocol(raw_root, int(seed))
        save_protocol_splits(protocol, target)
        print(f"[ready] ADE seed {seed}: {target}")


def prepare_controlled(
    seeds: Iterable[int], config_path: Path, output_root: Path, dry_run: bool
) -> None:
    config_path = config_path.expanduser()
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    data_config = MixtureConfig(**raw_config["data"]["config"])
    metadata = raw_config["data"].get("metadata", {})
    world_seed = int(metadata.get("world_seed", 2026072907))
    for seed in seeds:
        target = output_root.expanduser() / f"seed_{seed}.npz"
        metadata_path = target.with_suffix(".json")
        if target.is_file() and metadata_path.is_file():
            print(f"[skip] controlled seed {seed}: {target}")
            continue
        if dry_run:
            print(f"[dry-run] prepare controlled seed {seed}: {target}")
            continue
        payload = generate_dataset(
            int(seed),
            data_config,
            world_seed=world_seed,
            initialization_seed=int(seed) + 700001,
        )
        save_dataset(target, payload)
        print(f"[ready] controlled seed {seed}: {target}")


def _copy_missing_tree(source: Path, destination: Path) -> tuple[int, int]:
    copied = 0
    skipped = 0
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        if target.exists():
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        copied += 1
    return copied, skipped


def prepare_prsm(
    output_root: Path, models_root: Path, dry_run: bool
) -> None:
    output_root = output_root.expanduser()
    models_root = models_root.expanduser()
    if not (PROJECT_ROOT / "data/processed/prsm/train_files.csv").is_file():
        raise FileNotFoundError("Missing data/processed/prsm/train_files.csv")
    if not (PROJECT_ROOT / "data/processed/prsm/test_files.csv").is_file():
        raise FileNotFoundError("Missing data/processed/prsm/test_files.csv")
    train_target = output_root / "train.npz"
    test_target = output_root / "test"
    if train_target.is_file() and test_target.is_dir() and any(test_target.glob("*.npz")):
        print(f"[skip] PrSM feature cache already exists: {output_root}")
        return
    if dry_run:
        print(f"[dry-run] prepare PrSM cache: {output_root}")
        return

    with tempfile.TemporaryDirectory(prefix="boro-prsm-") as temporary_dir:
        command = [
            sys.executable,
            "-m",
            "boro.prsm.prepare",
            "--output-dir",
            temporary_dir,
            "--models",
            str(models_root),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SRC_ROOT)
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
        copied, skipped = _copy_missing_tree(Path(temporary_dir), output_root)
    print(f"[ready] PrSM cache: copied={copied} skipped_existing={skipped} root={output_root}")


def prepare_ir(
    datasets: Iterable[str], data_root: Path, dry_run: bool
) -> None:
    download_beir(datasets, data_root.expanduser(), dry_run=dry_run)
    if not dry_run:
        for name in datasets:
            folder = data_root.expanduser() / name
            for required in ("corpus.jsonl", "queries.jsonl"):
                if not (folder / required).is_file():
                    raise FileNotFoundError(f"BEIR {name} is missing {required}: {folder}")
            qrels_dir = folder / "qrels"
            if not qrels_dir.is_dir() or not any(qrels_dir.glob("*.tsv")):
                raise FileNotFoundError(f"BEIR {name} is missing qrels/*.tsv: {folder}")
            print(f"[ready] IR data {name}: {folder}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("all", "ade", "controlled", "prsm", "ir"),
        default="all",
        help="Preparation family to run (default: all).",
    )
    parser.add_argument("--seeds", nargs="*", default=None, help="Seeds for ADE or controlled preparation.")
    parser.add_argument("--ade-raw-root", type=Path, default=DEFAULT_ADE_ROOT)
    parser.add_argument("--ade-output-root", type=Path, default=DEFAULT_ADE_OUTPUT)
    parser.add_argument("--controlled-config", type=Path, default=DEFAULT_CONTROLLED_CONFIG)
    parser.add_argument("--controlled-output-root", type=Path, default=DEFAULT_CONTROLLED_OUTPUT)
    parser.add_argument("--prsm-output-root", type=Path, default=DEFAULT_PRSM_OUTPUT)
    parser.add_argument("--prsm-models", type=Path, default=DEFAULT_PRSM_MODELS)
    parser.add_argument("--beir-root", type=Path, default=DEFAULT_BEIR_ROOT)
    parser.add_argument("--ir-datasets", nargs="+", choices=BEIR_DATASETS, default=list(BEIR_DATASETS))
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without writing or downloading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task in {"all", "ade"}:
        prepare_ade(
            _parse_seed_list(args.seeds or [], ADE_SEEDS),
            args.ade_raw_root,
            args.ade_output_root,
            args.dry_run,
        )
    if args.task in {"all", "controlled"}:
        prepare_controlled(
            _parse_seed_list(args.seeds or [], CONTROLLED_SEEDS),
            args.controlled_config,
            args.controlled_output_root,
            args.dry_run,
        )
    if args.task in {"all", "prsm"}:
        prepare_prsm(args.prsm_output_root, args.prsm_models, args.dry_run)
    if args.task in {"all", "ir"}:
        prepare_ir(args.ir_datasets, args.beir_root, args.dry_run)


if __name__ == "__main__":
    main()
