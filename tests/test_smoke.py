"""Small, dependency-light checks for the released BORO snapshot."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@pytest.mark.parametrize(
    "module",
    [
        "boro.ade.train",
        "boro.prsm.train",
        "boro.ir.train",
        "boro.controlled.train",
    ],
)
def test_training_entrypoints_show_help(module: str) -> None:
    """All documented training entry points should be importable and expose CLI help."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage" in completed.stdout.lower()


def test_ade_metrics_return_boundary_and_ranking_fields() -> None:
    """The public ADE metric helper returns finite ranking and boundary metrics."""
    from boro.ade.train import all_metrics

    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    scores = np.array([0.10, 0.90, 0.20, 0.80], dtype=np.float64)
    metrics = all_metrics(labels, scores)

    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["best_f1"] == pytest.approx(1.0)
    for key in (
        "r_at_p90",
        "r_at_p95",
        "r_at_p98",
        "r_at_p99",
        "r_at_fpr0p01",
        "r_at_fpr0p05",
        "r_at_fpr0p1",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_boro_ade_json_matches_source_table() -> None:
    """The ten released BORO ADE JSON summaries agree with the analysis source table."""
    source_path = ROOT / "data/source/ade/main_per_seed.csv"
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] == "BORO"]

    assert len(rows) == 10
    field_map = {
        "auroc": "auroc",
        "auprc": "ap",
        "r_at_p90": "r_at_p90",
        "r_at_p95": "r_at_p95",
        "r_at_p98": "r_at_p98",
        "r_at_p99": "r_at_p99",
        "r_at_fpr0p01": "tpr_at_fpr1",
        "r_at_fpr0p05": "tpr_at_fpr5",
        "r_at_fpr0p1": "tpr_at_fpr10",
        "best_f1": "best_f1",
    }

    for row in rows:
        seed = row["seed"]
        metrics_path = ROOT / f"results/ade/boro/seed_{seed}/metrics.json"
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)["test_metrics"]
        assert "r_at_p99" in metrics
        for json_field, source_field in field_map.items():
            assert float(metrics[json_field]) == pytest.approx(
                float(row[source_field]), abs=2e-12
            )
