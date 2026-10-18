from __future__ import annotations
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np
EPS = 1e-12

@dataclass(frozen=True)
class OperatingPoint:
    target_precision: float
    threshold: float
    precision: float
    recall: float
    accepted: int
    accepted_positives: int
    frontier_percentile: float
    feasible: bool

def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))

def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out

def linear_scores(features: np.ndarray, weight: np.ndarray, bias: float | np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float64) @ np.asarray(weight, dtype=np.float64).reshape(-1) + float(np.asarray(bias).reshape(-1)[0])

def operating_point(labels: np.ndarray, scores: np.ndarray, target_precision: float) -> OperatingPoint:
    labels = np.asarray(labels).astype(int).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-scores, kind='mergesort')
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels)
    accepted = np.arange(1, labels.size + 1)
    precision = cumulative_tp / accepted
    total_positive = max(int(labels.sum()), 1)
    recall = cumulative_tp / total_positive
    valid = np.flatnonzero(precision >= float(target_precision))
    if valid.size == 0:
        return OperatingPoint(target_precision=float(target_precision), threshold=float('inf'), precision=0.0, recall=0.0, accepted=0, accepted_positives=0, frontier_percentile=0.0, feasible=False)
    valid_recall = recall[valid]
    best_recall = valid_recall.max()
    tied = valid[valid_recall == best_recall]
    best = int(tied[-1])
    return OperatingPoint(target_precision=float(target_precision), threshold=float(scores[order[best]]), precision=float(precision[best]), recall=float(recall[best]), accepted=int(best + 1), accepted_positives=int(cumulative_tp[best]), frontier_percentile=float(best / max(labels.size - 1, 1)), feasible=True)

def evaluate_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels = np.asarray(labels).astype(int).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    selected = scores >= float(threshold)
    accepted = int(selected.sum())
    tp = int(labels[selected].sum()) if accepted else 0
    precision = float(tp / accepted) if accepted else 0.0
    recall = float(tp / max(int(labels.sum()), 1))
    return {'precision': precision, 'recall': recall, 'accepted': accepted, 'accepted_positives': tp}

def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    order = np.argsort(-scores, kind='mergesort')
    sorted_labels = labels[order]
    total_positive = int(sorted_labels.sum())
    if total_positive == 0:
        return 0.0
    precision = np.cumsum(sorted_labels) / np.arange(1, labels.size + 1)
    return float((precision * sorted_labels).sum() / total_positive)

def rank_percentiles(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores).reshape(-1)
    order = np.argsort(-scores, kind='mergesort')
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(scores.size, dtype=np.float64)
    return ranks / max(scores.size - 1, 1)

def frontier_band_mask(score_ranks: np.ndarray, frontier_percentile: float, width: float=0.1) -> np.ndarray:
    half = width / 2.0
    lower = frontier_percentile - half
    upper = frontier_percentile + half
    if lower < 0.0:
        upper -= lower
        lower = 0.0
    if upper > 1.0:
        lower -= upper - 1.0
        upper = 1.0
    lower = max(lower, 0.0)
    return (score_ranks >= lower) & (score_ranks <= upper)

def pairwise_accuracy(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    positive = scores[mask & (labels == 1)]
    negative = scores[mask & (labels == 0)]
    if positive.size == 0 or negative.size == 0:
        return float('nan')
    comparisons = positive[:, None] - negative[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())

def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
