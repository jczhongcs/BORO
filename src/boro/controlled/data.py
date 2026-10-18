from __future__ import annotations
import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp

@dataclass(frozen=True)
class MixtureConfig:
    n_train: int = 10000
    n_validation: int = 10000
    n_test: int = 20000
    n_features: int = 64
    prevalence: float = 0.2
    positive_overlap_fraction: float = 0.3
    negative_overlap_fraction: float = 0.08
    easy_separation: float = 2.25
    overlap_location: float = 0.9
    secondary_separation: float = 1.0
    easy_secondary_conflict: float = 0.0
    informative_sd: float = 0.85
    nuisance_sd: float = 1.0

def random_orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs

def gaussian_log_density(values: np.ndarray, mean: np.ndarray, sd: float) -> np.ndarray:
    dimension = values.shape[1]
    squared = np.square((values - mean) / sd).sum(axis=1)
    return -0.5 * squared - dimension * math.log(sd) - 0.5 * dimension * math.log(2.0 * math.pi)

def class_log_density(values: np.ndarray, means: tuple[np.ndarray, np.ndarray], overlap_fraction: float, sd: float) -> np.ndarray:
    easy = gaussian_log_density(values, means[0], sd)
    overlap = gaussian_log_density(values, means[1], sd)
    return logsumexp(np.stack([math.log(1.0 - overlap_fraction) + easy, math.log(overlap_fraction) + overlap], axis=0), axis=0)

def draw_split(rng: np.random.Generator, config: MixtureConfig, means: dict[str, np.ndarray], rotation: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = rng.binomial(1, config.prevalence, size=n_samples).astype(np.int64)
    overlap_probability = np.where(labels == 1, config.positive_overlap_fraction, config.negative_overlap_fraction)
    overlap = rng.binomial(1, overlap_probability).astype(bool)
    informative = np.empty((n_samples, 2), dtype=np.float64)
    for label, name in ((0, 'negative'), (1, 'positive')):
        for is_overlap, suffix in ((False, 'easy'), (True, 'overlap')):
            mask = (labels == label) & (overlap == is_overlap)
            informative[mask] = rng.normal(loc=means[f'{name}_{suffix}'], scale=config.informative_sd, size=(int(mask.sum()), 2))
    nuisance = rng.normal(loc=0.0, scale=config.nuisance_sd, size=(n_samples, config.n_features - 2))
    latent = np.concatenate([informative, nuisance], axis=1)
    observed = latent @ rotation
    positive_log = math.log(config.prevalence) + class_log_density(informative, (means['positive_easy'], means['positive_overlap']), config.positive_overlap_fraction, config.informative_sd)
    negative_log = math.log(1.0 - config.prevalence) + class_log_density(informative, (means['negative_easy'], means['negative_overlap']), config.negative_overlap_fraction, config.informative_sd)
    posterior = np.exp(positive_log - np.logaddexp(positive_log, negative_log))
    return (observed.astype(np.float32), labels.astype(np.float32), posterior.astype(np.float32), overlap.astype(np.int8))

def generate_dataset(seed: int, config: MixtureConfig, *, world_seed: int | None=None, initialization_seed: int | None=None) -> dict[str, np.ndarray | dict]:
    world_rng = np.random.default_rng(int(seed) if world_seed is None else int(world_seed))
    sample_rng = np.random.default_rng(int(seed))
    easy_axis = world_rng.normal(size=2)
    easy_axis /= np.linalg.norm(easy_axis)
    secondary_axis = np.array([-easy_axis[1], easy_axis[0]])
    easy_jitter = world_rng.uniform(0.9, 1.1)
    overlap_jitter = world_rng.uniform(0.9, 1.1)
    secondary_sign = world_rng.choice((-1.0, 1.0))
    means = {'positive_easy': config.easy_separation * easy_jitter * easy_axis - secondary_sign * config.easy_secondary_conflict * secondary_axis, 'negative_easy': -config.easy_separation * easy_jitter * easy_axis + secondary_sign * config.easy_secondary_conflict * secondary_axis, 'positive_overlap': config.overlap_location * overlap_jitter * easy_axis + secondary_sign * config.secondary_separation * secondary_axis, 'negative_overlap': config.overlap_location * overlap_jitter * easy_axis - secondary_sign * config.secondary_separation * secondary_axis}
    rotation = random_orthogonal(world_rng, config.n_features)
    splits = {}
    for name, size in (('train', config.n_train), ('validation', config.n_validation), ('test', config.n_test)):
        values = draw_split(sample_rng, config, means, rotation, size)
        splits[name] = values
    train_x = splits['train'][0]
    train_mean = train_x.mean(axis=0, dtype=np.float64)
    train_sd = train_x.std(axis=0, dtype=np.float64)
    train_sd = np.where(train_sd < 1e-06, 1.0, train_sd)

    def standardize(values: np.ndarray) -> np.ndarray:
        return ((values - train_mean) / train_sd).astype(np.float32)
    for name in splits:
        values = splits[name]
        splits[name] = (standardize(values[0]), values[1], values[2], values[3])
    resolved_initialization_seed = int(seed) + 700001 if initialization_seed is None else int(initialization_seed)
    init_rng = np.random.default_rng(resolved_initialization_seed)
    bound = 1.0 / math.sqrt(config.n_features)
    payload: dict[str, np.ndarray | dict] = {'initial_weight': init_rng.uniform(-bound, bound, size=config.n_features).astype(np.float32), 'initial_bias': np.asarray(init_rng.uniform(-bound, bound), dtype=np.float32), 'config': asdict(config), 'metadata': {'world_seed': int(seed) if world_seed is None else int(world_seed), 'data_seed': int(seed), 'initialization_seed': resolved_initialization_seed}}
    for name, (x, y, probability, overlap) in splits.items():
        payload[f'{name}_x'] = x
        payload[f'{name}_y'] = y
        payload[f'{name}_probability'] = probability
        payload[f'{name}_overlap'] = overlap
    return payload

def save_dataset(path: Path, payload: dict[str, np.ndarray | dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in payload.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(path, **arrays)
    path.with_suffix('.json').write_text(json.dumps({'config': payload['config'], 'metadata': payload.get('metadata', {})}, indent=2), encoding='utf-8')
