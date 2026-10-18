from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BORO_SOURCE_DIR = PROJECT_ROOT / 'src/boro/ade'
from ..ade.loss import BOROLoss
from .metrics import average_precision, evaluate_at_threshold, frontier_band_mask, linear_scores, operating_point, pairwise_accuracy, rank_percentiles, seed_everything, write_json
BORO_METHODS = ('boro_full',)
ALL_METHODS = BORO_METHODS
DEFAULT_TARGETS = (0.95,)

class LinearScoreHead(nn.Module):

    def __init__(self, input_dimension: int):
        super().__init__()
        self.linear = nn.Linear(input_dimension, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(1)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def set_torch_seed(seed: int) -> None:
    seed_everything(seed)
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}

def make_model(input_dimension: int, initial_weight: np.ndarray, initial_bias: np.ndarray) -> LinearScoreHead:
    model = LinearScoreHead(input_dimension)
    with torch.no_grad():
        model.linear.weight.copy_(torch.from_numpy(initial_weight.reshape(1, -1)))
        model.linear.bias.copy_(torch.from_numpy(initial_bias.reshape(-1)))
    return model

def make_loss(method: str, target_precision: float | None) -> tuple[Any, dict[str, Any]]:
    if method not in BORO_METHODS:
        raise ValueError('Expected BORO')
    if target_precision is None:
        raise ValueError(f'{method} requires target_precision.')
    module = BOROLoss(target_precision=float(target_precision), initial_margin=0.2, positive_pair_weight=1.0, negative_pair_weight=1.0, current_pair_weight=0.2, history_pair_weight=1.25, scale_mode='boundary_band', upper_temperature=1.0, lower_temperature=1.0, lower_band=0.35, min_pos_weight=0.4, far_below_suppression=0.05, boundary_start_epoch=3, threshold_momentum=0.5, max_prev_positives=5000, max_prev_negatives=5000, use_negative_scale=False, current_hard_negatives=16, soft_rank_temperature=0.6, boundary_negative_weight=0.5, boundary_gap_weight=0.15)
    return (module, {'project_method': 'boro.ade.loss.BOROLoss', 'source_file': str(BORO_SOURCE_DIR / 'loss.py'), 'source_sha256': sha256_file(BORO_SOURCE_DIR / 'loss.py'), 'frozen_config_source': 'experiments/ade/config.json', 'frozen_config_name': 'boundary_tight_p99', 'target_precision': float(target_precision), 'current_pair_weight': 0.2, 'history_pair_weight': 1.25, 'current_hard_negatives': 16, 'soft_rank_temperature': 0.6, 'boundary_negative_weight': 0.5, 'boundary_gap_weight': 0.15, 'lower_band': 0.35, 'min_pos_weight': 0.4, 'far_below_suppression': 0.05, 'max_prev_positives': 5000, 'max_prev_negatives': 5000, 'intervened_field': 'target_precision only'})

def score_numpy(model: nn.Module, features: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(features).detach().cpu().numpy().astype(np.float64)

def metric_bundle(labels: np.ndarray, scores: np.ndarray, targets: list[float]) -> dict[str, Any]:
    payload: dict[str, Any] = {'average_precision': average_precision(labels, scores)}
    for target in targets:
        suffix = f'p{int(round(target * 100))}'
        payload[f'r_at_{suffix}'] = operating_point(labels, scores, target).recall
        point = operating_point(labels, scores, target)
        payload[f'precision_at_{suffix}'] = point.precision
        payload[f'accepted_at_{suffix}'] = point.accepted
        payload[f'frontier_percentile_{suffix}'] = point.frontier_percentile
    return payload

def validation_transfer_bundle(validation_labels: np.ndarray, validation_scores: np.ndarray, test_labels: np.ndarray, test_scores: np.ndarray, targets: list[float]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for target in targets:
        suffix = f'p{int(round(target * 100))}'
        validation_point = operating_point(validation_labels, validation_scores, target)
        test_transfer = evaluate_at_threshold(test_labels, test_scores, validation_point.threshold)
        payload[f'validation_threshold_{suffix}'] = validation_point.threshold
        for key, value in test_transfer.items():
            payload[f'test_transfer_{key}_{suffix}'] = value
    return payload

def initialize_boro_history(loss_module: BOROLoss, model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
    with torch.no_grad():
        initial_scores = model(train_x)
    loss_module.update_history(initial_scores, train_y)

def make_train_loader(train_x: torch.Tensor, train_y: torch.Tensor, batch_size: int) -> DataLoader:
    return DataLoader(TensorDataset(train_x, train_y), batch_size=int(batch_size), shuffle=True, drop_last=False)

def compute_initial_mechanism(method: str, model: LinearScoreHead, loss_module: BOROLoss, train_x: torch.Tensor, train_y: torch.Tensor, validation_x: torch.Tensor, validation_y: np.ndarray, target_precision: float, normalized_step_size: float) -> dict[str, Any]:
    model.train()
    scores = model(train_x)
    scores.retain_grad()
    loss = loss_module(scores, train_y)
    score_gradient = torch.autograd.grad(loss, scores, retain_graph=True)[0]
    parameter_gradient = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=False, allow_unused=False)
    scores_np = scores.detach().cpu().numpy()
    labels_np = train_y.detach().cpu().numpy().astype(int)
    gradient_np = score_gradient.detach().cpu().numpy().astype(np.float64)
    score_ranks = rank_percentiles(scores_np)
    frontier = operating_point(labels_np, scores_np, target_precision)
    band = frontier_band_mask(score_ranks, frontier.frontier_percentile, width=0.1)
    gradient_mass = np.abs(gradient_np)
    gradient_mass /= max(float(gradient_mass.sum()), 1e-12)
    signed_drive = (2 * labels_np - 1) * -gradient_np
    band_mass = max(float(gradient_mass[band].sum()), 1e-12)
    correct_sign_mass = float(gradient_mass[band & (signed_drive > 0)].sum() / band_mass)
    normalized_signed_drive = float(signed_drive[band].sum() / max(float(np.abs(gradient_np[band]).sum()), 1e-12))
    gradient_centroid = float((gradient_mass * score_ranks).sum())
    gradient_vector = torch.cat([gradient.detach().reshape(-1) for gradient in parameter_gradient])
    raw_norm = float(torch.linalg.vector_norm(gradient_vector).item())
    scale = normalized_step_size / max(raw_norm, 1e-12)
    one_step_model = copy.deepcopy(model)
    with torch.no_grad():
        for parameter, gradient in zip(one_step_model.parameters(), parameter_gradient):
            parameter.add_(gradient, alpha=-scale)
    validation_initial = score_numpy(model, validation_x)
    validation_after = score_numpy(one_step_model, validation_x)
    validation_frontier = operating_point(validation_y, validation_initial, target_precision)
    initial_rank = rank_percentiles(validation_initial)
    after_rank = rank_percentiles(validation_after)
    validation_band = frontier_band_mask(initial_rank, validation_frontier.frontier_percentile, width=0.1)
    positive_mask = validation_band & (validation_y.astype(int) == 1)
    negative_mask = validation_band & (validation_y.astype(int) == 0)
    positive_improvement = float(np.median(initial_rank[positive_mask] - after_rank[positive_mask])) if positive_mask.any() else float('nan')
    negative_improvement = float(np.median(after_rank[negative_mask] - initial_rank[negative_mask])) if negative_mask.any() else float('nan')
    pairwise_before = pairwise_accuracy(validation_y, validation_initial, validation_band)
    pairwise_after = pairwise_accuracy(validation_y, validation_after, validation_band)
    return {'method': method, 'target_precision': float(target_precision), 'loss': float(loss.detach().cpu().item()), 'train_frontier_percentile': frontier.frontier_percentile, 'gradient_centroid': gradient_centroid, 'centroid_frontier_distance': gradient_centroid - frontier.frontier_percentile, 'band_gradient_mass': float(gradient_mass[band].sum()), 'band_correct_sign_mass_fraction_q': correct_sign_mass, 'band_normalized_signed_drive_s': normalized_signed_drive, 'raw_parameter_gradient_norm': raw_norm, 'normalized_parameter_step_norm': normalized_step_size, 'validation_frontier_percentile': validation_frontier.frontier_percentile, 'validation_positive_median_rank_improvement': positive_improvement, 'validation_negative_median_rank_improvement': negative_improvement, 'validation_pairwise_accuracy_before': pairwise_before, 'validation_pairwise_accuracy_after': pairwise_after, 'validation_pairwise_accuracy_gain': pairwise_after - pairwise_before, 'validation_band_size': int(validation_band.sum()), 'validation_band_positives': int(positive_mask.sum()), 'validation_band_negatives': int(negative_mask.sum())}

def train_one(method: str, seed: int, dataset_path: Path, output_dir: Path, target_precision: float | None, evaluation_targets: list[float], learning_rate: float, steps: int, eval_every: int, batch_size: int, normalized_step_size: float, force: bool) -> dict[str, Any]:
    if method not in BORO_METHODS:
        raise ValueError('Expected BORO')
    target_token = 'agnostic' if target_precision is None else f'p{int(round(target_precision * 100))}'
    run_dir = output_dir / ('boro' if method in BORO_METHODS else method) / f'seed_{seed}'
    result_path = run_dir / 'result.json'
    if result_path.exists() and (not force):
        return json.loads(result_path.read_text(encoding='utf-8'))
    set_torch_seed(seed)
    dataset = load_dataset(dataset_path)
    train_x = torch.from_numpy(dataset['train_x']).float()
    train_y = torch.from_numpy(dataset['train_y']).float()
    validation_x = torch.from_numpy(dataset['validation_x']).float()
    test_x = torch.from_numpy(dataset['test_x']).float()
    validation_y = dataset['validation_y'].astype(int)
    test_y = dataset['test_y'].astype(int)
    model = make_model(dataset['train_x'].shape[1], dataset['initial_weight'], dataset['initial_bias'])
    loss_fn, provenance = make_loss(method, target_precision)
    if isinstance(loss_fn, BOROLoss):
        mechanism_loss = copy.deepcopy(loss_fn)
        initialize_boro_history(mechanism_loss, model, train_x, train_y)
        mechanism = compute_initial_mechanism(method=method, model=model, loss_module=mechanism_loss, train_x=train_x, train_y=train_y, validation_x=validation_x, validation_y=validation_y, target_precision=float(target_precision), normalized_step_size=normalized_step_size)
        loss_fn.initialize_history(train_y, seed=0)
    else:
        mechanism = None
    optimizer = torch.optim.AdamW([{'params': list(model.parameters()), 'lr': float(learning_rate), 'weight_decay': 0.0001}, {'params': [loss_fn.margin], 'lr': 1e-05, 'weight_decay': 0.0}])
    optimizer_name = 'torch.optim.AdamW'
    weight_decay = 0.0001
    initial_validation_scores = score_numpy(model, validation_x)
    initial_test_scores = score_numpy(model, test_x)
    trajectory: list[dict[str, Any]] = []
    validation_score_steps: list[int] = [0]
    validation_score_trajectory: list[np.ndarray] = [initial_validation_scores.astype(np.float32)]
    train_loader = make_train_loader(train_x, train_y, batch_size)
    for step in range(1, steps + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            train_scores = model(batch_x)
            loss = loss_fn(train_scores, batch_y)
            loss.backward()
            optimizer.step()
            if isinstance(loss_fn, BOROLoss):
                loss_fn.clamp_margin_()
            total_loss += float(loss.detach().cpu().item()) * len(batch_y)
            seen += len(batch_y)
        if isinstance(loss_fn, BOROLoss):
            with torch.no_grad():
                updated_scores = model(train_x)
            loss_fn.update_history(updated_scores, train_y)
        epoch_loss = total_loss / max(seen, 1)
        if step == 1 or step % eval_every == 0 or step == steps:
            validation_scores = score_numpy(model, validation_x)
            validation_metrics = metric_bundle(validation_y, validation_scores, evaluation_targets)
            row = {'step': step, 'loss': epoch_loss, **{f'validation_{key}': value for key, value in validation_metrics.items()}}
            if isinstance(loss_fn, BOROLoss):
                row.update({'boro_tau': loss_fn.tau_p, 'boro_raw_tau': loss_fn.raw_tau_p, 'boro_margin': float(loss_fn.margin.detach().item())})
            trajectory.append(row)
            validation_score_steps.append(int(step))
            validation_score_trajectory.append(validation_scores.astype(np.float32))
            print(f'[torch] method={method} seed={seed} target={target_token} lr={learning_rate:g} step={step}/{steps} loss={epoch_loss:.6f}', flush=True)
    final_validation_scores = score_numpy(model, validation_x)
    final_test_scores = score_numpy(model, test_x)
    validation_metrics = metric_bundle(validation_y, final_validation_scores, evaluation_targets)
    test_metrics = metric_bundle(test_y, final_test_scores, evaluation_targets)
    transfer_metrics = validation_transfer_bundle(validation_y, final_validation_scores, test_y, final_test_scores, evaluation_targets)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'loss_state_dict': loss_fn.state_dict() if isinstance(loss_fn, nn.Module) else None}, run_dir / 'final_state.pt')
    np.savez_compressed(run_dir / 'scores.npz', validation_initial=initial_validation_scores, validation_final=final_validation_scores, test_initial=initial_test_scores, test_final=final_test_scores, validation_score_steps=np.asarray(validation_score_steps, dtype=np.int32), validation_score_trajectory=np.stack(validation_score_trajectory, axis=0))
    with (run_dir / 'trajectory.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory[0]))
        writer.writeheader()
        writer.writerows(trajectory)
    result = {'method': method, 'seed': int(seed), 'target_precision': target_precision, 'evaluation_targets': evaluation_targets, 'learning_rate': float(learning_rate), 'steps': int(steps), 'unit_of_step': 'one complete training-data pass', 'batch_size': int(batch_size), 'eval_every': int(eval_every), 'optimizer': optimizer_name, 'weight_decay': weight_decay, 'fixed_horizon_primary': True, 'dataset_path': str(dataset_path), 'data_sha256': sha256_file(dataset_path), 'provenance': provenance, 'initial_validation_metrics': metric_bundle(validation_y, initial_validation_scores, evaluation_targets), 'initial_test_metrics': metric_bundle(test_y, initial_test_scores, evaluation_targets), 'final_validation_metrics': validation_metrics, 'final_test_metrics': test_metrics, 'validation_threshold_transfer': transfer_metrics, 'initial_mechanism': mechanism, 'run_dir': str(run_dir)}
    write_json(result_path, result)
    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data/processed/controlled')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'results/controlled')
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--methods', choices=ALL_METHODS, nargs='+', default=list(BORO_METHODS))
    parser.add_argument('--targets', type=float, nargs='+', default=list(DEFAULT_TARGETS))
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--normalized-step-size', type=float, default=0.01)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--smoke-data', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    data_dir = args.data_dir / 'smoke' if args.smoke_data else args.data_dir
    for seed in args.seeds:
        dataset_path = data_dir / f'seed_{seed}.npz'
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        for method in args.methods:
            for target in [float(value) for value in args.targets]:
                train_one(method=method, seed=int(seed), dataset_path=dataset_path, output_dir=args.output_dir, target_precision=target, evaluation_targets=[float(value) for value in args.targets], learning_rate=float(args.learning_rate if args.learning_rate is not None else 0.001), steps=int(args.steps if args.steps is not None else 45), eval_every=int(args.eval_every), batch_size=int(args.batch_size), normalized_step_size=float(args.normalized_step_size), force=bool(args.force))
if __name__ == '__main__':
    main()
