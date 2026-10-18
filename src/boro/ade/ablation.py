from __future__ import annotations
import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .loss import BOROLoss
from .train import DEFAULT_SEEDS, PRIMARY_METRIC, Config, SingleScoreHead, all_metrics, build_clean_protocol, fit_dense_encoder, fit_sparse_encoder, flatten_result, make_loader, predict_scores, sanitize_json, save_predictions, save_protocol_splits, selection_score, set_seed, transform_dense, transform_sparse
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data/raw/ade'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'results/ade/ablation'
DEFAULT_VARIANTS = ('no_history', 'no_current_boundary', 'no_boundary_local_selection', 'uniform_weights', 'no_lower_bound', 'hard_example_matched')
VARIANT_DISPLAY = {'full_boro': 'Full BORO', 'no_history': 'No history', 'no_current_boundary': 'No current boundary term', 'no_boundary_local_selection': 'No boundary-local selection', 'uniform_weights': 'Uniform weights', 'no_lower_bound': 'No lower bound', 'hard_example_matched': 'Hard-example matched'}
VARIANT_DESCRIPTIONS = {'full_boro': 'Original boundary_tight_p99 BORO configuration.', 'no_history': 'Disables the cross-epoch history pair term and keeps the current boundary-rank term.', 'no_current_boundary': 'Keeps the history ranking term and disables the current boundary-rank term.', 'no_boundary_local_selection': 'Replaces boundary-local selection with global hard-example pair mining.', 'uniform_weights': 'Uses equal pair weights instead of adaptive boundary weights.', 'no_lower_bound': 'Removes the lower bound on positive boundary weights.', 'hard_example_matched': 'Uses global hard-example mining with per-batch loss scale matched to Full BORO.'}

class UniformWeightBOROLoss(BOROLoss):

    def _adaptive_scale(self, current_pos: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(current_pos)

class NoLowerBoundBOROLoss(BOROLoss):

    def __init__(self, *args, **kwargs):
        kwargs['min_pos_weight'] = max(float(kwargs.get('min_pos_weight', 1e-08)), 1e-08)
        super().__init__(*args, **kwargs)

    def _adaptive_scale(self, current_pos: torch.Tensor) -> torch.Tensor:
        if current_pos.numel() == 0 or self.tau_p is None:
            return torch.ones_like(current_pos)
        tau = torch.tensor(float(self.tau_p), device=current_pos.device, dtype=current_pos.dtype)
        scale_scores = current_pos.detach()
        if self.scale_mode == 'operating_exp' or self.history_update_count < self.boundary_start_epoch:
            excess = torch.clamp(scale_scores - tau, min=0.0)
            return torch.exp(-excess / self.upper_temperature)
        if self.scale_mode != 'boundary_band':
            raise ValueError(f'Unsupported scale_mode: {self.scale_mode}')
        above_decay = torch.exp(-torch.clamp(scale_scores - tau, min=0.0) / self.upper_temperature)
        far_below_gate = torch.sigmoid((tau - scale_scores - self.lower_band) / self.lower_temperature)
        below_factor = 1.0 - self.far_below_suppression * far_below_gate
        return torch.clamp(below_factor * above_decay, min=0.0, max=1.0)

class HardExamplePairLoss(nn.Module):

    def __init__(self, initial_margin: float=0.2, max_hard_positives: int=64, max_hard_negatives: int=64, reduction_floor: float=1e-09):
        super().__init__()
        self.margin = nn.Parameter(torch.tensor(float(initial_margin), dtype=torch.float32))
        self.max_hard_positives = int(max_hard_positives)
        self.max_hard_negatives = int(max_hard_negatives)
        self.reduction_floor = float(reduction_floor)

    def initialize_history(self, labels: torch.Tensor, seed: int=0) -> None:
        return None

    def update_history(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        return float('nan')

    def clamp_margin_(self) -> None:
        with torch.no_grad():
            self.margin.clamp_(min=0.0)

    def _select_hard(self, scores: torch.Tensor, largest: bool, cap: int) -> torch.Tensor:
        if scores.numel() == 0 or cap <= 0 or scores.numel() <= cap:
            return scores
        return torch.topk(scores, k=cap, largest=largest).values

    def forward(self, current_scores: torch.Tensor, current_labels: torch.Tensor) -> torch.Tensor:
        labels = current_labels.float()
        pos = current_scores[labels >= 0.5]
        neg = current_scores[labels < 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            return current_scores.sum() * self.reduction_floor
        hard_pos = self._select_hard(pos, largest=False, cap=self.max_hard_positives)
        hard_neg = self._select_hard(neg, largest=True, cap=self.max_hard_negatives)
        diff = hard_neg.view(1, -1) - hard_pos.view(-1, 1) + torch.clamp(self.margin, min=0.0)
        return torch.clamp(diff, min=0.0).pow(2).mean()

class LossScaleMatchedHardExampleLoss(HardExamplePairLoss):

    def __init__(self, reference_loss: BOROLoss, min_scale: float=0.25, max_scale: float=4.0, **kwargs):
        super().__init__(**kwargs)
        self.reference_loss = reference_loss
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def initialize_history(self, labels: torch.Tensor, seed: int=0) -> None:
        self.reference_loss.initialize_history(labels, seed=seed)

    def update_history(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        return self.reference_loss.update_history(scores, labels)

    def clamp_margin_(self) -> None:
        super().clamp_margin_()
        self.reference_loss.clamp_margin_()

    def forward(self, current_scores: torch.Tensor, current_labels: torch.Tensor) -> torch.Tensor:
        hard_loss = super().forward(current_scores, current_labels)
        with torch.no_grad():
            reference = self.reference_loss(current_scores.detach(), current_labels)
            scale = reference / torch.clamp(hard_loss.detach(), min=1e-12)
            scale = torch.clamp(scale, min=self.min_scale, max=self.max_scale)
        return hard_loss * scale

def make_boro_loss(cfg: Config, loss_cls: type[BOROLoss]=BOROLoss, **overrides) -> BOROLoss:
    params = {'target_precision': cfg.loss_target_precision, 'initial_margin': 0.2, 'positive_pair_weight': 1.0, 'negative_pair_weight': 1.0, 'history_pair_weight': cfg.boro_history_pair_weight, 'current_pair_weight': cfg.boro_current_pair_weight, 'scale_mode': 'boundary_band', 'upper_temperature': 1.0, 'lower_temperature': 1.0, 'lower_band': cfg.boro_lower_band, 'min_pos_weight': cfg.boro_min_pos_weight, 'far_below_suppression': cfg.boro_far_below_suppression, 'boundary_start_epoch': 3, 'threshold_momentum': 0.5, 'max_prev_positives': 5000, 'max_prev_negatives': 5000, 'use_negative_scale': False, 'current_hard_negatives': cfg.boro_current_hard_negatives, 'soft_rank_temperature': cfg.boro_soft_rank_temperature, 'boundary_negative_weight': cfg.boro_boundary_negative_weight, 'boundary_gap_weight': cfg.boro_boundary_gap_weight}
    params.update(overrides)
    return loss_cls(**params)

def build_variant_loss(variant: str, base_cfg: Config):
    if variant == 'full_boro':
        return (make_boro_loss(base_cfg), base_cfg, {})
    if variant == 'no_history':
        cfg = replace(base_cfg, boro_history_pair_weight=0.0)
        return (make_boro_loss(cfg), cfg, {'disabled': 'history_pair_term'})
    if variant == 'no_current_boundary':
        cfg = replace(base_cfg, boro_current_pair_weight=0.0)
        return (make_boro_loss(cfg), cfg, {'disabled': 'current_boundary_rank_term'})
    if variant == 'uniform_weights':
        cfg = replace(base_cfg, boro_boundary_negative_weight=0.0)
        return (make_boro_loss(cfg, UniformWeightBOROLoss), cfg, {'disabled': 'adaptive_positive_and_negative_boundary_weights'})
    if variant == 'no_lower_bound':
        cfg = replace(base_cfg, boro_min_pos_weight=0.0)
        return (make_boro_loss(cfg, NoLowerBoundBOROLoss, min_pos_weight=1e-08), cfg, {'effective_min_pos_weight': 0.0})
    if variant == 'no_boundary_local_selection':
        loss = HardExamplePairLoss(max_hard_positives=64, max_hard_negatives=64)
        return (loss, base_cfg, {'replacement': 'global_hard_positive_negative_pair_mining'})
    if variant == 'hard_example_matched':
        reference = make_boro_loss(base_cfg)
        loss = LossScaleMatchedHardExampleLoss(reference_loss=reference, max_hard_positives=64, max_hard_negatives=64, min_scale=0.25, max_scale=4.0)
        return (loss, base_cfg, {'replacement': 'global_hard_example_mining', 'scale_match': 'per_batch_full_boro_loss_clipped_0.25_4.0'})
    raise ValueError(f'Unknown variant: {variant}')

def optimizer_for(model: nn.Module, loss_fn: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    groups = [{'params': list(model.parameters()), 'lr': cfg.learning_rate, 'weight_decay': cfg.weight_decay}]
    margin = getattr(loss_fn, 'margin', None)
    if isinstance(margin, nn.Parameter):
        groups.append({'params': [margin], 'lr': 1e-05, 'weight_decay': 0.0})
    return torch.optim.AdamW(groups)

def train_variant(x_train, y_train, x_val, y_val, variant: str, cfg: Config, output_dir: Path):
    model = SingleScoreHead(x_train.shape[1])
    loss_fn, effective_cfg, variant_params = build_variant_loss(variant, cfg)
    if hasattr(loss_fn, 'initialize_history'):
        loss_fn.initialize_history(torch.from_numpy(y_train), seed=0)
    optimizer = optimizer_for(model, loss_fn, effective_cfg)
    loader = make_loader(x_train, y_train, effective_cfg.batch_size, True)
    best_score = -np.inf
    best_payload = None
    patience = effective_cfg.patience
    rows = []
    for epoch in range(1, effective_cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            if hasattr(loss_fn, 'clamp_margin_'):
                loss_fn.clamp_margin_()
            total_loss += float(loss.item()) * xb.shape[0]
            seen += xb.shape[0]
        train_scores = predict_scores(model, x_train, effective_cfg.batch_size)
        if hasattr(loss_fn, 'update_history'):
            loss_fn.update_history(torch.from_numpy(train_scores), torch.from_numpy(y_train))
        val_scores = predict_scores(model, x_val, effective_cfg.batch_size)
        val_metrics = all_metrics(y_val.astype(int), val_scores)
        score = selection_score(y_val.astype(int), val_scores, effective_cfg.selection_precision)
        rows.append({'epoch': epoch, 'variant': variant, 'train_loss': total_loss / max(seen, 1), 'selection_score': score, **{f'val_{k}': v for k, v in val_metrics.items() if not k.startswith('threshold')}})
        if score > best_score:
            best_score = score
            patience = effective_cfg.patience
            best_payload = {'model_state_dict': copy.deepcopy(model.state_dict()), 'loss_state_dict': copy.deepcopy(loss_fn.state_dict()) if hasattr(loss_fn, 'state_dict') else {}, 'input_dim': int(x_train.shape[1]), 'epoch': int(epoch), 'selection_score': float(score), 'variant': variant, 'variant_display': VARIANT_DISPLAY[variant], 'variant_params': variant_params}
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_payload is None:
        raise RuntimeError(f'No checkpoint selected for variant {variant}')
    best_model = SingleScoreHead(best_payload['input_dim'])
    best_model.load_state_dict(best_payload['model_state_dict'])
    pd.DataFrame(rows).to_csv(output_dir / 'train_log.csv', index=False)
    return (best_model, best_payload, effective_cfg, variant_params)

def config_to_dict(cfg: Config) -> dict:
    return {'word_max_features': int(cfg.word_max_features), 'char_max_features': int(cfg.char_max_features), 'svd_components': int(cfg.svd_components), 'batch_size': int(cfg.batch_size), 'epochs': int(cfg.epochs), 'patience': int(cfg.patience), 'learning_rate': float(cfg.learning_rate), 'weight_decay': float(cfg.weight_decay), 'selection_precision': float(cfg.selection_precision), 'loss_target_precision': float(cfg.loss_target_precision), 'boro_history_pair_weight': float(cfg.boro_history_pair_weight), 'boro_current_pair_weight': float(cfg.boro_current_pair_weight), 'boro_current_hard_negatives': int(cfg.boro_current_hard_negatives), 'boro_soft_rank_temperature': float(cfg.boro_soft_rank_temperature), 'boro_boundary_negative_weight': float(cfg.boro_boundary_negative_weight), 'boro_boundary_gap_weight': float(cfg.boro_boundary_gap_weight), 'boro_lower_band': float(cfg.boro_lower_band), 'boro_min_pos_weight': float(cfg.boro_min_pos_weight), 'boro_far_below_suppression': float(cfg.boro_far_below_suppression)}

def run_one_variant(seed: int, variant: str, cfg: Config, data_dir: Path, output_dir: Path, force: bool) -> dict:
    run_dir = output_dir / variant / f'seed_{seed}'
    metrics_path = run_dir / 'metrics.json'
    if metrics_path.exists() and (not force):
        return json.loads(metrics_path.read_text(encoding='utf-8'))
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_clean_protocol(data_dir, seed)
    save_protocol_splits(protocol, output_dir / 'splits' / f'seed_{seed}')
    train_df, val_df, test_df = (protocol['train'], protocol['val'], protocol['test'])
    sparse_encoder, x_train_sparse = fit_sparse_encoder(train_df['text'].tolist(), cfg)
    x_val_sparse = transform_sparse(val_df['text'].tolist(), sparse_encoder)
    x_test_sparse = transform_sparse(test_df['text'].tolist(), sparse_encoder)
    dense_encoder, x_train = fit_dense_encoder(x_train_sparse, cfg, seed)
    x_val = transform_dense(x_val_sparse, dense_encoder)
    x_test = transform_dense(x_test_sparse, dense_encoder)
    y_train = train_df['label'].to_numpy(dtype=np.float32)
    y_val = val_df['label'].to_numpy(dtype=np.float32)
    y_test = test_df['label'].to_numpy(dtype=int)
    model, payload, effective_cfg, variant_params = train_variant(x_train, y_train, x_val, y_val, variant, cfg, run_dir)
    val_scores = predict_scores(model, x_val, effective_cfg.batch_size)
    test_scores = predict_scores(model, x_test, effective_cfg.batch_size)
    val_metrics = all_metrics(y_val.astype(int), val_scores)
    test_metrics = all_metrics(y_test, test_scores)
    save_predictions(val_df, val_scores, run_dir / 'val_predictions.csv')
    save_predictions(test_df, test_scores, run_dir / 'test_predictions.csv')
    torch.save(payload, run_dir / 'model.pt')
    joblib.dump({'sparse_encoder': sparse_encoder, 'dense_encoder': dense_encoder}, run_dir / 'encoder.joblib')
    result = {'protocol': 'ade_p0_component_ablation', 'method': VARIANT_DISPLAY[variant], 'variant': variant, 'variant_description': VARIANT_DESCRIPTIONS[variant], 'variant_params': variant_params, 'seed': int(seed), 'best_epoch': int(payload['epoch']), 'selection_score': float(selection_score(y_val.astype(int), val_scores, effective_cfg.selection_precision)), 'config': config_to_dict(effective_cfg), 'data': {'raw_train_size': protocol['raw_train_size'], 'raw_test_size': protocol['raw_test_size'], 'dedup_train_size': protocol['dedup_train_size'], 'dedup_test_size': protocol['dedup_test_size'], 'removed_test_overlap': protocol['removed_test_overlap'], 'train_size': int(len(train_df)), 'val_size': int(len(val_df)), 'test_size': int(len(test_df)), 'train_positive': int(train_df['label'].sum()), 'val_positive': int(val_df['label'].sum()), 'test_positive': int(test_df['label'].sum())}, 'val_metrics': val_metrics, 'test_metrics': test_metrics, 'run_dir': str(run_dir)}
    metrics_path.write_text(json.dumps(sanitize_json(result), indent=2, allow_nan=False), encoding='utf-8')
    return result

def summarize_ablation(rows: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    metric_cols = [c for c in rows.columns if c.startswith('test_') and rows[c].dtype.kind in 'fiu']
    order = {VARIANT_DISPLAY[v]: i for i, v in enumerate(variants)}
    out = []
    for method, group in rows.groupby('method', sort=False):
        row = {'method': method, 'n_seeds': int(group['seed'].nunique())}
        for col in metric_cols:
            row[f'{col}_mean'] = float(group[col].mean())
            row[f'{col}_std'] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        out.append(row)
    summary = pd.DataFrame(out)
    summary['_order'] = summary['method'].map(order).fillna(999)
    return summary.sort_values('_order').drop(columns=['_order'])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--variants', nargs='*', default=list(DEFAULT_VARIANTS), choices=sorted(VARIANT_DISPLAY))
    parser.add_argument('--seeds', type=int, nargs='*', default=[20260417, 20260418, 20260419, 20260420, 20260421])
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--word-max-features', type=int, default=Config.word_max_features)
    parser.add_argument('--char-max-features', type=int, default=Config.char_max_features)
    parser.add_argument('--svd-components', type=int, default=Config.svd_components)
    parser.add_argument('--batch-size', type=int, default=Config.batch_size)
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--patience', type=int, default=Config.patience)
    parser.add_argument('--learning-rate', type=float, default=Config.learning_rate)
    parser.add_argument('--weight-decay', type=float, default=Config.weight_decay)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = Config(word_max_features=args.word_max_features, char_max_features=args.char_max_features, svd_components=args.svd_components, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = list(dict.fromkeys(args.variants))
    seeds = [int(s) for s in args.seeds]
    results = []
    for variant in variants:
        for seed in seeds:
            print(f'[ade-ablation] variant={variant} seed={seed}', flush=True)
            results.append(run_one_variant(seed, variant, cfg, args.data_dir, args.output_dir, args.force))
    rows = pd.DataFrame([flatten_result(r) for r in results])
    rows.insert(2, 'variant', [r['variant'] for r in results])
    rows.to_csv(args.output_dir / 'all_seed_results.csv', index=False)
    summary = summarize_ablation(rows, variants)
    summary.to_csv(args.output_dir / 'summary.csv', index=False)
    primary_col = f'test_{PRIMARY_METRIC}_mean'
    print(json.dumps({'output_dir': str(args.output_dir), 'variants': variants, 'seeds': seeds, 'primary_metric': primary_col, 'best_primary_value': float(summary[primary_col].max()) if primary_col in summary else None}, indent=2))
if __name__ == '__main__':
    main()
