from __future__ import annotations
import argparse
import copy
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader, TensorDataset
from .loss import BOROLoss
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data/raw/ade'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'results/ade'
DEFAULT_SEEDS = (20260417, 20260418, 20260419, 20260420, 20260421, 20260919, 20260920, 20260921, 20260922, 20260923)
PRECISION_TARGETS = (0.9, 0.95, 0.98, 0.99)
FPR_TARGETS = (0.01, 0.05, 0.1)
PRIMARY_METRIC = 'r_at_p95'
PROTOCOL_NAME = 'boro'
METHOD_NAME = 'BORO'
OUTPUT_METHOD_DIR = 'boro'
LOSS_NAME = 'boro'

@dataclass(frozen=True)
class Config:
    word_max_features: int = 80000
    char_max_features: int = 60000
    svd_components: int = 512
    batch_size: int = 256
    epochs: int = 60
    patience: int = 12
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    selection_precision: float = 0.95
    loss_target_precision: float = 0.99
    boro_history_pair_weight: float = 1.25
    boro_current_pair_weight: float = 0.2
    boro_current_hard_negatives: int = 16
    boro_soft_rank_temperature: float = 0.6
    boro_boundary_negative_weight: float = 0.5
    boro_boundary_gap_weight: float = 0.15
    boro_lower_band: float = 0.35
    boro_min_pos_weight: float = 0.4
    boro_far_below_suppression: float = 0.05

class SingleScoreHead(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.float()).squeeze(-1)

def normalize_text(text: str) -> str:
    return re.sub('\\s+', ' ', str(text).strip()).lower()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)[['text', 'label']].copy()
    df['label'] = df['label'].astype(int)
    df['norm_text'] = df['text'].map(normalize_text)
    return df

def deduplicate_by_text(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    conflicts = df.groupby('norm_text')['label'].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(f'{split_name} has conflicting duplicate normalized texts: {len(conflicts)}')
    return df.sort_values(['norm_text', 'text']).drop_duplicates('norm_text', keep='first').reset_index(drop=True)

def build_clean_protocol(data_dir: Path, seed: int, validation_fraction: float=0.15) -> dict:
    raw_train = read_jsonl(data_dir / 'train.jsonl')
    raw_test = read_jsonl(data_dir / 'test.jsonl')
    full_train = deduplicate_by_text(raw_train, 'raw_train')
    test_unique = deduplicate_by_text(raw_test, 'raw_test')
    clean_test = test_unique[~test_unique['norm_text'].isin(set(full_train['norm_text']))].reset_index(drop=True)
    train_df, val_df = train_test_split(full_train, test_size=validation_fraction, random_state=seed, stratify=full_train['label'])
    return {'raw_train_size': int(len(raw_train)), 'raw_test_size': int(len(raw_test)), 'dedup_train_size': int(len(full_train)), 'dedup_test_size': int(len(test_unique)), 'removed_test_overlap': int(len(test_unique) - len(clean_test)), 'train': train_df.reset_index(drop=True), 'val': val_df.reset_index(drop=True), 'test': clean_test}

def save_protocol_splits(protocol: dict, split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ('train', 'val', 'test'):
        protocol[split_name][['text', 'label', 'norm_text']].to_json(split_dir / f'{split_name}.jsonl', orient='records', lines=True, force_ascii=False)
    metadata = {k: v for k, v in protocol.items() if k not in {'train', 'val', 'test'}}
    (split_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

def fit_sparse_encoder(train_texts: list[str], cfg: Config):
    word_vectorizer = TfidfVectorizer(lowercase=True, strip_accents='unicode', analyzer='word', ngram_range=(1, 2), min_df=2, max_features=cfg.word_max_features, sublinear_tf=True, norm='l2')
    char_vectorizer = TfidfVectorizer(lowercase=True, strip_accents='unicode', analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=cfg.char_max_features, sublinear_tf=True, norm='l2')
    sparse = hstack([word_vectorizer.fit_transform(train_texts), char_vectorizer.fit_transform(train_texts)], format='csr')
    return ({'word_vectorizer': word_vectorizer, 'char_vectorizer': char_vectorizer}, sparse)

def transform_sparse(texts: list[str], encoder: dict):
    return hstack([encoder['word_vectorizer'].transform(texts), encoder['char_vectorizer'].transform(texts)], format='csr')

def fit_dense_encoder(train_sparse, cfg: Config, seed: int):
    svd = TruncatedSVD(n_components=cfg.svd_components, random_state=seed, n_iter=7)
    normalizer = Normalizer(copy=False)
    train_dense = normalizer.fit_transform(svd.fit_transform(train_sparse)).astype(np.float32)
    return ({'svd': svd, 'normalizer': normalizer}, train_dense)

def transform_dense(sparse, dense_encoder: dict) -> np.ndarray:
    return dense_encoder['normalizer'].transform(dense_encoder['svd'].transform(sparse)).astype(np.float32)

def make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool):
    return DataLoader(TensorDataset(torch.from_numpy(features.astype(np.float32)), torch.from_numpy(labels.astype(np.float32))), batch_size=batch_size, shuffle=shuffle)

def predict_scores(model: nn.Module, features: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        zeros = np.zeros(features.shape[0], dtype=np.float32)
        for xb, _ in make_loader(features, zeros, batch_size, False):
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)

def constrained_recall(y_true: np.ndarray, scores: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    out = {}
    for target in PRECISION_TARGETS:
        valid = np.flatnonzero(precision[:-1] >= target)
        key = f'r_at_p{int(target * 100)}'
        if valid.size == 0:
            out[key] = 0.0
            out[f'threshold_p{int(target * 100)}'] = np.nan
            continue
        idx = valid[np.argmax(recall[:-1][valid])]
        out[key] = float(recall[idx])
        out[f'threshold_p{int(target * 100)}'] = float(thresholds[idx])
    return out

def fpr_constrained_recall(y_true: np.ndarray, scores: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    out = {}
    for target in FPR_TARGETS:
        suffix = str(target).replace('.', 'p')
        valid = np.flatnonzero(fpr <= target)
        if valid.size == 0:
            out[f'r_at_fpr{suffix}'] = 0.0
            out[f'threshold_fpr{suffix}'] = np.nan
            continue
        idx = valid[np.argmax(tpr[valid])]
        out[f'r_at_fpr{suffix}'] = float(tpr[idx])
        out[f'threshold_fpr{suffix}'] = float(thresholds[idx])
    return out

def best_f1_stats(y_true: np.ndarray, scores: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    pred = scores >= thresholds[idx]
    return {'best_f1': float(f1_score(y_true, pred, zero_division=0)), 'best_f1_precision': float(precision_score(y_true, pred, zero_division=0)), 'best_f1_recall': float(recall_score(y_true, pred, zero_division=0)), 'best_f1_threshold': float(thresholds[idx])}

def all_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    metrics = {'auroc': float(roc_auc_score(y_true, scores)), 'auprc': float(average_precision_score(y_true, scores))}
    metrics.update(constrained_recall(y_true, scores))
    metrics.update(fpr_constrained_recall(y_true, scores))
    metrics.update(best_f1_stats(y_true, scores))
    return metrics

def sanitize_json(value):
    if isinstance(value, dict):
        return {k: sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_json(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value

def selection_score(y_true: np.ndarray, scores: np.ndarray, target_precision: float=0.95) -> float:
    metrics = all_metrics(y_true, scores)
    return metrics[f'r_at_p{int(target_precision * 100)}'] + 0.01 * metrics['auprc'] + 0.001 * metrics['auroc']

def save_predictions(df: pd.DataFrame, scores: np.ndarray, path: Path) -> None:
    pred = df[['text', 'label', 'norm_text']].copy()
    pred.insert(0, 'sample_id', np.arange(len(pred)))
    pred['score_logit'] = scores
    pred['score_probability'] = 1.0 / (1.0 + np.exp(-scores))
    pred.to_csv(path, index=False)

def make_boro(cfg: Config) -> BOROLoss:
    return BOROLoss(target_precision=cfg.loss_target_precision, initial_margin=0.2, positive_pair_weight=1.0, negative_pair_weight=1.0, history_pair_weight=cfg.boro_history_pair_weight, current_pair_weight=cfg.boro_current_pair_weight, scale_mode='boundary_band', upper_temperature=1.0, lower_temperature=1.0, lower_band=cfg.boro_lower_band, min_pos_weight=cfg.boro_min_pos_weight, far_below_suppression=cfg.boro_far_below_suppression, boundary_start_epoch=3, threshold_momentum=0.5, max_prev_positives=5000, max_prev_negatives=5000, use_negative_scale=False, current_hard_negatives=cfg.boro_current_hard_negatives, soft_rank_temperature=cfg.boro_soft_rank_temperature, boundary_negative_weight=cfg.boro_boundary_negative_weight, boundary_gap_weight=cfg.boro_boundary_gap_weight)

def train_boro(x_train, y_train, x_val, y_val, cfg: Config, output_dir: Path):
    model = SingleScoreHead(x_train.shape[1])
    loss_fn = make_boro(cfg)
    loss_fn.initialize_history(torch.from_numpy(y_train), seed=0)
    optimizer = torch.optim.AdamW([{'params': list(model.parameters()), 'lr': cfg.learning_rate, 'weight_decay': cfg.weight_decay}, {'params': [loss_fn.margin], 'lr': 1e-05, 'weight_decay': 0.0}])
    loader = make_loader(x_train, y_train, cfg.batch_size, True)
    best_score = -np.inf
    best_payload = None
    patience = cfg.patience
    rows = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        stage = LOSS_NAME
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            loss_fn.clamp_margin_()
            total_loss += float(loss.item()) * xb.shape[0]
            seen += xb.shape[0]
        train_scores = predict_scores(model, x_train, cfg.batch_size)
        loss_fn.update_history(torch.from_numpy(train_scores), torch.from_numpy(y_train))
        val_scores = predict_scores(model, x_val, cfg.batch_size)
        val_metrics = all_metrics(y_val.astype(int), val_scores)
        score = selection_score(y_val.astype(int), val_scores, cfg.selection_precision)
        rows.append({'epoch': epoch, 'stage': stage, 'train_loss': total_loss / max(seen, 1), 'selection_score': score, **{f'val_{k}': v for k, v in val_metrics.items() if not k.startswith('threshold')}})
        if score > best_score:
            best_score = score
            patience = cfg.patience
            best_payload = {'model_state_dict': copy.deepcopy(model.state_dict()), 'boro_state_dict': copy.deepcopy(loss_fn.state_dict()), 'input_dim': int(x_train.shape[1]), 'epoch': int(epoch), 'selection_score': float(score), 'loss': LOSS_NAME}
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_payload is None:
        raise RuntimeError('No BORO checkpoint selected')
    best_model = SingleScoreHead(best_payload['input_dim'])
    best_model.load_state_dict(best_payload['model_state_dict'])
    pd.DataFrame(rows).to_csv(output_dir / 'train_log.csv', index=False)
    return (best_model, best_payload)

def run_one(seed: int, cfg: Config, data_dir: Path, output_dir: Path, force: bool) -> dict:
    run_dir = output_dir / OUTPUT_METHOD_DIR / f'seed_{seed}'
    metrics_path = run_dir / 'metrics.json'
    if metrics_path.exists() and (not force):
        return json.loads(metrics_path.read_text())
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
    model, payload = train_boro(x_train, y_train, x_val, y_val, cfg, run_dir)
    val_scores = predict_scores(model, x_val, cfg.batch_size)
    test_scores = predict_scores(model, x_test, cfg.batch_size)
    val_metrics = all_metrics(y_val.astype(int), val_scores)
    test_metrics = all_metrics(y_test, test_scores)
    save_predictions(val_df, val_scores, run_dir / 'val_predictions.csv')
    save_predictions(test_df, test_scores, run_dir / 'test_predictions.csv')
    torch.save(payload, run_dir / 'model.pt')
    joblib.dump({'sparse_encoder': sparse_encoder, 'dense_encoder': dense_encoder}, run_dir / 'encoder.joblib')
    result = {'protocol': PROTOCOL_NAME, 'method': METHOD_NAME, 'loss': LOSS_NAME, 'seed': int(seed), 'best_epoch': int(payload['epoch']), 'selection_score': float(selection_score(y_val.astype(int), val_scores, cfg.selection_precision)), 'config': {'word_max_features': int(cfg.word_max_features), 'char_max_features': int(cfg.char_max_features), 'svd_components': int(cfg.svd_components), 'batch_size': int(cfg.batch_size), 'epochs': int(cfg.epochs), 'patience': int(cfg.patience), 'learning_rate': float(cfg.learning_rate), 'weight_decay': float(cfg.weight_decay), 'selection_precision': float(cfg.selection_precision), 'loss_target_precision': float(cfg.loss_target_precision), 'boro_history_pair_weight': float(cfg.boro_history_pair_weight), 'boro_current_pair_weight': float(cfg.boro_current_pair_weight), 'boro_current_hard_negatives': int(cfg.boro_current_hard_negatives), 'boro_soft_rank_temperature': float(cfg.boro_soft_rank_temperature), 'boro_boundary_negative_weight': float(cfg.boro_boundary_negative_weight), 'boro_boundary_gap_weight': float(cfg.boro_boundary_gap_weight), 'boro_lower_band': float(cfg.boro_lower_band), 'boro_min_pos_weight': float(cfg.boro_min_pos_weight), 'boro_far_below_suppression': float(cfg.boro_far_below_suppression)}, 'data': {'raw_train_size': protocol['raw_train_size'], 'raw_test_size': protocol['raw_test_size'], 'dedup_train_size': protocol['dedup_train_size'], 'dedup_test_size': protocol['dedup_test_size'], 'removed_test_overlap': protocol['removed_test_overlap'], 'train_size': int(len(train_df)), 'val_size': int(len(val_df)), 'test_size': int(len(test_df)), 'train_positive': int(train_df['label'].sum()), 'val_positive': int(val_df['label'].sum()), 'test_positive': int(test_df['label'].sum())}, 'val_metrics': val_metrics, 'test_metrics': test_metrics, 'run_dir': str(run_dir)}
    metrics_path.write_text(json.dumps(sanitize_json(result), indent=2, allow_nan=False), encoding='utf-8')
    return result

def flatten_result(result: dict) -> dict:
    row = {'protocol': result['protocol'], 'method': result['method'], 'seed': result['seed'], 'best_epoch': result['best_epoch'], 'selection_score': result['selection_score'], 'run_dir': result['run_dir']}
    row.update({f'test_{k}': v for k, v in result['test_metrics'].items() if not k.startswith('threshold')})
    row.update({f'val_{k}': v for k, v in result['val_metrics'].items() if not k.startswith('threshold')})
    row.update(result['data'])
    return row

def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in rows.columns if c.startswith('test_') and rows[c].dtype.kind in 'fiu']
    out = []
    for method, group in rows.groupby('method', sort=False):
        row = {'method': method, 'n_seeds': int(group['seed'].nunique())}
        for col in metric_cols:
            row[f'{col}_mean'] = float(group[col].mean())
            row[f'{col}_std'] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        out.append(row)
    return pd.DataFrame(out).sort_values(f'test_{PRIMARY_METRIC}_mean', ascending=False)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--seeds', type=int, nargs='*', default=list(DEFAULT_SEEDS))
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--word-max-features', type=int, default=Config.word_max_features)
    parser.add_argument('--char-max-features', type=int, default=Config.char_max_features)
    parser.add_argument('--svd-components', type=int, default=Config.svd_components)
    parser.add_argument('--batch-size', type=int, default=Config.batch_size)
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--patience', type=int, default=Config.patience)
    parser.add_argument('--learning-rate', type=float, default=Config.learning_rate)
    parser.add_argument('--weight-decay', type=float, default=Config.weight_decay)
    parser.add_argument('--selection-precision', type=float, default=Config.selection_precision)
    parser.add_argument('--loss-target-precision', type=float, default=Config.loss_target_precision)
    parser.add_argument('--boro-history-pair-weight', type=float, default=Config.boro_history_pair_weight)
    parser.add_argument('--boro-current-pair-weight', type=float, default=Config.boro_current_pair_weight)
    parser.add_argument('--boro-current-hard-negatives', type=int, default=Config.boro_current_hard_negatives)
    parser.add_argument('--boro-soft-rank-temperature', type=float, default=Config.boro_soft_rank_temperature)
    parser.add_argument('--boro-boundary-negative-weight', type=float, default=Config.boro_boundary_negative_weight)
    parser.add_argument('--boro-boundary-gap-weight', type=float, default=Config.boro_boundary_gap_weight)
    parser.add_argument('--boro-lower-band', type=float, default=Config.boro_lower_band)
    parser.add_argument('--boro-min-pos-weight', type=float, default=Config.boro_min_pos_weight)
    parser.add_argument('--boro-far-below-suppression', type=float, default=Config.boro_far_below_suppression)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = Config(word_max_features=args.word_max_features, char_max_features=args.char_max_features, svd_components=args.svd_components, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience, learning_rate=args.learning_rate, weight_decay=args.weight_decay, selection_precision=args.selection_precision, loss_target_precision=args.loss_target_precision, boro_history_pair_weight=args.boro_history_pair_weight, boro_current_pair_weight=args.boro_current_pair_weight, boro_current_hard_negatives=args.boro_current_hard_negatives, boro_soft_rank_temperature=args.boro_soft_rank_temperature, boro_boundary_negative_weight=args.boro_boundary_negative_weight, boro_boundary_gap_weight=args.boro_boundary_gap_weight, boro_lower_band=args.boro_lower_band, boro_min_pos_weight=args.boro_min_pos_weight, boro_far_below_suppression=args.boro_far_below_suppression)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        print(f'[boro-final] seed={seed}', flush=True)
        results.append(run_one(int(seed), cfg, args.data_dir, args.output_dir, args.force))
    rows = pd.DataFrame([flatten_result(r) for r in results])
    summary = summarize(rows)
    rows.to_csv(args.output_dir / 'all_seed_results.csv', index=False)
    summary.to_csv(args.output_dir / 'summary.csv', index=False)
    print(json.dumps({'output_dir': str(args.output_dir), 'primary_metric': f'test_{PRIMARY_METRIC}_mean', 'primary_value': float(summary.iloc[0][f'test_{PRIMARY_METRIC}_mean'])}, indent=2))
if __name__ == '__main__':
    main()
