from __future__ import annotations
import csv
import glob
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping
BASE_ARL_PARAMS: Dict[str, Any] = {'max_prev_targets': 5000, 'max_prev_decoys': 5000, 'min_pos_weight': 0.35, 'boundary_start_epoch': 1, 'threshold_momentum': 0.5, 'fallback_quantile': 0.9, 'exact_scan_max_items': 200000, 'num_bins': 4096, 'min_required_targets': 1, 'margin_lr': 1e-05, 'score_mode': 'query_centered', 'include_inbatch_decoys': True, 'current_pair_weight': 1.0}

@dataclass(frozen=True)
class SelectorDiagnostics:
    top_gap: float = 0.5
    boundary_violation_rate: float = 0.5
    near_decoy_density: float = 0.5
    decoy_spread: float = 0.5
    inbatch_hardness: float = 0.5
    history_stability: float = 0.5
    score_entropy: float = 0.5
    score_spread: float = 0.5
    corpus_query_shift: float = 0.5
    top10_slope: float = 0.5
    tail_mass: float = 0.5
    top1_gap: float = 0.5
    query_set_scale: float = 0.5
    negative_protocol_hardness: float = 0.5
    explicit_negative_dominance: float = 0.5
    boundary_variance: float = 0.5
    label_sparsity: float = 0.5
    relevance_breadth: float = 0.5
    source_dev_primary: Mapping[str, float] | None = None

def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

def _mix(lo: float, hi: float, t: float) -> float:
    return float(lo + (hi - lo) * _clip01(t))

def _params(**updates: Any) -> Dict[str, Any]:
    values = dict(BASE_ARL_PARAMS)
    values.update(updates)
    return values

def _entropy(scores: List[float], temperature: float=0.05) -> float:
    if not scores:
        return 0.0
    pivot = max(scores)
    exp_scores = [math.exp((score - pivot) / temperature) for score in scores]
    total = sum(exp_scores)
    probs = [score / total for score in exp_scores]
    return -sum((p * math.log(p + 1e-12) for p in probs)) / math.log(len(scores))

def normalize_diagnostics(raw: Mapping[str, Any]) -> SelectorDiagnostics:
    values: Dict[str, Any] = {}
    for field in SelectorDiagnostics.__dataclass_fields__:
        if field == 'source_dev_primary':
            values[field] = raw.get(field)
        elif field == 'query_set_scale' and 'query_set_scale' not in raw and ('query_count' in raw):
            import math
            values[field] = _clip01(math.log10(max(1.0, float(raw['query_count']))) / 3.2)
        else:
            values[field] = _clip01(float(raw.get(field, 0.5)))
    return SelectorDiagnostics(**values)

def read_score_curve_rankings(path: str | Path, max_rank: int=20) -> Dict[str, List[float]]:
    per_query: Dict[str, List[float]] = {}
    with Path(path).open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rank = int(row['rank'])
            if rank <= max_rank:
                per_query.setdefault(str(row['qid']), []).append(float(row['score']))
    return {qid: sorted(scores, reverse=True) for qid, scores in per_query.items() if len(scores) >= 10}

def compute_unlabeled_score_diagnostics(scores_by_query: Mapping[str, List[float]]) -> Dict[str, float]:
    query_rows = list(scores_by_query.values())
    if not query_rows:
        raise ValueError('No ranking scores available for selector diagnostics')
    top1_gap_raw = []
    top10_slope_raw = []
    entropy_raw = []
    spread_raw = []
    tail_raw = []
    for scores in query_rows:
        top1_gap_raw.append(scores[0] - scores[min(4, len(scores) - 1)])
        top10_slope_raw.append((scores[0] - scores[min(9, len(scores) - 1)]) / 9.0)
        entropy_raw.append(_entropy(scores[:20]))
        row_mean = mean(scores[:20])
        spread_raw.append(math.sqrt(mean([(score - row_mean) ** 2 for score in scores[:20]])))
        boundary = scores[min(9, len(scores) - 1)]
        tail_raw.append(sum((1 for score in scores[:20] if abs(score - boundary) <= 0.01)) / min(20, len(scores)))
    top1_gap = _clip01(mean(top1_gap_raw) / 0.18)
    flat_top10 = _clip01(1.0 - mean(top10_slope_raw) / 0.025)
    entropy = _clip01(mean(entropy_raw))
    spread = _clip01(mean(spread_raw) / 0.08)
    tail_mass = _clip01(mean(tail_raw) / 0.5)
    return {'top_gap': top1_gap, 'top1_gap': top1_gap, 'top10_slope': flat_top10, 'score_entropy': entropy, 'score_spread': spread, 'tail_mass': tail_mass, 'query_count': float(len(query_rows)), 'near_decoy_density': 1.0 - top1_gap, 'boundary_violation_rate': 0.5, 'decoy_spread': spread, 'inbatch_hardness': 0.5, 'history_stability': 0.5, 'corpus_query_shift': 0.5, 'raw_top1_gap': mean(top1_gap_raw), 'raw_top10_slope': mean(top10_slope_raw), 'raw_score_entropy': mean(entropy_raw), 'raw_score_spread': mean(spread_raw), 'raw_tail_mass': mean(tail_raw)}

def _latent_factors(diagnostics: SelectorDiagnostics) -> Dict[str, float]:
    d = diagnostics
    hardness = _clip01(0.35 * d.near_decoy_density + 0.25 * d.boundary_violation_rate + 0.2 * d.negative_protocol_hardness + 0.2 * (1.0 - d.top1_gap))
    uncertainty = _clip01(0.4 * d.score_entropy + 0.3 * d.tail_mass + 0.3 * d.label_sparsity)
    stability_need = _clip01(0.45 * d.boundary_variance + 0.3 * (1.0 - d.history_stability) + 0.25 * d.label_sparsity)
    explicit_need = _clip01(0.45 * d.explicit_negative_dominance + 0.25 * d.score_entropy + 0.2 * d.tail_mass + 0.1 * d.corpus_query_shift)
    return {'hardness': hardness, 'uncertainty': uncertainty, 'stability_need': stability_need, 'explicit_need': explicit_need}

def _parameter_vector(diagnostics: SelectorDiagnostics) -> Dict[str, Any]:
    factors = _latent_factors(diagnostics)
    hardness = factors['hardness']
    uncertainty = factors['uncertainty']
    stability_need = factors['stability_need']
    explicit_need = factors['explicit_need']
    if explicit_need >= 0.55 and uncertainty >= 0.6 and (diagnostics.query_set_scale >= 0.8):
        return _params(score_mode='raw', decoy_mode='explicit_all', include_inbatch_decoys=True, hard_decoy_topk=12, constraint_value=0.08, lower_band=0.1, margin_init=0.03, history_pair_weight=0.05, boundary_decoy_weight=2.0, soft_rank_temperature=0.035)
    if explicit_need >= 0.65:
        return _params(decoy_mode='explicit_all', include_inbatch_decoys=False, hard_decoy_topk=16, constraint_value=0.15, lower_band=0.08, margin_init=0.03, history_pair_weight=0.2, boundary_decoy_weight=1.0, soft_rank_temperature=0.05)
    if stability_need >= 0.8:
        return _params(decoy_mode='all', hard_decoy_topk=12, constraint_value=0.15, lower_band=0.08, margin_init=0.04, history_pair_weight=0.52, boundary_decoy_weight=1.0, soft_rank_temperature=0.05)
    broad_relevance = max(diagnostics.relevance_breadth, diagnostics.label_sparsity)
    if broad_relevance >= 0.85 and diagnostics.negative_protocol_hardness >= 0.65:
        return _params(decoy_mode='all', hard_decoy_topk=16, constraint_value=2.0, lower_band=0.09, margin_init=0.007, history_pair_weight=0.06, boundary_decoy_weight=2.1, soft_rank_temperature=0.021, fallback_quantile=0.82, max_pos_weight=1.2)
    if broad_relevance >= 0.85:
        return _params(decoy_mode='all', hard_decoy_topk=8, constraint_value=0.0, lower_band=0.055, margin_init=0.009, history_pair_weight=0.045, boundary_decoy_weight=2.1, soft_rank_temperature=0.02, fallback_quantile=0.86, max_pos_weight=1.4, deterministic_seed=7)
    if diagnostics.near_decoy_density <= 0.25 and diagnostics.boundary_variance <= 0.2:
        return _params(decoy_mode='all', hard_decoy_topk=6, constraint_value=2.0, lower_band=0.055, margin_init=0.008, history_pair_weight=0.025, boundary_decoy_weight=1.2, soft_rank_temperature=0.022, fallback_quantile=0.88, max_pos_weight=1.5)
    return _params(decoy_mode='all', hard_decoy_topk=12, constraint_value=0.07, lower_band=0.07, margin_init=0.009, history_pair_weight=0.035, boundary_decoy_weight=1.6, soft_rank_temperature=0.021, fallback_quantile=0.84, max_pos_weight=1.0)

def select_arl_parameters(raw_diagnostics: Mapping[str, Any]) -> Dict[str, Any]:
    diagnostics = normalize_diagnostics(raw_diagnostics)
    return {'diagnostics': asdict(diagnostics), 'selector_factors': _latent_factors(diagnostics), 'arl_params': _parameter_vector(diagnostics)}

def select_zero_shot_suite_parameters(raw_diagnostics_list: list[Mapping[str, Any]]) -> Dict[str, Any]:
    if not raw_diagnostics_list:
        raise ValueError('raw_diagnostics_list must not be empty')
    numeric_fields = [field for field in SelectorDiagnostics.__dataclass_fields__ if field != 'source_dev_primary']
    averaged: Dict[str, Any] = {}
    for field in numeric_fields:
        values = [float(raw[field]) for raw in raw_diagnostics_list if field in raw]
        if values:
            averaged[field] = sum(values) / len(values)
    if 'query_count' in raw_diagnostics_list[0]:
        averaged['query_count'] = sum((float(raw.get('query_count', 0.0)) for raw in raw_diagnostics_list))
    return select_arl_parameters(averaged)

def select_arl_parameters_from_score_curves(ranking_paths: Iterable[str | Path]) -> Dict[str, Any]:
    diagnostics = [compute_unlabeled_score_diagnostics(read_score_curve_rankings(path)) for path in ranking_paths]
    return select_zero_shot_suite_parameters(diagnostics)

def apply_selected_parameters(config: Dict[str, Any], selection: Mapping[str, Any]) -> Dict[str, Any]:
    params = dict(selection['arl_params'])
    config.setdefault('loss_params', {}).setdefault('arl', {}).update(params)
    config.setdefault('selector', {})['diagnostics'] = selection['diagnostics']
    config['selector']['factors'] = selection['selector_factors']
    config['selector']['selected_arl_params'] = params
    return config

def _resolve_run_diagnostics(config: Mapping[str, Any], dataset_name: str | None=None) -> Mapping[str, Any]:
    selector = config.get('selector', {})
    if not isinstance(selector, Mapping):
        raise ValueError('selector must be a mapping when enabled')
    diagnostics = selector.get('diagnostics')
    if isinstance(diagnostics, Mapping):
        return diagnostics
    by_dataset = selector.get('diagnostics_by_dataset')
    if isinstance(by_dataset, Mapping) and dataset_name is not None:
        dataset_entry = by_dataset.get(dataset_name)
        if isinstance(dataset_entry, Mapping):
            negative_name = str(config.get('negative_sampling', {}).get('name', 'random'))
            if negative_name in dataset_entry and isinstance(dataset_entry[negative_name], Mapping):
                return dataset_entry[negative_name]
            return dataset_entry
    raise ValueError('selector.enabled requires selector.diagnostics or selector.diagnostics_by_dataset')

def _resolve_score_curve_paths(config: Mapping[str, Any]) -> list[Path]:
    selector = config.get('selector', {})
    if not isinstance(selector, Mapping):
        raise ValueError('selector must be a mapping when enabled')
    raw_patterns = selector.get('score_curve_rankings')
    if raw_patterns is None:
        raw_patterns = selector.get('score_curve_ranking_files')
    if isinstance(raw_patterns, (str, Path)):
        patterns = [raw_patterns]
    elif isinstance(raw_patterns, Iterable):
        patterns = list(raw_patterns)
    else:
        raise ValueError('score-curve selector requires selector.score_curve_rankings')
    base_dirs = [Path.cwd()]
    config_dir = config.get('_config_dir')
    if config_dir:
        base_dirs.insert(0, Path(str(config_dir)))
    resolved: list[Path] = []
    for pattern in patterns:
        path_pattern = Path(str(pattern))
        candidate_patterns = [str(path_pattern)] if path_pattern.is_absolute() else [str(base / path_pattern) for base in base_dirs]
        for candidate_pattern in candidate_patterns:
            for match in glob.glob(candidate_pattern):
                path = Path(match).resolve()
                if path.is_file() and path not in resolved:
                    resolved.append(path)
    if not resolved:
        raise FileNotFoundError(f'No score-curve ranking files matched: {patterns}')
    return sorted(resolved)

def apply_selector_to_config(config: Dict[str, Any], dataset_name: str | None, loss_name: str) -> Dict[str, Any]:
    selector = config.get('selector', {})
    if str(loss_name).lower() != 'arl' or not bool(selector.get('enabled', False)):
        return config
    diagnostics_source = str(selector.get('diagnostics_source', '')).lower()
    if diagnostics_source == 'auto':
        return config
    if diagnostics_source in {'score_curves', 'score_curve', 'unlabeled_score_curves'}:
        ranking_paths = _resolve_score_curve_paths(config)
        selection = select_arl_parameters_from_score_curves(ranking_paths)
        config = apply_selected_parameters(config, selection)
        config.setdefault('selector', {})['diagnostics_source'] = 'score_curves'
        config['selector']['score_curve_files'] = [str(path) for path in ranking_paths]
        return config
    diagnostics = _resolve_run_diagnostics(config, dataset_name)
    selection = select_arl_parameters(diagnostics)
    return apply_selected_parameters(config, selection)
