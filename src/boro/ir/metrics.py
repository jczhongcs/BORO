from __future__ import annotations
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font

def _metric_kind(metric: str) -> tuple[str, int]:
    name, k = metric.split('@', 1)
    return (name.upper(), int(k))

def _ranked_doc_ids(items: Sequence[Any]) -> List[str]:
    doc_ids = []
    for item in items:
        if isinstance(item, (tuple, list)):
            doc_ids.append(str(item[0]))
        else:
            doc_ids.append(str(item))
    return doc_ids

def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(ranked[:k])
    hits = sum((1 for doc_id, rel in relevant.items() if rel > 0 and doc_id in top))
    positives = sum((1 for rel in relevant.values() if rel > 0))
    return hits / positives if positives else 0.0

def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    dcg = 0.0
    for idx, doc_id in enumerate(ranked[:k], start=1):
        rel = float(relevant.get(doc_id, 0.0))
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(idx + 1)
    ideal_rels = sorted((float(v) for v in relevant.values() if float(v) > 0), reverse=True)[:k]
    idcg = sum(((2.0 ** rel - 1.0) / math.log2(idx + 1) for idx, rel in enumerate(ideal_rels, start=1)))
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    for idx, doc_id in enumerate(ranked[:k], start=1):
        if float(relevant.get(doc_id, 0.0)) > 0:
            return 1.0 / idx
    return 0.0

def map_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    positives = sum((1 for rel in relevant.values() if float(rel) > 0))
    if positives == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for idx, doc_id in enumerate(ranked[:k], start=1):
        if float(relevant.get(doc_id, 0.0)) > 0:
            hits += 1
            precision_sum += hits / idx
    return precision_sum / min(positives, k)

def _compute_ir_metrics_pytrec_eval(qrels: Mapping[str, Mapping[str, float]], rankings: Mapping[str, Sequence[Any]], metrics: Iterable[str]) -> Dict[str, float]:
    try:
        import pytrec_eval
    except ImportError as exc:
        raise ImportError('pytrec_eval metric backend requested but pytrec_eval is not installed') from exc
    metric_names = list(metrics)
    measures = set()
    needs_mrr = []
    for metric in metric_names:
        kind, k = _metric_kind(metric)
        if kind == 'R':
            measures.add(f'recall_{k}')
        elif kind == 'NDCG':
            measures.add(f'ndcg_cut_{k}')
        elif kind == 'MAP':
            measures.add(f'map_cut_{k}')
        elif kind == 'MRR':
            needs_mrr.append((metric, k))
        else:
            raise ValueError(f'Unsupported metric: {metric}')
    qrel_dict = {str(qid): {str(doc_id): int(float(rel)) for doc_id, rel in rels.items() if float(rel) > 0} for qid, rels in qrels.items()}
    run_dict = {}
    ranked_cache = {}
    for qid in qrel_dict:
        docs = rankings.get(str(qid), [])
        ranked = _ranked_doc_ids(docs)
        ranked_cache[str(qid)] = ranked
        scores = {}
        for rank, item in enumerate(docs, start=1):
            doc_id = str(item[0]) if isinstance(item, (tuple, list)) else str(item)
            if isinstance(item, (tuple, list)) and len(item) > 1:
                score = float(item[1])
            else:
                score = float(len(docs) - rank + 1)
            scores[doc_id] = score
        run_dict[str(qid)] = scores
    values = {name: [] for name in metric_names}
    if measures:
        evaluator = pytrec_eval.RelevanceEvaluator(qrel_dict, measures)
        results = evaluator.evaluate(run_dict)
        for qid, result in results.items():
            for metric in metric_names:
                kind, k = _metric_kind(metric)
                if kind == 'R':
                    values[metric].append(float(result.get(f'recall_{k}', 0.0)))
                elif kind == 'NDCG':
                    values[metric].append(float(result.get(f'ndcg_cut_{k}', 0.0)))
                elif kind == 'MAP':
                    values[metric].append(float(result.get(f'map_cut_{k}', 0.0)))
    for metric, k in needs_mrr:
        for qid, relevant in qrel_dict.items():
            values[metric].append(mrr_at_k(ranked_cache.get(qid, []), relevant, k))
    return {name: float(sum(vals) / len(vals)) if vals else 0.0 for name, vals in values.items()}

def _compute_ir_metrics_trec_eval(qrels: Mapping[str, Mapping[str, float]], rankings: Mapping[str, Sequence[Any]], metrics: Iterable[str]) -> Dict[str, float]:
    metric_names = list(metrics)
    trec_cuts: Dict[str, List[int]] = {'recall': [], 'ndcg_cut': [], 'map_cut': []}
    needs_mrr = []
    output_to_metric = {}
    for metric in metric_names:
        kind, k = _metric_kind(metric)
        if kind == 'R':
            trec_cuts['recall'].append(k)
            output_to_metric[f'recall_{k}'] = metric
        elif kind == 'NDCG':
            trec_cuts['ndcg_cut'].append(k)
            output_to_metric[f'ndcg_cut_{k}'] = metric
        elif kind == 'MAP':
            trec_cuts['map_cut'].append(k)
            output_to_metric[f'map_cut_{k}'] = metric
        elif kind == 'MRR':
            needs_mrr.append((metric, k))
        else:
            raise ValueError(f'Unsupported metric: {metric}')
    values: Dict[str, float] = {name: 0.0 for name in metric_names}
    with tempfile.TemporaryDirectory() as tmp:
        qrel_path = os.path.join(tmp, 'qrels.txt')
        run_path = os.path.join(tmp, 'run.txt')
        with open(qrel_path, 'w', encoding='utf-8') as f:
            for qid, rels in qrels.items():
                for doc_id, rel in rels.items():
                    if float(rel) > 0:
                        f.write(f'{qid} 0 {doc_id} {int(float(rel))}\n')
        with open(run_path, 'w', encoding='utf-8') as f:
            for qid, docs in rankings.items():
                for rank, item in enumerate(docs, start=1):
                    doc_id = str(item[0]) if isinstance(item, (tuple, list)) else str(item)
                    score = float(item[1]) if isinstance(item, (tuple, list)) and len(item) > 1 else float(len(docs) - rank + 1)
                    f.write(f'{qid} Q0 {doc_id} {rank} {score:.12f} IRarl\n')
        trec_measures = []
        for measure, cuts in trec_cuts.items():
            if cuts:
                trec_measures.append(f"{measure}.{','.join((str(k) for k in sorted(set(cuts))))}")
        if trec_measures:
            binary = os.environ.get('TREC_EVAL_BIN', 'trec_eval')
            cmd = [binary]
            for measure in trec_measures:
                cmd.extend(['-m', measure])
            cmd.extend([qrel_path, run_path])
            completed = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for line in completed.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[1] == 'all' and (parts[0] in output_to_metric):
                    values[output_to_metric[parts[0]]] = float(parts[2])
    for metric, k in needs_mrr:
        per_query = []
        for qid, relevant in qrels.items():
            ranked = _ranked_doc_ids(rankings.get(str(qid), []))
            rel = {str(doc_id): float(score) for doc_id, score in relevant.items() if float(score) > 0}
            per_query.append(mrr_at_k(ranked, rel, k))
        values[metric] = float(sum(per_query) / len(per_query)) if per_query else 0.0
    return values

def _compute_ir_metrics_custom(qrels: Mapping[str, Mapping[str, float]], rankings: Mapping[str, Sequence[Any]], metrics: Iterable[str]) -> Dict[str, float]:
    metric_names = list(metrics)
    values = {name: [] for name in metric_names}
    for qid, relevant in qrels.items():
        ranked = _ranked_doc_ids(rankings.get(str(qid), []))
        relevant = {str(doc_id): float(rel) for doc_id, rel in relevant.items() if float(rel) > 0}
        for metric in metric_names:
            kind, k = _metric_kind(metric)
            if kind == 'R':
                value = recall_at_k(ranked, relevant, k)
            elif kind == 'NDCG':
                value = ndcg_at_k(ranked, relevant, k)
            elif kind == 'MRR':
                value = mrr_at_k(ranked, relevant, k)
            elif kind == 'MAP':
                value = map_at_k(ranked, relevant, k)
            else:
                raise ValueError(f'Unsupported metric: {metric}')
            values[metric].append(value)
    return {name: float(sum(vals) / len(vals)) if vals else 0.0 for name, vals in values.items()}

def compute_ir_metrics(qrels: Mapping[str, Mapping[str, float]], rankings: Mapping[str, Sequence[Any]], metrics: Iterable[str], backend: str='custom') -> Dict[str, float]:
    backend_name = backend.lower()
    if backend_name in {'pytrec_eval', 'official'}:
        return _compute_ir_metrics_pytrec_eval(qrels, rankings, metrics)
    if backend_name in {'trec_eval', 'trec_eval_binary'}:
        binary = os.environ.get('TREC_EVAL_BIN', 'trec_eval')
        if shutil.which(binary) is None:
            try:
                return _compute_ir_metrics_pytrec_eval(qrels, rankings, metrics)
            except ImportError:
                return _compute_ir_metrics_custom(qrels, rankings, metrics)
        return _compute_ir_metrics_trec_eval(qrels, rankings, metrics)
    return _compute_ir_metrics_custom(qrels, rankings, metrics)
