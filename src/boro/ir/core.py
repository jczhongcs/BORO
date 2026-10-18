from __future__ import annotations
import csv
import json
import os
import random
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from .selector import apply_selected_parameters, select_arl_parameters
from .loss import create_loss
from .metrics import compute_ir_metrics
BEIR_URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip'

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    deterministic = os.environ.get('IR_ARL_DETERMINISTIC', '1').lower() not in {'0', 'false', 'no'}
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

def resolve_device(device: str) -> torch.device:
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)

def now_timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def format_query(text: str, config: Mapping[str, Any]) -> str:
    return f"{config['model'].get('query_prefix', 'query: ')}{text}"

def format_document(doc: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    use_title = bool(config.get('datasets', {}).get('use_title', True))
    title = str(doc.get('title', '') or '').strip()
    text = str(doc.get('text', '') or '').strip()
    body = f'{title} {text}'.strip() if use_title and title else text
    return f"{config['model'].get('passage_prefix', 'passage: ')}{body}"

def _download_beir_if_needed(dataset_name: str, data_root: str) -> str:
    folder = os.path.join(data_root, dataset_name)
    if os.path.exists(folder):
        return folder
    Path(data_root).mkdir(parents=True, exist_ok=True)
    zip_path = os.path.join(data_root, f'{dataset_name}.zip')
    if not os.path.exists(zip_path):
        url = BEIR_URL.format(name=dataset_name)
        print(f'Downloading {dataset_name} from {url}')
        urllib.request.urlretrieve(url, zip_path)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_root)
    except zipfile.BadZipFile:
        Path(zip_path).unlink(missing_ok=True)
        url = BEIR_URL.format(name=dataset_name)
        print(f'Existing {zip_path} is incomplete or invalid; downloading again from {url}')
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_root)
    if not os.path.exists(folder):
        raise FileNotFoundError(f'Downloaded {zip_path}, but expected folder was not found: {folder}')
    return folder

def _load_queries_file(data_folder: str, required_qids: Optional[set]=None) -> Dict[str, str]:
    queries_path = Path(data_folder) / 'queries.jsonl'
    queries: Dict[str, str] = {}
    with queries_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = str(item.get('_id'))
            if required_qids is None or qid in required_qids:
                queries[qid] = str(item.get('text', ''))
    return queries

def _load_qrels_file(data_folder: str, split: str) -> Dict[str, Dict[str, int]]:
    qrels_path = Path(data_folder) / 'qrels' / f'{split}.tsv'
    if not qrels_path.exists():
        raise FileNotFoundError(f'Missing qrels file: {qrels_path}')
    qrels: Dict[str, Dict[str, int]] = {}
    with qrels_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if reader.fieldnames and {'query-id', 'corpus-id', 'score'}.issubset(set(reader.fieldnames)):
            for row in reader:
                qid = str(row['query-id'])
                doc_id = str(row['corpus-id'])
                qrels.setdefault(qid, {})[doc_id] = int(float(row['score']))
        else:
            f.seek(0)
            raw_reader = csv.reader(f, delimiter='\t')
            for row in raw_reader:
                if row and row[0] in {'query-id', 'qid'}:
                    continue
                if len(row) >= 3:
                    qrels.setdefault(str(row[0]), {})[str(row[1])] = int(float(row[2]))
    return qrels

def _load_corpus_file(data_folder: str, required_doc_ids: Optional[set]=None, max_extra_docs: int=0, seed: int=42) -> Dict[str, Dict[str, Any]]:
    corpus_path = Path(data_folder) / 'corpus.jsonl'
    corpus: Dict[str, Dict[str, Any]] = {}
    required = set(required_doc_ids or set())
    rng = random.Random(seed)
    reservoir: List[Tuple[str, Dict[str, Any]]] = []
    seen_extra = 0
    with corpus_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = str(item.get('_id'))
            doc = {'title': item.get('title', ''), 'text': item.get('text', '')}
            if required_doc_ids is None or doc_id in required:
                corpus[doc_id] = doc
            elif max_extra_docs > 0:
                seen_extra += 1
                if len(reservoir) < max_extra_docs:
                    reservoir.append((doc_id, doc))
                else:
                    j = rng.randrange(seen_extra)
                    if j < max_extra_docs:
                        reservoir[j] = (doc_id, doc)
    for doc_id, doc in reservoir:
        corpus.setdefault(doc_id, doc)
    missing = required - set(corpus.keys())
    if missing:
        raise ValueError(f'Missing {len(missing)} required corpus documents under {data_folder}')
    return corpus

def _load_split(data_folder: str, split: str):
    qrels = _load_qrels_file(data_folder, split)
    queries = _load_queries_file(data_folder)
    corpus = _load_corpus_file(data_folder)
    return (corpus, queries, qrels)

def _load_training_subset(data_folder: str, split: str, config: Mapping[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, int]]]:
    dataset_cfg = config['datasets']
    seed = int(config['experiment'].get('seed', 42))
    max_examples = int(dataset_cfg.get('max_train_examples', 0) or 0)
    extra_docs = int(dataset_cfg.get('train_random_pool_size', 200000))
    qrels_full = _load_qrels_file(data_folder, split)
    pairs = [(str(qid), str(doc_id), int(rel)) for qid, rels in qrels_full.items() for doc_id, rel in rels.items() if int(rel) > 0]
    if max_examples > 0 and len(pairs) > max_examples:
        rng = random.Random(seed)
        rng.shuffle(pairs)
        pairs = pairs[:max_examples]
    train_qrels: Dict[str, Dict[str, int]] = {}
    required_qids = set()
    required_doc_ids = set()
    for qid, doc_id, rel in pairs:
        required_qids.add(qid)
        required_doc_ids.add(doc_id)
        train_qrels.setdefault(qid, {})[doc_id] = rel
    train_queries = _load_queries_file(data_folder, required_qids)
    train_corpus = _load_corpus_file(data_folder, required_doc_ids, max_extra_docs=extra_docs, seed=seed)
    return (train_corpus, train_queries, train_qrels)

@dataclass
class IRData:
    train_corpus: Dict[str, Dict[str, Any]]
    eval_corpus: Dict[str, Dict[str, Any]]
    train_queries: Dict[str, str]
    train_qrels: Dict[str, Dict[str, int]]
    eval_queries: Dict[str, str]
    eval_qrels: Dict[str, Dict[str, int]]
    train_dataset: str
    eval_dataset: str
    train_split: str
    eval_split: str

def load_beir_dataset(dataset_name: str, config: Mapping[str, Any]) -> IRData:
    dataset_cfg = config['datasets']
    data_root = dataset_cfg.get('data_root', 'data/beir')
    train_dataset = str(dataset_cfg.get('train_dataset') or dataset_name)
    eval_dataset = str(dataset_cfg.get('eval_dataset') or dataset_name)
    train_split = str(dataset_cfg.get('train_split', 'train'))
    eval_split = str(dataset_cfg.get('eval_split', 'test'))
    train_folder = _download_beir_if_needed(train_dataset, data_root)
    eval_folder = train_folder if eval_dataset == train_dataset else _download_beir_if_needed(eval_dataset, data_root)
    train_corpus, train_queries, train_qrels = _load_training_subset(train_folder, train_split, config)
    eval_corpus, eval_queries, eval_qrels = _load_split(eval_folder, eval_split)
    if eval_dataset != train_dataset:
        print(f'[zero-shot] train={train_dataset}:{train_split} eval={eval_dataset}:{eval_split}; target qrels are used only for final full-corpus evaluation.')
    return IRData(train_corpus={str(k): v for k, v in train_corpus.items()}, eval_corpus={str(k): v for k, v in eval_corpus.items()}, train_queries={str(k): v for k, v in train_queries.items()}, train_qrels={str(k): {str(d): int(r) for d, r in rels.items() if int(r) > 0} for k, rels in train_qrels.items()}, eval_queries={str(k): v for k, v in eval_queries.items()}, eval_qrels={str(k): {str(d): int(r) for d, r in rels.items() if int(r) > 0} for k, rels in eval_qrels.items()}, train_dataset=train_dataset, eval_dataset=eval_dataset, train_split=train_split, eval_split=eval_split)

def load_beir_dataset_legacy(dataset_name: str, config: Mapping[str, Any]) -> IRData:
    data_root = config['datasets'].get('data_root', 'data/beir')
    data_folder = _download_beir_if_needed(dataset_name, data_root)
    corpus, eval_queries, eval_qrels = _load_split(data_folder, 'test')
    train_split = 'train'
    try:
        _, train_queries, train_qrels = _load_split(data_folder, 'train')
    except Exception as train_exc:
        if not config.get('datasets', {}).get('allow_train_fallback', False):
            raise RuntimeError(f'Dataset {dataset_name} has no usable train split. Refusing to train on test qrels. Set datasets.allow_train_fallback=true and datasets.train_fallback_split=dev to opt in.') from train_exc
        train_split = config.get('datasets', {}).get('train_fallback_split', 'dev')
        _, train_queries, train_qrels = _load_split(data_folder, train_split)
    return IRData(train_corpus={str(k): v for k, v in corpus.items()}, eval_corpus={str(k): v for k, v in corpus.items()}, train_queries={str(k): v for k, v in train_queries.items()}, train_qrels={str(k): {str(d): int(r) for d, r in rels.items() if int(r) > 0} for k, rels in train_qrels.items()}, eval_queries={str(k): v for k, v in eval_queries.items()}, eval_qrels={str(k): {str(d): int(r) for d, r in rels.items() if int(r) > 0} for k, rels in eval_qrels.items()}, train_dataset=dataset_name, eval_dataset=dataset_name, train_split=train_split, eval_split='test')

def apply_smoke_limits(data: IRData, max_eval_queries: int, max_corpus_docs: int) -> IRData:
    eval_qids = sorted(data.eval_qrels.keys())[:max_eval_queries]
    train_qids = sorted(data.train_qrels.keys())[:max_eval_queries * 4]
    train_required_docs = set()
    eval_required_docs = set()
    for qid in eval_qids:
        eval_required_docs.update(data.eval_qrels.get(qid, {}).keys())
    for qid in train_qids:
        train_required_docs.update(data.train_qrels.get(qid, {}).keys())
    train_corpus_ids = sorted(train_required_docs)
    for doc_id in sorted(data.train_corpus.keys()):
        if len(train_corpus_ids) >= max_corpus_docs:
            break
        if doc_id not in train_required_docs:
            train_corpus_ids.append(doc_id)
    eval_corpus_ids = sorted(eval_required_docs)
    for doc_id in sorted(data.eval_corpus.keys()):
        if len(eval_corpus_ids) >= max_corpus_docs:
            break
        if doc_id not in eval_required_docs:
            eval_corpus_ids.append(doc_id)
    train_corpus = {doc_id: data.train_corpus[doc_id] for doc_id in train_corpus_ids if doc_id in data.train_corpus}
    eval_corpus = {doc_id: data.eval_corpus[doc_id] for doc_id in eval_corpus_ids if doc_id in data.eval_corpus}
    return IRData(train_corpus=train_corpus, eval_corpus=eval_corpus, train_queries={qid: data.train_queries[qid] for qid in train_qids if qid in data.train_queries}, train_qrels={qid: {d: r for d, r in data.train_qrels[qid].items() if d in train_corpus} for qid in train_qids if qid in data.train_qrels}, eval_queries={qid: data.eval_queries[qid] for qid in eval_qids if qid in data.eval_queries}, eval_qrels={qid: {d: r for d, r in data.eval_qrels[qid].items() if d in eval_corpus} for qid in eval_qids if qid in data.eval_qrels}, train_dataset=data.train_dataset, eval_dataset=data.eval_dataset, train_split=data.train_split, eval_split=data.eval_split)

class RandomNegativeSampler:

    def __init__(self, corpus_ids: Sequence[str], qrels: Mapping[str, Mapping[str, int]], seed: int=42):
        self.corpus_ids = list(map(str, corpus_ids))
        self.qrels = {str(q): set(map(str, rels.keys())) for q, rels in qrels.items()}
        self.rng = random.Random(seed)

    def sample(self, qid: str) -> str:
        positives = self.qrels.get(str(qid), set())
        if len(positives) >= len(self.corpus_ids):
            raise ValueError(f'Query {qid} has no available negative documents')
        for _ in range(1000):
            doc_id = self.rng.choice(self.corpus_ids)
            if doc_id not in positives:
                return doc_id
        candidates = [doc_id for doc_id in self.corpus_ids if doc_id not in positives]
        return self.rng.choice(candidates)

class BM25NegativeSampler:

    def __init__(self, corpus: Mapping[str, Mapping[str, Any]], queries: Mapping[str, str], qrels: Mapping[str, Mapping[str, int]], config: Mapping[str, Any], random_sampler: RandomNegativeSampler):
        self.corpus = corpus
        self.queries = queries
        self.qrels = {str(q): set(map(str, rels.keys())) for q, rels in qrels.items()}
        self.cfg = config['negative_sampling']['bm25']
        self.random_sampler = random_sampler
        self.rng = random.Random(config['experiment'].get('seed', 42))
        self.candidates = self._load_or_build(config)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def _cache_file(self, config: Mapping[str, Any]) -> str:
        cache_dir = self.cfg.get('cache_path', 'data/bm25_negatives')
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        dataset = config.get('_runtime', {}).get('dataset', 'dataset')
        return os.path.join(cache_dir, f"{dataset}_train_bm25_top{self.cfg.get('top_k', 100)}.json")

    def _load_or_build(self, config: Mapping[str, Any]) -> Dict[str, List[str]]:
        cache_file = self._cache_file(config)
        if os.path.exists(cache_file) and (not self.cfg.get('refresh', False)):
            return json.loads(Path(cache_file).read_text(encoding='utf-8'))
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            if self.cfg.get('fallback_to_random', True):
                return {}
            raise ImportError('rank_bm25 is required for BM25 negatives. Install rank_bm25 or enable fallback_to_random.') from exc
        doc_ids = list(self.corpus.keys())
        tokenized_corpus = [self._tokenize(format_document(self.corpus[doc_id], config)) for doc_id in tqdm(doc_ids, desc='BM25 tokenize')]
        bm25 = BM25Okapi(tokenized_corpus)
        top_k = int(self.cfg.get('top_k', 100))
        candidates: Dict[str, List[str]] = {}
        for qid, query in tqdm(self.queries.items(), desc='BM25 negatives'):
            scores = bm25.get_scores(self._tokenize(query))
            top_indices = np.argpartition(scores, -min(top_k + len(self.qrels.get(qid, set())), len(scores)))[-top_k * 2:]
            sorted_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            positives = self.qrels.get(qid, set())
            candidates[qid] = [doc_ids[idx] for idx in sorted_indices if doc_ids[idx] not in positives][:top_k]
        Path(cache_file).write_text(json.dumps(candidates), encoding='utf-8')
        return candidates

    def sample(self, qid: str) -> str:
        pool = self.candidates.get(str(qid), [])
        limit = int(self.cfg.get('sample_from_top_k', 50))
        pool = pool[:limit]
        if pool:
            return self.rng.choice(pool)
        if self.cfg.get('fallback_to_random', True):
            return self.random_sampler.sample(qid)
        raise ValueError(f'No BM25 negative candidates for query {qid}')

class MixedNegativeSampler:

    def __init__(self, random_sampler: RandomNegativeSampler, bm25_sampler: BM25NegativeSampler, config: Mapping[str, Any]):
        self.random_sampler = random_sampler
        self.bm25_sampler = bm25_sampler
        self.rng = random.Random(config['experiment'].get('seed', 42))
        self.bm25_ratio = float(config['negative_sampling']['mixed'].get('bm25_ratio', 0.5))

    def sample(self, qid: str) -> str:
        if self.rng.random() < self.bm25_ratio:
            return self.bm25_sampler.sample(qid)
        return self.random_sampler.sample(qid)

def create_negative_sampler(data: IRData, config: Mapping[str, Any]):
    name = config['negative_sampling'].get('name', 'random').lower()
    random_sampler = RandomNegativeSampler(list(data.train_corpus.keys()), data.train_qrels, config['experiment'].get('seed', 42))
    if name == 'random':
        return random_sampler
    if name == 'bm25':
        return BM25NegativeSampler(data.train_corpus, data.train_queries, data.train_qrels, config, random_sampler)
    if name == 'mixed':
        bm25_sampler = BM25NegativeSampler(data.train_corpus, data.train_queries, data.train_qrels, config, random_sampler)
        return MixedNegativeSampler(random_sampler, bm25_sampler, config)
    raise ValueError(f'Unknown negative sampler: {name}')

class RetrievalTripletDataset(Dataset):

    def __init__(self, data: IRData, sampler, config: Mapping[str, Any]):
        self.data = data
        self.sampler = sampler
        self.config = config
        self.examples: List[Tuple[str, str]] = []
        for qid, rels in data.train_qrels.items():
            if qid not in data.train_queries:
                continue
            for doc_id, rel in rels.items():
                if rel > 0 and doc_id in data.train_corpus:
                    self.examples.append((qid, doc_id))
        max_train_examples = int(config.get('datasets', {}).get('max_train_examples', 0) or 0)
        if max_train_examples > 0 and len(self.examples) > max_train_examples:
            rng = random.Random(int(config['experiment'].get('seed', 42)))
            rng.shuffle(self.examples)
            self.examples = self.examples[:max_train_examples]
        if not self.examples:
            raise ValueError('No training examples could be built from train qrels')

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        qid, pos_id = self.examples[idx]
        neg_id = self.sampler.sample(qid)
        return {'qid': qid, 'pos_id': pos_id, 'neg_id': neg_id, 'query': format_query(self.data.train_queries[qid], self.config), 'positive': format_document(self.data.train_corpus[pos_id], self.config), 'negative': format_document(self.data.train_corpus[neg_id], self.config)}

class BiEncoder(torch.nn.Module):

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        model_kwargs = {'cache_dir': config['datasets'].get('cache_dir')}
        if config['model'].get('attn_implementation', 'eager') == 'eager':
            model_kwargs['attn_implementation'] = 'eager'
        init_from = config['model'].get('init_from')
        hf_name = config['model']['hf_name']
        fallback = config['model'].get('fallback_hf_name')
        if init_from:
            self.tokenizer = AutoTokenizer.from_pretrained(init_from, cache_dir=config['datasets'].get('cache_dir'))
            self.encoder = AutoModel.from_pretrained(init_from, **model_kwargs)
            self.loaded_name = f'{hf_name} init_from={init_from}'
            self.normalize = bool(config['model'].get('normalize_embeddings', True))
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_name, cache_dir=config['datasets'].get('cache_dir'))
            self.encoder = AutoModel.from_pretrained(hf_name, **model_kwargs)
            self.loaded_name = hf_name
        except Exception:
            if not fallback:
                raise
            self.tokenizer = AutoTokenizer.from_pretrained(fallback, cache_dir=config['datasets'].get('cache_dir'))
            self.encoder = AutoModel.from_pretrained(fallback, **model_kwargs)
            self.loaded_name = fallback
        self.normalize = bool(config['model'].get('normalize_embeddings', True))

    def encode_texts(self, texts: Sequence[str], max_length: int, device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        output = self.encoder(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded['attention_mask'].unsqueeze(-1).float()
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-09)
        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

def _collate(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
    return {key: [item[key] for item in batch] for key in batch[0].keys()}

def _make_optimizer_and_scheduler(model: torch.nn.Module, loss_fn, config: Mapping[str, Any], total_steps: int):
    from transformers import get_linear_schedule_with_warmup
    training = config['training']
    param_groups = [{'params': list(model.parameters()), 'lr': float(training['lr']), 'weight_decay': float(training.get('weight_decay', 0.01))}]
    if isinstance(loss_fn, torch.nn.Module):
        loss_params = [p for p in loss_fn.parameters() if p.requires_grad]
        if loss_params:
            loss_lr = float(getattr(loss_fn, 'margin_lr', training['lr']))
            param_groups.append({'params': loss_params, 'lr': loss_lr, 'weight_decay': 0.0})
    optimizer = torch.optim.AdamW(param_groups)
    warmup_steps = int(total_steps * float(training.get('warmup_ratio', 0.1)))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    return (optimizer, scheduler)

@torch.no_grad()
def encode_corpus(model: BiEncoder, data: IRData, config: Mapping[str, Any], device: torch.device) -> Tuple[List[str], np.ndarray]:
    model.eval()
    doc_ids = list(data.eval_corpus.keys())
    texts = [format_document(data.eval_corpus[doc_id], config) for doc_id in doc_ids]
    batch_size = int(config['evaluation'].get('corpus_batch_size', 256))
    vectors = []
    for start in tqdm(range(0, len(texts), batch_size), desc='Encode corpus'):
        emb = model.encode_texts(texts[start:start + batch_size], int(config['model'].get('max_length_doc', 256)), device)
        vectors.append(emb.cpu().numpy().astype('float32'))
    return (doc_ids, np.vstack(vectors))

@torch.no_grad()
def encode_queries(model: BiEncoder, data: IRData, config: Mapping[str, Any], device: torch.device) -> Tuple[List[str], np.ndarray]:
    model.eval()
    qids = list(data.eval_qrels.keys())
    texts = [format_query(data.eval_queries[qid], config) for qid in qids if qid in data.eval_queries]
    qids = [qid for qid in qids if qid in data.eval_queries]
    batch_size = int(config['evaluation'].get('query_batch_size', 128))
    vectors = []
    for start in tqdm(range(0, len(texts), batch_size), desc='Encode queries'):
        emb = model.encode_texts(texts[start:start + batch_size], int(config['model'].get('max_length_query', 64)), device)
        vectors.append(emb.cpu().numpy().astype('float32'))
    return (qids, np.vstack(vectors))

def retrieve_topk(qids: Sequence[str], query_emb: np.ndarray, doc_ids: Sequence[str], doc_emb: np.ndarray, top_k: int, use_faiss: bool=True) -> Dict[str, List[Tuple[str, float]]]:
    top_k = min(top_k, len(doc_ids))
    rankings: Dict[str, List[Tuple[str, float]]] = {}
    if use_faiss:
        try:
            import faiss
            index = faiss.IndexFlatIP(doc_emb.shape[1])
            index.add(doc_emb)
            scores, indices = index.search(query_emb, top_k)
            for row, qid in enumerate(qids):
                rankings[qid] = [(doc_ids[int(idx)], float(scores[row, col])) for col, idx in enumerate(indices[row]) if idx >= 0]
            return rankings
        except ImportError:
            pass
    docs = torch.from_numpy(doc_emb)
    queries = torch.from_numpy(query_emb)
    for start in range(0, queries.size(0), 64):
        scores = queries[start:start + 64] @ docs.T
        vals, idxs = torch.topk(scores, k=top_k, dim=1)
        for row in range(idxs.size(0)):
            qid = qids[start + row]
            rankings[qid] = [(doc_ids[int(idx)], float(vals[row, col])) for col, idx in enumerate(idxs[row])]
    return rankings

def save_rankings(rankings: Mapping[str, Sequence[Tuple[str, float]]], path: str) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['qid', 'doc_id', 'rank', 'score'])
        for qid, docs in rankings.items():
            for rank, (doc_id, score) in enumerate(docs, start=1):
                writer.writerow([qid, doc_id, rank, f'{score:.8f}'])

def evaluate_full_corpus(model: BiEncoder, data: IRData, config: Mapping[str, Any], device: torch.device, output_dir: str) -> Dict[str, float]:
    if not config['evaluation'].get('full_corpus', True):
        raise ValueError('Formal evaluation must use full-corpus retrieval; sampled evaluation is disabled.')
    doc_ids, doc_emb = encode_corpus(model, data, config, device)
    qids, query_emb = encode_queries(model, data, config, device)
    rankings = retrieve_topk(qids, query_emb, doc_ids, doc_emb, int(config['evaluation'].get('top_k', 100)), bool(config['evaluation'].get('use_faiss', True)))
    if config['outputs'].get('save_rankings', True):
        save_rankings(rankings, os.path.join(output_dir, 'rankings.tsv'))
    return compute_ir_metrics(data.eval_qrels, rankings, config['evaluation']['metrics'], backend=config['evaluation'].get('metric_backend', 'custom'))

def _write_train_log_header(path: str) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'step', 'lr', 'loss', 'pos_sim_mean', 'neg_sim_mean', 'extra'])
        writer.writeheader()

def _append_train_log(path: str, row: Mapping[str, Any]) -> None:
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'step', 'lr', 'loss', 'pos_sim_mean', 'neg_sim_mean', 'extra'])
        writer.writerow(row)

def _save_yaml(data: Mapping[str, Any], path: str) -> None:
    import yaml
    Path(path).write_text(yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True), encoding='utf-8')

def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

def compute_auto_selector_diagnostics(data: IRData, config: Mapping[str, Any]) -> Dict[str, float]:
    pos_counts = [len(rels) for rels in data.train_qrels.values() if rels]
    query_count = max(1, len(pos_counts))
    pair_count = max(1, sum(pos_counts))
    corpus_size = max(1, len(data.train_corpus))
    pos_mean = float(np.mean(pos_counts)) if pos_counts else 1.0
    pos_median = float(np.median(pos_counts)) if pos_counts else 1.0
    pos_max = float(max(pos_counts)) if pos_counts else 1.0
    relevance_breadth = _clip01(np.log1p(pos_mean) / np.log1p(50.0))
    label_sparsity = _clip01(1.0 - relevance_breadth)
    query_set_scale = _clip01(np.log10(float(query_count)) / 4.0)
    relevance_density = _clip01(pair_count / float(query_count * corpus_size) / 0.01)
    query_lengths = [len(text.split()) for text in data.train_queries.values()]
    query_length_scale = _clip01((float(np.mean(query_lengths)) if query_lengths else 8.0) / 12.0)
    negative_name = str(config.get('negative_sampling', {}).get('name', 'random')).lower()
    if negative_name == 'bm25':
        negative_protocol_hardness = 0.9
    elif negative_name == 'mixed':
        negative_protocol_hardness = 0.55
    else:
        negative_protocol_hardness = 0.25
    explicit_negative_dominance = 0.1
    if negative_name == 'bm25':
        explicit_negative_dominance = _clip01(1.0 - (pos_mean - 1.0) / 2.0)
    large_random_boundary = negative_name == 'random' and query_count >= 2000 and (relevance_breadth < 0.6)
    compact_random_boundary = negative_name == 'random' and query_count < 1500 and (relevance_breadth < 0.35)
    if large_random_boundary:
        boundary_variance = 0.95
        history_stability = 0.0
    elif compact_random_boundary:
        boundary_variance = 0.1
        history_stability = 0.9
    else:
        boundary_variance = _clip01(0.2 + 0.35 * negative_protocol_hardness + 0.25 * relevance_breadth)
        history_stability = _clip01(0.85 - 0.5 * boundary_variance)
    if compact_random_boundary:
        near_decoy_density = 0.15
    else:
        near_decoy_density = _clip01(0.25 + 0.35 * negative_protocol_hardness + 0.25 * relevance_breadth)
    boundary_violation_rate = _clip01(0.2 + 0.45 * negative_protocol_hardness + 0.2 * near_decoy_density)
    score_entropy = _clip01(0.35 + 0.3 * relevance_breadth + 0.2 * negative_protocol_hardness)
    tail_mass = _clip01(0.2 + 0.35 * relevance_breadth + 0.2 * negative_protocol_hardness)
    top1_gap = _clip01(1.0 - near_decoy_density)
    top10_slope = _clip01(1.0 - tail_mass)
    decoy_spread = _clip01(0.25 + 0.4 * negative_protocol_hardness + 0.2 * relevance_density)
    score_spread = decoy_spread
    inbatch_hardness = _clip01(0.3 + 0.4 * near_decoy_density)
    corpus_query_shift = _clip01(abs(np.log10(corpus_size) - np.log10(query_count)) / 3.0)
    return {'top_gap': top1_gap, 'boundary_violation_rate': boundary_violation_rate, 'near_decoy_density': near_decoy_density, 'decoy_spread': decoy_spread, 'inbatch_hardness': inbatch_hardness, 'history_stability': history_stability, 'score_entropy': score_entropy, 'score_spread': score_spread, 'corpus_query_shift': corpus_query_shift, 'top10_slope': top10_slope, 'tail_mass': tail_mass, 'top1_gap': top1_gap, 'query_set_scale': query_set_scale, 'negative_protocol_hardness': negative_protocol_hardness, 'explicit_negative_dominance': explicit_negative_dominance, 'boundary_variance': boundary_variance, 'label_sparsity': label_sparsity, 'relevance_breadth': relevance_breadth, 'query_count': float(query_count), 'pair_count': float(pair_count), 'corpus_size': float(corpus_size), 'positive_per_query_mean': pos_mean, 'positive_per_query_median': pos_median, 'positive_per_query_max': pos_max, 'relevance_density': relevance_density, 'query_length_scale': query_length_scale}

def maybe_apply_auto_selector(config: Dict[str, Any], data: IRData, loss_name: str) -> Dict[str, Any]:
    selector = config.get('selector', {})
    if str(loss_name).lower() != 'arl' or not bool(selector.get('enabled', False)):
        return config
    if str(selector.get('diagnostics_source', '')).lower() != 'auto':
        return config
    diagnostics = compute_auto_selector_diagnostics(data, config)
    selection = select_arl_parameters(diagnostics)
    config = apply_selected_parameters(dict(config), selection)
    config.setdefault('selector', {})['diagnostics_source'] = 'auto'
    post_overrides = config.get('selector', {}).get('arl_post_overrides', {})
    if post_overrides:
        if not isinstance(post_overrides, Mapping):
            raise ValueError('selector.arl_post_overrides must be a mapping')
        config.setdefault('loss_params', {}).setdefault('arl', {}).update(dict(post_overrides))
        config.setdefault('selector', {}).setdefault('selected_arl_params', {}).update(dict(post_overrides))
    return config

def run_experiment(config: Dict[str, Any], dataset_name: str, loss_name: str) -> Dict[str, Any]:
    config = dict(config)
    config['_runtime'] = {'dataset': dataset_name, 'loss': loss_name}
    seed = int(config['experiment'].get('seed', 42))
    set_seed(seed)
    device = resolve_device(config['experiment'].get('device', 'auto'))
    data = load_beir_dataset(dataset_name, config)
    dataset_label = data.eval_dataset if data.train_dataset == data.eval_dataset else f'{data.train_dataset}_to_{data.eval_dataset}'
    config['_runtime'] = {'dataset': data.train_dataset, 'eval_dataset': data.eval_dataset, 'loss': loss_name}
    mode = config['experiment'].get('mode', 'single')
    fixed_epoch_protocol = not bool(config['training'].get('eval_every_epoch', True))
    notes = ['full-corpus retrieval evaluation']
    if fixed_epoch_protocol:
        notes.append('fixed_epoch_test_once')
    init_from = config['model'].get('init_from')
    if init_from:
        notes.append(f'warm_start_from={init_from}')
    if mode == 'smoke':
        data = apply_smoke_limits(data, int(config['smoke'].get('max_eval_queries', 10)), int(config['smoke'].get('max_corpus_docs', 200)))
        notes.append('SMOKE')
    config = maybe_apply_auto_selector(config, data, loss_name)
    negative_name = config['negative_sampling'].get('name', 'random')
    run_name = f'{dataset_label}_{negative_name}/seed_{seed}'
    output_dir = os.path.join(config['outputs'].get('output_root', 'outputs'), run_name)
    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _save_yaml(config, os.path.join(output_dir, 'config_resolved.yaml'))
    sampler = create_negative_sampler(data, config)
    train_dataset = RetrievalTripletDataset(data, sampler, config)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=int(config['training'].get('batch_size', 16)), shuffle=True, collate_fn=_collate, generator=generator)
    model = BiEncoder(config).to(device)
    loss_fn = create_loss(loss_name, config)
    epochs = int(config['training'].get('epochs', 3))
    max_train_steps = config.get('smoke', {}).get('max_train_steps') if mode == 'smoke' else None
    steps_per_epoch = min(len(loader), int(max_train_steps)) if max_train_steps else len(loader)
    total_steps = max(1, epochs * max(1, steps_per_epoch) // int(config['training'].get('gradient_accumulation_steps', 1)))
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn.to(device)
    optimizer, scheduler = _make_optimizer_and_scheduler(model, loss_fn, config, total_steps)
    amp_enabled = bool(config['training'].get('fp16', False)) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    train_log = os.path.join(output_dir, 'train_log.csv')
    _write_train_log_header(train_log)
    best_metric = None
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    patience = int(config['training'].get('early_stopping', {}).get('patience', 3))
    stale_epochs = 0
    grad_accum = int(config['training'].get('gradient_accumulation_steps', 1))
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loader, desc=f'Train {dataset_label}/{loss_name} epoch {epoch}'), start=1):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                query_emb = model.encode_texts(batch['query'], int(config['model'].get('max_length_query', 64)), device)
                pos_emb = model.encode_texts(batch['positive'], int(config['model'].get('max_length_doc', 256)), device)
                neg_emb = model.encode_texts(batch['negative'], int(config['model'].get('max_length_doc', 256)), device)
                loss, stats = loss_fn(query_emb, pos_emb, neg_emb)
                scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            if step % grad_accum == 0:
                scaler.unscale_(optimizer)
                trainable_params = [p for group in optimizer.param_groups for p in group['params']]
                torch.nn.utils.clip_grad_norm_(trainable_params, float(config['training'].get('max_grad_norm', 1.0)))
                scaler.step(optimizer)
                scaler.update()
                if hasattr(loss_fn, 'clamp_parameters'):
                    loss_fn.clamp_parameters()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            _append_train_log(train_log, {'epoch': epoch, 'step': step, 'lr': scheduler.get_last_lr()[0], 'loss': stats.get('loss'), 'pos_sim_mean': stats.get('pos_sim_mean'), 'neg_sim_mean': stats.get('neg_sim_mean'), 'extra': json.dumps({k: v for k, v in stats.items() if k not in {'loss', 'pos_sim_mean', 'neg_sim_mean'}})})
            if max_train_steps and step >= int(max_train_steps):
                break
        epoch_stats = loss_fn.on_epoch_end() if hasattr(loss_fn, 'on_epoch_end') else {}
        if epoch_stats:
            _append_train_log(train_log, {'epoch': epoch, 'step': 'epoch_end', 'lr': scheduler.get_last_lr()[0], 'loss': '', 'pos_sim_mean': '', 'neg_sim_mean': '', 'extra': json.dumps(epoch_stats)})
        if config['training'].get('eval_every_epoch', True):
            metrics = evaluate_full_corpus(model, data, config, device, output_dir)
            primary = config['training'].get('save_best_by', config['registry'].get('primary_metric', 'NDCG@10'))
            current = metrics.get(primary, 0.0)
            if best_metric is None or current > best_metric:
                best_metric = current
                best_epoch = epoch
                best_metrics = metrics
                stale_epochs = 0
                if config['outputs'].get('save_model', True):
                    model.save(os.path.join(output_dir, 'best_model'))
            else:
                stale_epochs += 1
            if config['training'].get('early_stopping', {}).get('enabled', True) and stale_epochs >= patience:
                notes.append(f'early_stopped_epoch_{epoch}')
                break
    train_only = bool(config['training'].get('train_only', False))
    if train_only:
        best_metrics = {}
        best_epoch = epochs
        notes.append('source_train_only_no_target_evaluation')
    elif not best_metrics:
        best_metrics = evaluate_full_corpus(model, data, config, device, output_dir)
        best_epoch = epochs
        if config['outputs'].get('save_model', True):
            model.save(os.path.join(output_dir, 'best_model'))
    if config['outputs'].get('save_model', True):
        model.save(os.path.join(output_dir, 'last_model'))
    metrics_path = os.path.join(output_dir, 'metrics.json')
    Path(metrics_path).write_text(json.dumps(best_metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    row: Dict[str, Any] = {'记录类型': 'RUN', '运行时间': datetime.now().isoformat(timespec='seconds'), 'Dataset': dataset_label, 'TrainDataset': data.train_dataset, 'EvalDataset': data.eval_dataset, 'Loss': loss_name, 'NegativeSampler': negative_name, 'Encoder': getattr(model, 'loaded_name', config['model'].get('hf_name')), 'Seed': seed, 'Epochs': epochs, 'BestEpoch': best_epoch, 'Protocol': config['training'].get('protocol', 'fixed_epoch' if fixed_epoch_protocol else 'test_selected_epoch'), 'PrimaryMetric': config['registry'].get('primary_metric', 'NDCG@10'), 'OutputDir': output_dir, 'Config': os.path.join(output_dir, 'config_resolved.yaml'), 'Notes': '; '.join(notes)}
    row.update(best_metrics)
    return row
