from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import yaml
from .selector import apply_selector_to_config
from .core import run_experiment

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['_config_dir'] = str(Path(path).resolve().parent)
    root = Path(__file__).resolve().parents[3]
    for section, keys in (('datasets', ('data_root', 'cache_dir')), ('outputs', ('output_root',))):
        for key in keys:
            if key in config[section] and (not Path(config[section][key]).is_absolute()):
                config[section][key] = str(root / config[section][key])
    bm25 = config['negative_sampling'].get('bm25', {})
    if 'cache_path' in bm25 and (not Path(bm25['cache_path']).is_absolute()):
        bm25['cache_path'] = str(root / bm25['cache_path'])
    if config['model']['hf_name'].startswith('data/'):
        config['model']['hf_name'] = str(root / config['model']['hf_name'])
    return config

def parse_override(raw: str) -> tuple[List[str], Any]:
    if '=' not in raw:
        raise ValueError(f'Override must be key=value, got: {raw}')
    key, value = raw.split('=', 1)
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed = value
    return (key.split('.'), parsed)

def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    for raw in overrides:
        keys, value = parse_override(raw)
        cursor = cfg
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = value
    return cfg

def selected_runs(config: Dict[str, Any]) -> List[tuple[str, str]]:
    datasets = list(config['datasets'].get('selected', []))
    losses = list(config['losses'].get('selected', []))
    mode = config['experiment'].get('mode', 'single')
    if not datasets:
        raise ValueError('datasets.selected is empty')
    if not losses:
        raise ValueError('losses.selected is empty')
    if mode in {'single', 'smoke'}:
        return [(datasets[0], losses[0])]
    if mode == 'grid':
        return [(dataset, loss) for dataset in datasets for loss in losses]
    raise ValueError(f'Unknown experiment.mode: {mode}')

def main() -> None:
    parser = argparse.ArgumentParser(description='BORO retrieval training')
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    parser.add_argument('overrides', nargs='*', help='Optional dotted overrides, e.g. experiment.seed=42')
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    Path(config['outputs'].get('output_root', 'outputs')).mkdir(parents=True, exist_ok=True)
    runs = selected_runs(config)
    print(f'Planned runs: {json.dumps(runs, ensure_ascii=False)}')
    registry_df = None
    for dataset_name, loss_name in runs:
        run_config = copy.deepcopy(config)
        if run_config.get('datasets', {}).get('train_dataset'):
            run_config['datasets']['eval_dataset'] = dataset_name
        run_config = apply_selector_to_config(run_config, dataset_name, loss_name)
        print(f"\n=== Running dataset={dataset_name} loss={loss_name} negatives={config['negative_sampling'].get('name', 'random')} ===")
        row = run_experiment(run_config, dataset_name, loss_name)
        print(f"Finished: {dataset_name}/{loss_name} -> {row.get(run_config['registry'].get('primary_metric', 'NDCG@10'))}")
if __name__ == '__main__':
    main()
