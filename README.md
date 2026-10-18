# BORO

This repository contains the BORO code, experiment configurations, source tables, and selected outputs for four studies: adverse drug event detection, proteoform–spectrum match (PrSM) rescoring, dense retrieval, and controlled boundary experiments.

## Data and downloads

Large raw inputs, downloaded collections, and pretrained weights are not tracked in Git. [`docs/data_sources.md`](docs/data_sources.md) lists their sources, expected paths, and preparation steps. The local data snapshot used for the current analyses is recorded in [`data/manifest.json`](data/manifest.json); it stores file sizes and SHA-256 digests for files under `data/raw`, `data/processed`, `data/pretrained`, and `data/source`.

Run the following commands from the repository root after installing the environment in [`docs/reproducibility.md`](docs/reproducibility.md).

### Download public resources

`download_data.py` fetches the supported BEIR collections and the E5 model files. It does not fetch the ADE or PrSM inputs, whose exact local files and access conditions are described in the data-source document.

```bash
# Download all supported BEIR collections and intfloat/e5-small-v2.
python scripts/download_data.py --resource all

# Download only the collections needed for an experiment.
python scripts/download_data.py --resource beir --datasets scifact fiqa
python scripts/download_data.py --resource e5

# Preview downloads and extraction without network or filesystem changes.
python scripts/download_data.py --resource all --dry-run
```

### Prepare derived inputs

`prepare_data.py` generates the derived ADE and controlled splits, builds the PrSM feature cache from local XML files and feature models, and prepares BEIR directories. Existing outputs are skipped rather than overwritten.

```bash
python scripts/prepare_data.py --task ade --seeds 20260417
python scripts/prepare_data.py --task controlled --seeds 7103 7121
python scripts/prepare_data.py --task prsm
python scripts/prepare_data.py --task ir --ir-datasets scifact

# Run every preparation family for which the required inputs are available.
python scripts/prepare_data.py --task all

# Preview all preparation actions without downloading or writing files.
python scripts/prepare_data.py --task all --dry-run
```

The manifest is a snapshot rather than a live index. To check an individual entry, compute `shasum -a 256 <path>` and compare it with the corresponding `sha256` value in `data/manifest.json`. The manifest excludes itself from the hash list.

## Reproduction

Run the commands from the repository root after installing the environments described in [`docs/reproducibility.md`](docs/reproducibility.md):

```bash
export PYTHONPATH="$PWD/src"
python -m boro.ade.train --output-dir "$RUN_OUTPUT/ade"
python -m boro.prsm.train --seed 8256 --output-dir "$RUN_OUTPUT/prsm"
python -m boro.ir.train --config experiments/ir/in_domain.yaml \
  'datasets.selected=["scifact"]' "outputs.output_root=$RUN_OUTPUT/ir"
python -m boro.controlled.train --seeds 7103 --output-dir "$RUN_OUTPUT/controlled"
```

The runnable training and evaluation paths in this checkout implement BORO. The files under `data/source` are fixed inputs used by the analysis and figure scripts. The `results` directory retains the predictions, metrics, and logs used for the reported reference runs.

## Citation

When citing a released version, use the repository commit or release tag that contains the code, configurations, source tables, and any result or checkpoint archives used for the reported analysis.
