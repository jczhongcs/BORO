# Data sources and downloads

This document lists the local inputs, public sources, and generated data used by BORO. Large raw inputs, downloaded collections, and pretrained weights are intentionally kept out of Git; the local snapshot used for the reported analyses is recorded in `data/manifest.json`. All paths below are relative to the repository root.

## Obtaining and preparing inputs

Run these commands from the repository root after installing the environment described in `docs/reproducibility.md`.

`download_data.py` covers the public resources that have stable direct download endpoints:

```bash
# All supported BEIR collections and intfloat/e5-small-v2.
python scripts/download_data.py --resource all

# A selected set of BEIR collections or only the E5 files.
python scripts/download_data.py --resource beir --datasets scifact fiqa
python scripts/download_data.py --resource e5

# Show the planned actions without downloading, extracting, or writing files.
python scripts/download_data.py --resource all --dry-run
```

The BEIR command writes archives and extracted collections below `data/raw/ir`. The E5 command writes the model files below `data/pretrained/e5`. Existing files are skipped. The downloader prints a SHA-256 digest for each newly downloaded file, but it does not rewrite the checked-in snapshot manifest.

`prepare_data.py` handles the derived inputs used by the experiments:

```bash
# ADE split files for one or more seeds.
python scripts/prepare_data.py --task ade --seeds 20260417

# Deterministic controlled datasets from experiments/controlled/config.json.
python scripts/prepare_data.py --task controlled --seeds 7103 7121

# PrSM feature cache from the local XML files and feature models.
python scripts/prepare_data.py --task prsm

# BEIR directory preparation; missing collections are downloaded through the same endpoint.
python scripts/prepare_data.py --task ir --ir-datasets scifact

# Preview all available preparation actions without writing or downloading.
python scripts/prepare_data.py --task all --dry-run
```

ADE preparation requires the local `data/raw/ade/train.jsonl` and `test.jsonl`. PrSM preparation requires the local candidate XML files, the two file manifests under `data/processed/prsm`, and the feature models under `data/pretrained/prsm`. The controlled task is generated locally from the committed configuration. Use `--task all` only after the required ADE and PrSM inputs are available. Existing derived outputs are skipped rather than overwritten; remove a target yourself when a rebuild is intended.

## Version and SHA-256 manifest

`data/manifest.json` uses schema `boro-data-manifest-v1` and currently labels the snapshot `boro-paper-snapshot-2026-10-18`. Each entry records a project-relative path, category, byte size, and SHA-256 digest. The current snapshot covers files under `data/raw`, `data/processed`, `data/pretrained`, and `data/source`; the manifest file itself is excluded. The manifest is a record of the data used for a result set, so replacing an input requires a new snapshot and a new manifest commit.

To verify the complete manifest without downloading anything:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('.')
manifest = json.loads((root / 'data/manifest.json').read_text())
for item in manifest['files']:
    path = root / item['path']
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    if path.stat().st_size != item['size_bytes'] or digest.hexdigest() != item['sha256']:
        raise SystemExit(f'checksum mismatch: {item["path"]}')
print(f"verified {len(manifest['files'])} files")
PY
```

## ADE detection

The ADE copy is in `data/raw/ade/train.jsonl` and `data/raw/ade/test.jsonl`. It is a binary classification version of ADE Corpus V2 (Gurulingappa et al., 2012). An upstream mirror is available at [ADE Benchmark Corpus v2](https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2). The local JSONL files preserve the splits and label fields used by `boro.ade`; downloading the upstream corpus alone does not recreate these files.

## PrSM rescoring

The candidate XML files used by the experiments are in `data/raw/prsm/train` and `data/raw/prsm/test`. File mappings and labels are recorded in `data/processed/prsm/train_files.csv` and `data/processed/prsm/test_files.csv`; both manifests include SHA-256 hashes. The original vendor mass-spectrometry files are not included.

The paper uses the dataset label `FB_TO`. The local files use the historical identifier `FB_TeO`, so `FB_TO_1`–`FB_TO_3` correspond to `FB_TeO_1.xml`–`FB_TeO_3.xml`.

| Dataset label | Local identifier | Subsets | Upstream access |
|---|---|---|---|
| `FB_TO` | `FB_TeO` | `FB_TO_1`–`FB_TO_3` | Zebrafish download: [`https://bit.ly/3pkGB03`](https://bit.ly/3pkGB03) |
| `Human_L` | `Human_L` | `Human_L1`–`Human_L2` | [PRIDE PXD026178](https://www.ebi.ac.uk/pride/archive/projects/PXD026178) |
| `Human_0` | `Human_0` | `Human_01`–`Human_04` | [PRIDE PXD026128](https://www.ebi.ac.uk/pride/archive/projects/PXD026128) |
| `Human_E` | `Human_E` | `Human_E1`–`Human_E2` | [PRIDE PXD029703](https://www.ebi.ac.uk/pride/archive/projects/PXD029703) |
| `AH` | `AH` | `AH_1`–`AH_6` | [PRIDE PXD042298](https://www.ebi.ac.uk/pride/archive/projects/PXD042298) |
| `TM` | `TM` | `TM_1`–`TM_2` | [PRIDE PXD018772](https://www.ebi.ac.uk/pride/archive/projects/PXD018772) |
| `PS` | `PS` | `PS_1`–`PS_5` | [PRIDE PXD017382](https://www.ebi.ac.uk/pride/archive/projects/PXD017382) |
| `AT` | `AT` | `AT_1`–`AT_5` | [PRIDE PXD034368](https://www.ebi.ac.uk/pride/archive/projects/PXD034368) |
| `Yeast` | `Yeast` | `Yeast_1`–`Yeast_6` | [PRIDE PXD046651](https://www.ebi.ac.uk/pride/archive/projects/PXD046651) |

The same dataset labels and subset names are listed in `data/source/prsm/datasets.csv`.

## Dense retrieval

The IR loader downloads a collection when its directory is absent from `data/raw/ir`. The download URL is:

```text
https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip
```

| Collection | Local archive | Download |
|---|---|---|
| MS MARCO | `data/raw/ir/msmarco.zip` (on demand) | [`msmarco.zip`](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip) |
| ArguAna | `data/raw/ir/arguana.zip` (on demand) | [`arguana.zip`](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip) |
| DBPedia-Entity | `data/raw/ir/dbpedia-entity.zip` (on demand) | [`dbpedia-entity.zip`](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip) |
| SciDocs | `data/raw/ir/scidocs.zip` (on demand) | [`scidocs.zip`](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip) |
| SciFact | `data/raw/ir/scifact.zip` | [`scifact.zip`](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip) |
| FiQA | `data/raw/ir/fiqa.zip` | [`fiqa.zip`](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip) |
| NFCorpus | `data/raw/ir/nfcorpus.zip` | [`nfcorpus.zip`](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip) |

The relevance files used for evaluation are in `data/processed/ir/qrels`.

## Controlled experiments

The controlled datasets are generated from `experiments/controlled/config.json`. The 20 generated datasets are in `data/processed/controlled`; their JSON metadata record the world, sampling, and initialization seeds.

## Pretrained models

The E5 tokenizer and weights are in `data/pretrained/e5`; the upstream model page is [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2). The four classical PrSM feature models are in `data/pretrained/prsm`.

## Release contents

A release should include the code, configurations, manifests, source tables, and any result or checkpoint files needed for the analyses being reported. The paper should cite the commit or release tag used for the analysis.
