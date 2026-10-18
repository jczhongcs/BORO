# Data sources and downloads

This document lists the data included in the repository and the data downloaded when an experiment needs them. All paths are relative to the repository root.

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
