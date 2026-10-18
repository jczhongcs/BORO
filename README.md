# BORO

This repository contains the BORO code, experiment configurations, source tables, and selected outputs for four studies: adverse drug event detection, proteoform–spectrum match (PrSM) rescoring, dense retrieval, and controlled boundary experiments.

## Data and downloads

[`docs/data_sources.md`](docs/data_sources.md) lists the data included in this checkout and the collections downloaded when an experiment needs them. The PrSM XML files are TopPIC-derived candidate files; the original vendor mass-spectrometry files are not included.

The ADE, PrSM, and controlled analyses use the local data and outputs under `data` and `results`. The IR experiments need the BEIR collections listed in the data-source document; the loader downloads a missing collection from the BEIR endpoint.

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
