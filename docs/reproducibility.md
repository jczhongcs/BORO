# Reproduction notes

Commands below assume the repository root as the working directory. The BORO package is under `src/boro` and experiment configurations are under `experiments`.

```bash
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## Environments

ADE, controlled experiments, and figure generation use Python 3.12 with NumPy 1.26.4, SciPy 1.13.1, pandas 2.2.2, scikit-learn 1.5.1, joblib 1.4.2, and PyTorch 2.8.0.

PrSM feature reconstruction uses Python 3.9.21 with NumPy 2.0.2, SciPy 1.13.1, pandas 2.2.3, scikit-learn 1.5.2, joblib 1.4.2, XGBoost 2.1.1, and PyTorch 2.5.1.

IR production runs use Python 3.8.10, PyTorch 2.4.1+cu121, Transformers 4.46.3, FAISS 1.8.0, NumPy 1.24.4, pandas 2.0.3, and rank_bm25 on an NVIDIA RTX 4090 D. In-domain runs use 32-bit floating point; MS MARCO source training uses mixed precision. E5 weights and the tokenizer are in `data/pretrained/e5`. Full-corpus retrieval uses FAISS `IndexFlatIP` when enabled and falls back to exact PyTorch blockwise inner products when FAISS is unavailable.

## Data and outputs

The data map and download locations are in [`data_sources.md`](data_sources.md).

- `data/raw`: ADE JSONL files, PrSM XML files, and the included BEIR archives.
- `data/processed`: ADE splits and features, PrSM feature caches, IR negatives and qrels, and controlled datasets.
- `data/pretrained`: PrSM feature models and E5 weights.
- `data/source`: Source tables used by the figure scripts.
- `results`: predictions, metrics, logs, figures, and selected checkpoints for BORO and the stored reference runs.

IR transfer runs need the MS MARCO, ArguAna, DBPedia-Entity, and SciDocs collections. The loader downloads a missing collection from the BEIR endpoint. The result directories contain resolved configurations and, where available, model checkpoints. Only BORO training and evaluation code is included in this checkout; stored reference outputs are retained for the reported analyses.

## Training and evaluation

`RUN_OUTPUT` below is an output directory chosen by the user; it keeps new runs separate from the supplied result directories.

```bash
python -m boro.ade.train --output-dir "$RUN_OUTPUT/ade"
python -m boro.ade.ablation --variants uniform_weights no_boundary_local_selection hard_example_matched --output-dir "$RUN_OUTPUT/ade_ablation"
python -m boro.prsm.train --seed 8256 --output-dir "$RUN_OUTPUT/prsm"
python -m boro.prsm.evaluate --model results/prsm/boro/seed_8256/model.pkl --output-dir "$RUN_OUTPUT/prsm_evaluation"
python -m boro.prsm.prepare --output-dir "$RUN_OUTPUT/prsm_features"
```

ADE uses ten seeds for the main analysis and five seeds for component analyses. The PrSM BORO entry point supports the five seeds listed in `docs/provenance.md`; feature reconstruction uses the four models in `data/pretrained/prsm` and produces eight input features. PrSM statistics are recomputed from the model score, TopPIC e-value, and target/decoy label using a stable sort and the requested FDR.

For ADE and controlled experiments, the code reports both the evaluated-set operating point and the validation-threshold transfer fields where available. The former measures the best operating point on the evaluated set; the latter applies a threshold selected on validation data to the test set.

### IR in-domain example

```bash
python -m boro.ir.train --config experiments/ir/in_domain.yaml 'datasets.selected=["scifact"]' experiment.seed=42 "outputs.output_root=$RUN_OUTPUT/ir"
```

Set `datasets.selected` to `scifact`, `fiqa`, or `nfcorpus`, and use seeds 42–46. Random and BM25 negatives are separate settings; BM25 uses `negative_sampling.name=bm25` and `negative_sampling.use_bm25=true`.

### IR source training

```bash
python -m boro.ir.train --config experiments/ir/transfer.yaml experiment.seed=42 "outputs.output_root=$RUN_OUTPUT/transfer"
```

For a target-domain evaluation, start from the resolved source configuration, set `datasets.selected` and `datasets.eval_split=test`, use `training.epochs=0`, `training.train_only=false`, `selector.enabled=false`, and `outputs.save_model=false`, and point `model.init_from` to the source-domain `last_model`. Keep the loss settings from source training and use a separate output directory for each target domain.

### Controlled experiments

```bash
python -m boro.controlled.train --seeds 7103 --output-dir "$RUN_OUTPUT/controlled"
```

Pass the 20 seeds listed in `docs/provenance.md` to reproduce the supplied repeats. The default configuration uses a precision target of 0.95, learning rate 0.001, 45 training passes, evaluation every five passes, and batch size 256. Data-generation code is in `boro.controlled.data`; the world seed and other seeds are in `experiments/controlled/config.json` and the generated JSON files.

## Figure inputs

Set `BORO_FIGURE_OUTPUT=results/figures` before running a figure script. The scripts use the following source tables:

| Figure | Entry point | Source data |
|---|---|---|
| Figure 2 | `boro.figures.fig2` | `data/source/controlled` |
| Figure 3 | `boro.figures.fig3` | `data/source/ade` |
| Figure 4 | `boro.figures.fig4` | `data/source/prsm` |
| Figure 5 | `boro.figures.fig5` | `data/source/ir` |
| Figure 6 | `boro.figures.fig6` | Boundary-analysis tables |
| Extended Data Figure 1 | `boro.figures.ed2` | PrSM file summaries |
| Extended Data Figure 2 | `boro.figures.ed3` | PrSM entity summaries |
| Extended Data Figure 3 | `boro.figures.ed4` | PrSM feature summaries |
| Extended Data Figure 4 | `boro.figures.boundary_controls` | ADE and PrSM ablation tables |
