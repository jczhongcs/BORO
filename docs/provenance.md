# Data and experiment provenance

This file summarizes the inputs and run identifiers used by the current source tables. Paths in the documents and result JSON files are relative to the repository root.

## Analysis units

| Task | Run identifiers | Inputs |
|---|---|---|
| ADE | 20260417–20260421 and 20260919–20260923 | ADE train/test JSONL files and the stored split files |
| PrSM | 8256, 1729, 3407, 4969, 7919 | 12 training XML files, 35 test XML files, and feature caches |
| IR | 42–46 | SciFact, FiQA, NFCorpus, and MS MARCO-to-target transfer settings |
| Controlled | 7103, 7121, 7151, 7177, 7207, 7237, 7253, 7283, 7307, 7331, 7351, 7387, 7411, 7433, 7457, 7481, 7507, 7529, 7559, 7583 | Generated train, validation, and test arrays |

ADE uses ten seeds for the main analysis and seeds 20260417–20260421 for component and candidate-localization analyses.

The PrSM manifests contain 40,492 training candidates and 90,188 test candidates. The four saved feature models are part of the feature construction used by BORO.

The IR source tables cover nine settings and five seeds. The in-domain settings combine SciFact, FiQA, and NFCorpus with random or BM25 negatives. The transfer settings train on MS MARCO and evaluate the target domains.

The controlled datasets use world seed 2026072907. Each dataset JSON records its sampling seed and initialization seed.

## Configuration records

ADE parameters are in `experiments/ade/config.json`; checkpoints are selected with validation metrics. The default arguments of the training entry point match that configuration.

PrSM parameters are in `experiments/prsm/config.json`. The main BORO runs use five seeds and 50 training epochs.

IR parameters are in `experiments/ir/in_domain.yaml` and `experiments/ir/transfer.yaml`. Resolved configurations are saved with each run. Transfer evaluation loads the source-domain checkpoint and evaluates the target domain without target-domain training.

Controlled data and training parameters are in `experiments/controlled/config.json`. The generated arrays and their metadata are stored under `data/processed/controlled`.
