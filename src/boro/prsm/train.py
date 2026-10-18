
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from . import core

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (8256, 1729, 3407, 4969, 7919)

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.data = data["inputs"].copy()
            self.label = data["labels"].copy()
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        return self.data[index], self.label[index], index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, default=8256)
    parser.add_argument("--config", type=Path, default=ROOT/"experiments/prsm/config.json")
    parser.add_argument("--data", type=Path, default=ROOT/"data/processed/prsm/train.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument('--method', choices=('boro',), default='boro')
    args = parser.parse_args()
    run = args.output_dir/f"seed_{args.seed}"
    if run.exists():
        raise FileExistsError(run)
    request = json.loads(args.config.read_text())
    data = CachedDataset(args.data)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    core.seed = args.seed
    core.num_epochs = request["epochs"]
    core.set_global_seed(args.seed)
    run.mkdir(parents=True)
    core.train_model(data, str(run/"model.pkl"), request["loss"], str(run),
                     loss_method='boro', optimizer_method='adam')
    (run/"config.json").write_text(json.dumps({**request, "seed":args.seed, "method":args.method},indent=2)+"\n")

if __name__ == "__main__":
    main()
