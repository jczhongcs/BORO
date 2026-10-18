
from pathlib import Path
import argparse
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from . import core

ROOT = Path(__file__).resolve().parents[3]

def accepted(labels, scores, threshold, ascending=False):
    order = np.argsort(scores if ascending else -scores, kind="stable")
    targets = np.cumsum(labels[order])
    decoys = np.cumsum(1-labels[order])
    valid = np.flatnonzero((targets > 0) & (decoys <= threshold * targets))
    return order[:int(valid[-1])+1] if len(valid) else order[:0]

def load_model(path):
    sys.modules["RERANKPRSM_upgrade_train"] = core
    for name in ("ResNeXt", "ResNeXtBlock", "ResidualBlock"):
        setattr(sys.modules["__main__"], name, getattr(core, name))
    return joblib.load(path)

def score_files(model_path, cache_dir, files_path):
    model = load_model(model_path)
    rows = []
    for item in pd.read_csv(files_path).to_dict("records"):
        with np.load(cache_dir/f"{item['file_id']}.npz", allow_pickle=False) as data:
            with torch.no_grad():
                scores = model(torch.tensor(data["inputs"], dtype=torch.float32)).cpu().numpy().reshape(-1)
            rows.append(pd.DataFrame({
                "file_order":item["file_order"], "file_id":item["file_id"],
                "species_group":item["species_group"],
                "generalization_regime":item["generalization_regime"],
                "prsm_id":data["prsm_ids"], "is_target":data["labels"],
                "model_score":scores, "toppic_e_value":data["evalues"],
                "toppic_xml_fdr":data["xml_fdr"],
                "matched_peaks_scaled":data["raw"][:,0],
                "matched_fragments_scaled":data["raw"][:,1],
                "normalized_matched_fragments":data["raw"][:,2],
                "p_value":data["raw"][:,3]}))
    return pd.concat(rows,ignore_index=True)

def summarize(data):
    rows=[]
    for file_id,frame in data.groupby("file_id",sort=False):
        y=frame.is_target.to_numpy(int)
        for threshold in (.005,.01,.02):
            kept=accepted(y,frame.model_score.to_numpy(),threshold)
            reference=accepted(y,frame.toppic_e_value.to_numpy(),threshold,True)
            n=int(y[kept].sum()); d=int(len(kept)-n); base=int(y[reference].sum())
            rows.append(dict(file_id=file_id,fdr=threshold,targets=n,decoys=d,
                             toppic_targets=base,additional_targets=n-base))
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--cache",type=Path,default=ROOT/"data/processed/prsm/test")
    parser.add_argument("--files",type=Path,default=ROOT/"data/processed/prsm/test_files.csv")
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    torch.set_num_threads(4)
    data=score_files(args.model,args.cache,args.files)
    args.output_dir.mkdir(parents=True)
    data.to_csv(args.output_dir/"scores.csv.gz",index=False,compression="gzip")
    summarize(data).to_csv(args.output_dir/"counts.csv",index=False)

if __name__=="__main__":
    main()
