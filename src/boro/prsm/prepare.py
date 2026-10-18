import xml.dom.minidom
import math
import numpy as np

def getevalue_and_fdr(name):
    dom = xml.dom.minidom.parse(name)
    e_value = dom.getElementsByTagName('e_value')
    fdr = dom.getElementsByTagName('fdr')
    c = list()
    for i in range(len(e_value)):
        c.append([float(e_value[i].firstChild.data), float(fdr[i].firstChild.data)])
    return np.array(c)

def inputfile_np_withSeq(name):
    dom = xml.dom.minidom.parse(name)
    match_peak_num = dom.getElementsByTagName('match_peak_num')
    match_fragment_num = dom.getElementsByTagName('match_fragment_num')
    norm_match_fragment_num = dom.getElementsByTagName('norm_match_fragment_num')
    p_value = dom.getElementsByTagName('p_value')
    e_value = dom.getElementsByTagName('e_value')
    ori_prec_mass = dom.getElementsByTagName('ori_prec_mass')
    seq_name = dom.getElementsByTagName('seq_name')
    prsm_id = dom.getElementsByTagName('prsm_id')
    target = 0
    decoy = 0
    logits = list()
    label = list()
    prsmid = list()
    for i in range(len(ori_prec_mass)):
        pe = math.log10(float(e_value[i].firstChild.data))
        logits.append([float(match_peak_num[i].firstChild.data) / 34, float(match_fragment_num[i].firstChild.data) / 24, float(norm_match_fragment_num[i].firstChild.data), float(p_value[i].firstChild.data)])
        prsmid.append(int(prsm_id[i].firstChild.data))
        if 'DECOY_' in seq_name[i].firstChild.data:
            label.append(0)
            decoy = decoy + 1
        else:
            label.append(1)
            target = target + 1
    print('filename:', name, 'target:', target, '   decoy:', decoy)
    return (np.array(logits), np.array(label), np.array(prsmid))

def inputfile_np(name):
    dom = xml.dom.minidom.parse(name)
    match_peak_num = dom.getElementsByTagName('match_peak_num')
    match_fragment_num = dom.getElementsByTagName('match_fragment_num')
    norm_match_fragment_num = dom.getElementsByTagName('norm_match_fragment_num')
    p_value = dom.getElementsByTagName('p_value')
    frac_feature_score = dom.getElementsByTagName('frac_feature_score')
    variable_ptm_num = dom.getElementsByTagName('variable_ptm_num')
    unexpected_ptm_num = dom.getElementsByTagName('unexpected_ptm_num')
    e_value = dom.getElementsByTagName('e_value')
    ori_prec_mass = dom.getElementsByTagName('ori_prec_mass')
    seq_name = dom.getElementsByTagName('seq_name')
    spectrum_id = dom.getElementsByTagName('spectrum_id')
    target = 0
    decoy = 0
    logits = list()
    label = list()
    for i in range(len(ori_prec_mass)):
        pe = math.log10(float(e_value[i].firstChild.data))
        logits.append([float(match_peak_num[i].firstChild.data) / 34, float(match_fragment_num[i].firstChild.data) / 24, float(norm_match_fragment_num[i].firstChild.data), float(p_value[i].firstChild.data)])
        if 'DECOY_' in seq_name[i].firstChild.data:
            label.append(0)
            decoy = decoy + 1
        else:
            label.append(1)
            target = target + 1
    print('filename:', name, 'target:', target, '   decoy:', decoy)
    return (np.array(logits), np.array(label))

def build_inputs(raw: np.ndarray, models: dict) -> np.ndarray:
    lr = models['lr'].predict(raw).reshape(-1, 1)
    tree = models['tree'].predict_proba(raw)[:, 1].reshape(-1, 1)
    svm = models['svm'].predict_proba(raw)[:, 0].reshape(-1, 1)
    xgb = models['xgb'].predict_proba(raw)[:, 1].reshape(-1, 1)
    return np.concatenate((raw, lr, tree, svm, xgb), axis=1)
from pathlib import Path
import argparse
import csv
import joblib
ROOT = Path(__file__).resolve().parents[3]

def models(path):
    return {key: joblib.load(path / name) for key, name in [('lr', 'LR.pkl'), ('tree', 'DecTree.pkl'), ('svm', 'svm.pkl'), ('xgb', 'xgboost.pkl')]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=ROOT / 'data/pretrained/prsm')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    fitted = models(args.models)
    args.output_dir.mkdir(parents=True)
    raw_rows = []
    inputs = []
    labels = []
    file_ids = []
    for row in csv.DictReader((ROOT / 'data/processed/prsm/train_files.csv').open()):
        number = int(row['file_order'])
        raw, y = inputfile_np(str(ROOT / f'data/raw/prsm/train/file_{number:02d}.xml'))
        raw_rows.append(raw)
        inputs.append(build_inputs(raw, fitted))
        labels.append(y)
        file_ids.append(np.full(len(y), number))
    np.savez_compressed(args.output_dir / 'train.npz', raw=np.concatenate(raw_rows), inputs=np.concatenate(inputs), labels=np.concatenate(labels), file_ids=np.concatenate(file_ids))
    (args.output_dir / 'test').mkdir()
    for row in csv.DictReader((ROOT / 'data/processed/prsm/test_files.csv').open()):
        path = ROOT / 'data/raw/prsm/test' / (row['file_id'] + '.xml')
        raw, y, ids = inputfile_np_withSeq(str(path))
        efdr = getevalue_and_fdr(str(path))
        np.savez_compressed(args.output_dir / 'test' / (row['file_id'] + '.npz'), raw=raw, labels=y, prsm_ids=ids, evalues=efdr[:, 0], xml_fdr=efdr[:, 1], inputs=build_inputs(raw, fitted))
if __name__ == '__main__':
    main()
