from __future__ import annotations
import json
from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import prsm_matrix_style as base
METHODS = ['BCE', 'Focal', 'Class-Balanced', 'IMMAX', 'pAUC', 'STACO', 'BORO']
INPUT = Path(__file__).resolve().parents[3] / 'data/source/prsm/file_summary.csv'

def load_matrix() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], list[str]]:
    data = pd.read_csv(INPUT)
    assert data.shape == (35 * len(METHODS), 11)
    assert set(data['method']) == set(METHODS)
    assert data['n_seeds'].eq(5).all()
    data['plot_group'] = data['species_group']
    human = data['species_group'].eq('Human')
    data.loc[human & data['file_id'].str.startswith('Human_0'), 'plot_group'] = 'Human_0'
    data.loc[human & data['file_id'].str.startswith('Human_E'), 'plot_group'] = 'Human_E'
    data.loc[human & data['file_id'].str.startswith('Human_L'), 'plot_group'] = 'Human_L'
    assert set(data['plot_group']) == set(base.GROUPS)
    file_meta = data[['file_order', 'file_id', 'plot_group']].drop_duplicates()
    file_meta['group_order'] = pd.Categorical(file_meta['plot_group'], categories=base.GROUPS, ordered=True)
    file_meta = file_meta.sort_values(['group_order', 'file_order'])
    assert len(file_meta) == 35
    file_ids = file_meta['file_id'].tolist()
    file_groups = file_meta['plot_group'].tolist()
    mean = data.pivot(index='method', columns='file_id', values='additional_target_prsms_mean').reindex(index=METHODS, columns=file_ids).to_numpy()
    sd = data.pivot(index='method', columns='file_id', values='additional_target_prsms_sd').reindex(index=METHODS, columns=file_ids).to_numpy()
    assert np.isfinite(mean).all() and np.isfinite(sd).all()
    return (data, mean, sd, file_ids, file_groups)
