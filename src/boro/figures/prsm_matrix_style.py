from __future__ import annotations
import json
import os
from pathlib import Path
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
MM = 1 / 25.4
GROUPS = ['Human_0', 'Human_E', 'Human_L', 'FB_TeO', 'PS', 'TM', 'AH', 'AT', 'Yeast']
GROUP_LABELS = {'Human_0': '01–04', 'Human_E': 'E1–E2', 'Human_L': 'L1–L2', 'FB_TeO': 'FB–TeO', 'PS': 'PS', 'TM': 'TM', 'AH': 'AH', 'AT': 'AT', 'Yeast': 'Yeast'}
REGIME = {'Human_0': 'represented', 'Human_E': 'represented', 'Human_L': 'represented', 'FB_TeO': 'represented', 'PS': 'unseen', 'TM': 'unseen', 'AH': 'unseen', 'AT': 'unseen', 'Yeast': 'unseen'}
REGIME_COLORS = {'represented': '#93BED1', 'unseen': '#D7B28A'}
CMAP = mcolors.LinearSegmentedColormap.from_list('signed_prsm', ['#426A92', '#F7F7F7', '#C94B53'], N=256)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 5.5, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

def short_label(file_id: str, group: str) -> str:
    if group.startswith('Human_'):
        return file_id.removeprefix('Human_')
    if group == 'FB_TeO':
        return file_id.removeprefix('FB_TeO_')
    return file_id.removeprefix(f'{group}_')
