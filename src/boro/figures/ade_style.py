from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd
ORDER = ['BORO', 'MR-SVM', 'Focal', 'ERO-FPOR', 'E2SC_US', 'BCE-NoSel', 'WCE', 'UBR', 'SigmoidF1', 'TFCO-FPOR']
RED, INK, GREY = ('#BE3857', '#252E38', '#737F8D')
COLORS = {'BORO': RED, 'MR-SVM': '#446783', 'Focal': '#709EB1', 'ERO-FPOR': '#8C83A5', 'WCE': '#8397AD', 'BCE-NoSel': '#A7ABB1', 'E2SC_US': '#73968C', 'UBR': '#ABA490', 'SigmoidF1': '#A6A0B5', 'TFCO-FPOR': '#B4BBC2'}
STYLES = {'BORO': '-', 'MR-SVM': '--', 'Focal': '-.', 'ERO-FPOR': ':', 'WCE': (0, (5, 2)), 'BCE-NoSel': (0, (1, 2)), 'E2SC_US': '-', 'UBR': '--', 'SigmoidF1': '-.', 'TFCO-FPOR': ':'}

def setup():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 6.5, 'axes.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.55, 'axes.edgecolor': '#6E7A84', 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.2, 'ytick.major.size': 2.2, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'path.simplify': False})
