from __future__ import annotations
import json
import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image
METHODS = ['BCE', 'Focal', 'Class-Balanced', 'pAUC', 'IMMAX', 'BORO']
COLORS = {'BCE': '#9B6A9A', 'Focal': '#B69AC5', 'Class-Balanced': '#AFC2D6', 'pAUC': '#78AFC7', 'IMMAX': '#5E9E8D', 'BORO': '#D74370'}
BORO_BAND = '#FBF0F4'
INK = '#262626'
MID_GREY = '#6F6F6F'
GUIDE = '#E3E6E9'
SEEDS = [1729, 3407, 4969, 7919, 8256]
SPECIES_ORDER = ['Human', 'FB_TeO', 'PS', 'TM', 'AH', 'AT', 'Yeast']
SPECIES_DISPLAY = {'Human': ('Human', 'Homo sapiens', 'seen'), 'FB_TeO': ('FB–TeO', 'Danio rerio', 'seen'), 'PS': ('PS', 'Pisum sativum', 'unseen'), 'TM': ('TM', 'Tenebrio molitor', 'unseen'), 'AH': ('AH', 'Mus musculus', 'unseen'), 'AT': ('AT', 'Arabidopsis thaliana', 'unseen'), 'Yeast': ('Yeast', 'Saccharomyces cerevisiae', 'unseen')}

def apply_style() -> None:
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 6.2, 'axes.titlesize': 6.9, 'axes.labelsize': 6.4, 'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'legend.fontsize': 5.6, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5, 'xtick.major.width': 0.65, 'ytick.major.width': 0.65, 'xtick.major.size': 2.3, 'ytick.major.size': 0, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
