from __future__ import annotations
import os
import glob
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
MM_TO_IN = 1 / 25.4
BLACK = '#242424'
GREY = '#747474'
TOPPIC_FILL = '#7099BD'
TOPPIC_EDGE = '#7099BD'
BORO_FILL = '#D99B55'
BORO_EDGE = '#D99B55'
GROUPS = ['TopPIC-only', 'BORO-only']
METRICS = ['Matched peaks', 'Matched fragments', 'Normalized fragments']
METRIC_LABELS = ['Peaks', 'Fragments', 'Normalized\nfragment count']
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 7, 'axes.titlesize': 7.5, 'axes.titleweight': 'bold', 'axes.labelsize': 7, 'xtick.labelsize': 6.7, 'ytick.labelsize': 6.7, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})

def draw_box(ax: plt.Axes, values: np.ndarray, position: float, fill: str, edge: str) -> None:
    artists = ax.boxplot([values], positions=[position], widths=0.3, patch_artist=True, showfliers=False, whis=1.5, medianprops={'color': BLACK, 'linewidth': 1.05}, boxprops={'facecolor': 'white', 'edgecolor': BLACK, 'linewidth': 0.95}, whiskerprops={'color': BLACK, 'linewidth': 0.85}, capprops={'color': BLACK, 'linewidth': 0.85})
    artists['boxes'][0].set_alpha(1.0)

def make_figure(source: pd.DataFrame, selected: list[str]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM_TO_IN, 132 * MM_TO_IN))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.82, wspace=0.22, hspace=0.55)
    offsets = {'TopPIC-only': -0.18, 'BORO-only': 0.18}
    fills = {'TopPIC-only': TOPPIC_FILL, 'BORO-only': BORO_FILL}
    edges = {'TopPIC-only': TOPPIC_EDGE, 'BORO-only': BORO_EDGE}
    rng = np.random.default_rng(20260721)
    for panel_index, (ax, file_id) in enumerate(zip(axes.ravel(), selected)):
        panel = source.loc[source['file_id'].eq(file_id)]
        organism = panel['organism'].iloc[0]
        regime = panel['generalization_regime'].iloc[0]
        for metric_index, metric in enumerate(METRICS):
            for label in GROUPS:
                values = panel.loc[panel['metric'].eq(metric) & panel['group'].eq(label), 'value'].to_numpy(float)
                position = metric_index + offsets[label]
                draw_box(ax, values, position, fills[label], edges[label])
                point_values = values if len(values) <= 60 else rng.choice(values, size=60, replace=False)
                jitter = rng.uniform(-0.085, 0.085, size=len(point_values))
                ax.scatter(np.full(len(point_values), position) + jitter, point_values, s=8.2, marker='o', facecolor=fills[label], edgecolor='none', alpha=0.68, zorder=3)
        counts = panel.loc[panel['metric'].eq(METRICS[0])].groupby('group').size()
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(-0.5, 44.5)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_xticks(range(3))
        ax.set_xticklabels(METRIC_LABELS)
        ax.tick_params(direction='out', length=2.8, width=0.7)
        if panel_index % 2 == 0:
            ax.set_ylabel('Feature count')
        species_label = organism.replace(' ', r'\ ')
        regime_label = 'Seen species' if regime == 'species-represented held-out' else 'Unseen species'
        ax.set_title(f'{file_id} · $\\mathit{{{species_label}}}$\n{regime_label}', loc='left', pad=7, fontsize=7.2)
        ax.text(-0.13, 1.14, chr(ord('a') + panel_index), transform=ax.transAxes, fontsize=8, fontweight='bold', ha='left', va='bottom')
        ax.text(1.0, 1.02, f"n = {int(counts.get('TopPIC-only', 0))} / {int(counts.get('BORO-only', 0))}", transform=ax.transAxes, fontsize=6.5, color='#4D5055', ha='right', va='bottom')
    fig.legend(handles=[Patch(facecolor=TOPPIC_FILL, edgecolor=TOPPIC_EDGE, label='TopPIC-only'), Patch(facecolor=BORO_FILL, edgecolor=BORO_EDGE, label='BORO-only')], loc='upper center', bbox_to_anchor=(0.53, 0.975), ncol=2, columnspacing=1.8, handlelength=1.8, fontsize=7.0)
    return fig

def main():
    root = Path(__file__).resolve().parents[3]
    out = Path(os.environ.get('BORO_FIGURE_OUTPUT', str(root / 'results/figures')))
    data = pd.read_csv(root / 'data/source/prsm/features.csv')
    files = ['AT_3', 'Yeast_1', 'FB_TeO_3', 'TM_2']
    fig = make_figure(data, files)
    for ax, file in zip(fig.axes, files):
        maximum = data.loc[data.file_id.eq(file), 'value'].max()
        if maximum > ax.get_ylim()[1]:
            ax.set_ylim(-0.5, maximum + 2)
    for ext in ['pdf', 'svg', 'png', 'tiff']:
        fig.savefig(out / f'ed4.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
if __name__ == '__main__':
    main()
