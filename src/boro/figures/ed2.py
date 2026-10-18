from __future__ import annotations
import json
import os
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from . import prsm_matrix_style as base
from . import prsm_matrix as source
BORO_COLOR = '#24272B'
GROUP_LABELS = {'Human_0': 'Human 01–04', 'Human_E': 'Human E1–E2', 'Human_L': 'Human L1–L2', 'FB_TeO': 'FB–TeO', 'PS': 'PS', 'TM': 'TM', 'AH': 'AH', 'AT': 'AT', 'Yeast': 'Yeast'}
METHOD_LABELS = ['BCE', 'Focal', 'Class-\nBalanced', 'IMMAX', 'pAUC', 'STACO', 'BORO']

def build_portrait(mean: np.ndarray, file_ids: list[str], file_groups: list[str]) -> plt.Figure:
    raw = mean.T
    row_min = raw.min(axis=1, keepdims=True)
    row_max = raw.max(axis=1, keepdims=True)
    row_span = row_max - row_min
    normalized = np.divide(raw - row_min, row_span, out=np.full_like(raw, 0.5, dtype=float), where=row_span > 0)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(7.0, 8.26))
    ax = fig.add_axes([0.3, 0.045, 0.675, 0.795])
    image = ax.imshow(normalized, cmap=base.CMAP, norm=norm, aspect='auto', interpolation='none')
    n_files, n_methods = normalized.shape
    ax.set_xticks(np.arange(n_methods))
    ax.set_xticklabels(METHOD_LABELS, fontsize=7.0)
    ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=4)
    ax.get_xticklabels()[-1].set_color(BORO_COLOR)
    ax.get_xticklabels()[-1].set_fontweight('bold')
    ax.set_yticks(np.arange(n_files))
    ax.set_yticklabels([base.short_label(file_id, group) for file_id, group in zip(file_ids, file_groups)], fontsize=6.6)
    ax.tick_params(axis='y', length=0, pad=3.0)
    ax.set_xticks(np.arange(-0.5, n_methods, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_files, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8)
    ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(n_files):
        for col in range(n_methods):
            rgba = base.CMAP(norm(normalized[row, col]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = 'white' if luminance < 0.56 else '#24272B'
            rounded = int(np.rint(mean[col, row]))
            is_within_dataset_best = np.isclose(raw[row, col], row_max[row, 0])
            ax.text(col, row, f'{rounded:+d}' if rounded != 0 else '0', ha='center', va='center', fontsize=6.6, fontweight='bold' if is_within_dataset_best else 'regular', color=text_color)
    starts: list[int] = []
    ends: list[int] = []
    for group in base.GROUPS:
        idx = [i for i, value in enumerate(file_groups) if value == group]
        starts.append(min(idx))
        ends.append(max(idx))
    for boundary in [end + 0.5 for end in ends[:-1]]:
        ax.axhline(boundary, color='#3D4146', linewidth=0.9, zorder=6)
    for group, start, end in zip(base.GROUPS, starts, ends):
        center = (start + end) / 2
        ax.text(-0.16, center, GROUP_LABELS[group], ha='right', va='center', fontsize=6.8, fontweight='bold', clip_on=False, transform=ax.get_yaxis_transform())
        ax.plot([-0.095, -0.095], [start - 0.45, end + 0.45], color=base.REGIME_COLORS[base.REGIME[group]], linewidth=3.0, solid_capstyle='butt', clip_on=False, transform=ax.get_yaxis_transform())
    fig.text(0.045, 0.957, 'Per-dataset results for seven PrSM training objectives', ha='left', va='top', fontsize=8.8, fontweight='bold', color='#2F3135')
    cax = fig.add_axes([0.045, 0.895, 0.36, 0.014])
    colorbar = fig.colorbar(image, cax=cax, orientation='horizontal')
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(['Lower', 'Higher'])
    colorbar.ax.tick_params(labelsize=6.6, length=1.8, pad=1.2)
    colorbar.outline.set_linewidth(0.45)
    cax.set_title('Relative performance within dataset', fontsize=6.7, loc='left', pad=2.0)
    legend_ax = fig.add_axes([0.49, 0.884, 0.485, 0.03])
    legend_ax.set_axis_off()
    legend_ax.plot([0.0, 0.08], [0.5, 0.5], color=base.REGIME_COLORS['represented'], lw=3.2, solid_capstyle='butt')
    legend_ax.text(0.095, 0.5, 'Seen species', va='center', fontsize=6.7)
    legend_ax.plot([0.55, 0.63], [0.5, 0.5], color=base.REGIME_COLORS['unseen'], lw=3.2, solid_capstyle='butt')
    legend_ax.text(0.645, 0.5, 'Unseen species', va='center', fontsize=6.7)
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    return fig

def main():
    root = Path(__file__).resolve().parents[3]
    out = Path(os.environ.get('BORO_FIGURE_OUTPUT', str(root / 'results/figures')))
    data, mean, sd, files, groups = source.load_matrix()
    fig = build_portrait(mean, files, groups)
    for ext in ['pdf', 'svg', 'png', 'tiff']:
        fig.savefig(out / f'ed2.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
if __name__ == '__main__':
    main()
