from . import prsm_style as old
from pathlib import Path
import os, importlib.util, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.text import Text
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('BORO_FIGURE_OUTPUT', str(ROOT / 'results/figures')))
main = pd.read_csv(ROOT / 'data/source/prsm/main_per_seed.csv')
fdr = pd.read_csv(ROOT / 'data/source/prsm/fdr_per_seed.csv')
methods = old.METHODS[:-1] + ['STACO', 'BORO']
colors = {**old.COLORS, 'STACO': '#71879C'}
for frame, keys in [(main, ['panel', 'group', 'method']), (fdr, ['fdr_threshold', 'method'])]:
    assert frame.groupby(keys).size().eq(5).all()
    assert set(frame.method) == set(methods)
    assert set(frame.seed) == set(old.SEEDS)
old.apply_style()
plt.rcParams.update({'font.size': 6, 'axes.labelsize': 6.5, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5})
fig = plt.figure(figsize=(180 / 25.4, 170 / 25.4), dpi=150)

def text(x, y, s, **kw):
    return fig.text(x / 180, 1 - y / 170, s, **kw)

def axes(x, y, w, h):
    return fig.add_axes([x / 180, 1 - (y + h) / 170, w / 180, h / 170])

def horizontal(x, y, w, h, group, panel, labels=False):
    ax = axes(x, y, w, h)
    sub = main[main.panel.eq(panel) & main.group.eq(group)]
    ax.axhspan(-0.5, 0.5, color=old.BORO_BAND, zorder=0)
    allvals = []
    for i, m in enumerate(methods):
        vals = sub[sub.method.eq(m)].sort_values('seed').additional_target_prsms_per_dataset.to_numpy()
        assert len(vals) == 5
        allvals.extend(vals)
        pos = 6 - i
        bp = ax.boxplot([vals], positions=[pos], vert=False, widths=0.5, patch_artist=True, manage_ticks=False, showfliers=False, whis=1.5, boxprops=dict(facecolor=colors[m], edgecolor=colors[m], linewidth=0.65), medianprops=dict(color='#333333', linewidth=0.8), whiskerprops=dict(color='#707780', linewidth=0.6), capprops=dict(color='#707780', linewidth=0.6))
        ax.scatter(vals, pos + np.linspace(-0.15, 0.15, 5), s=6, facecolor='white', edgecolor='#65717C', linewidth=0.4, zorder=4)
        assert np.allclose(bp['medians'][0].get_xdata(), np.median(vals))
    span = max(np.ptp(allvals), 1)
    ax.set_xlim(min(allvals) - 0.08 * span, max(allvals) + 0.1 * span)
    ax.set_ylim(-0.6, 6.6)
    ax.set_yticks(range(7), methods[::-1] if labels else [''] * 7)
    if labels:
        ax.get_yticklabels()[0].set_color(colors['BORO'])
        ax.get_yticklabels()[0].set_weight('bold')
    ax.tick_params(axis='y', pad=2, length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    lo, hi = ax.get_xlim()
    ax.set_xticks([v for v in ax.get_xticks() if lo <= v <= hi])
    ax.tick_params(axis='x', pad=1.5)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color('#929AA1')
        ax.spines[s].set_linewidth(0.5)
    return (ax, sub)
effects = {}
for j, group in enumerate(['Overall', 'Seen-species internal', 'Unseen-species transfer']):
    x = 23 + j * 53
    ax, sub = horizontal(x, 10, 48, 29, group, 'a', j == 0)
    text(x + 24, 4, 'Seen-species held-out' if group == 'Seen-species internal' else group, ha='center', va='center', fontsize=6.5)
    n = int(sub.dataset_count.iloc[0])
    means = sub.groupby('method').additional_target_prsms_per_dataset.mean()
    best = means.drop('BORO').idxmax()
    delta = means.BORO - means[best]
    effects[group] = {'baseline': best, 'delta': delta, 'relative_percent': 100 * delta / means[best]}
    boro_right = sub.loc[sub.method.eq('BORO'), 'additional_target_prsms_per_dataset'].max()
    ax.annotate(f'{delta:+.2f} ({100 * delta / means[best]:+.1f}%)', xy=(boro_right, 0.3), xytext=(0, 2), textcoords='offset points', ha='right', va='bottom', fontsize=7, fontweight='bold', color=colors['BORO'])
for j, g in enumerate(old.SPECIES_ORDER):
    row = j // 4
    col = j % 4
    x = 23 + col * 40
    y = 56 + row * 42
    ax, sub = horizontal(x, y, 33, 28, g, 'b', col == 0)
    common, latin, status = old.SPECIES_DISPLAY[g]
    text(x + 16.5, y - 6, f'{common} ({status})', ha='center', fontsize=6)
legend = axes(145, 94, 33, 29)
legend.axis('off')
legend.legend(handles=[Patch(facecolor=colors[m], edgecolor=colors[m], label=m) for m in methods], loc='upper left', fontsize=5.5, handlelength=1, labelspacing=0.3, borderaxespad=0)
text(100, 132, 'Additional target PrSMs per dataset at 1% FDR', ha='center', fontsize=6.5)
ax = axes(23, 139, 153, 23)
centers = np.arange(3) * 3
offsets = np.linspace(-0.85, 0.85, 7)
boro_x = []
boro_y = []
allv = []
for j, t in enumerate([0.005, 0.01, 0.02]):
    for i, m in enumerate(methods):
        vals = fdr[fdr.method.eq(m) & np.isclose(fdr.fdr_threshold, t)].sort_values('seed').additional_target_prsms_per_dataset.to_numpy()
        assert len(vals) == 5
        p = centers[j] + offsets[i]
        allv.extend(vals)
        ax.boxplot([vals], positions=[p], widths=0.2, manage_ticks=False, patch_artist=True, showfliers=False, boxprops=dict(facecolor=colors[m], edgecolor=colors[m], alpha=0.6, linewidth=0.6), medianprops=dict(color='#333333', linewidth=0.7), whiskerprops=dict(color='#707780', linewidth=0.5), capprops=dict(color='#707780', linewidth=0.5))
        ax.scatter(p + np.linspace(-0.035, 0.035, 5), vals, s=6, marker='D', facecolor=colors[m], edgecolor='#424B53', linewidth=0.3, zorder=4)
        if m == 'BORO':
            boro_x.append(p)
            boro_y.append(np.median(vals))
ax.plot(boro_x, boro_y, color=colors['BORO'], lw=1)
ax.set_xlim(-1.15, 7.15)
ax.set_ylim(min(allv) - 2, max(allv) + 2)
ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
lo, hi = ax.get_ylim()
ax.set_yticks([v for v in ax.get_yticks() if lo <= v <= hi])
ax.set_xticks(centers, ['0.5', '1', '2'])
ax.set_xlabel('FDR (%)', labelpad=2)
ax.set_ylabel('Additional target PrSMs\n(per dataset)', fontsize=6, labelpad=3)
for x in [1.5, 4.5]:
    ax.axvline(x, color='#E3E6E9', lw=0.5, zorder=0)
for lab, y in [('a', 4), ('b', 50), ('c', 136)]:
    text(3, y, lab, fontsize=8, fontweight='bold', va='center')
fig.canvas.draw()
outside = []
for item in fig.findobj(Text):
    if item.get_visible() and item.get_text():
        b = item.get_window_extent(fig.canvas.get_renderer())
        if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > fig.bbox.width + 0.5 or (b.y1 > fig.bbox.height + 0.5):
            outside.append(item.get_text())
assert not outside, outside
for ext in ['pdf', 'svg', 'png', 'tiff']:
    fig.savefig(OUT / f'fig4.{ext}', dpi=600 if ext == 'tiff' else 300)
plt.close(fig)
