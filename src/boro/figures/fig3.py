from . import ade_style as style
from pathlib import Path
import os, json, importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('BORO_FIGURE_OUTPUT', str(ROOT / 'results/figures')))
SD = ROOT / 'data/source/ade'
style.setup()
order = style.ORDER
colors = style.COLORS
main = pd.read_csv(SD / 'main_per_seed.csv', float_precision='round_trip')
d = pd.read_csv(SD / 'threshold_transfer.csv', float_precision='round_trip')
curves = pd.read_csv(SD / 'curves_summary.csv', float_precision='round_trip')
P = 'test_locked_p95_precision'
R = 'test_locked_p95_recall'
Q = 'test_locked_p95_meets_point_target'
assert len(main) == len(d) == 100 and d.groupby('method').size().eq(10).all()
W, H = (180.0, 190.0)
fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=150)

def text(x, top, s, **kw):
    return fig.text(x / W, 1 - top / H, s, va='center', **kw)

def axis(x, top, w, h):
    return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H])
text(3, 4, 'a', fontsize=8, weight='bold')
text(26, 4, 'High-precision recall', fontsize=7, weight='bold')
aa = axis(26, 10, 105, 35)
aa.set_xlim(95, 99.2)
aa.set_ylim(0, 15)
aa.set_xticks([95, 98, 99])
aa.set_yticks([0, 5, 10, 15])
aa.set_xlabel('Minimum required precision (%)', labelpad=3)
aa.set_ylabel('Recall (%)', labelpad=3)
aa.grid(axis='y', color='#E4E8EB', lw=0.4)
aa.set_axisbelow(True)
means = main.groupby('method').mean(numeric_only=True)
aa.add_patch(Rectangle((94.84, 0), 0.32, means.loc['BORO', 'r_at_p95'] * 100, facecolor='#E8CDD2', edgecolor='none', alpha=0.65, zorder=0.5))
curve_counts = {}
for method in list(reversed(order[1:])) + ['BORO']:
    local = curves[(curves.method == method) & (curves.family == 'precision') & curves.target.ge(0.95)].sort_values('target')
    line, = aa.plot(local.target * 100, local['mean'] * 100, color=colors[method], ls=style.STYLES[method], lw=1.7 if method == 'BORO' else 0.85, alpha=1 if method == 'BORO' else 0.95, zorder=7 if method == 'BORO' else 2)
    assert np.array_equal(line.get_ydata(), local['mean'].to_numpy() * 100)
    assert local['mean'].mul(100).between(*aa.get_ylim()).all()
    curve_counts[method] = len(local)
gains = []
for target in [0.95, 0.99]:
    metric = f'r_at_p{round(target * 100)}'
    best = means.drop('BORO')[metric].idxmax()
    gain = 100 * (means.loc['BORO', metric] / means.loc[best, metric] - 1)
    y = 100 * means.loc['BORO', metric]
    aa.plot(target * 100, y, 'o', ms=3.2, color=style.RED, markeredgewidth=0, zorder=8)
    aa.annotate(f'+{gain:.1f}%', xy=(target * 100, y), xytext=(0, 8), textcoords='offset points', ha='left' if target == 0.95 else 'center', va='bottom', fontsize=7, color=style.RED, weight='bold')
    gains.append(dict(target=target, best_baseline=best, relative_gain=gain))
handles = [Line2D([], [], color=colors[m], ls=style.STYLES[m], lw=1.7 if m == 'BORO' else 0.85, label='BCE' if m == 'BCE-NoSel' else m) for m in order]
legend = fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(139 / W, 1 - 11.5 / H), frameon=False, borderaxespad=0, handlelength=2.1, handletextpad=0.55, labelspacing=0.45, fontsize=6.5)
for label in legend.get_texts():
    if label.get_text() == 'BORO':
        label.set_color(style.RED)
        label.set_weight('bold')
text(3, 59.05, 'b', fontsize=8, weight='bold')
ap = axis(26.1, 63.1, 130.5, 26.1)
ar = axis(26.1, 96.4, 130.5, 26.1)
ap.set_title('Realized precision', loc='left', fontsize=7, pad=8)
ar.set_title('Recall conditional on precision ≥95%', loc='left', fontsize=7, pad=8)
ap.axhspan(0.95, 1.02, color='#F3F5F7', zorder=0)
ap.axhline(0.95, color='#7F8993', lw=0.65, ls=(0, (3, 2)), zorder=1)
boro = d.loc[d.method.eq('BORO') & d[Q], R].mean()
ar.axhline(boro, color=style.RED, lw=0.7, ls=(0, (3, 2)), zorder=1, alpha=0.8)
pmin = min(0.5, float(np.floor(d[P].min() * 10) / 10)) - 0.025
rmax = max(0.188, float(np.ceil(d[R].max() / 0.05) * 0.05 + 0.015))
summary = []
point_counts = {'precision': 0, 'recall': 0, 'undefined': 0}
for i, m in enumerate(order):
    local = d[d.method.eq(m)].sort_values('seed')
    c = colors[m]
    for offset, (_, row) in zip(np.linspace(-0.25, 0.25, 10), local.iterrows()):
        x = i + offset
        valid = pd.notna(row[P])
        passed = bool(row[Q])
        if valid:
            ap.scatter(x, row[P], s=8, marker='D', facecolor=to_rgba(c, 0.78), edgecolor=c, linewidth=0.4, zorder=3)
            point_counts['precision'] += 1
        else:
            ap.plot(x, 0.025, marker='x', color='#89929B', ms=3.1, mew=0.65, transform=ap.get_xaxis_transform(), clip_on=False)
            point_counts['undefined'] += 1
        ar.scatter(x, row[R], s=8 if valid else 13, marker='D' if valid else 'x', color=c if passed else '#D3D7DC', linewidth=0.5, zorder=3, **{'edgecolor': c if passed else '#969FA8'} if valid else {})
        point_counts['recall'] += 1
    pv = local[P].dropna()
    qv = local.loc[local[Q], R]
    if len(pv):
        ap.scatter(i, pv.mean(), s=43, marker='D', facecolor=to_rgba(c, 0.32), edgecolor=c, lw=0.9, zorder=5)
    if len(qv):
        ar.scatter(i, qv.mean(), s=43, marker='D', facecolor=to_rgba(c, 0.32), edgecolor=c, lw=0.9, zorder=5)
    ar.text(i, rmax - 0.014, f'n={len(qv)}', ha='center', va='bottom', fontsize=5.3, color=style.RED if m == 'BORO' else '#737F8D', weight='bold' if m == 'BORO' else 'normal')
    summary.append(dict(method=m, n_total=10, n_defined_precision=len(pv), mean_defined_precision=pv.mean(), n_qualified=len(qv), mean_qualified_recall=qv.mean()))
for ax in [ap, ar]:
    ax.set_xlim(-0.65, 9.65)
    ax.set_xticks(range(10), ['BCE' if m == 'BCE-NoSel' else m for m in order], rotation=35, ha='right')
    ax.tick_params(axis='x', labelsize=5.8, pad=3)
    ax.tick_params(axis='y', labelsize=6)
    ax.get_xticklabels()[0].set_color(style.RED)
    ax.get_xticklabels()[0].set_weight('bold')
ap.tick_params(axis='x', bottom=False, labelbottom=False)
ap.set_ylim(pmin, 1.02)
pticks = [v for v in [0.2, 0.4, 0.6, 0.8, 0.95, 1.0] if v >= pmin]
ap.set_yticks(pticks, [f'{v:.2f}' for v in pticks])
ap.set_ylabel('Test precision', fontsize=6.5, labelpad=3)
ar.set_ylim(-0.008, rmax)
rticks = np.arange(0, rmax - 0.012, 0.05)
ar.set_yticks(rticks, [f'{v:.2f}' for v in rticks])
ar.set_ylabel('Test recall', fontsize=6.5, labelpad=3)
assert d[P].dropna().between(*ap.get_ylim()).all() and d[R].between(*ar.get_ylim()).all()
assert point_counts['precision'] + point_counts['undefined'] == point_counts['recall'] == 100
handles = [Line2D([], [], ls='none', marker='D', color='#667889', ms=3, label='Qualified seed'), Line2D([], [], ls='none', marker='D', mfc='#D3D7DC', mec='#969FA8', ms=3, label='Constraint-violating seed'), Line2D([], [], ls='none', marker='x', color='#89929B', ms=3, label='No valid selection'), Line2D([], [], ls='none', marker='D', mfc='#D3DCE3', mec='#667889', ms=4.5, label='Mean of qualified seeds')]
fig.legend(handles=handles, ncol=4, loc='lower center', bbox_to_anchor=(0.505, (45 + 2.7) / H), fontsize=5.5, frameon=False, columnspacing=1.5, handletextpad=0.5, title='Recall panel', title_fontsize=5.5)
text(3, 149, 'c', fontsize=8, weight='bold')
text(26, 149, 'Average precision', fontsize=7, weight='bold')
ac = axis(26, 153, 130, 24)
ac.set_xlim(-0.6, 9.6)
ac.set_ylim(35, 75)
ac.set_yticks([40, 50, 60, 70])
ac.set_ylabel('AP (%)', fontsize=7)
ac.set_xticks(range(10), ['BCE' if m == 'BCE-NoSel' else m for m in order], rotation=30, ha='right', rotation_mode='anchor')
ac.tick_params(axis='x', labelsize=6.5, length=2, width=0.5, pad=2)
ac.tick_params(axis='y', labelsize=6, length=2, width=0.5, pad=2)
ac.grid(axis='y', color='#E8EBEE', lw=0.4, zorder=0)
box_stats = []
for i, m in enumerate(order):
    vals = main.loc[main.method.eq(m)].sort_values('seed').ap.to_numpy() * 100
    c = colors[m]
    bp = ac.boxplot([vals], positions=[i], widths=0.44, whis=1.5, patch_artist=True, manage_ticks=False, showmeans=False, showfliers=True, boxprops=dict(facecolor=to_rgba(c, 0.3), edgecolor=c, lw=0.8), medianprops=dict(color=c, lw=0.9), whiskerprops=dict(color=c, lw=0.7), capprops=dict(color=c, lw=0.7), flierprops=dict(marker='o', markersize=3.8, markerfacecolor='none', markeredgecolor=c, markeredgewidth=0.7))
    q1, q3 = np.percentile(vals, [25, 75])
    assert len(vals) == 10 and np.allclose(bp['medians'][0].get_ydata(), np.median(vals))
    assert np.isclose(bp['boxes'][0].get_path().vertices[:, 1].min(), q1) and np.isclose(bp['boxes'][0].get_path().vertices[:, 1].max(), q3)
    assert min(vals) >= ac.get_ylim()[0] and max(vals) <= ac.get_ylim()[1]
    box_stats.append(dict(method=m, n=10, q1=q1, median=np.median(vals), q3=q3))
ac.get_xticklabels()[0].set_color(style.RED)
ac.get_xticklabels()[0].set_weight('bold')
fig.text(26 / W, 1 / H, 'Boxes: IQR; line: median; whiskers: 1.5×IQR; circles: outliers; n = 10 seeds', fontsize=5.5, color='#737F8D', va='bottom')
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
outside = []
for item in fig.findobj(Text):
    if not item.get_visible() or not item.get_text():
        continue
    bb = item.get_window_extent(renderer)
    if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > fig.bbox.width + 0.5 or (bb.y1 > fig.bbox.height + 0.5):
        outside.append(item.get_text())
assert not outside, outside
dest = OUT
stem = dest / 'fig3'
for ext in ['pdf', 'svg']:
    fig.savefig(stem.with_suffix('.' + ext))
fig.savefig(stem.with_suffix('.png'), dpi=300)
fig.savefig(stem.with_suffix('.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)
