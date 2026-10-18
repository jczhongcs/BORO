from pathlib import Path
import os
import json
import hashlib
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('BORO_FIGURE_OUTPUT', str(ROOT / 'results/figures')))
W, H = (180, 210)
FIGURE_TITLE = 'Ranking and selection changes near operating cutoffs'
INK, MUTED = ('#2C3037', '#626C78')
ROSE, ROSE_FILL, ROSE_SOFT = ('#B84F70', '#E2B7C6', '#D29AAA')
SLATE, SOFT_SLATE, LIGHT = ('#637A91', '#92A1B2', '#CDD4DC')
BINS = ['11–20', '21–30', '31–50', '51–100', '>100']
COLORS = [ROSE, ROSE_SOFT, '#AEB9C6', '#CAD1D9', '#E1E5EA']
SETTINGS = ['scifact|random', 'fiqa|random', 'nfcorpus|random', 'scifact|bm25', 'fiqa|bm25', 'nfcorpus|bm25', 'arguana|source-only', 'dbpedia-entity|source-only', 'scidocs|source-only']
LABELS = ['SciFact · random', 'FiQA · random', 'NFCorpus · random', 'SciFact · BM25', 'FiQA · BM25', 'NFCorpus · BM25', 'ArguAna · source-only', 'DBPedia-Entity · source-only', 'SciDocs · source-only']

def load_data():
    return {key: pd.read_csv(ROOT / 'data/source' / path) for key, path in {'ade_curve': 'boundary/cumulative_frontier_rescue.csv', 'ade_bands': 'boundary/frontier_band_summary.csv', 'ade_seeds': 'boundary/frontier_band_by_seed.csv', 'prsm_similarity': 'prsm/boundary_metrics.csv', 'prsm_units': 'prsm/boundary_per_file.csv', 'ir_origins': 'boundary/relevant_entrant_origin_by_setting_seed.csv'}.items()}

def style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 7, 'axes.labelsize': 6.5, 'axes.titlesize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 5.5, 'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': MUTED, 'xtick.color': INK, 'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.55, 'axes.grid': False, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.4, 'ytick.major.size': 2.4, 'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5, 'legend.frameon': False, 'legend.handlelength': 1.15, 'legend.handletextpad': 0.5, 'legend.columnspacing': 0.9, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'axes.unicode_minus': True})

def ax_mm(fig, x, y, w, h, label):
    return fig.add_axes([x / W, y / H, w / W, h / H], label=label)

def text_mm(fig, x, y, text, **kwargs):
    return fig.text(x / W, y / H, text, **kwargs)

def header(fig, letter, title, y, detail=''):
    text_mm(fig, 3, y, letter, fontsize=8, weight='bold', va='center')
    text_mm(fig, 10, y, title, fontsize=7, weight='bold', va='center')
    if detail:
        text_mm(fig, 176, y, detail, fontsize=6, color=MUTED, ha='right', va='center')

def draw_ade(fig, d):
    header(fig, 'a', 'ADE | precision ≥95%', 202)
    curve = ax_mm(fig, 16, 155, 59, 35, 'ade_cumulative')
    band = ax_mm(fig, 94, 155, 82, 35, 'ade_bands')
    curve.set_title('Origins of recovered positives', pad=6)
    band.set_title('Class-conditional selection rates', pad=6)
    c = d['ade_curve'].sort_values('focal_precision_at_rank_threshold', ascending=False)
    x = 100 * c.focal_precision_at_rank_threshold.to_numpy()
    y = 100 * c.pooled_cumulative_rescue_fraction.to_numpy()
    curve.plot(x, y, '-o', color=SLATE, lw=1.15, ms=3, zorder=3)
    for i in [0, 2]:
        curve.annotate(f'{y[i]:.1f}%', (x[i], y[i]), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=6, weight='bold')
    curve.set(xlim=(93, 58), ylim=(0, 115), xticks=[90, 80, 70, 60], yticks=[0, 50, 100], xlabel='Focal precision\nat rank (%)', ylabel='Cumulative share (%)')
    b = d['ade_bands'].sort_values('band_order')
    x = np.arange(len(b))
    pos = 100 * b.pooled_target_rescue_rate.to_numpy()
    neg = 100 * b.pooled_decoy_entry_rate.to_numpy()
    band.bar(x, pos, width=0.65, color=ROSE_FILL, edgecolor=ROSE, lw=0.6, zorder=1)
    band.plot(x, neg, '-s', color=SLATE, lw=0.9, ms=2.7, zorder=4)
    for i, label in enumerate(b.band_label):
        vals = 100 * d['ade_seeds'].loc[d['ade_seeds'].band_label.eq(label)].sort_values('seed').target_rescue_rate.to_numpy()
        assert len(vals) == 5 and np.all((vals >= 0) & (vals <= 85))
        band.scatter(i + np.linspace(-0.16, 0.16, 5), vals, s=8.5, facecolor='white', edgecolor=ROSE, lw=0.55, zorder=5, clip_on=False)
    band.set(xlim=(-0.6, 6.6), ylim=(0, 85), yticks=[0, 20, 40, 60, 80], xticks=x, xticklabels=b.display_band, ylabel='Within-class rate (%)', xlabel='Focal precision at rank (%)')
    band.tick_params(axis='x', labelsize=5.5)
    plt.setp(band.get_xticklabels(), rotation=40, ha='right')
    band.legend(handles=[Patch(facecolor=ROSE_FILL, edgecolor=ROSE, label='Positive (pooled)'), Line2D([], [], color=SLATE, marker='s', lw=0.9, ms=2.7, label='Negative (pooled)'), Line2D([], [], color=ROSE, mfc='white', marker='o', lw=0, ms=3, label='Positive (seed)')], loc='upper right', borderaxespad=0.1, labelspacing=0.23, fontsize=5.3)
    return {'ade_pooled_rescued_events': int(c.pooled_total_rescued_positive.iloc[0]), 'ade_max_seed_entry_pct': float(100 * d['ade_seeds'].target_rescue_rate.max()), 'ade_ablation_panel_removed': True}

def box_points(ax, units, column):
    groups = [units, units.loc[units.analysis_group.eq('Seen species')], units.loc[units.analysis_group.eq('Unseen species')]]
    values = [g[column].dropna().to_numpy() for g in groups]
    ax.boxplot(values, positions=[0, 1, 2], widths=0.44, patch_artist=True, showfliers=False, whis=1.5, medianprops={'color': INK, 'linewidth': 1}, boxprops={'edgecolor': INK, 'linewidth': 0.65, 'facecolor': 'none'}, whiskerprops={'color': INK, 'linewidth': 0.65}, capprops={'color': INK, 'linewidth': 0.65}, zorder=3)
    for i, v in enumerate(values):
        jitter = np.random.default_rng(20260909 + i).uniform(-0.18, 0.18, len(v))
        ax.scatter(i + jitter, v, s=5, color=[SLATE, SOFT_SLATE, SOFT_SLATE][i], alpha=0.52, edgecolor='none', zorder=2)
    ns = [len(v) for v in values]
    ax.set(xlim=(-0.55, 2.55), xticks=[0, 1, 2], xticklabels=[f'{label}\nn={n}' for label, n in zip(['Overall\n', 'Seen\nspecies', 'Unseen\nspecies'], ns)])
    ax.tick_params(axis='x', labelsize=5.8)
    return ns

def draw_prsm(fig, d):
    header(fig, 'b', 'PrSM | 1% target–decoy FDR', 132)
    text_mm(fig, 40, 125.5, 'Agreement with BORO', ha='center', fontsize=6.5)
    text_mm(fig, 125.5, 125.5, 'BORO vs BCE', ha='center', fontsize=6.5, weight='bold')
    spear = ax_mm(fig, 24, 86, 13, 32, 'prsm_spearman')
    jacc = ax_mm(fig, 43, 86, 13, 32, 'prsm_jaccard')
    change = ax_mm(fig, 75, 83, 42, 35, 'prsm_changes')
    shell = ax_mm(fig, 134, 83, 42, 35, 'prsm_shell')
    g = d['prsm_similarity']
    assert g.baseline.tolist() == ['BCE', 'Focal', 'Class-Balanced', 'pAUC', 'IMMAX', 'STACO']
    for ax, col, marker, title in [(spear, 'rank_spearman', 'o', 'Spearman ρ'), (jacc, 'accepted_set_jaccard', 's', 'Jaccard')]:
        ax.scatter(g[col], range(6), marker=marker, s=18, color=SLATE, edgecolor='white', lw=0.4, zorder=3)
        lo = min(0.929, float(g[col].min()) - 0.004)
        hi = max(0.963, float(g[col].max()) + 0.004)
        ax.set(xlim=(lo, hi), ylim=(5.5, -0.5), yticks=range(6), yticklabels=g.baseline if ax is spear else [''] * 6,
               xticks=[np.ceil(lo*100)/100, np.floor(hi*100)/100])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        ax.tick_params(axis='y', length=0, labelsize=6.2)
        ax.spines['left'].set_visible(False)
        ax.set_title(title, fontsize=6, pad=5)
    change.set_title('Selection disagreement', pad=5)
    shell.set_title('Changed candidates near cutoffs', pad=5)
    units = d['prsm_units']
    ns = box_points(change, units, 'changed_candidates_pct')
    nws = box_points(shell, units, 'within_10pct_shell_pct')
    assert ns == [175, 55, 120] and sum(nws[1:]) == nws[0]
    change.set(ylim=(-0.1, max(6, float(units.changed_candidates_pct.max()) + 0.2)), yticks=[0, 2, 4, 6], ylabel='All candidates (%)')
    shell.set(ylim=(-2, 105), yticks=[0, 25, 50, 75, 100], ylabel='Changed candidates (%)')
    return {'prsm_baselines': g.baseline.tolist(), 'prsm_change_group_n': ns, 'prsm_shell_group_n': nws, 'prsm_shell_definition': 'intersection of both model-specific ±10% cutoff-rank windows', 'prsm_change_max_pct': float(units.changed_candidates_pct.max()), 'prsm_shell_minmax_pct': [float(units.within_10pct_shell_pct.min()), float(units.within_10pct_shell_pct.max())]}

def draw_ir(fig, d):
    header(fig, 'c', 'Dense retrieval | ranks of newly retrieved relevant documents', 63)
    ax = ax_mm(fig, 42, 12, 99, 41, 'ir_origin')
    rows = []
    o = d['ir_origins']
    assert len(o) == 45 and (not o.duplicated(['Setting', 'Seed']).any())
    for setting in SETTINGS:
        ss = o.loc[o.Setting.eq(setting)]
        assert len(ss) == 5
        counts = ss[[f'Count_{b}' for b in BINS]].sum().to_numpy(float)
        events = int(ss.RelevantEntrantEvents.sum())
        assert events > 0 and np.isclose(counts.sum(), events)
        rows.append({'Setting': setting, 'Events': events, **dict(zip(BINS, 100 * counts / events))})
    agg = pd.DataFrame(rows)
    ypos = np.array([10, 9, 8, 6.8, 5.8, 4.8, 3.6, 2.6, 1.6])
    left = np.zeros(9)
    for rank, color in zip(BINS, COLORS):
        values = agg[rank].to_numpy()
        ax.barh(ypos, values, left=left, height=0.67, color=color, edgecolor='white', lw=0.4)
        left += values
    assert np.allclose(left, 100)
    ax.set(xlim=(0, 100), ylim=(0.85, 10.7), xticks=[0, 25, 50, 75, 100], yticks=ypos, yticklabels=LABELS, xlabel='New top-10 relevant documents (%)')
    ax.tick_params(axis='y', length=0, labelsize=6.2)
    ax.spines['left'].set_visible(False)
    text_mm(fig, 42, 56.5, 'Rank under InfoNCE', fontsize=6, va='center')
    fig.legend(handles=[Patch(facecolor=color, label=rank) for color, rank in zip(COLORS, BINS)], loc='center left', bbox_to_anchor=(69 / W, 56.5 / H), ncol=5, borderaxespad=0, fontsize=5.8, columnspacing=0.8, handlelength=1)
    text_mm(fig, 154, 56.5, '11–30 (%)', fontsize=6, ha='center', va='center')
    text_mm(fig, 172, 56.5, 'Events', fontsize=6, ha='center', va='center')
    near = agg['11–20'] + agg['21–30']
    for y, share, events in zip(ypos, near, agg.Events):
        yf = (12 + (y - 0.85) / (10.7 - 0.85) * 41) / H
        fig.text(154 / W, yf, f'{share:.1f}', ha='center', va='center', fontsize=6.5)
        fig.text(172 / W, yf, f'{events:,}', ha='center', va='center', fontsize=6.2)
    return {'ir_setting_count': 9, 'ir_seed_setting_count': 45, 'ir_total_entrant_events': int(agg.Events.sum()), 'ir_row_events': agg.Events.tolist(), 'ir_row_sums_pct': left.tolist(), 'ir_near_rank_11_30_pct': near.tolist()}

def main():
    d = load_data()
    style()
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=150)
    checks = {'canvas_mm': [W, H]}
    checks.update(draw_ade(fig, d))
    checks.update(draw_prsm(fig, d))
    checks.update(draw_ir(fig, d))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fonts, outside = (set(), [])
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text().strip():
            continue
        fonts.add(obj.get_fontsize())
        bb = obj.get_window_extent(renderer)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > fig.bbox.width + 0.5 or (bb.y1 > fig.bbox.height + 0.5):
            outside.append(obj.get_text())
    checks['font_sizes_pt'] = sorted(fonts)
    checks['text_outside_canvas'] = outside
    assert not outside, outside
    base = OUT / 'fig6'
    for fmt in ['svg', 'pdf', 'png', 'tiff']:
        kw = {'dpi': 600 if fmt == 'tiff' else 300}
        if fmt == 'tiff':
            kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        if fmt == 'pdf':
            kw['metadata'] = {'Title': FIGURE_TITLE, 'Author': '', 'Subject': ''}
        fig.savefig(base.with_suffix('.' + fmt), **kw)
    plt.close(fig)
if __name__ == '__main__':
    main()
