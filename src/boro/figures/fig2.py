import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, t
from .regions import draw_regional_panel
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('BORO_FIGURE_OUTPUT', str(ROOT / 'results/figures')))
SOURCE = ROOT / 'data/source/controlled'
BUNDLE = SOURCE
STEM = OUT / 'fig2'
WIDTH, HEIGHT = (180.0, 180.0)
METHODS = ('BORO', 'BCE', 'Focal', 'TFCO')
CURVE_METHODS = METHODS[:3]
COLORS = {'BORO': '#B84F70', 'BCE': '#969BA4', 'Focal': '#5B9FBF', 'TFCO': '#454F78'}
STYLES = {'BORO': '-', 'BCE': (0, (1.5, 2)), 'Focal': (0, (5, 2)), 'TFCO': '-'}
MARKERS = {'BORO': 'o', 'BCE': 's', 'Focal': '^', 'TFCO': 'D'}
INK = '#2C3037'
MUTED = '#626C78'
BAND = '#EEF2F6'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def mean_ci(values):
    values = np.asarray(values, float)
    assert values.size == 20 and np.isfinite(values).all()
    mean = values.mean()
    half = t.ppf(0.975, 19) * values.std(ddof=1) / np.sqrt(20)
    return (float(mean), float(mean - half), float(mean + half))

def axes_mm(fig, x, y, width, height):
    return fig.add_axes([x / WIDTH, y / HEIGHT, width / WIDTH, height / HEIGHT])

def title(fig, x, y, letter, text):
    fig.text((x - 7) / WIDTH, y / HEIGHT, letter, fontsize=8, weight='bold', va='bottom')
    fig.text(x / WIDTH, y / HEIGHT, text, fontsize=7, weight='bold', va='bottom')

def quiet(ax):
    ax.spines[['top', 'right']].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(INK)
    ax.tick_params(direction='out', pad=2.3)

def keyed_text(ax, x, y, method, text, coordinates='data', size=6.4):
    trans = ax.transAxes if coordinates == 'axes' else ax.transData
    dx = 0.055 if coordinates == 'axes' else 1.0
    ax.plot([x, x + dx], [y, y], transform=trans, color=COLORS[method], linestyle=STYLES[method], lw=1.35, clip_on=False)
    ax.text(x + dx * 1.5, y, text, transform=trans, va='center', fontsize=size, color=INK)

def draw_boundary_panel(fig, data):
    params = data['reference_boundary.csv'].iloc[0]
    tau = float(params.reference_cutoff)
    lower_band = float(params.nearest10_score_low) - tau
    upper_band = float(params.nearest10_score_high) - tau
    scores = data['reference_scores.csv'].copy()
    scores['score_minus_cutoff'] = scores.reference_score - tau
    selected = scores.score_minus_cutoff.ge(0)
    positive_count = int((selected & scores['class'].eq('Positive')).sum())
    negative_count = int((selected & scores['class'].eq('Negative')).sum())
    precision = positive_count / (positive_count + negative_count)
    assert (positive_count, negative_count) == (159, 8)
    cls_colors = {'Positive': (0.443, 0.58, 0.847), 'Negative': (0.91, 0.757, 0.78)}
    cls_lines = {'Positive': (0.192, 0.518, 0.98), 'Negative': (0.835, 0.333, 0.431)}
    detail_background = tuple((1 - 0.55 * (1 - v) for v in (0.82, 0.886, 0.945)))
    title(fig, 15, 164, 'a', 'Operating boundary')
    overview = axes_mm(fig, 15, 148, 52, 13)
    quiet(overview)
    zoom_low, zoom_high = (-1.5, 1.5)
    overview.axvspan(zoom_low, zoom_high, color=detail_background, zorder=0)
    grid = np.linspace(-14, 5, 600)
    for cls in ('Negative', 'Positive'):
        values = scores.loc[scores['class'].eq(cls), 'score_minus_cutoff'].to_numpy(float)
        density = gaussian_kde(values, bw_method=lambda kde: 1.6 * kde.scotts_factor())(grid)
        overview.fill_between(grid, density, color=cls_colors[cls], alpha=0.17, lw=0)
        overview.plot(grid, density, color=cls_lines[cls], lw=1.05)
    overview.axvline(0, ymax=0.8, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=3)
    overview.text(-10.8, 0.227, 'Negative', fontsize=6)
    overview.text(-5.8, 0.255, 'Positive', fontsize=6)
    overview.text(0, 0.323, 'Near boundary', ha='center', va='bottom', fontsize=5.8, clip_on=False)
    overview.set(xlim=(-14, 5), ylim=(0, 0.31), xticks=[-12, -6, 0], yticks=[0, 0.2], ylabel='Density')
    overview.tick_params(axis='both', labelsize=5.8, pad=1.4, length=2)
    overview.yaxis.label.set_fontsize(6)
    detail = axes_mm(fig, 15, 122, 52, 15)
    quiet(detail)
    detail.spines['left'].set_visible(False)
    detail.set_facecolor(detail_background)
    detail.axvline(0, color=INK, lw=0.85, ls=(0, (3, 2)), zorder=2)
    detail.set(xlim=(-1.5, 1.5), ylim=(-1.0, 1.0), xticks=[-1.5, 0, 1.5], yticks=[], xlabel='Score − boundary')
    detail.tick_params(axis='x', labelsize=6.2, pad=2.3, length=2.5)
    from matplotlib.patches import Polygon
    upper_edges = overview.transData.transform([(zoom_low, 0), (zoom_high, 0)])
    lower_edges = detail.transData.transform([(zoom_high, 1), (zoom_low, 1)])
    vertices = fig.transFigure.inverted().transform(np.vstack([upper_edges, lower_edges]))
    fig.add_artist(Polygon(vertices, transform=fig.transFigure, facecolor=detail_background, edgecolor='none', linewidth=0, zorder=-1, clip_on=False))
    fig.canvas.draw()
    plotted = scores[scores.score_minus_cutoff.between(-1.5, 1.5)].sort_values('score_minus_cutoff', kind='mergesort').copy()
    x = plotted.score_minus_cutoff.to_numpy(float)
    x_pixels = detail.transData.transform(np.column_stack([x, np.zeros(len(x))]))[:, 0]
    candidates_y = np.array([0] + [v for k in range(1, 80) for v in (k * 7.0, -k * 7.0)])
    offsets_px = np.zeros(len(x))
    separation = 8.0
    active_start = 0
    for i, xp in enumerate(x_pixels):
        while active_start < i and xp - x_pixels[active_start] >= separation:
            active_start += 1
        dx = xp - x_pixels[active_start:i]
        used_y = offsets_px[active_start:i]
        for yy in candidates_y:
            if np.all(dx * dx + (yy - used_y) ** 2 >= separation ** 2):
                offsets_px[i] = yy
                break
        else:
            raise RuntimeError('Display packing exceeded available lanes')
    max_allowed = detail.bbox.height * 0.38
    factor = min(1.0, max_allowed / max(1.0, np.abs(offsets_px).max()))
    y = offsets_px * factor * 2.0 / detail.bbox.height
    plotted['display_y_offset'] = y
    plotted['selected'] = plotted.score_minus_cutoff.ge(0)
    assert np.abs(y).max() <= 0.761
    for cls in ('Positive', 'Negative'):
        take = plotted['class'].eq(cls).to_numpy(bool)
        detail.scatter(x[take], y[take], s=4.2 if cls == 'Positive' else 5.3, marker='o' if cls == 'Positive' else 'D', facecolors=cls_colors[cls], edgecolors=cls_lines[cls], linewidths=0.27, alpha=0.78 if cls == 'Positive' else 0.96, zorder=4 if cls == 'Positive' else 6)
    display_positive = int((plotted.selected & plotted['class'].eq('Positive')).sum())
    display_negative = int((plotted.selected & plotted['class'].eq('Negative')).sum())
    assert len(plotted) == 719
    assert display_positive == 123 and display_negative == negative_count
    audit = {'reference_cutoff': tau, 'positive_total': int(scores['class'].eq('Positive').sum()), 'negative_total': int(scores['class'].eq('Negative').sum()), 'selected_positive': positive_count, 'selected_negative': negative_count, 'full_reference_precision_percent': 100 * precision, 'detail_score_window': [-1.5, 1.5], 'display_selected_positive': display_positive, 'display_selected_negative': display_negative, 'full_selection_numbers_displayed': False, 'distribution_overview_displayed': True, 'detail_candidates_plotted': len(plotted), 'nearest10_band': [lower_band, upper_band], 'overview_zoom_band': [zoom_low, zoom_high], 'overview_band_matches_detail_window': True, 'score_coordinates_jittered': False, 'KDE_displayed': True, 'upper_KDEs_normalized_within_class': True, 'local_KDE_displayed': False, 'upper_KDE_bandwidth': '1.6 times Scott factor for both classes', 'artificial_tail_rescaling': False, 'detail_vertical_offsets_are_layout_only': True, 'shared_candidate_band': True, 'vertical_packing_is_class_blind': True, 'vertical_packing_uniform_compression': float(factor)}
    return (audit, plotted)

def main():
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 6.7, 'axes.labelsize': 6.7, 'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2, 'axes.linewidth': 0.65, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    files = {name: SOURCE / name for name in ['reference_scores.csv', 'reference_boundary.csv', 'gain_localization_summary.csv', 'gain_localization_per_dataset.csv', 'score_gradient_summary.csv', 'score_gradient_per_dataset.csv', 'ordering_summary.csv', 'ordering_per_dataset.csv', 'test_recall_per_dataset.csv', 'test_effect_source.csv', 'regional_gradient_per_dataset.csv', 'regional_gradient_mean_ci.csv', 'regional_gradient_plotted_values.csv']}
    initial_hashes = {name: digest(path) for name, path in files.items()}
    data = {name: pd.read_csv(path) for name, path in files.items()}
    recall = data['test_recall_per_dataset.csv']
    assert set(recall.method) == set(METHODS)
    assert recall.required_precision_percent.eq(95).all()
    wide = recall.pivot(index='seed', columns='method', values='maximum_recall_percent')
    assert wide.shape == (20, 4) and wide.notna().all().all()
    summary = {m: mean_ci(wide[m]) for m in METHODS}
    paired = {m: mean_ci(wide.BORO - wide[m]) for m in METHODS[1:]}
    assert np.isclose(summary['BORO'][0], 8.017806455223687)
    assert np.isclose(paired['BCE'][0], 1.6926560306389686)
    effect_source = data['test_effect_source.csv']
    for m in METHODS[1:]:
        row = effect_source[(effect_source.comparison == 'BORO - ' + m) & effect_source.endpoint.eq('test_r_at_p95')].iloc[0]
        assert np.allclose(paired[m], row[['mean_difference', 'ci95_low', 'ci95_high']].to_numpy(float))
    fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=300)
    handles = [Line2D([0], [0], color=COLORS[m], lw=1.5, linestyle=STYLES[m], marker=MARKERS[m], markersize=3.4, label=m) for m in CURVE_METHODS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.71, 0.991), ncol=3, fontsize=6.7, handlelength=2.6, columnspacing=2.5)
    panel_a_audit, panel_a_candidates = draw_boundary_panel(fig, data)
    b = axes_mm(fig, 83, 122, 91, 39)
    quiet(b)
    title(fig, 83, 164, 'b', 'Recall-gain localization')
    b.axvspan(0, 10, color=BAND, zorder=0)
    b.axvline(10, color='#A4B0BD', lw=0.7, ls=(0, (3, 2)))
    local = data['gain_localization_summary.csv']
    endpoints = {}
    for m in CURVE_METHODS[::-1]:
        block = local[local.method.eq(m)].sort_values('nearest_sample_percent')
        x = block.nearest_sample_percent.to_numpy(float)
        y = block.mean_share_percent.to_numpy(float)
        b.fill_between(x, block.ci95_low_share_percent.to_numpy(float), block.ci95_high_share_percent.to_numpy(float), color=COLORS[m], alpha=0.08, lw=0)
        b.plot(x, y, color=COLORS[m], ls=STYLES[m], lw=1.7 if m == 'BORO' else 1.25)
        endpoints[m] = float(block.loc[block.nearest_sample_percent.eq(10), 'mean_share_percent'].iloc[0])
        b.scatter(10, endpoints[m], s=15, marker=MARKERS[m], color=COLORS[m], edgecolors='white', linewidths=0.4, zorder=5)
    b.set(xlim=(0, 20), ylim=(0, 106), xticks=[0, 5, 10, 20], yticks=[0, 50, 100], xlabel='Candidates nearest to the boundary (%)', ylabel='Validation recall-gain share (%)')
    gradient10, region_audit = draw_regional_panel(fig, ROOT, axes_mm, quiet, title, COLORS, STYLES, MARKERS, BAND)
    ratio = gradient10['BORO'] / gradient10['Focal']
    d = axes_mm(fig, 15, 10, 68, 37)
    quiet(d)
    title(fig, 15, 51, 'd', 'Near-boundary ordering')
    order = data['ordering_summary.csv']
    d.axhline(0, color='#C2C6CC', lw=0.65, ls=(0, (2, 2)))
    for m in CURVE_METHODS[::-1]:
        block = order[order.method.eq(m)].sort_values('epoch')
        x = block.epoch.to_numpy(float)
        d.fill_between(x, block.ci95_low.to_numpy(float), block.ci95_high.to_numpy(float), color=COLORS[m], alpha=0.1, lw=0)
        d.plot(x, block['mean'].to_numpy(float), color=COLORS[m], ls=STYLES[m], lw=1.7 if m == 'BORO' else 1.25)
        endpoint = float(block.loc[block.epoch.eq(25), 'mean'].iloc[0])
        d.scatter(25, endpoint, color=COLORS[m], marker=MARKERS[m], s=15, edgecolors='white', linewidths=0.4, zorder=5)
        yy = {'BORO': 18.5, 'BCE': 15.5, 'Focal': 12.8}[m]
        d.annotate(f'{endpoint:.1f}', xy=(25, endpoint), xytext=(26.6, yy), fontsize=6.2, va='center', arrowprops={'arrowstyle': '-', 'lw': 0.5, 'color': COLORS[m]})
    d.set(xlim=(0, 30), ylim=(-0.6, 22), xticks=[0, 5, 10, 25], yticks=[0, 10, 20], xlabel='Training epoch', ylabel='Pairwise ordering gain (pp)')
    e = axes_mm(fig, 99, 10, 64, 35)
    quiet(e)
    title(fig, 99, 51, 'e', 'Held-out recall at precision ≥95%')
    bar_methods = ('BORO', 'BCE', 'Focal', 'TFCO')
    for yi, m in enumerate(bar_methods):
        center, low, high = summary[m]
        fill = mpl.colors.to_rgba(COLORS[m], 0.9 if m == 'BORO' else 0.35)
        e.barh(yi, center, height=0.52, color=fill, edgecolor=COLORS[m], linewidth=0.85, zorder=3)
        e.errorbar(center, yi, xerr=[[center - low], [high - center]], fmt='none', ecolor=INK, elinewidth=0.9, capsize=3, capthick=0.9, zorder=4)
        e.text(10.6, yi, f'{center:.2f}', ha='left', va='center', fontsize=6.7, weight='bold' if m == 'BORO' else 'normal', color=INK)
    e.set(xlim=(0, 13.0), ylim=(3.6, -0.6), yticks=range(4), yticklabels=bar_methods, xticks=[0, 5, 10], xlabel='Test recall (%)')
    e.tick_params(axis='y', length=0, pad=5)
    e.spines['left'].set_visible(False)
    for label, m in zip(e.get_yticklabels(), bar_methods):
        if m == 'BORO':
            label.set_color(INK)
            label.set_weight('bold')
    relative_gain = 100 * (summary['BORO'][0] / summary['BCE'][0] - 1)
    significance = json.loads((SOURCE / 'statistics.json').read_text())
    assert np.allclose(significance['means'], [summary[m][0] for m in bar_methods])
    assert all((p < 0.001 for p in significance['holm_adjusted_p']))
    for yi, x in zip((1, 2, 3), (1.02, 1.08, 1.14)):
        trans = e.get_yaxis_transform()
        e.plot([x - 0.012, x, x, x - 0.012], [0, 0, yi, yi], transform=trans, color=MUTED, lw=0.65, clip_on=False)
        e.text(x + 0.01, yi / 2, '***', transform=trans, ha='left', va='center', fontsize=7, color=INK, clip_on=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_outside = []
    text_sizes = []
    from matplotlib.text import Text
    for obj in fig.findobj(match=Text):
        if not obj.get_visible() or not obj.get_text():
            continue
        box = obj.get_window_extent(renderer)
        text_sizes.append(float(obj.get_fontsize()))
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1 + 1 or (box.y1 > fig.bbox.y1 + 1):
            text_outside.append(obj.get_text())
    assert not text_outside, text_outside
    assert min(text_sizes) >= 5 and max(text_sizes) <= 8
    assert max((v[2] for v in summary.values())) < e.get_xlim()[1]
    for ext in ('png', 'pdf', 'svg', 'tiff'):
        kw = {'dpi': 600 if ext == 'tiff' else 300, 'facecolor': 'white'}
        if ext == 'tiff':
            kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(STEM.with_suffix('.' + ext), **kw)
    plt.close(fig)
if __name__ == '__main__':
    main()
