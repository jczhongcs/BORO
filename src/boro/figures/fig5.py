from __future__ import annotations
import argparse
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from . import ir_ranks as rank_frequency
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 6.2, 'axes.titlesize': 7.0, 'axes.labelsize': 6.5, 'axes.linewidth': 0.65, 'axes.spines.right': False, 'axes.spines.top': False, 'xtick.labelsize': 5.9, 'ytick.labelsize': 6.2, 'xtick.major.width': 0.55, 'ytick.major.width': 0.55, 'xtick.major.size': 2.4, 'ytick.major.size': 2.4, 'legend.frameon': False, 'legend.fontsize': 6.1, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
METHODS = ['InfoNCE', 'Triplet', 'ADNCE', 'AoE', 'DMCL', 'BORO']
METRICS = ['R@1', 'R@5', 'R@10', 'R@20', 'NDCG@1', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'MRR@10', 'MAP@100']
DISPLAY_METRICS = ['R@10', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'MRR@10']
METHOD_STYLE = {'InfoNCE': {'color': '#9DA4AA', 'edge': '#727980'}, 'Triplet': {'color': '#AF96CA', 'edge': '#8466A6'}, 'ADNCE': {'color': '#879CC1', 'edge': '#5F78A6'}, 'AoE': {'color': '#5DA394', 'edge': '#3A7E70'}, 'DMCL': {'color': '#78B7C7', 'edge': '#4F8E9F'}, 'BORO': {'color': '#E1546B', 'edge': '#B62F49'}}
GROUP_STYLE = {'In-domain / random': {'color': '#4B77B3', 'band': '#F7F9FC', 'label': 'In-domain /\nrandom'}, 'In-domain / BM25': {'color': '#278C88', 'band': '#F6FAFA', 'label': 'In-domain /\nBM25'}, 'Zero-shot': {'color': '#7660A3', 'band': '#FAF8FC', 'label': 'Source-only\ntransfer'}}
GROUPS = [('In-domain / random', 0, 2), ('In-domain / BM25', 3, 5), ('Zero-shot', 6, 8)]
GROUP_LABEL_RIGHT_X = 0.086
HEAT_CMAP = mcolors.LinearSegmentedColormap.from_list('boro_delta', ['#6E97C5', '#F7F7F7', '#E45A74'])
HEAT_NORM = mcolors.TwoSlopeNorm(vmin=-0.015, vcenter=0.0, vmax=0.023)

def _signed(value: float, decimals: int=3) -> str:
    return f'{value:+.{decimals}f}'.replace('-', '−')

def _tick_two_decimals(value: float, _position: int) -> str:
    if abs(value) < 5e-08:
        return '0.00'
    return f'{value:.2f}'.replace('-', '−')

def _validate_inputs(paired: pd.DataFrame, settings: pd.DataFrame, effects: pd.DataFrame) -> None:
    if settings['setting_index'].tolist() != list(range(9)):
        raise ValueError('Panel-a setting order must be 0–8')
    if set(paired['method']) != set(METHODS):
        raise ValueError('Panel-a methods do not match the six main-text methods')
    counts = paired.groupby(['setting_index', 'method'])['seed'].nunique()
    if len(counts) != 9 * len(METHODS) or not (counts == 5).all():
        raise ValueError('Panel a requires five paired seeds for each setting and method')
    if effects.shape[0] != 90 or set(effects['metric']) != set(METRICS):
        raise ValueError('Panel b requires 9 settings × 10 metrics')
    if int((settings['boro_ndcg10_delta'] > 0).sum()) != 9:
        raise ValueError('Expected BORO to win NDCG@10 in all nine settings')
    if int(effects['win'].sum()) != 86:
        raise ValueError('Expected 86 of 90 metric–setting wins')
    ndcg10 = effects.loc[effects['metric'] == 'NDCG@10']
    if len(ndcg10) != 9 or not ndcg10['win'].all():
        raise ValueError('Expected NDCG@10 wins in all nine settings')

def _load_multimetric_runs(base_path: Path, indomain_path: Path, zeroshot_path: Path, paired: pd.DataFrame, settings: pd.DataFrame, per_seed_path: Path | None=None) -> pd.DataFrame:
    if per_seed_path is not None:
        runs = pd.read_csv(per_seed_path).rename(columns={'Method': 'method', 'Seed': 'seed'})
        runs = runs.loc[runs['method'].isin(METHODS)].copy()
        runs['dataset_key'] = runs['Eval Dataset'].str.lower().replace({'dbpedia-entity': 'dbpedia'})
        runs['sampler_key'] = runs['Negative Sampler'].str.lower()
        runs = runs[['dataset_key', 'sampler_key', 'method', 'seed', *DISPLAY_METRICS]]
    else:
        base = pd.read_csv(base_path)
        base = base.loc[base['Variant'].isin(['infonce', 'triplet', 'dmcl', 'boro_source_auto'])].copy()
        base['method'] = base['Variant'].map({'infonce': 'InfoNCE', 'triplet': 'Triplet', 'dmcl': 'DMCL', 'boro_source_auto': 'BORO'})
        base['dataset_key'] = base['TargetDataset'].str.lower().replace({'dbpedia-entity': 'dbpedia'})
        base['sampler_key'] = base['NegativeSampler'].str.lower()
        base = base.rename(columns={'Seed': 'seed'})
        indomain = pd.read_csv(indomain_path)
        indomain['method'] = indomain['loss'].map({'adnce': 'ADNCE', 'aoe': 'AoE'})
        indomain['dataset_key'] = indomain['dataset'].str.lower()
        indomain['sampler_key'] = indomain['sampler'].str.lower()
        zeroshot = pd.read_csv(zeroshot_path)
        zeroshot['method'] = zeroshot['loss'].map({'adnce': 'ADNCE', 'aoe': 'AoE'})
        zeroshot['dataset_key'] = zeroshot['target'].str.lower().replace({'dbpedia-entity': 'dbpedia'})
        zeroshot['sampler_key'] = 'random'
        columns = ['dataset_key', 'sampler_key', 'method', 'seed', *DISPLAY_METRICS]
        runs = pd.concat([base[columns], indomain[columns], zeroshot[columns]], ignore_index=True)
    setting_keys = settings[['setting_index', 'setting', 'group', 'dataset', 'sampler']].copy()
    setting_keys['dataset_key'] = setting_keys['dataset'].str.lower().replace({'dbpedia': 'dbpedia'})
    setting_keys['sampler_key'] = setting_keys['sampler'].str.lower()
    runs = runs.merge(setting_keys, on=['dataset_key', 'sampler_key'], how='left', validate='many_to_one')
    if len(runs) != 9 * len(METHODS) * 5 or runs['setting_index'].isna().any():
        raise ValueError('Expected 270 complete multimetric run rows')
    if runs[DISPLAY_METRICS].isna().any().any():
        raise ValueError('Multimetric run table contains missing metric values')
    counts = runs.groupby(['setting_index', 'method'])['seed'].nunique()
    if len(counts) != 9 * len(METHODS) or not counts.eq(5).all():
        raise ValueError('Expected five seeds for every setting and method')
    ndcg_check = runs.merge(paired[['setting_index', 'method', 'seed', 'method_value']], on=['setting_index', 'method', 'seed'], how='outer', validate='one_to_one', indicator=True)
    if not ndcg_check['_merge'].eq('both').all():
        raise ValueError('Multimetric runs do not align with panel-a run identities')
    if not np.allclose(ndcg_check['NDCG@10'], ndcg_check['method_value'], atol=1e-12, rtol=0):
        raise ValueError('Multimetric NDCG@10 values do not match panel-a source data')
    return runs

def _summarize_multimetric_runs(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    setting_counts = runs.groupby(['group', 'method', 'seed'])['setting_index'].nunique()
    if len(setting_counts) != 3 * len(METHODS) * 5 or not setting_counts.eq(3).all():
        raise ValueError('Each group-level seed average must contain three settings')
    seed_macro = runs.groupby(['group', 'method', 'seed'], as_index=False)[DISPLAY_METRICS].mean().melt(id_vars=['group', 'method', 'seed'], value_vars=DISPLAY_METRICS, var_name='metric', value_name='seed_macro_mean')
    summary = seed_macro.groupby(['group', 'metric', 'method'], as_index=False).agg(mean_score=('seed_macro_mean', 'mean'), sd_score=('seed_macro_mean', 'std'), n=('seed_macro_mean', 'size'))
    summary['ci95_half_width'] = 2.7764451051977987 * summary['sd_score'] / np.sqrt(summary['n'])
    summary['rank'] = summary.groupby(['group', 'metric'])['mean_score'].rank(method='first', ascending=False)
    top3 = summary.loc[summary['rank'].le(3)].copy()
    top3['rank'] = top3['rank'].astype(int)
    if len(top3) != 54:
        raise ValueError('Panel b requires three methods × six metrics × three groups')
    boro = top3.loc[top3['method'].eq('BORO')]
    if len(boro) != 18 or not boro['rank'].eq(1).all():
        raise ValueError('Expected BORO to rank first for all displayed group–metric pairs')
    group_order = [group for group, _start, _end in GROUPS]
    top3['group'] = pd.Categorical(top3['group'], group_order, ordered=True)
    top3['metric'] = pd.Categorical(top3['metric'], DISPLAY_METRICS, ordered=True)
    top3 = top3.sort_values(['group', 'metric', 'rank'], ascending=[True, True, False])
    source = seed_macro.merge(summary[['group', 'metric', 'method', 'rank', 'mean_score', 'sd_score', 'ci95_half_width', 'n']], on=['group', 'metric', 'method'], how='inner', validate='many_to_one')
    source['displayed_top3'] = source['rank'].le(3)
    source = source.sort_values(['group', 'metric', 'rank', 'seed'])
    return (top3, source)

def _draw_group_bracket(ax: plt.Axes, start: float, end: float, label: str, color: str, x: float, arm: float, label_x: float, rotation: float=0, fontsize: float=6.4) -> None:
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    y0, y1 = (start - 0.43, end + 0.43)
    ax.plot([x, x], [y0, y1], color='#70757A', lw=0.7, transform=transform, clip_on=False)
    ax.plot([x, x + arm], [y0, y0], color='#70757A', lw=0.7, transform=transform, clip_on=False)
    ax.plot([x, x + arm], [y1, y1], color='#70757A', lw=0.7, transform=transform, clip_on=False)
    ax.text(label_x, (start + end) / 2, label, color=color, fontsize=fontsize, fontweight='bold', ha='center' if rotation else 'right', va='center', rotation=rotation, transform=transform, clip_on=False, linespacing=0.92)

def _panel_marker(ax: plt.Axes, panel: str, headline: str | None=None) -> None:
    ax.set_axis_off()
    ax.text(-0.217, 0.52, panel, transform=ax.transAxes, ha='left', va='center', fontsize=9.0, fontweight='bold', clip_on=False)
    if headline:
        ax.text(1.0, 0.52, headline, transform=ax.transAxes, ha='right', va='center', fontsize=7.2, fontweight='bold', color=METHOD_STYLE['BORO']['color'])

def _plot_panel_a(fig: plt.Figure, spec, paired: pd.DataFrame, settings: pd.DataFrame) -> None:
    grid = spec.subgridspec(3, 2, height_ratios=[0.07, 0.085, 0.845], width_ratios=[0.205, 0.795], hspace=0.0, wspace=0.035)
    title_ax = fig.add_subplot(grid[0, :])
    _panel_marker(title_ax, 'a')
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.set_axis_off()
    handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=4.8, markerfacecolor=METHOD_STYLE[method]['color'], markeredgecolor='white', markeredgewidth=0.45, label=method) for method in METHODS]
    legend_ax.legend(handles=handles, labels=METHODS, ncol=len(METHODS), loc='center', bbox_to_anchor=(0.55, 0.45), handletextpad=0.45, columnspacing=1.45, borderaxespad=0)
    ax_left = fig.add_subplot(grid[2, 0])
    ax_right = fig.add_subplot(grid[2, 1], sharey=ax_left)
    for group, start, end in GROUPS:
        band = GROUP_STYLE[group]['band']
        for ax in (ax_left, ax_right):
            ax.axhspan(start - 0.5, end + 0.5, color=band, zorder=-10)
    for y in range(9):
        for ax in (ax_left, ax_right):
            ax.axhline(y, color='#E2E5E8', lw=0.45, zorder=-2)
    for y in (2.5, 5.5):
        for ax in (ax_left, ax_right):
            ax.axhline(y, color='#C6C9CD', lw=0.7, zorder=-1)
    left_limit = min(-0.27, float(np.floor((min(paired['delta'].min(), paired['ci95_low'].min()) - 0.01) / 0.05) * 0.05))
    method_offsets = dict(zip(METHODS, np.linspace(-0.29, 0.29, len(METHODS))))
    seed_jitter = dict(zip(sorted(paired['seed'].unique()), np.linspace(-0.025, 0.025, 5)))
    for (setting_index, method), block in paired.groupby(['setting_index', 'method'], sort=False):
        block = block.sort_values('seed')
        style = METHOD_STYLE[method]
        y_mean = float(setting_index) + method_offsets[method]
        mean = float(block['mean_delta'].iloc[0])
        ci_low = float(block['ci95_low'].iloc[0])
        ci_high = float(block['ci95_high'].iloc[0])
        is_reference = np.allclose(block['delta'].to_numpy(), 0.0, atol=1e-14)
        segments = [(ax_left, left_limit, -0.06, False), (ax_right, -0.06, 0.04, True)]
        for ax, lower, upper, include_lower in segments:
            if include_lower:
                seed_mask = block['delta'].ge(lower) & block['delta'].le(upper)
                mean_in_segment = lower <= mean <= upper
            else:
                seed_mask = block['delta'].ge(lower) & block['delta'].lt(upper)
                mean_in_segment = lower <= mean < upper
            visible = block.loc[seed_mask]
            if not is_reference and (not visible.empty):
                ys = [y_mean + seed_jitter[int(seed)] for seed in visible['seed']]
                ax.scatter(visible['delta'], ys, s=10.5, facecolors='white', edgecolors=style['color'], linewidths=0.62, alpha=0.86, zorder=3, clip_on=True)
            if mean_in_segment:
                ax.errorbar(mean, y_mean, xerr=np.array([[mean - ci_low], [ci_high - mean]]), fmt='o', ms=4.8 if method == 'BORO' else 4.2, mfc=style['color'], mec=style['edge'], mew=0.65, ecolor=style['color'], elinewidth=0.9 if method == 'BORO' else 0.75, capsize=2.1, capthick=0.75, zorder=5, clip_on=True)
            elif ci_low < upper and ci_high > lower:
                ax.plot([max(ci_low, lower), min(ci_high, upper)], [y_mean, y_mean], color=style['color'], lw=0.9 if method == 'BORO' else 0.75, zorder=5, clip_on=True)
                for endpoint in (ci_low, ci_high):
                    if lower <= endpoint <= upper:
                        ax.plot(endpoint, y_mean, marker='|', markersize=4.2, markeredgewidth=0.75, color=style['color'], zorder=5, clip_on=True)
    ax_left.set_xlim(left_limit, -0.06)
    ax_right.set_xlim(-0.06, 0.04)
    ax_left.set_xticks(np.arange(np.ceil(left_limit * 10) / 10, -0.06, 0.1))
    ax_right.set_xticks([-0.04, -0.02, 0.0, 0.02, 0.03])
    ax_left.xaxis.set_major_formatter(FuncFormatter(_tick_two_decimals))
    ax_right.xaxis.set_major_formatter(FuncFormatter(_tick_two_decimals))
    ax_right.axvline(0, color='#55595E', lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_left.set_ylim(8.55, -0.55)
    labels = settings.sort_values('setting_index')['dataset'].replace({'DBPedia': 'DBPedia-Entity'}).tolist()
    ax_left.set_yticks(range(9))
    ax_left.set_yticklabels(labels)
    ax_left.tick_params(axis='y', length=0, pad=8.5)
    ax_right.tick_params(axis='y', left=False, labelleft=False)
    for ax in (ax_left, ax_right):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2D2D2D')
        ax.spines['bottom'].set_linewidth(0.7)
        ax.tick_params(axis='x', pad=2.5)
    ax_left.spines['left'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    d = 0.012
    ax_left.plot((1 - d, 1 + d), (-d, +d), transform=ax_left.transAxes, color='#2D2D2D', lw=0.75, clip_on=False)
    ax_right.plot((-d, +d), (-d, +d), transform=ax_right.transAxes, color='#2D2D2D', lw=0.75, clip_on=False)
    for group, start, end in GROUPS:
        style = GROUP_STYLE[group]
        label_axes_x = (GROUP_LABEL_RIGHT_X - ax_left.get_position().x0) / ax_left.get_position().width
        _draw_group_bracket(ax_left, start, end, style['label'], style['color'], x=-0.72, arm=0.04, label_x=label_axes_x, fontsize=6.4)
    right_transform = blended_transform_factory(ax_right.transAxes, ax_right.transData)
    for row in settings.itertuples(index=False):
        ax_right.text(0.985, row.setting_index, f'vs {row.reference_method}', transform=right_transform, ha='right', va='center', fontsize=5.4, color='#777B80')
    left_pos = ax_left.get_position()
    right_pos = ax_right.get_position()
    center_x = (left_pos.x0 + right_pos.x1) / 2
    fig.text(center_x, left_pos.y0 - 0.028, 'Paired ΔNDCG@10 versus the strongest non-BORO baseline', ha='center', va='center', fontsize=6.5, color='#252525')
    fig.text(left_pos.x0, left_pos.y0 - 0.048, 'Values below −0.06 are displayed on a compressed scale.', ha='left', va='center', fontsize=5.2, color='#858585')

def _plot_panel_b(fig: plt.Figure, spec, multimetric_runs: pd.DataFrame) -> None:
    observations, frequency = rank_frequency._rank_runs(multimetric_runs)
    settings = multimetric_runs[['setting_index', 'setting', 'group', 'dataset', 'sampler']].drop_duplicates().sort_values('setting_index')
    panel_box = spec.get_position(fig)
    grid = fig.add_gridspec(4, 3, height_ratios=[0.13, 0.29, 0.29, 0.29], left=0.17, right=0.98, bottom=panel_box.y0, top=panel_box.y1, hspace=0.52, wspace=0.1)
    header_ax = fig.add_subplot(grid[0, :])
    header_ax.set_axis_off()
    header_ax.text(-0.165, 0.62, 'b', transform=header_ax.transAxes, ha='left', va='center', fontsize=9.0, fontweight='bold', clip_on=False)
    header_ax.text(0.0, 0.18, 'Rank (1 = best)', transform=header_ax.transAxes, ha='left', va='center', fontsize=6.3, color='#222222')
    method_handles = [Line2D([0], [0], marker='o', linestyle='-' if method == 'BORO' else 'none', linewidth=1.15 if method == 'BORO' else 0, markersize=4.2, markerfacecolor=METHOD_STYLE[method]['color'], markeredgecolor=METHOD_STYLE[method]['edge'], markeredgewidth=0.5, color=METHOD_STYLE[method]['edge'], alpha=1.0 if method == 'BORO' else 0.72, label=method) for method in METHODS]
    method_legend = header_ax.legend(handles=method_handles, labels=METHODS, ncol=len(METHODS), loc='upper left', bbox_to_anchor=(0.015, 1.0), handletextpad=0.32, columnspacing=0.75, borderaxespad=0)
    for label in method_legend.get_texts():
        if label.get_text() == 'BORO':
            label.set_fontweight('bold')
    header_ax.add_artist(method_legend)
    shares = [20, 40, 60, 80, 100]
    size_handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=np.sqrt(rank_frequency._bubble_area(share)), markerfacecolor='#3A3A3A', markeredgecolor='#3A3A3A', label=str(share)) for share in shares]
    size_legend = header_ax.legend(handles=size_handles, labels=[str(share) for share in shares], ncol=5, loc='upper right', bbox_to_anchor=(1.0, 1.0), handletextpad=0.12, columnspacing=0.45, borderaxespad=0, title='Seed share (%)', title_fontsize=5.8)
    header_ax.add_artist(size_legend)
    header_ax.plot([0.53, 0.53], [0.08, 0.34], transform=header_ax.transAxes, color='#5A5A5A', lw=0.7, clip_on=False)
    header_ax.scatter([0.53], [0.21], transform=header_ax.transAxes, marker='x', s=12, linewidths=0.72, color='#111111', clip_on=False)
    header_ax.text(0.545, 0.21, 'mean rank; line, observed range', transform=header_ax.transAxes, ha='left', va='center', fontsize=5.45, color='#333333')
    metric_positions = dict(zip(DISPLAY_METRICS, range(len(DISPLAY_METRICS))))
    data_axes: list[plt.Axes] = []
    row_centers: list[tuple[str, float]] = []
    for row_index, (group, _start, _end) in enumerate(GROUPS, start=1):
        group_settings = settings.loc[settings['group'].eq(group)]
        row_axes: list[plt.Axes] = []
        for col_index, setting in enumerate(group_settings.itertuples(index=False)):
            ax = fig.add_subplot(grid[row_index, col_index])
            data_axes.append(ax)
            row_axes.append(ax)
            style = GROUP_STYLE[group]
            ax.set_facecolor(style['band'])
            ax.axhspan(0.62, 1.45, color='#FBECEF', alpha=0.88, zorder=0)
            panel = frequency.loc[frequency['setting_index'].eq(int(setting.setting_index))]
            boro_profile = panel.loc[panel['method'].astype(str).eq('BORO')].drop_duplicates('metric').copy()
            boro_profile['metric_order'] = boro_profile['metric'].map(metric_positions)
            boro_profile = boro_profile.sort_values('metric_order')
            ax.plot(boro_profile['metric_order'].to_numpy(dtype=float) + rank_frequency.METHOD_OFFSETS['BORO'], boro_profile['mean_rank'], color=METHOD_STYLE['BORO']['edge'], lw=1.2, alpha=0.94, solid_capstyle='round', solid_joinstyle='round', zorder=1.6)
            for (metric, method), block in panel.groupby(['metric', 'method'], observed=True, sort=False):
                metric = str(metric)
                method = str(method)
                method_style = METHOD_STYLE[method]
                x = metric_positions[metric] + rank_frequency.METHOD_OFFSETS[method]
                mean_rank = float(block['mean_rank'].iloc[0])
                min_rank = float(block['min_rank'].iloc[0])
                max_rank = float(block['max_rank'].iloc[0])
                ax.plot([x, x], [min_rank, max_rank], color=method_style['edge'], lw=0.76 if method == 'BORO' else 0.46, alpha=0.9 if method == 'BORO' else 0.32, solid_capstyle='round', zorder=1)
                observed = block.loc[block['seed_count'].gt(0)]
                ax.scatter(np.full(len(observed), x), observed['rank'], s=[rank_frequency._bubble_area(value) for value in observed['seed_share_percent']], facecolors=method_style['color'], edgecolors=method_style['edge'], linewidths=0.74 if method == 'BORO' else 0.4, alpha=0.96 if method == 'BORO' else 0.52, zorder=2)
                ax.scatter([x], [mean_rank], marker='x', s=13 if method == 'BORO' else 9, linewidths=0.76 if method == 'BORO' else 0.55, color='#111111' if method == 'BORO' else '#4F5459', alpha=1.0 if method == 'BORO' else 0.68, zorder=4)
            for rank in range(1, len(METHODS) + 1):
                ax.axhline(rank, color='#DDE1E5', lw=0.42, zorder=0)
            for boundary_x in np.arange(0.5, len(DISPLAY_METRICS) - 0.49, 1.0):
                ax.axvline(boundary_x, color='#D8DDE2', lw=0.5, alpha=0.72, zorder=0.2)
            ax.set_xlim(-0.58, len(DISPLAY_METRICS) - 0.42)
            ax.set_ylim(len(METHODS) + 0.38, 0.62)
            ax.set_yticks(range(1, len(METHODS) + 1))
            ax.set_xticks(range(len(DISPLAY_METRICS)))
            if row_index == 3:
                ax.set_xticklabels(DISPLAY_METRICS, rotation=38, ha='right', rotation_mode='anchor')
                ax.tick_params(axis='x', pad=1.2)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', length=0)
            if col_index == 0:
                ax.set_yticklabels([str(rank) for rank in range(1, len(METHODS) + 1)])
                ax.tick_params(axis='y', pad=2.0)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', length=0)
            ax.spines['left'].set_color('#42464A')
            ax.spines['bottom'].set_color('#42464A')
            ax.spines['left'].set_linewidth(0.65)
            ax.spines['bottom'].set_linewidth(0.65)
            strip = patches.Rectangle((0, 1.015), 1, 0.145, transform=ax.transAxes, facecolor=style['band'], edgecolor='#C9CDD1', linewidth=0.58, clip_on=False, zorder=6)
            ax.add_patch(strip)
            ax.plot([0, 1], [1.16, 1.16], transform=ax.transAxes, color=style['color'], lw=1.15, clip_on=False, zorder=7)
            ax.text(0.5, 1.085, 'DBPedia-Entity' if setting.dataset == 'DBPedia' else setting.dataset, transform=ax.transAxes, ha='center', va='center', fontsize=6.35, fontweight='bold', color='#292929', clip_on=False, zorder=8)
        first_pos = row_axes[0].get_position()
        row_centers.append((group, (first_pos.y0 + first_pos.y1) / 2))
    for group, center_y in row_centers:
        fig.text(GROUP_LABEL_RIGHT_X, center_y, GROUP_STYLE[group]['label'], ha='right', va='center', fontsize=6.25, fontweight='bold', color=GROUP_STYLE[group]['color'], linespacing=0.92)

def render_figure(paired_path: Path, settings_path: Path, effects_path: Path, base_metrics_path: Path, indomain_metrics_path: Path, zeroshot_metrics_path: Path, output_dir: Path, dpi: int, per_seed_path: Path | None=None) -> list[Path]:
    paired = pd.read_csv(paired_path)
    settings = pd.read_csv(settings_path).sort_values('setting_index')
    effects = pd.read_csv(effects_path)
    if effects['win'].dtype != bool:
        effects['win'] = effects['win'].astype(str).str.lower().eq('true')
    _validate_inputs(paired, settings, effects)
    multimetric_runs = _load_multimetric_runs(base_metrics_path, indomain_metrics_path, zeroshot_metrics_path, paired, settings, per_seed_path)
    panel_b_observations, panel_b_frequency = rank_frequency._rank_runs(multimetric_runs)
    width_in = 180.0 / 25.4
    height_in = 225.0 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), facecolor='white')
    outer = fig.add_gridspec(2, 1, height_ratios=[0.94, 1.06], left=0.205, right=0.965, top=0.985, bottom=0.055, hspace=0.205)
    _plot_panel_a(fig, outer[0], paired, settings)
    _plot_panel_b(fig, outer[1], multimetric_runs)
    base = output_dir / 'fig5'
    outputs = [base.with_suffix('.svg'), base.with_suffix('.pdf'), base.with_suffix('.png'), base.with_suffix('.tiff')]
    fig.savefig(outputs[0], format='svg', facecolor='white')
    fig.savefig(outputs[1], format='pdf', facecolor='white')
    fig.savefig(outputs[2], format='png', dpi=dpi, facecolor='white')
    fig.savefig(outputs[3], format='tiff', dpi=dpi, facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    return outputs

def main():
    root = Path(__file__).resolve().parents[3]
    data = root / 'data/source/ir'
    out = Path(os.environ.get('BORO_FIGURE_OUTPUT', str(root / 'results/figures')))
    render_figure(data / 'panel_a_paired_deltas.csv', data / 'panel_a_setting_summary.csv', data / 'panel_b_metric_effects.csv', None, None, None, out, 300, per_seed_path=data / 'main_per_seed.csv')
if __name__ == '__main__':
    main()
