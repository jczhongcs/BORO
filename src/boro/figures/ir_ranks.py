from __future__ import annotations
import argparse
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 6.3, 'axes.labelsize': 6.7, 'axes.linewidth': 0.65, 'axes.spines.right': False, 'axes.spines.top': False, 'xtick.labelsize': 5.6, 'ytick.labelsize': 6.1, 'xtick.major.width': 0.55, 'ytick.major.width': 0.55, 'xtick.major.size': 2.2, 'ytick.major.size': 2.2, 'legend.frameon': False, 'legend.fontsize': 6.0, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
METHODS = ['InfoNCE', 'Triplet', 'ADNCE', 'AoE', 'DMCL', 'BORO']
METRICS = ['R@10', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'MRR@10']
METHOD_STYLE = {'InfoNCE': {'color': '#9DA4AA', 'edge': '#727980'}, 'Triplet': {'color': '#AF96CA', 'edge': '#8466A6'}, 'ADNCE': {'color': '#879CC1', 'edge': '#5F78A6'}, 'AoE': {'color': '#5DA394', 'edge': '#3A7E70'}, 'DMCL': {'color': '#78B7C7', 'edge': '#4F8E9F'}, 'BORO': {'color': '#E1546B', 'edge': '#B62F49'}}
GROUP_STYLE = {'In-domain / random': {'color': '#4B77B3', 'band': '#F7F9FC', 'label': 'In-domain /\nrandom'}, 'In-domain / BM25': {'color': '#278C88', 'band': '#F6FAFA', 'label': 'In-domain /\nBM25'}, 'Zero-shot': {'color': '#7660A3', 'band': '#FAF8FC', 'label': 'Zero-shot'}}
GROUP_ORDER = ['In-domain / random', 'In-domain / BM25', 'Zero-shot']
METHOD_OFFSETS = dict(zip(METHODS, np.linspace(-0.34, 0.34, len(METHODS))))

def _rank_runs(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    identifiers = ['setting_index', 'setting', 'group', 'dataset', 'sampler', 'method', 'seed']
    observations = runs.melt(id_vars=identifiers, value_vars=METRICS, var_name='metric', value_name='score')
    comparison_keys = ['setting_index', 'metric', 'seed']
    comparison_sizes = observations.groupby(comparison_keys)['method'].nunique()
    expected_comparisons = 9 * len(METRICS) * 5
    if len(comparison_sizes) != expected_comparisons or not comparison_sizes.eq(len(METHODS)).all():
        raise ValueError(f'Expected {expected_comparisons} six-method seed-level ranking comparisons')
    observations['tied_score'] = observations.groupby(comparison_keys)['score'].transform(lambda values: values.duplicated(keep=False))
    observations['rank'] = observations.groupby(comparison_keys)['score'].rank(method='min', ascending=False).astype(int)
    if not observations['rank'].between(1, len(METHODS)).all():
        raise ValueError('A seed-level rank lies outside 1–6')
    summary_keys = ['setting_index', 'setting', 'group', 'dataset', 'sampler', 'metric', 'method']
    rank_summary = observations.groupby(summary_keys, as_index=False).agg(mean_rank=('rank', 'mean'), min_rank=('rank', 'min'), max_rank=('rank', 'max'), n_seeds=('seed', 'nunique'), tied_seed_count=('tied_score', 'sum'))
    expected_summaries = 9 * len(METRICS) * len(METHODS)
    if len(rank_summary) != expected_summaries or not rank_summary['n_seeds'].eq(5).all():
        raise ValueError(f'Expected {expected_summaries} setting–metric–method summaries with five seeds each')
    ranks = pd.DataFrame({'rank': range(1, len(METHODS) + 1)})
    frequency = rank_summary[summary_keys + ['mean_rank', 'min_rank', 'max_rank', 'n_seeds', 'tied_seed_count']].merge(ranks, how='cross').merge(observations.groupby(summary_keys + ['rank'], as_index=False).size().rename(columns={'size': 'seed_count'}), on=summary_keys + ['rank'], how='left', validate='one_to_one')
    frequency['seed_count'] = frequency['seed_count'].fillna(0).astype(int)
    frequency['seed_share_percent'] = 100.0 * frequency['seed_count'] / frequency['n_seeds']
    sums = frequency.groupby(summary_keys)['seed_share_percent'].sum()
    if not np.allclose(sums.to_numpy(), 100.0, atol=1e-12, rtol=0):
        raise ValueError('Rank-frequency percentages do not sum to 100%')
    observations['metric'] = pd.Categorical(observations['metric'], METRICS, ordered=True)
    observations['method'] = pd.Categorical(observations['method'], METHODS, ordered=True)
    observations = observations.sort_values(['setting_index', 'metric', 'method', 'seed'])
    frequency['metric'] = pd.Categorical(frequency['metric'], METRICS, ordered=True)
    frequency['method'] = pd.Categorical(frequency['method'], METHODS, ordered=True)
    frequency = frequency.sort_values(['setting_index', 'metric', 'method', 'rank'])
    return (observations, frequency)

def _bubble_area(share_percent: float) -> float:
    return 7.5 + 0.34 * share_percent

def _draw_header(fig: plt.Figure, spec) -> None:
    ax = fig.add_subplot(spec)
    ax.set_axis_off()
    ax.text(-0.055, 0.66, 'b', transform=ax.transAxes, ha='left', va='center', fontsize=9.0, fontweight='bold', clip_on=False)
    method_handles = [Line2D([0], [0], marker='o', linestyle='-' if method == 'BORO' else 'none', linewidth=1.25 if method == 'BORO' else 0, markersize=4.8, markerfacecolor=METHOD_STYLE[method]['color'], markeredgecolor=METHOD_STYLE[method]['edge'], markeredgewidth=0.55, color=METHOD_STYLE[method]['edge'], alpha=1.0 if method == 'BORO' else 0.72, label=method) for method in METHODS]
    method_legend = ax.legend(handles=method_handles, labels=METHODS, ncol=len(METHODS), loc='upper left', bbox_to_anchor=(0.0, 0.98), handletextpad=0.35, columnspacing=1.08, borderaxespad=0)
    for label in method_legend.get_texts():
        if label.get_text() == 'BORO':
            label.set_fontweight('bold')
    ax.add_artist(method_legend)
    shares = [20, 40, 60, 80, 100]
    size_handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=np.sqrt(_bubble_area(share)), markerfacecolor='#3A3A3A', markeredgecolor='#3A3A3A', label=str(share)) for share in shares]
    size_legend = ax.legend(handles=size_handles, labels=[str(share) for share in shares], ncol=5, loc='upper right', bbox_to_anchor=(1.0, 0.98), handletextpad=0.15, columnspacing=0.55, borderaxespad=0, title='Seed share (%)', title_fontsize=6.0)
    ax.add_artist(size_legend)
    ax.plot([0.355, 0.355], [0.1, 0.34], transform=ax.transAxes, color='#5A5A5A', lw=0.75, clip_on=False)
    ax.scatter([0.355], [0.22], transform=ax.transAxes, marker='x', s=13, linewidths=0.75, color='#111111', clip_on=False)
    ax.text(0.37, 0.22, 'mean rank; line, observed range', transform=ax.transAxes, ha='left', va='center', fontsize=5.7, color='#333333')

def _plot_grid(fig: plt.Figure, spec, settings: pd.DataFrame, frequency: pd.DataFrame) -> None:
    grid = spec.subgridspec(3, 3, hspace=0.62, wspace=0.26)
    metric_positions = dict(zip(METRICS, range(len(METRICS))))
    for setting in settings.itertuples(index=False):
        row, col = divmod(int(setting.setting_index), 3)
        ax = fig.add_subplot(grid[row, col])
        style = GROUP_STYLE[setting.group]
        ax.set_facecolor(style['band'])
        panel = frequency.loc[frequency['setting_index'].eq(int(setting.setting_index))]
        ax.axhspan(0.62, 1.45, color='#FBECEF', alpha=0.88, zorder=0)
        boro_profile = panel.loc[panel['method'].astype(str).eq('BORO')].drop_duplicates('metric').copy()
        boro_profile['metric_order'] = boro_profile['metric'].map(metric_positions)
        boro_profile = boro_profile.sort_values('metric_order')
        ax.plot(boro_profile['metric_order'].to_numpy(dtype=float) + METHOD_OFFSETS['BORO'], boro_profile['mean_rank'], color=METHOD_STYLE['BORO']['edge'], lw=1.25, alpha=0.92, solid_capstyle='round', solid_joinstyle='round', zorder=1.6)
        for (metric, method), block in panel.groupby(['metric', 'method'], observed=True, sort=False):
            metric = str(metric)
            method = str(method)
            method_style = METHOD_STYLE[method]
            x = metric_positions[metric] + METHOD_OFFSETS[method]
            mean_rank = float(block['mean_rank'].iloc[0])
            min_rank = float(block['min_rank'].iloc[0])
            max_rank = float(block['max_rank'].iloc[0])
            ax.plot([x, x], [min_rank, max_rank], color=method_style['edge'], lw=0.78 if method == 'BORO' else 0.48, alpha=0.88 if method == 'BORO' else 0.34, solid_capstyle='round', zorder=1)
            observed = block.loc[block['seed_count'].gt(0)]
            ax.scatter(np.full(len(observed), x), observed['rank'], s=[_bubble_area(value) for value in observed['seed_share_percent']], facecolors=method_style['color'], edgecolors=method_style['edge'], linewidths=0.76 if method == 'BORO' else 0.42, alpha=0.95 if method == 'BORO' else 0.54, zorder=2)
            ax.scatter([x], [mean_rank], marker='x', s=13.5 if method == 'BORO' else 9.5, linewidths=0.78 if method == 'BORO' else 0.58, color='#111111' if method == 'BORO' else '#4F5459', alpha=1.0 if method == 'BORO' else 0.7, zorder=4)
        for rank in range(1, len(METHODS) + 1):
            ax.axhline(rank, color='#DDE1E5', lw=0.45, zorder=0)
        for metric_x in range(len(METRICS)):
            ax.axvline(metric_x, color='#E8EAED', lw=0.38, zorder=0)
        ax.set_xlim(-0.58, len(METRICS) - 0.42)
        ax.set_ylim(len(METHODS) + 0.38, 0.62)
        ax.set_yticks(range(1, len(METHODS) + 1))
        ax.set_xticks(range(len(METRICS)))
        if row == 2:
            ax.set_xticklabels(METRICS, rotation=38, ha='right', rotation_mode='anchor')
            ax.tick_params(axis='x', pad=1.4)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', length=0)
        if col == 0:
            ax.set_yticklabels([str(rank) for rank in range(1, len(METHODS) + 1)])
            ax.tick_params(axis='y', pad=2.2)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_color('#42464A')
        ax.spines['bottom'].set_color('#42464A')
        ax.spines['left'].set_linewidth(0.65)
        ax.spines['bottom'].set_linewidth(0.65)
        strip = patches.Rectangle((0, 1.015), 1, 0.145, transform=ax.transAxes, facecolor=style['band'], edgecolor='#C9CDD1', linewidth=0.6, clip_on=False, zorder=6)
        ax.add_patch(strip)
        ax.plot([0, 1], [1.16, 1.16], transform=ax.transAxes, color=style['color'], lw=1.2, clip_on=False, zorder=7)
        ax.text(0.5, 1.085, setting.dataset, transform=ax.transAxes, ha='center', va='center', fontsize=6.5, fontweight='bold', color='#292929', clip_on=False, zorder=8)
        if col == 0:
            ax.text(-0.17, 0.5, style['label'], transform=ax.transAxes, ha='right', va='center', fontsize=6.3, fontweight='bold', color=style['color'], linespacing=0.92, clip_on=False)
    fig.text(0.03, 0.505, 'Rank (1 = best)', rotation=90, ha='center', va='center', fontsize=6.8, color='#222222')
