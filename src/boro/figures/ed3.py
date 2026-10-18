from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 7.0, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'svg.fonttype': 'none', 'axes.titlesize': 7.5, 'axes.titleweight': 'bold', 'axes.labelsize': 7.0, 'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'xtick.major.width': 0.65, 'ytick.major.width': 0.65, 'xtick.major.size': 2.6, 'ytick.major.size': 2.6, 'legend.fontsize': 6.5, 'legend.frameon': False, 'lines.linewidth': 1.0, 'savefig.facecolor': 'white', 'savefig.transparent': False})
MM_TO_IN = 1.0 / 25.4
BLACK = '#222222'
MID_GREY = '#666666'
LIGHT_GREY = '#D3D6DA'
RAW_GREY = '#7F8790'
BORO_CORAL = '#D94368'
PRSM_ANALYSIS_SEED = 20260716
PRSM_BOOTSTRAP_REPLICATES = 10000

def add_panel_label(ax: plt.Axes, label: str, x: float=-0.14, y: float=1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight='bold', ha='left', va='bottom', color=BLACK)

def finish_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction='out', color=BLACK, labelcolor=BLACK)
    ax.spines['left'].set_color(BLACK)
    ax.spines['bottom'].set_color(BLACK)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def derive_standard_dt_entity_differences(accepted: pd.DataFrame) -> pd.DataFrame:
    entities = ['spectrum_id', 'proteoform_key', 'proteo_cluster_id']
    keys = ['seed', 'file_id', 'species_group', 'generalization_regime']
    required = {'method', *keys, *entities, 'is_target', 'is_target_xml', 'accepted_standard_dt_1pct'}
    missing = required.difference(accepted.columns)
    if missing:
        raise ValueError(f'Corrected accepted-target table is missing columns: {sorted(missing)}')
    if set(accepted['method']) != {'BORO', 'pAUC'}:
        raise ValueError('Expected accepted targets for BORO and pAUC')
    if not accepted['accepted_standard_dt_1pct'].astype(bool).all():
        raise ValueError('Corrected accepted-target table contains rejected candidates')
    if not accepted['is_target'].eq(1).all() or not accepted['is_target_xml'].eq(1).all():
        raise ValueError('Corrected accepted-target table contains decoys or XML label mismatches')
    if accepted[entities].isna().any().any():
        raise ValueError('Corrected accepted-target table has missing de-duplication keys')
    counts = accepted.groupby(['method', *keys], as_index=False).agg(spectrum_id=('spectrum_id', 'nunique'), proteoform_key=('proteoform_key', 'nunique'), proteo_cluster_id=('proteo_cluster_id', 'nunique'))
    file_counts = pd.read_csv(DATA / 'fdr_per_file.csv')
    grid = file_counts[file_counts.method.isin(['BORO', 'pAUC']) & file_counts.fdr.eq(.01)].rename(
        columns={'species': 'species_group', 'regime': 'generalization_regime'})
    counts = grid[['method', *keys, 'targets']].merge(counts, on=['method', *keys], how='left', validate='one_to_one')
    missing = counts[entities].isna().all(axis=1)
    assert counts.loc[missing, 'targets'].eq(0).all()
    assert not counts.loc[~missing, entities].isna().any().any()
    counts.loc[missing, entities] = 0
    counts = counts.drop(columns='targets')
    for method in ['BORO', 'pAUC']:
        method_counts = counts.loc[counts['method'].eq(method)]
        if len(method_counts) != 175:
            raise ValueError(f'Expected 175 seed--file units for {method}; found {len(method_counts)}')
        if method_counts['seed'].nunique() != 5 or method_counts['file_id'].nunique() != 35:
            raise ValueError(f'Incomplete corrected standard D/T coverage for {method}')
    boro = counts.loc[counts['method'].eq('BORO')].drop(columns='method')
    pauc = counts.loc[counts['method'].eq('pAUC')].drop(columns='method')
    boro = boro.rename(columns={entity: f'{entity}_boro' for entity in entities})
    pauc = pauc.rename(columns={entity: f'{entity}_pauc' for entity in entities})
    subset = boro.merge(pauc, on=keys, how='inner', validate='one_to_one')
    for entity in entities:
        subset[f'{entity}_net'] = subset[f'{entity}_boro'] - subset[f'{entity}_pauc']
    subset.insert(0, 'comparison', 'BORO − pAUC')
    subset.insert(1, 'comparator', 'pAUC')
    subset.insert(6, 'fdr_threshold', 0.01)
    return subset

def stable_rng(label: str) -> np.random.Generator:
    suffix = int(hashlib.sha256(label.encode('utf-8')).hexdigest()[:8], 16)
    return np.random.default_rng(PRSM_ANALYSIS_SEED + suffix)

def hierarchical_entity_interval(subset: pd.DataFrame, value: str, entity: str) -> tuple[float, float]:
    seeds = np.array(sorted(subset['seed'].unique()))
    species = np.array(sorted(subset['species_group'].unique()))
    files = {group: np.array(sorted(subset.loc[subset['species_group'].eq(group), 'file_id'].unique())) for group in species}
    lookup = subset.set_index(['seed', 'species_group', 'file_id'])[value].to_dict()
    label = f'entity|BORO − pAUC|0.01|{entity}'
    rng = stable_rng(label)
    draws = np.empty(PRSM_BOOTSTRAP_REPLICATES, dtype=float)
    for iteration in range(PRSM_BOOTSTRAP_REPLICATES):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_species = rng.choice(species, size=len(species), replace=True)
        values: list[float] = []
        for group in sampled_species:
            sampled_files = rng.choice(files[group], size=len(files[group]), replace=True)
            values.extend((lookup[int(seed), group, file_id] for seed in sampled_seeds for file_id in sampled_files))
        draws[iteration] = np.mean(values)
    low, high = np.quantile(draws, [0.025, 0.975])
    return (float(low), float(high))

def validate_entity_data(accepted: pd.DataFrame, effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = derive_standard_dt_entity_differences(accepted)
    effect_subset = effects.loc[effects['comparison'].eq('BORO − pAUC')].set_index('entity')
    columns = {'spectrum_id': 'spectrum_id_net', 'proteoform_key': 'proteoform_key_net', 'proteo_cluster_id': 'proteo_cluster_id_net'}
    if set(columns).difference(effect_subset.index):
        raise ValueError('Corrected standard D/T effect summary is incomplete')
    for entity, column in columns.items():
        values = subset[column].to_numpy(float)
        row = effect_subset.loc[entity]
        if not np.isclose(values.mean(), row['mean'], atol=1e-09):
            raise ValueError(f'Corrected mean mismatch for {entity}')
        low, high = hierarchical_entity_interval(subset, column, entity)
        if not np.isclose(low, row['ci_low'], atol=1e-09) or not np.isclose(high, row['ci_high'], atol=1e-09):
            raise ValueError(f'Corrected hierarchical interval mismatch for {entity}')
        checks = {'positive': int((values > 0).sum()), 'negative': int((values < 0).sum()), 'zero': int((values == 0).sum())}
        for field, observed in checks.items():
            if observed != int(row[field]):
                raise ValueError(f'Corrected {field}-count mismatch for {entity}')
        if int(row['n_datasets']) != 35:
            raise ValueError(f'Corrected file count mismatch for {entity}')
    return (subset, effect_subset)

def make_extended_data_figure_3(audit: dict) -> dict[str, str]:
    accepted_path = DATA / 'accepted_candidates.csv.gz'
    effects_path = DATA / 'entity_summary.csv'
    accepted = pd.read_csv(accepted_path)
    effects = pd.read_csv(effects_path)
    subset, effect_subset = validate_entity_data(accepted, effects)
    entity_specs = [('spectrum_id', 'spectrum_id_net', 'Unique spectra'), ('proteoform_key', 'proteoform_key_net', 'Unique proteoform signatures'), ('proteo_cluster_id', 'proteo_cluster_id_net', 'Unique proteoform clusters')]
    fig = plt.figure(figsize=(7.2, 3.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.65], left=0.205, right=0.985, bottom=0.18, top=0.82, wspace=0.03)
    ax = fig.add_subplot(grid[0, 0])
    ax_stats = fig.add_subplot(grid[0, 1], sharey=ax)
    rng = np.random.default_rng(20260720)
    y_positions = [2, 1, 0]
    stats_rows = []
    for y, (entity, column, label) in zip(y_positions, entity_specs):
        values = subset[column].to_numpy(float)
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.scatter(values, np.full(len(values), y) + jitter, s=8, color=RAW_GREY, alpha=0.32, edgecolor='none', rasterized=False, zorder=2)
        row = effect_subset.loc[entity]
        mean = float(row['mean'])
        low = float(row['ci_low'])
        high = float(row['ci_high'])
        positive = int(row['positive'])
        negative = int(row['negative'])
        zero = int(row['zero'])
        ax.plot([low, high], [y, y], color=BORO_CORAL, linewidth=2.2, zorder=5)
        ax.plot([low, low], [y - 0.06, y + 0.06], color=BORO_CORAL, linewidth=1.1, zorder=5)
        ax.plot([high, high], [y - 0.06, y + 0.06], color=BORO_CORAL, linewidth=1.1, zorder=5)
        ax.scatter(mean, y, s=43, marker='D', facecolor=BORO_CORAL, edgecolor=BLACK, linewidth=0.55, zorder=6)
        ax_stats.text(0.03, y + 0.08, f'{mean:.2f}  [{low:.2f}, {high:.2f}]', transform=ax_stats.get_yaxis_transform(), ha='left', va='center', fontsize=6.8, color=BLACK)
        ax_stats.text(0.03, y - 0.12, f'{positive} / {negative} / {zero}  (+ / − / 0)', transform=ax_stats.get_yaxis_transform(), ha='left', va='center', fontsize=6.3, color=MID_GREY)
        stats_rows.append({'entity': entity, 'label': label, 'mean': mean, 'ci95_low': low, 'ci95_high': high, 'positive': positive, 'negative': negative, 'zero': zero})
    for boundary in [0.5, 1.5]:
        ax.axhline(boundary, color='#E4E6E8', linewidth=0.55, zorder=0)
        ax_stats.axhline(boundary, color='#E4E6E8', linewidth=0.55, zorder=0)
    ax.axvline(0, color=MID_GREY, linestyle='--', linewidth=0.8, zorder=1)
    extent = subset[[spec[1] for spec in entity_specs]].to_numpy(float)
    padding = .05 * max(float(np.ptp(extent)), 1.)
    ax.set_xlim(float(extent.min()) - padding, float(extent.max()) + padding)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    ax.set_ylim(-0.42, 2.42)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([spec[2] for spec in entity_specs])
    ax.set_xlabel('Paired difference (BORO − pAUC)')
    ax.set_title('De-duplicated coverage at 1% FDR', loc='left', pad=6)
    finish_axis(ax)
    handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=3.2, markerfacecolor=RAW_GREY, markeredgecolor='none', alpha=0.55, label='seed–dataset unit'), Line2D([0], [0], marker='D', color=BORO_CORAL, linewidth=1.8, markersize=4.5, markerfacecolor=BORO_CORAL, markeredgecolor=BLACK, markeredgewidth=0.4, label='mean and 95% CI')]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 1.18), ncol=2, handlelength=1.8, columnspacing=1.2, borderaxespad=0)
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(-0.42, 2.42)
    ax_stats.set_xticks([])
    ax_stats.tick_params(axis='y', left=False, labelleft=False)
    for spine in ax_stats.spines.values():
        spine.set_visible(False)
    ax_stats.text(0.03, 1.08, 'Mean [95% CI]', transform=ax_stats.transAxes, ha='left', va='bottom', fontsize=6.8, fontweight='bold', color=BLACK)
    audit['extended_data_figure_3'] = {'rows': int(len(subset)), 'seeds': int(subset['seed'].nunique()), 'files': int(subset['file_id'].nunique()), 'comparison': 'BORO − pAUC', 'fdr_threshold': 0.01, 'acceptance_rule': 'inclusive cumulative D/T through the last feasible ranked candidate', 'bootstrap_analysis_seed': PRSM_ANALYSIS_SEED, 'bootstrap_replicates': PRSM_BOOTSTRAP_REPLICATES, 'bootstrap_rng_rule': 'analysis seed plus first 8 SHA-256 hex digits of the entity label', 'accepted_target_source_sha256': sha256_file(accepted_path), 'effect_summary_source_sha256': sha256_file(effects_path), 'effects': stats_rows}
    return save_bundle(fig, OUT / 'Extended_Data_Fig_3_PrSM_entity_coverage', 'Extended Data Figure 3: PrSM de-duplicated entity coverage')
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data/source/prsm'
OUT = Path(os.environ.get('BORO_FIGURE_OUTPUT', str(ROOT / 'results/figures')))

def save_bundle(fig, stem, title):
    paths = {}
    for ext in ['pdf', 'svg', 'png', 'tiff']:
        path = OUT / f'ed3.{ext}'
        fig.savefig(path, dpi=300)
        paths[ext] = str(path)
    plt.close(fig)
    return paths
if __name__ == '__main__':
    make_extended_data_figure_3({})
