import numpy as np
import pandas as pd
from scipy.stats import t
from matplotlib.colors import to_rgba

def draw_regional_panel(fig, root, axes_mm, quiet, title, colors, styles, markers, band):
    source = root / 'data/source/controlled'
    stats = pd.read_csv(source / 'regional_gradient_mean_ci.csv', float_precision='round_trip')
    raw = pd.read_csv(root / 'data/source/controlled/score_gradient_per_dataset.csv', float_precision='round_trip')
    records = []
    for (seed, method), g in raw.groupby(['seed', 'method']):
        curve = g.set_index('nearest_sample_percent')['cumulative_absolute_gradient_percent'].sort_index()
        shares = np.diff(curve.loc[list(range(0, 101, 10))].to_numpy())
        np.testing.assert_allclose(shares.sum(), 100, rtol=0, atol=1e-10)
        records.extend(((seed, method, lo, share) for lo, share in zip(range(0, 100, 10), shares)))
    bins = pd.DataFrame(records, columns=['seed', 'method', 'region_start_percent', 'share'])
    expected = bins.groupby(['method', 'region_start_percent']).share.agg(['mean', 'std', 'count'])
    for row in stats.itertuples():
        check = expected.loc[row.method, row.region_start_percent]
        assert check['count'] == row.count == 20
        half = t.ppf(0.975, 19) * check['std'] / np.sqrt(20)
        np.testing.assert_allclose([row.mean, row.ci95_low, row.ci95_high], [check['mean'], check['mean'] - half, check['mean'] + half], rtol=0, atol=1e-11)
    main = axes_mm(fig, 15, 63, 159, 42)
    quiet(main)
    title(fig, 15, 109, 'c', 'Score-gradient allocation')
    small = main.inset_axes([0.52, 0.4, 0.455, 0.56])
    quiet(small)
    labels = [f'{lo}–{lo + 10}' for lo in range(0, 100, 10)]
    plotted = []
    for ax, count, kind in [(main, 10, 'line'), (small, 5, 'bar')]:
        ax.axvspan(-0.49, 0.49, color=band, zorder=0)
        for mi, method in enumerate(['BORO', 'BCE', 'Focal']):
            s = stats.loc[stats.method.eq(method)].sort_values('region_start_percent').iloc[:count]
            x = np.arange(count, dtype=float)
            y = s['mean'].to_numpy()
            err = np.vstack([y - s.ci95_low.to_numpy(), s.ci95_high.to_numpy() - y])
            if kind == 'line':
                obj = ax.errorbar(x, y, yerr=err, color=colors[method], linestyle=styles[method], linewidth=1.15, marker=markers[method], markersize=3.3, markeredgewidth=0.55, markerfacecolor='white', elinewidth=0.65, capsize=1.3, capthick=0.65, zorder=4 if method == 'BORO' else 3)
                np.testing.assert_array_equal(obj.lines[0].get_ydata(), y)
            else:
                x += (mi - 1) * 0.23
                obj = ax.bar(x, y, width=0.23 * 0.92, color=to_rgba(colors[method], 0.7), edgecolor=colors[method], linewidth=0.6, zorder=3)
                np.testing.assert_array_equal([p.get_height() for p in obj], y)
                ax.errorbar(x, y, yerr=err, fmt='none', ecolor=colors[method], elinewidth=0.6, capsize=1.1, capthick=0.6, zorder=4)
            for row in s.itertuples():
                plotted.append(dict(panel=kind, method=method, lower_percentile=row.region_start_percent, upper_percentile=row.region_end_percent, mean=row.mean, ci95_low=row.ci95_low, ci95_high=row.ci95_high))
        ax.set(xlim=(-0.65, count - 0.35), ylim=(0, 45), yticks=[0, 10, 20, 30, 40], xticks=range(count), xticklabels=labels[:count], xlabel='Distance to boundary (percentile)', ylabel='Gradient contribution (%)')
        if kind == 'bar':
            ax.tick_params(labelsize=5.6, length=2, pad=2)
            ax.set_xlabel('Distance to boundary (percentile)', fontsize=5.8, labelpad=3)
            ax.set_ylabel('Gradient contribution (%)', fontsize=5.8, labelpad=3)
    small.set_title('Nearest 50% of candidates', loc='left', pad=4, fontsize=6.3, weight='bold')
    approved = pd.read_csv(source / 'regional_gradient_plotted_values.csv', float_precision='round_trip')
    pd.testing.assert_frame_equal(pd.DataFrame(plotted), approved, check_exact=True)
    gradient10 = stats.loc[stats.region_start_percent.eq(0)].set_index('method')['mean'].to_dict()
    return (gradient10, {})
