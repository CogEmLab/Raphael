"""Quality-control figure for one sorted session, saved next to the data."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from raphael.style import styled

COLOURS = {'good': 'tab:green', 'mua': 'tab:orange', 'noise': 'tab:grey',
           'non_soma': 'tab:purple'}


@styled
def session_figure(units, channels, analyzer, out_dir, title='', max_spikes=50_000):
    """Units on the probe by label; spike depth over time (drift); firing
    rate against depth."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_probe, ax_drift, ax_rate) = plt.subplots(
        1, 3, figsize=(10, 3.5), gridspec_kw={'width_ratios': [1, 1.6, 1]})

    used = channels['used'].to_numpy()
    ax_probe.scatter(channels['x_um'][used], channels['y_um'][used], s=1, c='0.85',
                     marker='s', label='channel')
    ax_probe.scatter(channels['x_um'][~used], channels['y_um'][~used], s=3, c='k',
                     marker='x', label='bad channel')
    for name, colour in COLOURS.items():
        rows = units['bombcell_label'] == name
        ax_probe.scatter(units['x'][rows], units['y'][rows], s=6,
                         c=colour, label=f'{name} ({rows.sum()})')
    ax_probe.set_xlabel('x (µm)')
    ax_probe.set_ylabel('depth from tip (µm)')
    ax_probe.legend(fontsize=6, markerscale=2, ncol=2, loc='upper center',
                    bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax_probe.set_title('units on the probe', fontsize=9)

    spikes = analyzer.sorting.to_spike_vector()
    locations = analyzer.get_extension('spike_locations').get_data()
    amplitudes = np.abs(analyzer.get_extension('spike_amplitudes').get_data())
    pick = np.random.default_rng(0).choice(len(spikes), min(max_spikes, len(spikes)),
                                           replace=False)
    t = spikes['sample_index'][pick] / analyzer.sampling_frequency
    order = np.argsort(amplitudes[pick])
    lo, hi = np.percentile(amplitudes[pick], [5, 95]) if len(pick) else (0, 1)
    ax_drift.scatter(t[order], locations['y'][pick][order], s=0.5, c=amplitudes[pick][order],
                     cmap='Greys', vmin=lo, vmax=hi, rasterized=True)
    ax_drift.set_xlabel('time (s)')
    ax_drift.set_title('spike depth over time (drift check)', fontsize=9)
    ax_drift.sharey(ax_probe)

    for name, colour in COLOURS.items():
        rows = units['bombcell_label'] == name
        ax_rate.scatter(units['firing_rate'][rows], units['y'][rows], s=6, c=colour)
    ax_rate.set_xscale('log')
    ax_rate.xaxis.set_minor_formatter(NullFormatter())
    ax_rate.set_xlabel('firing rate (Hz)')
    ax_rate.set_title('rate by depth', fontsize=9)
    ax_rate.sharey(ax_probe)

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path = out_dir / 'units.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
