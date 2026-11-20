"""Quality-control figure for one photometry session, saved next to the data."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from raphael.photometry import processing


def session_figure(rec, times, out, fits, settings, out_dir, title=''):
    """Per channel: raw signal (after trimming and filtering) with its
    bleaching fit; per signal: the reference fit and dF/F."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n in fits if n in rec.channels]
    rows = len(names) + len(out)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 1.6 * rows), sharex=True, squeeze=False)
    axes = axes[:, 0]
    s = {**processing.DEFAULTS, **settings}
    for ax, name in zip(axes, names, strict=False):
        c = rec.channels[name]
        keep = processing.trim(c.times, s['trim_start_s'])
        filtered = processing.lowpass(c.values[keep], c.rate, s['lowpass_hz'], s['filter_order'])
        baseline, _ = processing.fit_bleaching(c.times[keep], filtered, s['bleaching'])
        ax.plot(c.times, c.values, lw=0.3, color='0.75', label='raw')
        ax.plot(c.times[keep], filtered, lw=0.4, color='k', label='filtered')
        ax.plot(c.times[keep], baseline, lw=1, color='tab:red', label='bleaching fit')
        ax.set_ylabel(f'{name} (V)', fontsize=8)
        ax.axvspan(c.times[0], c.times[keep][0], color='0.9', lw=0)
    axes[0].legend(fontsize=6, loc='upper right', ncol=3)
    for ax, (name, series) in zip(axes[len(names):], out.items(), strict=True):
        ref = fits[name].get('reference')
        r = fits[name].get('motion_r')
        ax.plot(times, series['dff'], lw=0.4, color='tab:green' if 'green' in name else 'k')
        ax.axhline(0, color='0.6', lw=0.5)
        ax.set_ylabel(f'{name} dF/F (%)', fontsize=8)
        ax.set_title(f'{name}: reference {ref or "none"}'
                     + (f', r = {r:.2f}' if r is not None else '')
                     + f', sd {np.std(series["dff"]):.2f}%', fontsize=8, loc='left')
    axes[-1].set_xlabel('time (s)')
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path = out_dir / f'{title or "photometry"}_photometry.png'
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
