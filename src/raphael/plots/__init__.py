"""Standard neuroscience plots, drawn in the labs' style.

    from raphael import plots, population, style
    style.use()
    pop = population.load('.../sub-M25_ses-D18_typ-OF1_srt-kilosort4').accepted()
    plots.rate_map(ax, pop.data[7], beh, bounds=(0, 100, 0, 100))

Every function takes the axes to draw on and processed data as pynapple
objects: spikes as a `Ts` (one unit) or `TsGroup`, continuous signals
(imaging dF/F, photometry, LFP-derived values) as a `Tsd` or `TsdFrame`.
Behaviour is the session dict the tuning scores use (P_x, P_y, H, S,
`moving`). Each returns what it plotted, so numbers in a figure can be
reported from the same computation.

`example_data()` returns a synthetic session with units of known tuning,
for trying these out. Gallery: docs/plots.md.
"""

import warnings

import numpy as np
import pandas as pd
import pynapple as nap

from raphael import style
from raphael.plots.example import example_data
from raphael.util import gaussian_filter_nan

__all__ = ['autocorrelogram', 'example_data', 'head_direction', 'isi_histogram', 'peth',
           'phase_histogram', 'raster', 'rate_map', 'score_histogram', 'traces',
           'tuning_curve']


# ------------------------------------------------------------- helpers

def _unwrap(activity):
    """A raphael.population.Population's data; anything else unchanged."""
    from raphael.population import Population
    return activity.data if isinstance(activity, Population) else activity


def _is_spikes(activity):
    data = _unwrap(activity)
    return isinstance(data, (nap.Ts, nap.TsGroup)) and not isinstance(data, nap.Tsd)


def _one(activity):
    """A single unit/signal: spikes as a one-unit TsGroup, a signal as a
    one-column TsdFrame. Accepts raphael.population.Population too."""
    data = _unwrap(activity)
    if isinstance(data, nap.Ts) and not isinstance(data, nap.Tsd):
        return nap.TsGroup({0: data}, time_support=data.time_support)
    if isinstance(data, nap.TsGroup):
        if len(data) != 1:
            raise ValueError(f'expected one unit, got a TsGroup of {len(data)}: index it')
        return data
    if isinstance(data, nap.Tsd):
        return nap.TsdFrame(t=data.t, d=data.values[:, None], columns=[0],
                            time_support=data.time_support)
    if isinstance(data, nap.TsdFrame):
        if data.shape[1] != 1:
            raise ValueError(f'expected one signal, got {data.shape[1]} columns: index it')
        return data
    raise TypeError(f'expected spikes (Ts/TsGroup) or a signal (Tsd/TsdFrame), got {type(data)}')


def _ep(ep):
    """pynapple rejects ep=None, so pass it only when given."""
    return {} if ep is None else {'ep': ep}


def _epoch(behaviour, ep):
    if isinstance(ep, str):
        return behaviour[ep]
    return ep


# ------------------------------------------------------------ rate maps

def rate_map(ax, activity, behaviour, bounds, bins=40, ep='moving', smooth=1.5, cmap=None,
             outline=True, peak_label=True, colorbar=False, vmax=None):
    """Firing rate (or mean signal) over the arena, from position P_x, P_y
    in `behaviour`. `bounds`: (x min, x max, y min, y max) in cm, so the map
    has the arena's real shape. Unvisited bins are left blank. Returns the
    map (y x x, Hz or signal units)."""
    one = _one(activity)
    features = nap.TsdFrame(t=behaviour['P_x'].t,
                            d=np.c_[behaviour['P_x'].values, behaviour['P_y'].values],
                            columns=['x', 'y'])
    epoch = _epoch(behaviour, ep) if behaviour is not None else None
    x0, x1, y0, y1 = bounds
    nb = (bins, max(1, round(bins * (y1 - y0) / (x1 - x0))))
    minmax = (x0, x1, y0, y1)
    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')   # unvisited bins: blank
        if _is_spikes(activity):
            maps, _ = nap.compute_2d_tuning_curves(one, features, nb, minmax=minmax, **_ep(epoch))
        else:
            maps, _ = nap.compute_2d_tuning_curves_continuous(one, features, nb, minmax=minmax,
                                                              **_ep(epoch))
    m = np.asarray(next(iter(maps.values())), float).T          # rows = y
    if smooth:
        m = gaussian_filter_nan(m, sigma=smooth)
    cm = (cmap if cmap is not None else style.CMAPS['rate'])
    import matplotlib as mpl
    cm = (mpl.colormaps[cm] if isinstance(cm, str) else cm).with_extremes(bad='white')
    image = ax.imshow(m, origin='lower', extent=bounds, cmap=cm, vmin=0 if _is_spikes(activity)
                      else None, vmax=vmax, interpolation='nearest', aspect='equal')
    style.bare(ax)
    if outline:
        ax.add_patch(mpl.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.5,
                                           ec='0.6'))
    if peak_label:
        peak = np.nanmax(m)
        unit = 'Hz' if _is_spikes(activity) else ''
        ax.text(1.0, -0.02, f'{peak:.1f} {unit}'.strip(), transform=ax.transAxes, ha='right',
                va='top', fontsize=style.FONT['small'])
    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label('Rate (Hz)' if _is_spikes(activity) else 'Signal')
        bar.outline.set_visible(False)
    return m


# -------------------------------------------------------- tuning curves

def _tuning(activity, feature, bins, minmax, ep, min_occupancy_s=0.5):
    """1-D tuning curve; bins visited for under `min_occupancy_s` are NaN
    (blank), not 0."""
    one = _one(activity)
    with np.errstate(invalid='ignore', divide='ignore'):
        if _is_spikes(activity):
            tc = nap.compute_1d_tuning_curves(one, feature, bins, minmax=minmax, **_ep(ep))
        else:
            tc = nap.compute_1d_tuning_curves_continuous(one, feature, bins, minmax=minmax,
                                                         **_ep(ep))
    curve = tc.iloc[:, 0].astype(float)
    sampled = feature.restrict(ep) if ep is not None else feature
    lo, hi = minmax if minmax is not None else (np.nanmin(sampled.values), np.nanmax(sampled.values))
    occupancy, _ = np.histogram(sampled.values, bins=bins, range=(lo, hi))
    dt = np.median(np.diff(sampled.t)) if len(sampled) > 1 else 0
    if len(occupancy) == len(curve):
        curve[occupancy * dt < min_occupancy_s] = np.nan
    return curve


def tuning_curve(ax, activity, feature, bins=20, minmax=None, ep=None, color=None, label=None,
                 xlabel=None, ylabel=None, smooth=None, min_occupancy_s=0.5):
    """Rate (or mean signal) against any 1-D variable, e.g. speed or track
    position. `feature`: a Tsd. Bins visited for under `min_occupancy_s`
    are left blank. Returns the curve (pandas Series, index = bin centres)."""
    curve = _tuning(activity, feature, bins, minmax, ep, min_occupancy_s)
    values = curve.values
    if smooth:
        values = gaussian_filter_nan(values, sigma=smooth)
        curve = pd.Series(values, index=curve.index)
    ax.plot(curve.index, values, color=color, label=label)
    ax.set_xlabel(xlabel or '')
    ax.set_ylabel(ylabel or ('Rate (Hz)' if _is_spikes(activity) else 'Signal'))
    if ax.name != 'polar' and _is_spikes(activity):
        ax.set_ylim(bottom=0)
    return curve


def _circular_stats(angles, weights):
    w = np.nan_to_num(weights)
    vector = np.sum(w * np.exp(1j * angles)) / max(np.sum(w), 1e-12)
    return float(np.angle(vector) % (2 * np.pi)), float(np.abs(vector))


def head_direction(ax, activity, behaviour, bins=36, ep='moving', color=None, annotate=True,
                   variable='H'):
    """Head-direction tuning on polar axes (create `ax` with
    projection='polar'). Returns (curve, preferred direction in rad, mean
    vector length)."""
    if ax.name != 'polar':
        raise ValueError("head_direction needs polar axes: "
                         "fig.add_subplot(..., projection='polar')")
    curve = _tuning(activity, behaviour[variable], bins, (0, 2 * np.pi), _epoch(behaviour, ep))
    angles, values = curve.index.values, np.nan_to_num(curve.values)
    preferred, mvl = _circular_stats(angles, values)
    loop_a, loop_v = np.r_[angles, angles[0]], np.r_[values, values[0]]
    (line,) = ax.plot(loop_a, loop_v, color=color)
    ax.fill(loop_a, loop_v, color=line.get_color(), alpha=0.15, linewidth=0)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]), ['0°', '90°', '180°', '270°'])
    ax.tick_params(axis='x', pad=-2)
    ax.set_yticks([values.max()], [f'{values.max():.0f}'])
    ax.set_rlabel_position(45)
    ax.grid(True, lw=0.4, color='0.85')
    ax.spines['polar'].set_linewidth(0.5)
    if annotate:
        ax.set_title(f'{np.rad2deg(preferred):.0f}°, MVL {mvl:.2f}', fontsize=style.FONT['small'],
                     pad=8)
    return curve, preferred, mvl


def phase_histogram(ax, spikes, phase, bins=18, cycles=2, ep=None, color=None, reference=True):
    """Distribution of spike phases (e.g. theta phase from the lfp stage, in
    rad, 0 to 2 pi), drawn over `cycles` cycles. Returns (counts per bin,
    preferred phase in rad, mean vector length)."""
    one = _one(spikes)
    unit = one[one.keys()[0]]
    if ep is not None:
        unit = unit.restrict(ep)
    phases = np.asarray(unit.value_from(phase).values) % (2 * np.pi)
    counts, edges = np.histogram(phases, bins=bins, range=(0, 2 * np.pi))
    centres = (edges[:-1] + edges[1:]) / 2
    preferred, mvl = _circular_stats(phases, np.ones_like(phases))
    width = edges[1] - edges[0]
    deg = np.rad2deg
    for k in range(cycles):
        ax.bar(deg(centres + 2 * np.pi * k), counts, width=deg(width) * 0.9,
               color=color or style.CYCLE[0])
    if reference:
        x = np.linspace(0, 2 * np.pi * cycles, 200)
        top = counts.max()
        ax.plot(deg(x), top * 1.15 + top * 0.12 * np.cos(x), color='0.5', lw=0.6)
    ax.set_xlim(0, 360 * cycles)
    ax.set_xticks(np.arange(0, 360 * cycles + 1, 180))
    ax.set(xlabel='Phase (°)', ylabel='Spikes')
    return counts, preferred, mvl


# ---------------------------------------------------------- peri-event

def peth(ax_trials, ax_mean, activity, events, window=(-1.0, 2.0), binsize=0.05, color=None,
         label=None, cmap=None):
    """Activity around events. Spikes: a raster of trials (`ax_trials`) and
    the mean rate +/- SEM (`ax_mean`). A signal (dF/F, photometry): a trial x
    time heatmap and the mean +/- SEM. Either axes may be None. Returns
    (bin times, mean, SEM)."""
    events = events if isinstance(events, nap.Ts) else nap.Ts(t=np.asarray(events, float))
    if _is_spikes(activity):
        one = _one(activity)
        unit = one[one.keys()[0]]
        trials = nap.compute_perievent(unit, events, minmax=window)
        edges = np.arange(window[0], window[1] + binsize / 2, binsize)
        keys = list(trials.keys())
        counts = np.array([np.histogram(trials[k].t, bins=edges)[0] for k in keys])
        rates = counts / binsize
        times = (edges[:-1] + edges[1:]) / 2
        if ax_trials is not None:
            ax_trials.eventplot([trials[k].t for k in keys], colors='0.15',
                                linelengths=0.8, linewidths=0.6)
            ax_trials.set(ylabel='Trial', xlim=window)
            ax_trials.set_ylim(-0.5, len(trials) - 0.5)
        ylabel = 'Rate (Hz)'
    else:
        one = _one(activity)
        frame = nap.compute_perievent_continuous(one[:, 0] if one.ndim == 2 else one, events,
                                                 minmax=window)
        times = frame.t
        rates = frame.values.T                           # trials x time
        if ax_trials is not None:
            ax_trials.imshow(rates, aspect='auto', origin='lower', interpolation='nearest',
                             extent=(times[0], times[-1], -0.5, rates.shape[0] - 0.5),
                             cmap=cmap or style.CMAPS['rate'])
            ax_trials.set(ylabel='Trial')
        ylabel = 'Signal'
    mean = np.nanmean(rates, axis=0)
    sem = np.nanstd(rates, axis=0, ddof=1) / np.sqrt(max(rates.shape[0], 1))
    if ax_mean is not None:
        style.mean_with_error(ax_mean, times, mean, sem, color=color, label=label)
        ax_mean.set(xlabel='Time from event (s)', ylabel=ylabel, xlim=window)
    for ax in (ax_trials, ax_mean):
        if ax is not None:
            ax.axvline(0, color='0.5', ls='--', lw=0.6, zorder=0)
    if ax_trials is not None and ax_mean is not None:
        ax_trials.tick_params(labelbottom=False)
    return times, mean, sem


# ------------------------------------------------------------ populations

def raster(ax, spikes, start=None, stop=None, order=None, color='0.15', linewidth=0.6):
    """Spike raster of a population (TsGroup or Population), one row per
    unit, over [start, stop] s. `order`: unit ids, top to bottom. Returns
    the unit ids in row order."""
    group = _unwrap(spikes)
    ids = list(order) if order is not None else list(group.keys())
    t0 = group.time_support.start[0] if start is None else start
    t1 = group.time_support.end[-1] if stop is None else stop
    window = nap.IntervalSet(t0, t1)
    rows = [group[i].restrict(window).t for i in ids][::-1]
    ax.eventplot(rows, colors=color, linelengths=0.8, linewidths=linewidth)
    ax.set(xlim=(t0, t1), ylim=(-0.5, len(ids) - 0.5), xlabel='Time (s)', ylabel='Unit')
    ax.set_yticks(range(len(ids)), [str(i) for i in ids[::-1]])
    return ids


def traces(ax, signals, start=None, stop=None, columns=None, spacing=None, color='0.15',
           scale=None, scale_label=None):
    """Stacked traces (e.g. dF/F of several ROIs), first column on top, with
    a vertical scale bar instead of a y axis. Returns the offsets used."""
    frame = _unwrap(signals)
    cols = list(columns) if columns is not None else list(frame.columns)
    t0 = frame.t[0] if start is None else start
    t1 = frame.t[-1] if stop is None else stop
    part = frame.restrict(nap.IntervalSet(t0, t1))
    values = np.column_stack([np.asarray(part.loc[c]) for c in cols])   # .loc: by column
    gap = spacing if spacing is not None else 1.1 * np.nanpercentile(
        np.nanmax(values, 0) - np.nanmin(values, 0), 75)
    offsets = gap * np.arange(len(cols))[::-1]
    for k, c in enumerate(cols):
        ax.plot(part.t, values[:, k] + offsets[k], color=color, lw=0.6)
        ax.text(t0, offsets[k] + np.nanmedian(values[:, k]), f'{c} ', ha='right',
                va='center', fontsize=style.FONT['small'], color='0.3')
    style.tidy(ax, left=False)
    ax.set(xlim=(t0, t1), xlabel='Time (s)')
    size = scale if scale is not None else float(f'{gap / 2:.1g}')
    style.scale_bar(ax, size, scale_label or f'{size:g}', where='lower right', vertical=True)
    return offsets


# ------------------------------------------------------- spike statistics

def autocorrelogram(ax, spikes, binsize=0.005, window=0.5, ep=None, color=None):
    """Spike autocorrelogram (counts per bin, the zero-lag bin removed):
    theta-rhythmic cells show side peaks every ~125 ms. Returns the
    correlogram (pandas Series, index = lag in s)."""
    one = _one(spikes)
    cc = nap.compute_autocorrelogram(one, binsize, window, norm=False, **_ep(ep)).iloc[:, 0]
    cc[np.isclose(cc.index, 0)] = np.nan
    ax.bar(cc.index * 1000, cc.values, width=binsize * 1000, color=color or style.CYCLE[0])
    ax.set(xlabel='Lag (ms)', ylabel='Count', xlim=(-window * 1000, window * 1000))
    return cc


def isi_histogram(ax, spikes, max_ms=1000, bins=60, refractory_ms=1.5, color=None):
    """Inter-spike intervals on log-spaced bins, with the refractory period
    marked; violations show as counts left of the line. Returns (counts,
    bin edges in ms, fraction of intervals under the refractory period)."""
    one = _one(spikes)
    isi = np.diff(one[one.keys()[0]].t) * 1000
    edges = np.logspace(np.log10(0.5), np.log10(max_ms), bins)
    counts, _ = np.histogram(isi, bins=edges)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge',
           color=color or style.CYCLE[0])
    ax.set_xscale('log')
    ax.axvline(refractory_ms, color=style.CYCLE[1], ls='--', lw=0.6)
    ax.set(xlabel='Inter-spike interval (ms)', ylabel='Count')
    return counts, edges, float(np.mean(isi < refractory_ms)) if isi.size else np.nan


def score_histogram(ax, scores, null=None, threshold=None, bins=30, color=None,
                    label='Cells', null_label='Shuffled', xlabel='Score'):
    """Distribution of a score across cells, against its shuffled null
    (outline) and a threshold (e.g. the null's 95th percentile). Returns
    the fraction of cells above the threshold (or None)."""
    scores = np.asarray(scores, float)
    scores = scores[np.isfinite(scores)]
    pool = scores if null is None else np.r_[scores, np.asarray(null, float).ravel()]
    edges = np.histogram_bin_edges(pool[np.isfinite(pool)], bins=bins)
    ax.hist(scores, bins=edges, color=color or style.CYCLE[0], label=label, density=True,
            rwidth=0.9)
    if null is not None:
        ax.hist(np.asarray(null, float).ravel(), bins=edges, histtype='step', color='0.3',
                lw=0.7, density=True, label=null_label)
    fraction = None
    if threshold is not None:
        ax.axvline(threshold, color='0.2', ls='--', lw=0.6)
        fraction = float(np.mean(scores > threshold))
        ax.text(threshold, ax.get_ylim()[1], f' {100 * fraction:.0f}%', ha='left', va='top',
                fontsize=style.FONT['small'])
    ax.set(xlabel=xlabel, ylabel='Density')
    ax.legend()
    return fraction
