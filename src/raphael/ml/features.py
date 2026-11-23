"""Turning processed data into the matrices models learn from.

- `bin_counts`: spike counts (or mean signal) per time bin, neurons as columns.
- `behaviour_at`: behaviour variables sampled at the same bin centres.
- `spatial_basis`, `circular_basis`: smooth bumps tiling the arena or the
  circle, so a linear model or GLM can learn a place field or a tuning
  curve (raw x, y or an angle can't be fitted linearly).
- `history`: past values seen through raised-cosine filters (covariate or
  spike history), as in Benjamin et al. 2018.
"""

import numpy as np
import pandas as pd


def _unwrap(activity):
    return getattr(activity, 'data', activity) if type(activity).__name__ == 'Population' \
        else activity


def bin_counts(activity, binsize, ep=None):
    """Spike counts per bin (neurons as columns), or the mean of a
    continuous signal (TsdFrame, e.g. dF/F) per bin. Returns (bin centres
    in s, DataFrame of bins x neurons)."""
    activity = _unwrap(activity)
    kwargs = {'ep': ep} if ep is not None else {}
    if type(activity).__name__ == 'TsGroup':
        binned = activity.count(binsize, **kwargs)
    else:
        binned = activity.bin_average(binsize, **kwargs)
    values = np.asarray(binned.values, float)
    columns = list(getattr(binned, 'columns', range(values.shape[1])))
    return np.asarray(binned.t), pd.DataFrame(values, columns=columns)


def behaviour_at(behaviour, names, times, circular=()):
    """Behaviour variables (pynapple Tsd in `behaviour`, a dict) linearly
    interpolated at `times`. Circular variables (radians) are interpolated
    through sin and cos and returned in [0, 2 pi)."""
    out = {}
    for name in names:
        tsd = behaviour[name]
        t, v = np.asarray(tsd.t), np.asarray(tsd.values, float)
        if name in circular:
            s = np.interp(times, t, np.sin(v))
            c = np.interp(times, t, np.cos(v))
            out[name] = np.arctan2(s, c) % (2 * np.pi)
        else:
            out[name] = np.interp(times, t, v)
    return pd.DataFrame(out)


def spatial_basis(x, y, bounds, n=8, width=None):
    """Gaussian bumps on an n x n grid over `bounds` (x min, x max, y min,
    y max), evaluated at positions (x, y): one column per bump."""
    x0, x1, y0, y1 = bounds
    cx, cy = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    width = width or (x1 - x0) / (n - 1)
    gx, gy = np.meshgrid(cx, cy)
    d2 = ((np.asarray(x)[:, None] - gx.ravel()) ** 2 + (np.asarray(y)[:, None] - gy.ravel()) ** 2)
    return pd.DataFrame(np.exp(-d2 / (2 * width ** 2)),
                        columns=[f'place_{i}' for i in range(n * n)])


def circular_basis(angle, n=8, kappa=None, name='angle'):
    """Von Mises bumps evenly spaced around the circle, at `angle` (rad)."""
    centres = np.linspace(0, 2 * np.pi, n, endpoint=False)
    kappa = kappa or n / 2
    values = np.exp(kappa * (np.cos(np.asarray(angle)[:, None] - centres) - 1))
    return pd.DataFrame(values, columns=[f'{name}_{i}' for i in range(n)])


def sin_cos(angle, name='angle'):
    angle = np.asarray(angle, float)
    return pd.DataFrame({f'{name}_sin': np.sin(angle), f'{name}_cos': np.cos(angle)})


def raised_cosines(n_bases, max_lag_bins):
    """`n_bases` causal filters (rows) over lags 1..max_lag_bins, log-spaced
    so recent lags are finely resolved, each normalised to sum to 1."""
    lags = np.arange(1, max_lag_bins + 1)
    nt = np.log(lags + 0.5)
    centres = np.linspace(nt[0], nt[-1], n_bases)
    width = (centres[1] - centres[0]) if n_bases > 1 else nt[-1] - nt[0] + 1
    arg = np.clip((nt[None, :] - centres[:, None]) * np.pi / (2 * width), -np.pi, np.pi)
    filters = (np.cos(arg) + 1) / 2
    return filters / filters.sum(axis=1, keepdims=True)


def history(values, n_bases=5, max_lag_bins=10, name=None):
    """Each column's past (bins t-1 ... t-max_lag_bins, never the present)
    through `n_bases` raised-cosine filters: covariate history, or a
    neuron's own spike history for encoding. Returns a DataFrame."""
    frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(
        np.asarray(values, float).reshape(len(values), -1),
        columns=[name or 'x'] if np.ndim(values) == 1 else None)
    filters = raised_cosines(n_bases, max_lag_bins)
    out = {}
    for col in frame.columns:
        v = frame[col].to_numpy(float)
        for k, f in enumerate(filters):
            kernel = np.concatenate([[0.0], f])      # lag 0 excluded: no peeking
            out[f'{col}_hist{k}'] = np.convolve(v, kernel)[:len(v)]
    return pd.DataFrame(out, index=frame.index)
