"""Generic helpers, ported from the spatial-manifolds package.

Kept verbatim where possible so analysis code moves between the two repos
without edits.
"""

from collections import defaultdict

import numpy as np
import pynapple as nap
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


def tree():
    return defaultdict(tree)


def in_leaves(X, slice):
    return {key: value[slice] for key, value in X.items()}


def get_X(X, model_type):
    if model_type == '':
        return np.ones((len(next(iter(X.values()))), 1))
    return np.concatenate([X[sub_type] for sub_type in model_type], axis=-1)


def interpolate_nans(tsd, pkind='cubic'):
    """Fill NaNs in a Tsd by interpolating over the finite samples."""
    times = tsd.times()
    arr = tsd.values
    indexes = np.arange(arr.shape[0])
    (good,) = np.where(np.isfinite(arr))
    f = interp1d(
        good,
        arr[good],
        bounds_error=False,
        copy=False,
        fill_value='extrapolate',
        kind=pkind,
    )
    return nap.Tsd(d=f(indexes), t=times)


def gaussian_filter_nan(X, sigma, mode='reflect'):
    """Gaussian smoothing that ignores NaNs instead of spreading them."""
    V = X.copy()
    V[np.isnan(X)] = 0
    VV = gaussian_filter(V, sigma=sigma, mode=mode)

    W = 0 * X.copy() + 1
    W[np.isnan(X)] = 0
    WW = gaussian_filter(W, sigma=sigma, mode=mode)
    Y = VV / WW
    Y[np.isnan(X)] = np.nan
    return Y


def wrap(angle):
    """Wrap radians to [-pi, pi), matching the clark2025 H convention."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
