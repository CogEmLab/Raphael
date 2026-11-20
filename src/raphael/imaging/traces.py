"""Trace-level processing shared by every pipeline: neuropil and dF/F."""

import numpy as np
from scipy.ndimage import percentile_filter


def neuropil_corrected(F, neuropil, coefficient):
    return F - coefficient * neuropil


def baseline(F, rate, percentile=8, window_s=60):
    """Running low-percentile baseline F0, per ROI (columns).

    The percentile filter runs on ~1 s bin means and is interpolated back, so
    a 40,000 x 1,500 recording takes seconds rather than minutes. Bin means
    lift the percentile slightly, which matters little over a 60 s window.
    """
    F = np.asarray(F, dtype=np.float32)
    n = len(F)
    bin_frames = max(1, int(round(rate)))
    n_bins = n // bin_frames
    if n_bins < 2:
        return np.repeat(np.percentile(F, percentile, axis=0, keepdims=True), n, axis=0)
    binned = F[: n_bins * bin_frames].reshape(n_bins, bin_frames, -1).mean(axis=1)
    window = max(1, int(round(window_s * rate / bin_frames)))
    smooth = percentile_filter(binned, percentile, size=(window, 1), mode='nearest')
    centres = (np.arange(n_bins) + 0.5) * bin_frames
    frames = np.arange(n)
    return np.stack(
        [np.interp(frames, centres, smooth[:, i]) for i in range(F.shape[1])], axis=1
    ).astype(np.float32)


def dff(F, rate, percentile=8, window_s=60):
    """(F - F0) / F0, with F0 from `baseline`. F is frames x ROIs."""
    F0 = baseline(F, rate, percentile, window_s)
    # Guard against ROIs whose baseline sits at or below zero (e.g. after
    # neuropil subtraction): they get NaN rather than a huge ratio.
    F0 = np.where(F0 > 1e-6, F0, np.nan)
    return ((F - F0) / F0).astype(np.float32)
