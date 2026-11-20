"""Local field potential and theta phase from a Neuropixels recording. With
the default settings (configs/ephys.yaml, `lfp`) these are the arrays the
lab's NWB files hold (mecpy, `data/preprocessing/nwb.py`):

1. band-pass 1-400 Hz and resample to 1 kHz;
2. on each shank, order channels by depth and average them in blocks of 8
   neighbouring contacts (~60 µm along the shank); the remainder of a shank
   that does not fill a block is dropped;
3. theta phase per LFP channel: band-pass 6-10 Hz, find peaks at least
   80 ms apart, and let the phase run linearly from 0 to 2π between
   successive peaks (NaN before the first and after the last peak).

Size and detail are settings: the rate, the band, the block size (1 keeps
every contact), every Nth LFP channel, which shanks, the stored dtype and
the compression, and whether theta phase is written.

Differences from the lab's files: values are µV (from the recording's
gains), stored with NWB conversion 1e-6 so readers get volts; the lab's
files hold float64 labelled volts with conversion 1. Theta phase is in
radians.
"""

from itertools import pairwise

import numpy as np
import spikeinterface.full as si


def blocks(recording, block_size=8, every=1, shanks=None):
    """[(shank, [channel ids])]: per shank, channels in depth order cut into
    consecutive blocks of `block_size`, then every `every`-th block, on
    `shanks` only (None = all)."""
    if 'group' not in recording.get_property_keys():
        recording.set_property('group', np.zeros(recording.get_num_channels(), int))
    out = []
    for shank, part in sorted(recording.split_by('group').items(), key=lambda kv: int(kv[0])):
        if shanks is not None and int(shank) not in shanks:
            continue
        ordered = np.asarray(si.depth_order(part).channel_ids)
        n = len(ordered) // block_size * block_size
        kept = ordered[:n].reshape(-1, block_size)[::every]
        out += [(int(shank), list(b)) for b in kept]
    return out


def extract(recording, folder, settings):
    """LFP of one recording: (samples x LFP channels) float32 µV at
    `settings['rate_hz']`, the rate, and the LFP channels (shank, channel
    ids, mean position x/y in µm) in column order. The resampled recording
    is cached in `folder` (the needed channels, float32)."""
    channels = settings['channels']
    chosen = blocks(recording, channels['block_size'], channels['every'], channels['shanks'])
    needed = [c for _, ids in chosen for c in ids]
    low, high = settings['band_hz']
    filtered = si.bandpass_filter(recording.select_channels(needed), freq_min=low,
                                  freq_max=high, ignore_low_freq_error=True)
    resampled = si.resample(filtered, settings['rate_hz'])
    cached = resampled.save(folder=folder, overwrite=True, dtype='float32')
    gains = (dict(zip(cached.channel_ids, cached.get_channel_gains(), strict=True))
             if cached.has_scaleable_traces() else None)
    positions = dict(zip(cached.channel_ids, cached.get_channel_locations(), strict=True))
    columns, info = [], []
    for shank, ids in chosen:
        traces = cached.get_traces(channel_ids=ids).astype(np.float32)
        if gains is not None:   # µV; the offset cancels after band-passing
            traces = traces * np.array([gains[c] for c in ids], dtype=np.float32)
        columns.append(traces.mean(axis=1))
        info.append({'shank': shank, 'channel_ids': [str(c) for c in ids],
                     'x_um': float(np.mean([positions[c][0] for c in ids])),
                     'y_um': float(np.mean([positions[c][1] for c in ids]))})
    return np.stack(columns, axis=1), float(cached.get_sampling_frequency()), info


def theta_phase(lfp, rate, band_hz=(6.0, 10.0), min_peak_distance_s=0.08):
    """Theta phase (radians, 0 to 2π from one theta peak to the next) of
    every column of `lfp` (samples x channels); NaN outside the first and
    last peak. Filtered with spikeinterface's band-pass, as the lab's code."""
    from scipy.signal import find_peaks

    lfp_recording = si.NumpyRecording([lfp], sampling_frequency=rate)
    theta = si.bandpass_filter(lfp_recording, freq_min=band_hz[0], freq_max=band_hz[1],
                               ignore_low_freq_error=True).get_traces()
    phase = np.full(lfp.shape, np.nan, dtype=np.float32)
    distance = int(min_peak_distance_s * rate)
    for i in range(lfp.shape[1]):
        peaks, _ = find_peaks(theta[:, i], distance=distance)
        for start, end in pairwise(peaks):
            phase[start:end, i] = np.linspace(0, 2 * np.pi, end - start)
    return phase
