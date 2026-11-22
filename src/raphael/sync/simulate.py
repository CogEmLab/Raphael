"""Simulated sessions for testing sync: a behaviour CSV and a light
Neuropixels 2.0 recording in Open Ephys binary format, sharing one TTL
pulse train, with known clock drift and offset and optional faults.

The rig sends pulses at random intervals (0.5-1.5 s, 50 ms wide). The
behaviour CSV samples the TTL with each row (~60 Hz, jittered, like a game
engine's frames). The recording starts a few seconds after the rig and
stops before it, and logs the pulses as Open Ephys TTL events on line 1 of
the probe stream, on a clock that drifts against the rig's.

"Light": few channels of noise instead of 384 (sync reads only the events
and the sample clock), but the folder layout, structure.oebin, sample
numbers (not starting at 0) and event files are as Open Ephys writes them.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

STREAM = 'Neuropix-PXI-100.ProbeA'


def pulse_train(duration_s, rng, interval=(0.5, 1.5), width=0.05, start=0.5):
    onsets, t = [], start
    while t < duration_s - 1.0:
        onsets.append(t)
        t += rng.uniform(*interval)
    return np.array(onsets), width


def write_behaviour(path, onsets, width, duration_s, rng, rate_hz=60.0, jitter_s=0.002,
                    arena_cm=100.0):
    """Behaviour CSV: time (rig clock, s), x, y (cm), ttl (0/1)."""
    dt = np.clip(1 / rate_hz + rng.normal(0, jitter_s, int(duration_s * rate_hz * 1.1)),
                 0.5 / rate_hz, None)
    t = np.cumsum(dt)
    t = t[t < duration_s]
    xy = np.cumsum(rng.normal(0, 0.8, (len(t), 2)), axis=0)
    xy = arena_cm / 2 + (arena_cm / 2 - 1) * np.sin(xy / 40)          # stays in the arena
    k = np.searchsorted(onsets, t, side='right') - 1
    ttl = ((k >= 0) & (t - onsets[np.clip(k, 0, None)] < width)).astype(int)
    df = pd.DataFrame({'time': np.round(t, 6), 'x': xy[:, 0].round(2), 'y': xy[:, 1].round(2),
                       'ttl': ttl})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def write_openephys(folder, onsets, width, start_s, duration_s, drift_ppm, rng,
                    fs=30000.0, n_channels=4, first_sample=4_812_345, extra_pulses=()):
    """A light Open Ephys Neuropixels recording of the rig's time span
    [start_s, start_s + duration_s], with its clock running (1 + drift)
    times the rig's. Returns the Record Node folder."""
    node = Path(folder) / 'Record Node 101'
    rec = node / 'experiment1' / 'recording1'
    cont = rec / 'continuous' / STREAM
    ev = rec / 'events' / STREAM / 'TTL'
    cont.mkdir(parents=True, exist_ok=True)
    ev.mkdir(parents=True, exist_ok=True)
    (node / 'settings.xml').write_text('<SETTINGS><INFO><VERSION>0.6.7</VERSION></INFO>'
                                       '</SETTINGS>\n')
    scale = 1 + drift_ppm * 1e-6
    n = int(duration_s * scale * fs)
    noise = rng.normal(0, 20, (n, n_channels)).astype(np.int16)
    noise.tofile(cont / 'continuous.dat')
    samples = np.arange(first_sample, first_sample + n, dtype=np.int64)
    np.save(cont / 'sample_numbers.npy', samples)
    np.save(cont / 'timestamps.npy', samples / fs)

    seen = onsets[(onsets >= start_s) & (onsets + width <= start_s + duration_s)]
    rises = first_sample + np.round((seen - start_s) * scale * fs).astype(np.int64)
    rises = np.sort(np.concatenate([rises, first_sample + np.round(
        np.asarray(extra_pulses, float) * fs).astype(np.int64)]))
    falls = rises + round(width * scale * fs)
    sample_numbers = np.concatenate([rises, falls])
    states = np.concatenate([np.ones(len(rises)), -np.ones(len(falls))]).astype(np.int16)
    order = np.argsort(sample_numbers, kind='stable')
    sample_numbers, states = sample_numbers[order], states[order]
    np.save(ev / 'sample_numbers.npy', sample_numbers)
    np.save(ev / 'states.npy', states)
    np.save(ev / 'timestamps.npy', sample_numbers / fs)
    np.save(ev / 'full_words.npy', (states > 0).astype(np.uint64))

    structure = {
        'GUI version': '0.6.7',
        'continuous': [{
            'folder_name': f'{STREAM}/', 'sample_rate': fs,
            'source_processor_name': 'Neuropix-PXI', 'source_processor_id': 100,
            'stream_name': 'ProbeA', 'recorded_processor': 'Record Node',
            'recorded_processor_id': 101, 'num_channels': n_channels,
            'channels': [{'channel_name': f'CH{i + 1}', 'description': 'Neuropixels 2.0 channel',
                          'identifier': 'neuropixels.channel', 'history': 'Neuropix-PXI',
                          'bit_volts': 0.195, 'units': 'uV'} for i in range(n_channels)],
        }],
        'events': [{
            'folder_name': f'{STREAM}/TTL/', 'channel_name': 'Neuropixels PXI Sync',
            'description': 'Status of SMA sync line on PXI card', 'identifier': 'neuropixels.sync',
            'sample_rate': fs, 'type': 'int16', 'num_channels': 1,
            'source_processor': 'Neuropix-PXI', 'stream_name': 'ProbeA',
        }],
        'spikes': [],
    }
    (rec / 'structure.oebin').write_text(json.dumps(structure, indent=2))
    return node


def simulate_session(root, duration_s=120.0, seed=0, drift_ppm=20.0, recording_start_s=3.2,
                     recording_end_gap_s=2.0, rate_hz=60.0, n_channels=4,
                     drop_behaviour_pulses=(), extra_recording_pulses=(), interval=(0.5, 1.5)):
    """Write `root`/behaviour.csv and `root`/recording/Record Node 101/.

    drop_behaviour_pulses: indices of pulses the behaviour CSV misses.
    extra_recording_pulses: recording-clock times (s) of spurious pulses.
    interval: (min, max) s between pulses; (1, 1) for regular pulses.
    Returns paths and the truth: recording time = slope * rig time + intercept."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    onsets, width = pulse_train(duration_s, rng, interval=interval)
    beh_onsets = np.delete(onsets, list(drop_behaviour_pulses))
    behaviour = root / 'behaviour.csv'
    write_behaviour(behaviour, beh_onsets, width, duration_s, rng, rate_hz=rate_hz)
    rec_duration = duration_s - recording_start_s - recording_end_gap_s
    node = write_openephys(root / 'recording', onsets, width, recording_start_s, rec_duration,
                           drift_ppm, rng, n_channels=n_channels,
                           extra_pulses=extra_recording_pulses)
    scale = 1 + drift_ppm * 1e-6
    return {'behaviour': behaviour, 'recording': node, 'onsets': onsets,
            'slope': scale, 'intercept': -recording_start_s * scale, 'drift_ppm': drift_ppm}
