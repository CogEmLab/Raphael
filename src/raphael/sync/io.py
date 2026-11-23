"""Reading TTL pulses: a behaviour CSV with a TTL column, and the TTL
events of an Open Ephys (binary format) Neuropixels recording."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def read_behaviour(path, time_column='time', ttl_column='ttl'):
    """A behaviour CSV: one row per sample, a time column (s, the rig's
    clock) and a TTL column (0/1 or volts). Other columns are kept."""
    df = pd.read_csv(path)
    missing = {time_column, ttl_column} - set(df.columns)
    if missing:
        raise ValueError(f'{path}: no column(s) {sorted(missing)}; columns are '
                         f'{list(df.columns)}. Pass time_column= / ttl_column=.')
    if not df[time_column].is_monotonic_increasing:
        raise ValueError(f'{path}: {time_column} is not increasing')
    return df


@dataclass
class Recording:
    """The TTL line of a recording, on its own clock: seconds from the
    recording's first sample (the clock spike times use)."""
    pulses: object
    duration: float
    sample_rate: float
    first_sample: int
    stream: str
    folder: Path


def _recording_folders(folder):
    folder = Path(folder)
    found = sorted(folder.rglob('structure.oebin'))
    if not found:
        raise ValueError(f'{folder}: no structure.oebin, so not an Open Ephys binary recording')
    if len(found) > 1:
        raise ValueError(f'{folder} holds {len(found)} recordings; pass one recording folder')
    return found[0].parent


def read_openephys(folder, line=1, stream=None):
    """TTL `line` (1-based, as Open Ephys numbers them) of the probe stream
    of the Open Ephys recording under `folder`. With Neuropixels PXI, the
    probe's SYNC input is logged on line 1 of the probe stream."""
    from raphael.sync import Pulses

    rec = _recording_folders(folder)
    meta = json.loads((rec / 'structure.oebin').read_text())
    streams = {c['folder_name'].strip('/'): c for c in meta['continuous']}
    if stream is None:
        probes = [n for n in streams if 'Probe' in n and not n.endswith('-LFP')]
        if len(probes) != 1:
            raise ValueError(f'{rec}: cannot tell which stream is the probe among '
                             f'{list(streams)}; pass stream=')
        stream = probes[0]
    cont = streams[stream]
    fs = float(cont['sample_rate'])
    samples = np.load(rec / 'continuous' / stream / 'sample_numbers.npy', mmap_mode='r')
    first, n = int(samples[0]), len(samples)

    events = [e for e in meta['events'] if e['folder_name'].strip('/').startswith(stream + '/')
              and e['folder_name'].strip('/').endswith('TTL')]
    if not events:
        raise ValueError(f'{rec}: stream {stream} has no TTL events folder')
    ev = rec / 'events' / events[0]['folder_name'].strip('/')
    sample_numbers = np.load(ev / 'sample_numbers.npy').astype(np.int64)
    states = np.load(ev / 'states.npy').astype(np.int64)
    rises = sample_numbers[states == line]
    falls = sample_numbers[states == -line]
    if not len(rises):
        lines = sorted(set(np.abs(states).tolist()))
        raise ValueError(f'{rec}: no pulses on TTL line {line}; lines with pulses: {lines}')
    nxt = np.searchsorted(falls, rises, side='right')
    widths = np.full(len(rises), np.nan)
    ok = nxt < len(falls)
    widths[ok] = (falls[nxt[ok]] - rises[ok]) / fs
    pulses = Pulses((rises - first) / fs, widths, 0.0)
    return Recording(pulses, n / fs, fs, first, stream, rec)
