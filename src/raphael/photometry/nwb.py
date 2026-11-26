"""Write and read processed photometry.

The NWB file uses core types only, so any NWB reader opens it: each raw
demodulated channel as an acquisition TimeSeries (volts), and each signal's
dF/F and z-score as TimeSeries in a `photometry` processing module, with
the channel's indicator, reference and fit parameters in its description
and comments. (The ndx-fiber-photometry extension could later replace this
with typed fibre, LED and indicator metadata.)

The pynapple TsdFrame (`_photometry.npz`) and the channel table
(`_photometry.csv`) are what analysis reads; `load` returns both.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap
from pynwb import TimeSeries


def channel_table(fits, roles, rec):
    rows = []
    for name, fit in fits.items():
        c, spec = rec.channels[name], roles.get(name, {})
        rows.append({'channel': name, 'role': fit['role'], 'reference': fit.get('reference'),
                     'indicator': spec.get('indicator'), 'location': spec.get('location'),
                     'excitation_nm': spec.get('excitation_nm'),
                     'emission_nm': spec.get('emission_nm'),
                     'source': c.source, 'detector': c.detector, 'led': c.led,
                     'carrier_hz': c.carrier_hz, 'led_level': c.led_level,
                     **{f'fit_{k}': v for k, v in fit.items()
                        if k not in ('role', 'reference')}})
    return pd.DataFrame(rows)


def build_photometry_nwb(nwbfile, rec, times, out, table):
    for name, c in rec.channels.items():
        row = table.set_index('channel').loc[name] if name in set(table['channel']) else None
        what = (f'{row["role"]} channel' if row is not None else 'unused channel')
        nwbfile.add_acquisition(TimeSeries(
            name=f'{name}_raw', data=c.values.astype('float32'), timestamps=c.times,
            unit='volts',
            description=(f'Demodulated photodetector signal "{name}" ({what}): '
                         f'{c.source}, LED {c.led} carrier {c.carrier_hz} Hz, level '
                         f'{c.led_level}. Doric lock-in output, decimated.')))
    module = nwbfile.create_processing_module(
        'photometry', 'Fibre photometry dF/F and z-score per signal channel (Simpson et al. '
                      '2024 pre-processing: trim, low-pass, double-exponential bleaching fit, '
                      'reference regression).')
    by_name = table.set_index('channel')
    for name, series in out.items():
        row = by_name.loc[name]
        params = json.dumps({k: (None if pd.isna(v) else v) for k, v in row.items()
                             if str(k).startswith('fit_')}, default=float)
        ref = row['reference'] if isinstance(row['reference'], str) else 'none'
        indicator = row['indicator'] if isinstance(row['indicator'], str) else 'unknown'
        for kind, unit, label in (('dff', 'percent', 'dF/F (%)'), ('zscore', 'n.a.', 'z-score')):
            module.add(TimeSeries(
                name=f'{name}_{kind}', data=series[kind].astype('float32'), timestamps=times,
                unit=unit, comments=params,
                description=f'{label} of channel "{name}" (indicator {indicator}), motion '
                            f'reference: {ref}. Fit parameters in comments (JSON).'))


def export_tables(times, out, table, traces_path, channels_path):
    columns = {f'{name}_{kind}': series[kind] for name, series in out.items()
               for kind in ('dff', 'zscore')}
    frame = nap.TsdFrame(t=times, d=np.column_stack(list(columns.values())),
                         columns=list(columns))
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(str(traces_path))
    table.to_csv(channels_path, index=False)
    return frame


@dataclass
class Photometry:
    traces: nap.TsdFrame       # columns {channel}_dff, {channel}_zscore
    channels: pd.DataFrame     # one row per channel

    def dff(self, channel):
        return self.traces[f'{channel}_dff']

    def zscore(self, channel):
        return self.traces[f'{channel}_zscore']

    @property
    def signals(self):
        return self.channels.loc[self.channels['role'] == 'signal', 'channel'].tolist()


def load(prefix):
    """`prefix`: the session's file path up to `_photometry`, e.g.
    `.../sub-M25_day-18_ses-OF1`."""
    prefix = str(Path(prefix).expanduser())
    return Photometry(nap.load_file(f'{prefix}_photometry.npz'),
                      pd.read_csv(f'{prefix}_photometry.csv'))
