"""Write a sorted session as publishable NWB, plus its pynapple and CSV
tables. File-level metadata (subject, lab, start time) comes from
`imaging.nwb.new_nwbfile`, so both modalities describe sessions alike.

Processed ephys file:

    general/        subject, lab, procedures, keywords
    devices/        the probe
    extracellular_ephys/
        shank{i}               one ElectrodeGroup per shank
        electrodes             every channel: probe position, shank,
                               bad-channel label, whether it was sorted
    units/                     one row per unit: spike_times, obs_intervals,
                               electrodes (the unit's 16 nearest channels,
                               extremum channel first), waveform_mean and waveform_sd
                               (samples x those channels, µV), and one column
                               per units-table column (curation, position,
                               quality and template metrics)

The raw voltage is not copied in; `notes` name the source recordings.
"""

import numpy as np
import pandas as pd
import pynapple as nap
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb.ecephys import LFP, ElectricalSeries, FilteredEphys
from pynwb.misc import Units

from raphael.imaging.nwb import _describe, _scalar

DESCRIPTIONS = {
    'accepted': 'True if the unit passed curation (bombcell good or mua, ISI rule) '
                'and is used in analysis.',
    'bombcell_label': 'bombcell label: good, mua, noise or non_soma.',
    'isi_excluded': 'True if the ISI violation ratio exceeds 10 in any session of the day.',
    'KSLabel': "Kilosort's own label (good or mua).",
    'KSLabel_repeat': "Kilosort's label, as written a second time in its output.",
    'Amplitude': "Kilosort's template amplitude.",
    'ContamPct': "Kilosort's contamination estimate (%).",
    'original_cluster_id': "Cluster id in Kilosort's output for this shank.",
    'extremum_channel': 'Channel with the largest template amplitude.',
    'group': 'Shank the unit was sorted on.',
    'x': 'Unit position across the probe (µm), monopolar triangulation.',
    'y': 'Unit position along the probe (µm, from the tip).',
    'z': 'Unit distance from the probe plane (µm).',
    'coord_probe_x': 'Probe position (µm) of the extremum channel, across.',
    'coord_probe_y': 'Probe position (µm) of the extremum channel, from the tip.',
}


def build_ecephys_nwb(nwbfile, units, spikes, waveforms, channels, duration, rate, probe):
    """`units`: the units table; `spikes`: unit -> times (s); `waveforms`:
    (mean, sd, channel ids) from `sorting.unit_waveforms`; `channels`: the
    channel table from
    `sorting.prepare`; `rate`: sampling rate (Hz); `probe`: the `probe`
    section of configs/ephys.yaml. Metrics that are NaN for every unit
    (e.g. drift on a short recording) are left out; the CSV keeps them."""
    device = nwbfile.create_device(
        name='Probe', description=probe.get('description'),
        manufacturer=probe.get('manufacturer'))
    groups = {
        shank: nwbfile.create_electrode_group(
            name=f'shank{shank}', description=f'Electrodes on shank {shank}.',
            location=probe.get('location', 'unknown'), device=device)
        for shank in sorted(channels['shank'].unique())
    }
    for name, description in [
        ('channel_name', 'Channel id in the acquisition software.'),
        ('rel_x', 'Position across the probe (µm).'),
        ('rel_y', 'Position along the probe (µm, from the tip).'),
        ('shank', 'Shank number.'),
        ('label', 'Bad-channel detection: good, dead, noise or out (of the brain).'),
        ('used', 'True if the channel was given to the sorter.'),
    ]:
        nwbfile.add_electrode_column(name=name, description=description)
    for row in channels.itertuples(index=False):
        nwbfile.add_electrode(
            group=groups[row.shank], location=probe.get('location', 'unknown'),
            channel_name=row.channel_id, rel_x=float(row.x_um), rel_y=float(row.y_um),
            shank=int(row.shank), label=str(row.label), used=bool(row.used))
    electrode = {c: i for i, c in enumerate(channels['channel_id'])}

    nwbfile.units = Units(name='units', description='Units from spike sorting.',
                          resolution=1.0 / rate)
    columns = [c for c in units.columns if c != 'unit' and not units[c].isna().all()]
    for column in columns:
        nwbfile.add_unit_column(name=column,
                                description=DESCRIPTIONS.get(column, _describe(column)))
    mean, sd, channel_ids = waveforms
    for i, row in enumerate(units.itertuples(index=False)):
        values = row._asdict()
        nwbfile.add_unit(
            id=int(values['unit']),
            spike_times=np.asarray(spikes[int(values['unit'])], dtype=float),
            obs_intervals=[[0.0, float(duration)]],
            electrodes=[electrode[c] for c in channel_ids[i]],
            waveform_mean=mean[i],
            waveform_sd=sd[i],
            **{c: _scalar(values[c]) for c in columns},
        )
    return nwbfile


def build_lfp_nwb(nwbfile, lfp, rate, channels, probe, settings, phase=None):
    """The LFP file, laid out as the lab's: `processing/ecephys/LFP/lfp`
    (µV, conversion 1e-6 to volts) and, if given,
    `processing/ecephys/Processed/theta` (radians), both samples x LFP
    channels from 0 s at `rate`. One electrodes row per LFP channel, named by
    the contacts it averages (`CH49_CH50_...`), at their mean position.

    `channels`: from `lfp.extract`; `settings`: configs/ephys.yaml `lfp`."""
    device = nwbfile.create_device(name='Probe', description=probe.get('description'),
                                   manufacturer=probe.get('manufacturer'))
    location = probe.get('location', 'unknown')
    groups = {shank: nwbfile.create_electrode_group(
                  name=f'shank{shank}', description=f'Electrodes on shank {shank}.',
                  location=location, device=device)
              for shank in sorted({c['shank'] for c in channels})}
    for name, description in [
        ('channel_name', 'Contacts averaged into this LFP channel, joined by _.'),
        ('rel_x', 'Mean position of those contacts across the probe (µm).'),
        ('rel_y', 'Mean position of those contacts along the probe (µm, from the tip).'),
        ('shank', 'Shank number.'),
        ('n_contacts', 'Number of contacts averaged.'),
    ]:
        nwbfile.add_electrode_column(name=name, description=description)
    for c in channels:
        nwbfile.add_electrode(group=groups[c['shank']], location=location,
                              channel_name='_'.join(c['channel_ids']), rel_x=c['x_um'],
                              rel_y=c['y_um'], shank=c['shank'],
                              n_contacts=len(c['channel_ids']))
    region = nwbfile.create_electrode_table_region(
        list(range(len(channels))), 'LFP channels (contact blocks)')

    def stored(data):
        return H5DataIO(np.asarray(data, dtype=settings['dtype']), compression='gzip',
                        compression_opts=settings['compression_level'], chunks=True)

    low, high = settings['band_hz']
    ch = settings['channels']
    ecephys = nwbfile.create_processing_module('ecephys', 'Local field potential.')
    ecephys.add(LFP(electrical_series=ElectricalSeries(
        name='lfp', data=stored(lfp), electrodes=region, starting_time=0.0, rate=float(rate),
        conversion=1e-6, filtering=f'band-pass {low}-{high} Hz, resampled to {rate:g} Hz',
        description=(f'LFP, band-passed {low}-{high} Hz and resampled to {rate:g} Hz; each '
                     f'channel the mean of {ch["block_size"]} neighbouring contacts in depth '
                     'order on one shank. Stored in µV (conversion 1e-6 to volts).'))))
    if phase is not None:
        theta = settings['theta']
        ecephys.add(FilteredEphys(name='Processed', electrical_series=ElectricalSeries(
            name='theta', data=stored(phase), electrodes=region, starting_time=0.0,
            rate=float(rate), conversion=1.0,
            filtering=f'band-pass {theta["band_hz"][0]}-{theta["band_hz"][1]} Hz',
            description=('Theta phase of the LFP, in radians (not volts): 0 to 2π from one '
                         f'peak of the {theta["band_hz"][0]}-{theta["band_hz"][1]} Hz '
                         'band-passed LFP to the next; NaN outside the first and last peak.'))))
    return nwbfile


def read_waveforms(path):
    """Every unit's waveforms from a processed `_ecephys.nwb` file: `unit`
    ids; `mean` and `sd` (units x samples x channels, µV, the unit's
    extremum channel first); `channel_ids` and `positions_um` (x, y on the
    probe) of those channels, the extremum channel first; `sampling_frequency`."""
    import pynwb

    with pynwb.NWBHDF5IO(str(path), 'r') as io:
        f = io.read()
        units = f.units
        rows = np.asarray(units['electrodes'].target.data[:]).reshape(len(units), -1)
        electrodes = f.electrodes.to_dataframe()
        return {
            'unit': np.asarray(units.id[:]),
            'mean': np.asarray(units['waveform_mean'].data[:]),
            'sd': np.asarray(units['waveform_sd'].data[:]),
            'channel_ids': electrodes['channel_name'].to_numpy()[rows],
            'positions_um': electrodes[['rel_x', 'rel_y']].to_numpy()[rows],
            'sampling_frequency': 1.0 / units.resolution,
        }


def export_clusters(units, spikes, duration, path, waveforms=None, channels=None,
                    sampling_frequency=None, nbefore=None):
    """The session's units as a pynapple TsGroup saved to `path`, in the
    form of clark2025's `_clusters.npz`: one spike train per unit (keys =
    `unit`), with the units table as metadata.

    With `waveforms` ((mean, sd, channel ids) from `sorting.unit_waveforms`)
    and the `channels` table, each unit also carries `waveform_mean` and
    `waveform_sd` (samples x channels, µV, its extremum channel first),
    `waveform_channels` and `waveform_positions_um` (x, y of those
    channels), and the scalars `waveform_sampling_frequency` and
    `waveform_nbefore` (the sample at which the spike is aligned)."""
    group = nap.TsGroup({int(u): nap.Ts(t=np.asarray(spikes[int(u)], dtype=float))
                         for u in units['unit']},
                        time_support=nap.IntervalSet(0, float(duration)))
    metadata = units.set_index('unit')
    group.set_info(metadata[[c for c in metadata.columns
                             if pd.api.types.is_numeric_dtype(metadata[c])
                             or pd.api.types.is_bool_dtype(metadata[c])]])
    group.set_info(metadata[[c for c in metadata.columns
                             if c not in group.metadata_columns]].astype(str))
    if waveforms is not None:
        mean, sd, channel_ids = waveforms
        xy = channels.set_index('channel_id')[['x_um', 'y_um']]
        index = metadata.index

        def per_unit(arrays):
            return pd.Series(list(arrays), index=index, dtype=object)

        group.set_info(
            waveform_mean=per_unit(mean),
            waveform_sd=per_unit(sd),
            waveform_channels=per_unit(np.asarray(channel_ids).astype(str)),
            waveform_positions_um=per_unit(
                np.stack([xy.loc[list(ids)].to_numpy() for ids in channel_ids])),
            waveform_sampling_frequency=pd.Series(float(sampling_frequency), index=index),
            waveform_nbefore=pd.Series(int(nbefore), index=index))
    path.parent.mkdir(parents=True, exist_ok=True)
    group.save(str(path))
    return group
