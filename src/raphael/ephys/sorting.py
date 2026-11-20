"""Neuropixels recordings -> Kilosort 4 -> curated units, per session.

Follows the Methods ("Offline spike sorting") and the lab's sorting code
(MattNolanLab/nolanlab-ephys, protocol `kilosort4A`, and Harry's scripts in
chrishalcrow/nolanlab-ephys), with every setting in configs/ephys.yaml:

1. The day's recordings are joined and split by shank. On each shank, bad
   and noisy channels are removed (spikeinterface's IBL method, seed 1205)
   and the phase shift is corrected.
2. Kilosort 4.0.16 sorts each shank, default parameters, motion correction
   off (`sort_step`). Kilosort 4.0.16 needs numpy < 2 and spikeinterface
   <= 0.104, which pins the whole environment (pyproject.toml).
3. For the analyser, the same channels are phase-shifted, common-median
   referenced and band-passed (300-6000 Hz) per shank; the joined recording
   and sorting are then cut back into sessions (spike times start at 0 in
   each), and each session gets a sorting analyser (peak_sign both,
   radius 70 µm) with the lab's postprocessing extensions. Unit locations
   are monopolar triangulation, spikeinterface's default.
4. Curation: bombcell labels; good and mua are kept; a unit whose ISI
   violation ratio exceeds 10 in any session is excluded.

"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface.full as si

FORMATS = ('spikeglx', 'openephys', 'spikeinterface')
BOMBCELL_LABELS = ['good', 'mua', 'noise', 'non_soma']


# ------------------------------------------------------------------ reading

def detect_format(folder):
    folder = Path(folder)
    if any(folder.rglob('*.ap.meta')):
        return 'spikeglx'
    if any(folder.rglob('structure.oebin')):
        return 'openephys'
    if (folder / 'si_folder.json').exists():
        return 'spikeinterface'
    raise ValueError(f'{folder}: not a SpikeGLX, Open Ephys or spikeinterface recording')


def _probe_stream(names, fmt, folder):
    """The wideband (NP2.0) or AP (NP1.0) stream of the one probe."""
    if fmt == 'spikeglx':
        picks = [n for n in names if n.endswith('.ap')]
    else:
        picks = [n for n in names if 'Probe' in n and not n.endswith(('-LFP', '-LF'))]
    if len(picks) != 1:
        raise ValueError(f'{folder}: cannot tell which stream is the probe among {names}; '
                         "pass it with make --param stream=...")
    return picks[0]


def read(folder, fmt=None, stream=None):
    """One recording: a SpikeGLX run folder (`*_g0`), an Open Ephys record
    node folder, or a folder saved by spikeinterface."""
    fmt = fmt or detect_format(folder)
    if fmt not in FORMATS:
        raise ValueError(f'format must be one of {FORMATS}, got {fmt!r}')
    if fmt == 'spikeinterface':
        recording = _load_binary_folder(folder)
    else:
        neo = 'spikeglx' if fmt == 'spikeglx' else 'openephysbinary'
        names, ids = si.get_neo_streams(neo, folder)
        name = stream or _probe_stream(names, fmt, folder)
        recording = (si.read_spikeglx(folder, stream_id=ids[names.index(name)])
                     if fmt == 'spikeglx' else si.read_openephys(folder, stream_name=name))
    if recording.get_num_segments() != 1:
        raise ValueError(f'{folder} holds {recording.get_num_segments()} recordings; give '
                         'each its own input so each gets its own session')
    return recording


def _load_binary_folder(folder):
    """A binary folder saved by spikeinterface (`recording.save()`), rebuilt
    from its parts (raw file, probe, channel properties) instead of with
    `si.load`, whose description file changes between versions (a folder
    written by 0.105 cannot be `si.load`ed by 0.104)."""
    from probeinterface import read_probeinterface

    folder = Path(folder)
    kwargs = json.loads((folder / 'binary.json').read_text())['kwargs']
    recording = si.read_binary(
        [str(folder / f) for f in kwargs['file_paths']], kwargs['sampling_frequency'],
        kwargs['dtype'], num_channels=kwargs['num_channels'],
        channel_ids=kwargs['channel_ids'], time_axis=kwargs['time_axis'],
        file_offset=kwargs['file_offset'], gain_to_uV=kwargs['gain_to_uV'],
        offset_to_uV=kwargs['offset_to_uV'], is_filtered=kwargs['is_filtered'])
    if (folder / 'probegroup.json').exists():
        recording.set_probegroup(read_probeinterface(folder / 'probegroup.json'), in_place=True)
    for path in sorted((folder / 'properties').glob('*.npy')):
        recording.set_property(path.stem, np.load(path, allow_pickle=True))
    return recording


def join(recordings):
    return si.concatenate_recordings(recordings) if len(recordings) > 1 else recordings[0]


def _shanks(recording):
    """shank -> recording. Recordings without a `group` property are one shank."""
    if 'group' not in recording.get_property_keys():
        recording.set_property('group', np.zeros(recording.get_num_channels(), int))
    return {int(k): v for k, v in recording.split_by('group').items()}


def _phase_shift(recording, settings):
    if settings.get('phase_shift', True) and 'inter_sample_shift' in recording.get_property_keys():
        return si.phase_shift(recording)
    return recording


# ------------------------------------------------------------------ sorting

def prepare(recordings, settings):
    """Join the day's recordings, split by shank, remove bad channels and
    correct the phase shift on each shank (in that order, as in the
    Methods). Returns shank -> recording, and a table of every channel with
    its bad-channel label and whether the sorter saw it."""
    rows, shanks = [], {}
    for shank, recording in _shanks(join(recordings)).items():
        ids, positions = recording.channel_ids, recording.get_channel_locations()
        labels = np.full(len(ids), 'good', dtype=object)
        if settings.get('remove_bad_channels', True):
            bad, labels = si.detect_bad_channels(recording, seed=settings['bad_channel_seed'])
            recording = recording.remove_channels(bad)
        shanks[shank] = _phase_shift(recording, settings)
        rows.append(pd.DataFrame({'channel_id': ids.astype(str), 'x_um': positions[:, 0],
                                  'y_um': positions[:, 1], 'shank': shank,
                                  'label': np.asarray(labels, dtype=str)}))
    channels = pd.concat(rows, ignore_index=True)
    channels['used'] = channels['label'] == 'good'
    return shanks, channels


def sort(shanks, folder, device, params=None):
    """Kilosort 4 on each shank. Returns shank -> sorting."""
    sortings = si.run_sorter('kilosort4', shanks, folder=str(folder),
                             remove_existing_folder=True, verbose=True,
                             torch_device=device, **(params or {}))
    return sortings if isinstance(sortings, dict) else {next(iter(shanks)): sortings}


# Kilosort's per-cluster properties, as spikeinterface reads them; kept with
# clark2025's names.
KILOSORT_PROPERTIES = ['KSLabel', 'KSLabel_repeat', 'Amplitude', 'ContamPct']


def write_sorting(sortings, folder, fs):
    """All shanks' units numbered 0..N-1 (shank by shank) in one npz, with a
    table of each unit's shank (`group`), Kilosort cluster id on that shank
    (`original_cluster_id`) and Kilosort's properties, as plain numpy and
    CSV."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    samples, units, rows = [], [], []
    for shank, sorting in sorted(sortings.items()):
        keys = sorting.get_property_keys()
        props = {k: sorting.get_property(k) for k in KILOSORT_PROPERTIES if k in keys}
        for i, cluster in enumerate(sorting.unit_ids):
            unit = len(rows)
            train = sorting.get_unit_spike_train(cluster)
            samples.append(train)
            units.append(np.full(len(train), unit))
            rows.append({'unit': unit, 'group': shank, 'original_cluster_id': int(cluster),
                         **{k: v[i] for k, v in props.items()}})
    samples = np.concatenate(samples) if samples else np.array([], int)
    units = np.concatenate(units) if units else np.array([], int)
    order = np.argsort(samples, kind='stable')
    np.savez(folder / 'spikes.npz', sample_index=samples[order], unit=units[order],
             sampling_frequency=fs)
    pd.DataFrame(rows).reindex(columns=['unit', 'group', 'original_cluster_id',
                                        *KILOSORT_PROPERTIES]).to_csv(
        folder / 'units.csv', index=False)


def read_sorting(folder):
    """The sorting `write_sorting` saved, with every column of its unit
    table (group, original_cluster_id, Kilosort's properties) as unit
    properties."""
    folder = Path(folder)
    spikes = np.load(folder / 'spikes.npz')
    table = pd.read_csv(folder / 'units.csv')
    for column in ('KSLabel', 'KSLabel_repeat'):
        table[column] = table[column].fillna('').astype(str)
    sorting = si.NumpySorting.from_samples_and_labels(
        [spikes['sample_index']], [spikes['unit']], float(spikes['sampling_frequency']),
        unit_ids=table['unit'].to_numpy())
    for column in table.columns.drop('unit'):
        sorting.set_property(column, table[column].to_numpy())
    return sorting


# ------------------------------------------------------------ postprocessing

def analyzer_recording(recordings, channels, settings):
    """The joined recording the analysers are built on: on each shank, the
    channels the sorter saw, phase-shifted, common-median referenced and
    band-passed; shanks then put back together."""
    used = set(channels.loc[channels['used'], 'channel_id'])
    shanks = []
    for recording in _shanks(join(recordings)).values():
        recording = recording.select_channels(
            [c for c in recording.channel_ids if str(c) in used])
        recording = _phase_shift(recording, settings)
        recording = si.common_reference(recording)
        low, high = settings['bandpass_hz']
        shanks.append(si.bandpass_filter(recording, freq_min=low, freq_max=high))
    return si.aggregate_channels(shanks) if len(shanks) > 1 else shanks[0]


def split(sorting, recording, lengths):
    """(sorting, recording) per session, cut from the joined ones. Every unit
    is kept in every session, as in the lab's split scripts."""
    sorting = si.remove_excess_spikes(sorting, recording)
    out, start = [], 0
    for n in lengths:
        out.append((sorting.frame_slice(start_frame=start, end_frame=start + n),
                    recording.frame_slice(start_frame=start, end_frame=start + n)))
        start += n
    return out


def analyze(sorting, recording, folder, settings):
    """A sorting analyser saved as zarr with the lab's postprocessing
    extensions. The waveforms extension is deleted once everything that
    needs it is computed (GBs per session), as in the lab's scripts."""
    analyzer = si.create_sorting_analyzer(sorting, recording, format='zarr', folder=str(folder),
                                          overwrite=True, **settings['sparsity'])
    analyzer.compute(settings['extensions'])
    if not settings.get('keep_waveforms', False) and analyzer.has_extension('waveforms'):
        analyzer.delete_extension('waveforms')
    return analyzer


def label(analyzer):
    """bombcell labels (spikeinterface's default thresholds), per unit."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*thresholds.*')
        return si.bombcell_label_units(analyzer)['bombcell_label']


def write_curation(labels, path):
    """The labels as a spikeinterface curation file, mecpy's format: reapply
    with `si.apply_curation(analyzer, json.load(open(path)))`."""
    from spikeinterface.curation.curation_model import Curation

    curation = Curation(
        format_version='2',
        unit_ids=list(labels.index),
        label_definitions={'bombcell_label': {'name': 'bombcell_label',
                                              'label_options': BOMBCELL_LABELS,
                                              'exclusive': True}},
        manual_labels=[{'unit_id': u, 'labels': {'bombcell_label': [str(v)]}}
                       for u, v in labels.items()],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(curation.model_dump_json(indent=1))
    return path


def units_table(analyzer, labels, sorting, channels):
    """One row per unit, with clark2025's column names (as in its
    `_clusters.npz` metadata): unit position `x`, `y`, `z` (monopolar
    triangulation, µm; y from the tip); every quality and template metric,
    the waveform measures among them; Kilosort's `group` (shank),
    `original_cluster_id`, `KSLabel`, `Amplitude`, ...; `bombcell_label`;
    the `extremum_channel` and its probe position `coord_probe_x`,
    `coord_probe_y`. `unit` matches the keys of the clusters TsGroup.

    `sorting` is the sorting given to the analyser: unit properties are
    read from there, because a zarr analyser drops text properties such as
    `KSLabel`. `channels` is the channel table from `prepare`."""
    ids = sorting.unit_ids
    metrics = analyzer.get_metrics_extension_data().reindex(ids).reset_index(drop=True)
    locations = analyzer.get_extension('unit_locations').get_data()
    extremum = si.get_template_extremum_channel(analyzer, outputs='id')
    position = channels.set_index('channel_id')[['x_um', 'y_um']]
    main = [str(extremum[u]) for u in ids]
    table = pd.DataFrame({
        'unit': [int(u) for u in ids],
        'x': locations[:, 0],
        'y': locations[:, 1],
        'z': locations[:, 2] if locations.shape[1] > 2 else np.nan,
    })
    properties = pd.DataFrame({k: sorting.get_property(k, ids=ids)
                               for k in ['group', 'original_cluster_id', *KILOSORT_PROPERTIES]
                               if k in sorting.get_property_keys()})
    labels = pd.DataFrame({
        'bombcell_label': labels.reindex(ids).to_numpy(),
        'extremum_channel': main,
        'coord_probe_x': position.loc[main, 'x_um'].to_numpy(),
        'coord_probe_y': position.loc[main, 'y_um'].to_numpy(),
    })
    return pd.concat([table, metrics, properties, labels], axis=1)


# ------------------------------------------------------------------ curation

def curate(tables, settings):
    """Apply the Methods' curation to every session of a day at once
    (`tables`: session -> units table). Adds `isi_excluded` and `accepted`
    columns in place.

    - keep bombcell's `keep` labels (good, mua);
    - exclude a unit, in every session, when its ISI violation ratio exceeds
      `isi_violations_ratio_max` in any session (in `isi_sessions` if set).
    """
    sessions = settings.get('isi_sessions') or list(tables)
    violators = set()
    for name in sessions:
        t = tables[name]
        violators |= set(t.loc[t['isi_violations_ratio'] > settings['isi_violations_ratio_max'],
                               'unit'])
    for t in tables.values():
        t['isi_excluded'] = t['unit'].isin(violators)
        t['accepted'] = t['bombcell_label'].isin(settings['keep']) & ~t['isi_excluded']
    return tables


# --------------------------------------------------------------- exporting

def spike_times(sorting):
    """unit id -> spike times in seconds from the session's first sample."""
    fs = sorting.get_sampling_frequency()
    return {int(u): sorting.get_unit_spike_train(u) / fs for u in sorting.unit_ids}


def unit_waveforms(analyzer, n_channels=16):
    """Each unit's mean and SD waveform (µV) on the `n_channels` channels
    nearest its extremum channel (the units table's `extremum_channel`,
    spikeinterface's choice), nearest first: (units x samples x channels)
    arrays, and those channels' ids (units x channels). Computed from the
    analyser's random spike sample. Channels outside the unit's sparsity
    radius (70 µm) are zero; with 16 channels that happens only at the ends
    of a shank."""
    templates = analyzer.get_extension('templates')
    mean = templates.get_data(operator='average')
    sd = templates.get_data(operator='std')
    positions = analyzer.get_channel_locations()
    main = si.get_template_extremum_channel(analyzer, outputs='index')
    k = min(n_channels, len(analyzer.channel_ids))
    picks = np.stack([
        np.argsort(np.linalg.norm(positions - positions[main[u]], axis=1), kind='stable')[:k]
        for u in analyzer.unit_ids])
    take = picks[:, None, :]
    return (np.take_along_axis(mean, take, axis=2).astype(np.float32),
            np.take_along_axis(sd, take, axis=2).astype(np.float32),
            np.asarray(analyzer.channel_ids).astype(str)[picks])


def kilosort_peak_gpu_gb(folder):
    """Highest GPU memory in use that Kilosort logged (GB), over all shank
    runs. Kilosort resets torch's own peak counter, so torch cannot report
    it afterwards."""
    import re

    used = []
    for log in Path(folder).rglob('kilosort4.log'):
        used += re.findall(r'GPU memory:.*\|\s*([\d.]+)\s*/', log.read_text())
    return max(map(float, used)) if used else None


def sorter_params(folder):
    """The Kilosort parameters actually used, as spikeinterface recorded them."""
    for path in sorted(Path(folder).rglob('spikeinterface_params.json')):
        return json.loads(path.read_text()).get('sorter_params', {})
    return {}
