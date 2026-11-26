"""Ephys pipeline on a simulated Neuropixels 2.0 recording. Kilosort is not
run, and the sorting step runs in this environment instead of the Kilosort
4.0.16 one: the sorter's output is replaced by the ground truth, so
everything around it (bad channels and phase shift, the hand-over between
environments, splitting, analysers, bombcell, the Methods' curation rules,
NWB, tables, QC, the job stage) is tested in seconds on any machine."""

import json
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

si = pytest.importorskip('spikeinterface.full')

from raphael import population
from raphael.cluster import jobs
from raphael.ephys import benchmark
from raphael.ephys import sorting as srt
from raphael.ephys.layout import DEFAULT_CONFIG, load_settings

FS = 30000.0


@pytest.fixture(scope='module')
def day(tmp_path_factory):
    """Two 'sessions' saved as spikeinterface folders, and the ground-truth
    sorting of the two joined."""
    root = tmp_path_factory.mktemp('ephys')
    recording, truth = si.generate_ground_truth_recording(
        durations=[40.0], sampling_frequency=FS, num_units=8,
        probe=benchmark.np2_probe(per_shank=8), seed=2)
    truth = truth.rename_units(np.arange(len(truth.unit_ids)))
    half = recording.get_num_samples() // 2
    folders = []
    for i, (a, b) in enumerate([(0, half), (half, recording.get_num_samples())]):
        folder = root / 'data' / f'session{i}'
        recording.frame_slice(a, b).save(folder=folder, progress_bar=False)
        folders.append(folder)
    return folders, truth, half


def by_shank(truth, shanks):
    """The truth as Kilosort would return it: one sorting per shank, each
    numbered from 0, with Kilosort's labels."""
    out = {}
    for k, shank in enumerate(sorted(shanks)):
        units = truth.unit_ids[k::len(shanks)]
        part = truth.select_units(units, renamed_unit_ids=np.arange(len(units)))
        part.set_property('KSLabel', np.array(['good', 'mua'] * len(units))[:len(units)])
        out[shank] = part
    return out


def test_detect_format(tmp_path):
    (tmp_path / 'a_g0' / 'a_g0_imec0').mkdir(parents=True)
    (tmp_path / 'a_g0' / 'a_g0_imec0' / 'a_g0_t0.imec0.ap.meta').touch()
    (tmp_path / 'oe' / 'experiment1' / 'recording1').mkdir(parents=True)
    (tmp_path / 'oe' / 'experiment1' / 'recording1' / 'structure.oebin').touch()
    assert srt.detect_format(tmp_path / 'a_g0') == 'spikeglx'
    assert srt.detect_format(tmp_path / 'oe') == 'openephys'
    (tmp_path / 'empty').mkdir()
    with pytest.raises(ValueError):
        srt.detect_format(tmp_path / 'empty')


def test_probe_stream_choice():
    names = ['Record Node 101#Neuropix-PXI-100.ProbeA', 'Record Node 101#NI-DAQmx-102.PXIe']
    assert srt._probe_stream(names, 'openephys', '.') == names[0]   # NP2.0: one wideband stream
    np1 = ['Record Node 101#Neuropix-PXI-100.ProbeA-AP',
           'Record Node 101#Neuropix-PXI-100.ProbeA-LFP']
    assert srt._probe_stream(np1, 'openephys', '.') == np1[0]
    assert srt._probe_stream(['imec0.ap', 'imec0.lf', 'nidq'], 'spikeglx', '.') == 'imec0.ap'
    with pytest.raises(ValueError, match='stream'):
        srt._probe_stream(['imec0.ap', 'imec1.ap'], 'spikeglx', '.')


def test_prepare_works_per_shank(day):
    folders, _, _ = day
    recordings = [si.load(f) for f in folders]
    settings = load_settings(DEFAULT_CONFIG)['preprocessing']
    shanks, channels = srt.prepare(recordings, settings)
    assert sorted(shanks) == [0, 1, 2, 3] and len(channels) == 32
    assert all(r.get_num_samples() == sum(x.get_num_samples() for x in recordings)
               for r in shanks.values())
    # Every channel the bad-channel detector did not reject reaches the sorter.
    assert sum(r.get_num_channels() for r in shanks.values()) == channels['used'].sum()


def test_sorting_hand_over_keeps_units_and_labels(tmp_path, day):
    _, truth, _ = day
    sortings = by_shank(truth, [0, 1, 2, 3])
    srt.write_sorting(sortings, tmp_path, FS)
    back = srt.read_sorting(tmp_path)
    assert list(back.unit_ids) == list(range(8))
    assert sum(back.count_num_spikes_per_unit().values()) == sum(
        sum(s.count_num_spikes_per_unit().values()) for s in sortings.values())
    table = pd.read_csv(tmp_path / 'units.csv')
    first = table.iloc[1]   # unit 1: shank 0, Kilosort cluster 1
    np.testing.assert_array_equal(
        back.get_unit_spike_train(1),
        sortings[first['group']].get_unit_spike_train(first['original_cluster_id']))
    assert list(back.get_property('KSLabel'))[:2] == ['good', 'mua']


def test_split_restores_session_times(day):
    folders, truth, half = day
    recording = srt.join([si.load(f) for f in folders])
    parts = srt.split(truth, recording, [half, half])
    for i, (part, rec) in enumerate(parts):
        assert rec.get_num_samples() == half
        assert list(part.unit_ids) == list(truth.unit_ids)   # every unit, every session
        for u in part.unit_ids:
            train = truth.get_unit_spike_train(u)
            expected = train[(train >= i * half) & (train < (i + 1) * half)] - i * half
            np.testing.assert_array_equal(part.get_unit_spike_train(u), expected)


def table(units, y, shank=0, label='good', isi=0.0):
    return pd.DataFrame({'unit': units, 'shank': shank, 'probe_y_um': y,
                         'bombcell_label': label, 'isi_violations_ratio': isi})


def test_isi_rule_excludes_in_every_session():
    settings = load_settings(DEFAULT_CONFIG)['curation']
    tables = {'OF1': table([0, 1, 2], [100, 200, 300], isi=[0.5, 12.0, 0.1]),
              'VR': table([0, 1, 2], [100, 200, 300], isi=[11.0, 0.2, 0.1],
                          label=['good', 'good', 'noise'])}
    srt.curate(tables, settings)
    for t in tables.values():
        assert t['isi_excluded'].tolist() == [True, True, False]
    assert tables['OF1']['accepted'].tolist() == [False, False, True]
    assert tables['VR']['accepted'].tolist() == [False, False, False]   # unit 2 is noise there
    srt.curate(tables, {**settings, 'isi_sessions': ['OF1']})
    assert tables['VR']['isi_excluded'].tolist() == [False, True, False]


def test_ephys_metadata():
    check = jobs._ephys_metadata
    with pytest.raises(jobs.JobError, match='start time'):
        check({}, ['a'])
    with pytest.raises(jobs.JobError, match='one value per input'):
        check({'session_start': '2026-11-03T14:05', 'session_type': 'OF1'}, ['a', 'b'])
    with pytest.raises(jobs.JobError, match='differently'):
        check({'session_start': ['2026-11-03T14:05'] * 2, 'session_type': ['OF', 'OF']},
              ['a', 'b'])
    sessions, subject = check({'session_start': ['2026-11-03T14:05', '2026-11-03T15:00'],
                               'session_type': ['OF1', 'VR'], 'subject': {'age': 'P90D'}},
                              ['a', 'b'])
    assert [s for s, _ in sessions] == ['OF1', 'VR'] and subject == {'age': 'P90D'}
    assert sessions[0][1].utcoffset().total_seconds() == 3600   # Vienna in November


def test_folder_inputs_are_checked_file_by_file(tmp_path, day):
    folders, _, _ = day
    data_root = folders[0].parent
    job = jobs.make(tmp_path / 'jobs' / 'f', 'smoke', ['session0'], data_root,
                    resources={'gpu': False}, site=jobs.SITE_CONFIG)
    record = jobs.load(job)['inputs'][0]
    assert record['files'] and record['bytes'] == sum(f['bytes'] for f in record['files'])
    assert jobs.verify_inputs(jobs.load(job), data_root) == [folders[0]]
    (folders[0] / 'stray.txt').write_text('x')
    with pytest.raises(jobs.JobError, match='not there when the job was made'):
        jobs.verify_inputs(jobs.load(job), data_root)
    (folders[0] / 'stray.txt').unlink()


def test_stage_end_to_end(tmp_path, day, monkeypatch):
    """The kilosort4 stage, with the sorting step run here and Kilosort
    replaced by the ground truth. Needs the gpu extra (Kilosort's version
    is recorded)."""
    pytest.importorskip('kilosort')
    folders, truth, half = day
    data_root = folders[0].parent
    seen = {}

    def fake_sort(shanks, folder, device, params=None):
        seen.update(params=params, shanks=sorted(shanks))
        for shank in shanks:   # Kilosort's per-shank output folders
            (folder / str(shank) / 'sorter_output').mkdir(parents=True)
            (folder / str(shank) / 'sorter_output' / 'spike_times.npy').write_bytes(b'')
        return by_shank(truth, shanks)

    monkeypatch.setattr(srt, 'sort', fake_sort)
    job = jobs.make(tmp_path / 'jobs' / 'M1-D2', 'kilosort4', ['session0', 'session1'],
                    data_root, resources={'gpu': False}, configs=[DEFAULT_CONFIG],
                    params={'session_start': ['2026-11-03T14:05', '2026-11-03T14:30'],
                            'session_type': ['OF1', 'OF2'], 'mouse': 1, 'day': 2,
                            'subject': {'age': 'P90D', 'sex': 'F'},
                            'preprocessing': {'remove_bad_channels': False},
                            'postprocessing': {'keep_analyzer': True,
                                               'write_ecephys_nwb': True}},
                    site=jobs.SITE_CONFIG)
    jobs.run(job, data_root)
    assert jobs.read_status(job)['state'] == 'DONE'
    # Methods: Kilosort defaults with motion correction off, per shank.
    assert seen == {'params': {'do_correction': False, 'use_binary_file': False},
                    'shanks': [0, 1, 2, 3]}

    out = job / 'outputs'
    summary = json.loads((out / 'kilosort4_summary.json').read_text())
    assert [s['session_type'] for s in summary['sessions']] == ['OF1', 'OF2']
    assert summary['units_sorted'] == 8 and summary['channels_used'] == 32
    assert summary['preprocessing']['remove_bad_channels'] is False
    assert (out / 'kilosort4' / '0' / 'sorter_output' / 'spike_times.npy').exists()
    assert (out / 'kilosort4' / 'channels.csv').exists()
    assert not (job / 'work' / 'sorted').exists()

    folder = out / 'M1' / 'D2' / 'OF2'
    base = folder / 'sub-M1_ses-D2_typ-OF2_srt-kilosort4'
    # The processed tree holds the bombcell curation file (as clark2025) but
    # no zarr; with keep_analyzer, that sits in outputs/kilosort4/analyzers/.
    # _ecephys.nwb is there only because this job asks for it (write_ecephys_nwb).
    assert sorted(p.name.removeprefix('sub-M1_ses-D2_typ-OF2_srt-kilosort4')
                  for p in folder.iterdir()) == ['_bombcell_curation.json', '_clusters.npz',
                                                 '_ecephys.nwb', 'provenance', 'qc']
    kept = out / 'kilosort4' / 'analyzers' / 'M1' / 'D2' / 'OF2'
    analyzer_path = kept / 'sub-M1_ses-D2_typ-OF2_srt-kilosort4_analyzer.zarr'
    curation_path = folder / 'sub-M1_ses-D2_typ-OF2_srt-kilosort4_bombcell_curation.json'
    import pynapple as nap
    group = nap.load_file(f'{base}_clusters.npz')
    units = group.metadata.rename_axis('unit').reset_index()
    assert {'unit', 'accepted', 'bombcell_label', 'isi_excluded', 'group',
            'original_cluster_id', 'KSLabel', 'y', 'isi_violations_ratio',
            'firing_rate'} <= set(units.columns)
    assert len(units) == 8 and set(units['group']) == {0, 1, 2, 3}
    assert set(units['KSLabel']) == {'good', 'mua'}   # survives the zarr analyser

    # clark2025's _clusters.npz metadata, column for column, except what
    # needs histology (coordinates, brain_region), manual curation
    # (chris_good) or Kilosort properties the fake sorter does not set.
    clark2025 = {
        'x', 'y', 'z', 'amplitude_cutoff', 'amplitude_cv_median', 'amplitude_cv_range',
        'amplitude_median', 'drift_mad', 'drift_ptp', 'drift_std', 'firing_range',
        'firing_rate', 'isi_violations_count', 'isi_violations_ratio', 'noise_cutoff',
        'noise_ratio', 'num_spikes', 'presence_ratio', 'rp_contamination', 'rp_violations',
        'sd_ratio', 'sliding_rp_violation', 'snr', 'sync_spike_2', 'sync_spike_4',
        'sync_spike_8', 'exp_decay', 'main_peak_to_trough_ratio',
        'main_to_next_extremum_duration', 'num_negative_peaks', 'num_positive_peaks',
        'peak_after_to_trough_ratio', 'peak_after_width', 'peak_before_to_peak_after_ratio',
        'peak_before_to_trough_ratio', 'peak_before_width', 'peak_half_width',
        'peak_to_trough_duration', 'recovery_slope', 'repolarization_slope', 'spread',
        'trough_half_width', 'trough_width', 'velocity_above', 'velocity_below',
        'waveform_baseline_flatness', 'KSLabel', 'group', 'original_cluster_id',
        'bombcell_label', 'extremum_channel', 'coord_probe_x', 'coord_probe_y'}
    assert clark2025 <= set(group.metadata_columns), clark2025 - set(group.metadata_columns)
    keep = units['bombcell_label'].isin(['good', 'mua']) & ~units['isi_excluded'].astype(bool)
    assert (units['accepted'].astype(bool) == keep).all()
    assert (folder / 'qc' / 'units.png').exists()

    # The waveforms extension is gone once the rest is computed.
    analyzer = si.load_sorting_analyzer(analyzer_path)
    assert not analyzer.has_extension('waveforms') and analyzer.has_extension('templates')
    assert analyzer.get_extension('unit_locations').params['method'] == 'monopolar_triangulation'

    import pynwb
    with pynwb.NWBHDF5IO(f'{base}_ecephys.nwb', 'r') as io:
        nwbfile = io.read()
        nwb_units = nwbfile.units.to_dataframe()
        assert len(nwbfile.electrodes) == 32 and set(nwbfile.electrode_groups) == {
            'shank0', 'shank1', 'shank2', 'shank3'}
        assert nwb_units['spike_times'].map(len).sum() == units['num_spikes'].sum()
        assert max(t.max() for t in nwb_units['spike_times'] if len(t)) < half / FS
        assert nwbfile.subject.age == 'P90D'
        assert nwbfile.session_start_time.hour == 14 and nwbfile.session_start_time.minute == 30
        assert nwb_units['electrodes'].map(len).eq(16).all()
        notes = json.loads(nwbfile.notes)
        assert notes['sorted_with'] == ['session0', 'session1']
        assert notes['curation']['keep'] == ['good', 'mua']

    # Waveforms: mean and SD on each unit's 16 nearest channels, largest
    # first, matching the analyser's templates.
    from raphael.ephys import nwb as ephys_nwb
    w = ephys_nwb.read_waveforms(f'{base}_ecephys.nwb')
    samples = analyzer.get_extension('templates').get_data().shape[1]
    assert w['mean'].shape == w['sd'].shape == (8, samples, 16)
    assert w['sampling_frequency'] == FS and list(w['unit']) == units['unit'].tolist()
    assert list(w['channel_ids'][:, 0]) == units['extremum_channel'].astype(str).tolist()
    distance = np.linalg.norm(w['positions_um'] - w['positions_um'][:, :1], axis=2)
    assert (np.diff(distance, axis=1) >= 0).all()          # then by distance

    # The same waveforms ride in _clusters.npz, one array per unit.
    np.testing.assert_allclose(np.stack(units['waveform_mean']), w['mean'], rtol=1e-6)
    np.testing.assert_allclose(np.stack(units['waveform_sd']), w['sd'], rtol=1e-6)
    assert np.stack(units['waveform_channels']).tolist() == w['channel_ids'].tolist()
    np.testing.assert_allclose(np.stack(units['waveform_positions_um']), w['positions_um'])
    nbefore = analyzer.get_extension('templates').nbefore
    assert set(units['waveform_sampling_frequency']) == {FS}
    assert set(units['waveform_nbefore']) == {nbefore}

    # The curation file reapplies with spikeinterface, as mecpy does.
    curation = json.loads(curation_path.read_text())
    curated = si.apply_curation(analyzer, curation)
    assert list(curated.sorting.get_property('bombcell_label')) == units['bombcell_label'].tolist()

    # The shared loader: same calls as for imaging.
    pop = population.load(base)
    assert pop.kind == 'spikes' and list(pop.ids) == units['unit'].tolist()
    rates = pop.binned(1.0)
    assert rates.shape == (20, len(units))
    np.testing.assert_allclose(rates.values.sum(0), units['num_spikes'], rtol=1e-6)


def test_population_loads_imaging_tables(tmp_path):
    import pynapple as nap

    base = tmp_path / 'sub-M1_ses-D2_typ-OF1_src-suite2p'
    pd.DataFrame({'roi': [0, 1, 2], 'accepted': [True, False, True]}).to_csv(
        f'{base}_rois.csv', index=False)
    nap.TsdFrame(t=np.arange(100) / 10, d=np.ones((100, 3)), columns=[0, 1, 2]).save(
        f'{base}_traces.npz')
    pop = population.load(base).accepted()
    assert pop.kind == 'dff' and list(pop.ids) == [0, 2]
    assert pop.binned(1.0).shape == (10, 2)


def test_config_matches_methods():
    settings = load_settings(DEFAULT_CONFIG)
    assert settings['preprocessing'] == {'remove_bad_channels': True, 'bad_channel_seed': 1205,
                                         'phase_shift': True}
    assert settings['sorter']['params'] == {'do_correction': False, 'use_binary_file': False}
    assert settings['postprocessing']['sparsity'] == {'peak_sign': 'both', 'radius_um': 70}
    assert settings['curation']['keep'] == ['good', 'mua']
    assert settings['postprocessing']['keep_analyzer'] is False   # no zarr by default
    assert settings['postprocessing']['waveform_channels'] == 16
    assert settings['postprocessing']['write_ecephys_nwb'] is False   # beh + clusters + lfp
    assert settings['lfp']['rate_hz'] == 1000 and settings['lfp']['band_hz'] == [1, 400]
    assert settings['lfp']['channels']['block_size'] == 8     # the lab's 48 LFP channels
    assert settings['postprocessing']['extensions']['template_metrics'] == {
        'include_multi_channel_metrics': True}   # spread, exp_decay, velocities
    assert settings['curation']['isi_violations_ratio_max'] == 10


def test_one_environment_pins_methods_versions():
    """Kilosort 4.0.16 (the Methods' version) in the one locked environment."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = tomllib.loads((root / 'pyproject.toml').read_text())
    assert 'kilosort==4.0.16' in spec['project']['optional-dependencies']['gpu']
    assert 'spikeinterface[full]==0.104.9' in spec['project']['dependencies']
    assert not (root / 'envs' / 'kilosort4.0.16').exists()


def test_benchmark_scores_truth_as_perfect(tmp_path):
    root = benchmark.simulate(tmp_path, durations=(5.0, 5.0), num_units=5, per_shank=4)
    truth = si.load(root / 'truth' / 'session1')
    recording = si.load(root / 'session1')
    assert recording.get_dtype() == np.int16 and recording.get_num_channels() == 16
    units = pd.DataFrame({'unit': range(5), 'accepted': [True, True, False, False, False]})
    summary, _ = benchmark.score(truth, truth, units)
    assert summary['well_detected'] == 5 and summary['mean_accuracy'] == 1.0
    assert summary['accepted_matched'] == 2


# ---------------------------------------------------------------------- LFP

def test_lfp_blocks_follow_depth_per_shank(day):
    from raphael.ephys import lfp

    folders, _, _ = day
    recording = si.load(folders[0])
    blocks = lfp.blocks(recording, block_size=4)
    assert [shank for shank, _ in blocks] == [0, 0, 1, 1, 2, 2, 3, 3]
    y = dict(zip(recording.channel_ids, recording.get_channel_locations()[:, 1], strict=True))
    for _, ids in blocks:
        assert len(ids) == 4 and all(y[a] <= y[b] for a, b in pairwise(ids))
    assert len(lfp.blocks(recording, block_size=4, every=2)) == 4
    assert [s for s, _ in lfp.blocks(recording, block_size=8, shanks=[0, 3])] == [0, 3]
    assert len(lfp.blocks(recording, block_size=1)) == 32   # every contact
    assert len(lfp.blocks(recording, block_size=3)) == 8    # 2 per shank, 2 contacts dropped


def test_lfp_values(tmp_path, day):
    """Each LFP channel is the mean of its contacts, band-passed and
    resampled; theta phase runs 0 to 2π between peaks."""
    from raphael.ephys import lfp

    folders, _, _ = day
    recording = si.load(folders[0])
    settings = load_settings(DEFAULT_CONFIG)['lfp']
    data, rate, channels = lfp.extract(recording, tmp_path / 'cache', settings)
    assert rate == 1000 and data.shape == (20000, 4) and data.dtype == np.float32
    expected = si.resample(si.bandpass_filter(recording, 1, 400, ignore_low_freq_error=True),
                           1000).get_traces(channel_ids=channels[2]['channel_ids'],
                                            return_in_uV=True).mean(axis=1)
    np.testing.assert_allclose(data[:, 2], expected, rtol=1e-3, atol=1e-2)
    phase = lfp.theta_phase(data, rate)
    finite = phase[np.isfinite(phase)]
    assert finite.size and finite.min() >= 0 and finite.max() <= 2 * np.pi + 1e-6


def lfp_job(tmp_path, data_root, name, overrides=None):
    params = {'session_start': ['2026-11-03T14:05', '2026-11-03T14:30'],
              'session_type': ['OF1', 'OF2'], 'mouse': 1, 'day': 2, 'location': 'CA1'}
    if overrides:
        params['lfp'] = overrides
    job = jobs.make(tmp_path / 'jobs' / name, 'lfp', ['session0', 'session1'], data_root,
                    resources={'gpu': False}, configs=[DEFAULT_CONFIG], params=params,
                    site=jobs.SITE_CONFIG)
    jobs.run(job, data_root)
    assert jobs.read_status(job)['state'] == 'DONE'
    return job / 'outputs'


def test_lfp_stage_writes_the_lab_layout(tmp_path, day):
    import pynwb

    folders, _, _ = day
    out = lfp_job(tmp_path, folders[0].parent, 'lfp-default')
    summary = json.loads((out / 'lfp_summary.json').read_text())
    assert [s['lfp_channels'] for s in summary['sessions']] == [4, 4]
    path = out / 'M1' / 'D2' / 'OF2' / 'sub-M1_ses-D2_typ-OF2_lfp.nwb'
    assert sorted(p.name for p in path.parent.iterdir()) == sorted([path.name, 'provenance'])
    assert (path.parent / 'provenance' / 'lfp-default.json').exists()
    with pynwb.NWBHDF5IO(str(path), 'r') as io:
        f = io.read()
        lfp = f.processing['ecephys']['LFP']['lfp']
        theta = f.processing['ecephys']['Processed']['theta']
        assert lfp.data.shape == theta.data.shape == (20000, 4)
        assert lfp.rate == 1000 and lfp.starting_time == 0 and lfp.conversion == 1e-6
        assert lfp.data.dtype == np.float32
        names = f.electrodes['channel_name'][:]
        assert len(names) == 4 and all(len(n.split('_')) == 8 for n in names)
        assert set(f.electrodes['shank'][:]) == {0, 1, 2, 3}
        assert f.electrodes['location'][0] == 'CA1'
        assert f.session_start_time.minute == 30
    assert not (out.parent / 'work' / 'lfp' / 'OF2').exists()


def test_lfp_size_settings(tmp_path, day):
    """Every contact, every second one kept, float16, no theta: lighter."""
    import pynwb

    folders, _, _ = day
    out = lfp_job(tmp_path, folders[0].parent, 'lfp-light',
                  {'rate_hz': 500, 'band_hz': [1, 200], 'dtype': 'float16',
                   'channels': {'block_size': 1, 'every': 2}, 'theta': {'enabled': False}})
    path = out / 'M1' / 'D2' / 'OF1' / 'sub-M1_ses-D2_typ-OF1_lfp.nwb'
    with pynwb.NWBHDF5IO(str(path), 'r') as io:
        f = io.read()
        lfp = f.processing['ecephys']['LFP']['lfp']
        assert lfp.data.shape == (10000, 16) and lfp.rate == 500
        assert lfp.data.dtype == np.float16
        assert 'Processed' not in f.processing['ecephys'].data_interfaces
        assert all(len(n.split('_')) == 1 for n in f.electrodes['channel_name'][:])


def test_lfp_settings_are_checked_when_made(tmp_path, day):
    folders, _, _ = day
    params = {'session_start': ['2026-11-03T14:05', '2026-11-03T14:30'],
              'session_type': ['OF1', 'OF2']}
    for bad, message in [({'rate_hz': 500}, 'below half'),     # band 1-400 Hz at 500 Hz
                         ({'dtype': 'int8'}, 'float32 or float16'),
                         ({'channels': {'block_size': 0}}, 'at least 1')]:
        with pytest.raises(jobs.JobError, match=message):
            jobs.make(tmp_path / 'jobs' / 'bad', 'lfp', ['session0', 'session1'],
                      folders[0].parent, params={**params, 'lfp': bad},
                      resources={'gpu': False}, site=jobs.SITE_CONFIG)
