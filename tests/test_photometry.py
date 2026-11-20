"""Fibre photometry: reading Doric files, recovering a known signal from
simulated recordings with bleaching and motion, writing the processed
session, and the job stage."""

import os
from pathlib import Path

import numpy as np
import pynwb
import pytest

from raphael.cluster import jobs
from raphael.photometry import benchmark, doric, processing
from raphael.photometry import nwb as fp_nwb
from raphael.photometry.layout import DEFAULT_CONFIG, Session, load_settings
from raphael.photometry.run import process_file

ROLES = {'green': {'role': 'signal', 'reference': 'iso'}, 'iso': {'role': 'reference'},
         'red': {'role': 'signal', 'reference': None}}


@pytest.fixture(scope='module')
def simulated(tmp_path_factory):
    path = tmp_path_factory.mktemp('fp') / 'raw' / 'M1_D2_OF1.doric'
    t, truth = benchmark.simulate(path, duration_s=300.0)
    return path, t, truth


def test_read_doric(simulated):
    path, t, _ = simulated
    rec = doric.read(path)
    assert sorted(rec.channels) == ['green', 'iso', 'red']
    green = rec.channels['green']
    assert green.detector == 'AIN01' and green.carrier_hz == 572.205 and green.led == 'AOUT02'
    assert rec.channels['red'].detector == 'AIN02'
    assert abs(green.rate - benchmark.RATE) < 0.01 and len(green.values) == len(t)
    assert str(rec.start_time) == '2025-08-28 14:08:22'
    # The console-generated ~60 Hz TTL, reconstructed on the signal clock.
    ttl = rec.ttl_times('DIO01')
    assert abs(1 / np.median(np.diff(ttl)) - 59.874) < 0.01 and ttl[-1] <= t[-1]
    assert rec.ttl_times('DIO02') is None
    assert 'green' in doric.describe(rec)


def test_processing_recovers_the_true_signal(simulated):
    path, t, truth = simulated
    rec = doric.read(path)
    _, out, fits = processing.process(t, {k: c.values for k, c in rec.channels.items()}, ROLES)
    true = truth[processing.trim(t, processing.DEFAULTS['trim_start_s'])]
    raw = rec.channels['green'].values[processing.trim(t, 5.0)]
    recovered = np.corrcoef(out['green']['dff'], true)[0, 1]
    uncorrected = np.corrcoef(raw, true)[0, 1]
    assert recovered > 0.9 and recovered > uncorrected + 0.1
    assert fits['green']['reference'] == 'iso' and fits['green']['motion_r'] > 0.5
    assert fits['red']['reference'] is None and 'red' in out and 'iso' not in out
    assert abs(np.std(out['green']['zscore']) - 1) < 1e-6
    assert fits['green']['method'] == 'double_exponential'


def test_bleaching_fit_falls_back_to_linear():
    t = np.linspace(0, 10, 200)
    with pytest.warns(UserWarning, match='linear'):
        _, fit = processing.fit_bleaching(t, np.full_like(t, np.nan))
    assert fit['method'] == 'linear'


def test_process_file_writes_the_session(simulated, tmp_path):
    path, _, _ = simulated
    settings = load_settings(DEFAULT_CONFIG)
    session = Session('1', '2', 'OF1')
    summary = process_file(path, tmp_path, session, settings,
                           subject={'sex': 'F', 'age': 'P90D'})
    assert set(summary['signals']) == {'green', 'red'}

    fp = fp_nwb.load(session.folder(tmp_path) / session.stem)
    assert set(fp.traces.columns) == {'green_dff', 'green_zscore', 'red_dff', 'red_zscore'}
    assert fp.signals == ['green', 'red'] and len(fp.dff('green')) > 0
    assert set(fp.channels['channel']) == {'green', 'iso', 'red'}

    with pynwb.NWBHDF5IO(str(session.photometry(tmp_path)), 'r') as io:
        f = io.read()
        assert set(f.acquisition) == {'green_raw', 'iso_raw', 'red_raw'}
        assert set(f.processing['photometry'].data_interfaces) == {
            'green_dff', 'green_zscore', 'red_dff', 'red_zscore'}
        assert f.subject.sex == 'F' and f.session_start_time.year == 2025
    assert Path(summary['qc']).exists()

    nwbinspector = pytest.importorskip('nwbinspector')
    bad = [m for m in nwbinspector.inspect_nwbfile(nwbfile_path=str(session.photometry(tmp_path)))
           if m.importance.name in ('CRITICAL', 'PYNWB_VALIDATION')]
    assert not bad, bad


def test_missing_signal_channel_is_an_error(simulated, tmp_path):
    path, _, _ = simulated
    settings = load_settings(DEFAULT_CONFIG)
    settings['channels']['blue'] = {'role': 'signal', 'reference': None}
    with pytest.raises(ValueError, match='blue'):
        process_file(path, tmp_path, Session('1', '2', 'OF1'), settings)


def test_photometry_stage(simulated, tmp_path):
    path, _, _ = simulated
    data_root = path.parents[1]
    with pytest.raises(jobs.JobError, match='session_type'):
        jobs.make(tmp_path / 'jobs' / 'bad', 'photometry',
                  ['raw/M1_D2_OF1.doric', 'raw/M1_D2_OF1.doric'], data_root,
                  resources={'gpu': False})
    job = jobs.make(tmp_path / 'jobs' / 'fp', 'photometry', ['raw/M1_D2_OF1.doric'], data_root,
                    resources={'gpu': False}, configs=[DEFAULT_CONFIG],
                    params={'mouse': 1, 'day': 2, 'session_type': 'OF1',
                            'processing': {'lowpass_hz': 5.0}})
    jobs.run(job, data_root)
    assert jobs.read_status(job)['state'] == 'DONE'
    assert (job / 'outputs' / 'M1' / 'D2' / 'OF1' / 'sub-M1_ses-D2_typ-OF1_photometry.nwb').exists()


EXAMPLE = Path(os.environ.get('RAPHAEL_FP_EXAMPLE', '~/Desktop/example FP data')).expanduser()


@pytest.mark.skipif(not EXAMPLE.exists(), reason='real Doric example files not on this machine')
def test_real_doric_recordings(tmp_path):
    """The example GRAB-NE recordings (not in git): every file reads and
    processes, and the isosbestic channel tracks the green one."""
    files = sorted(EXAMPLE.glob('*.doric'))
    assert files
    for i, path in enumerate(files):
        summary = process_file(path, tmp_path, Session('1', '1', f'S{i}'),
                               load_settings(DEFAULT_CONFIG))
        assert summary['signals']['green']['motion_r'] > 0.5
