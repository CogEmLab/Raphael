"""Syncing a behaviour CSV with a Neuropixels recording from one TTL channel,
on simulated sessions with known clock drift and offset: the alignment is
recovered, and every fault the checks are for is caught."""

import json

import numpy as np
import pandas as pd
import pytest

from raphael import sync
from raphael.cluster import jobs
from raphael.ephys.layout import Session
from raphael.sync.run import sync_session
from raphael.sync.simulate import simulate_session

SESSION = Session('1', '2', 'OF1')


def run_align(sim, **settings):
    beh = sync.read_behaviour(sim['behaviour'])
    rec = sync.read_openephys(sim['recording'])
    return sync.align(sync.edges(beh['time'], beh['ttl']), rec.pulses,
                      behaviour_span=(beh['time'].iloc[0], beh['time'].iloc[-1]),
                      recording_span=(0, rec.duration), **settings)


@pytest.fixture(scope='module')
def clean(tmp_path_factory):
    return simulate_session(tmp_path_factory.mktemp('clean'), duration_s=90, n_channels=2)


def test_edges_are_dated_between_samples():
    t = np.arange(0, 1, 0.1)
    ttl = [0, 0, 1, 1, 1, 0, 0, 1, 0, 0]
    p = sync.edges(t, ttl)
    assert np.allclose(p.onsets, [0.15, 0.65]) and np.allclose(p.widths, [0.3, 0.1])


def test_simulated_recording_reads_as_open_ephys(clean):
    """The light NP2 recording is a real Open Ephys folder: Raphael's
    recording reader and spikeinterface's event reader both open it, and
    agree with sync's own reader on the pulses."""
    si = pytest.importorskip('spikeinterface.full')
    from raphael.ephys import sorting as srt

    recording = srt.read(clean['recording'])
    assert recording.get_sampling_frequency() == 30000.0
    rec = sync.read_openephys(clean['recording'])
    assert recording.get_total_duration() == pytest.approx(rec.duration)
    events = si.read_openephys_event(clean['recording'])
    times = events.get_events(channel_id=events.channel_ids[0])['time']
    assert np.allclose(times - rec.first_sample / rec.sample_rate, rec.pulses.onsets)


def test_clean_session_recovers_the_clocks(clean):
    result = run_align(clean)
    assert result.passed, result.checks
    # Every rig time lands within a few ms of where the recording saw it,
    # despite the behaviour CSV sampling the TTL only every ~17 ms.
    truth = clean['slope'] * clean['onsets'] + clean['intercept']
    assert np.max(np.abs(result.to_recording(clean['onsets']) - truth)) < 0.005
    assert abs(result.drift_ppm - clean['drift_ppm']) < 4 * result.drift_se_ppm
    assert result.checks['counts_agree']['passed']      # pulses before the recording explained
    assert len(result.pairs) == len(sync.read_openephys(clean['recording']).pulses)


@pytest.mark.parametrize('fault, where', [
    ({'drop_behaviour_pulses': [30]}, 'recording_unpaired_s'),
    ({'extra_recording_pulses': [40.123]}, 'recording_unpaired_s'),
])
def test_dropped_or_extra_pulses_fail_and_are_located(tmp_path, fault, where):
    sim = simulate_session(tmp_path, duration_s=90, n_channels=2, **fault)
    result = run_align(sim)
    assert not result.passed
    assert result.checks['counts_agree']['passed'] is False
    assert len(result.checks['counts_agree']['value'][where]) == 1
    if 'extra_recording_pulses' in fault:
        assert result.checks['counts_agree']['value'][where][0] == pytest.approx(40.123)
    truth = sim['slope'] * sim['onsets'] + sim['intercept']           # the fit survives
    assert np.max(np.abs(result.to_recording(sim['onsets']) - truth)) < 0.005


def test_too_much_drift_fails(tmp_path):
    result = run_align(simulate_session(tmp_path, duration_s=90, n_channels=2, drift_ppm=500))
    assert not result.passed and result.checks['drift']['passed'] is False
    assert result.checks['residuals']['passed']


def test_files_from_different_sessions_are_refused(tmp_path):
    a = simulate_session(tmp_path / 'a', duration_s=60, n_channels=1, seed=1)
    b = simulate_session(tmp_path / 'b', duration_s=60, n_channels=1, seed=2)
    with pytest.raises(sync.SyncError, match='same session'):
        run_align({'behaviour': a['behaviour'], 'recording': b['recording']})


def test_regular_pulses_are_ambiguous_when_the_recording_starts_late(tmp_path):
    late = run_align(simulate_session(tmp_path / 'late', duration_s=60, n_channels=1,
                                      interval=(1, 1)))
    assert late.checks['unambiguous']['passed'] is False and not late.passed
    whole = simulate_session(tmp_path / 'whole', duration_s=60, n_channels=1, interval=(1, 1),
                             recording_start_s=0.2, recording_end_gap_s=0.3)
    result = run_align(whole, expected_interval_s=1.0)
    assert result.passed, result.checks
    assert result.checks['intervals']['passed']


def test_sync_session_writes_behaviour_on_the_recording_clock(clean, tmp_path):
    result, written = sync_session(clean['behaviour'], clean['recording'], tmp_path, SESSION)
    assert result.passed and set(written) == {'report', 'figure', 'behaviour'}
    beh = pd.read_csv(written['behaviour'])
    assert list(beh.columns[:2]) == ['time', 'time_behaviour']
    rec = sync.read_openephys(clean['recording'])
    assert beh['time'].min() >= 0 and beh['time'].max() <= rec.duration
    # Rows where the TTL goes high now sit on the recording's pulses.
    rises = beh['time'][beh['ttl'].diff() == 1].to_numpy()
    nearest = np.abs(rises[:, None] - rec.pulses.onsets[None, :]).min(axis=1)
    assert np.max(nearest) < 0.02                                   # one behaviour frame
    report = json.loads(written['report'].read_text())
    assert report['passed'] and report['sources']['ttl_line'] == 1


def test_failed_sync_writes_only_the_report_unless_waived(tmp_path):
    sim = simulate_session(tmp_path / 'sim', duration_s=60, n_channels=1,
                           extra_recording_pulses=[20.5])
    result, written = sync_session(sim['behaviour'], sim['recording'], tmp_path / 'a', SESSION)
    assert not result.usable and 'behaviour' not in written
    result, written = sync_session(sim['behaviour'], sim['recording'], tmp_path / 'b', SESSION,
                                   waive='spurious pulse from cable test at 20 s')
    assert result.usable and 'behaviour' in written
    assert json.loads(written['report'].read_text())['waived'].startswith('spurious')


def test_sync_stage(clean, tmp_path):
    data_root = clean['behaviour'].parent
    with pytest.raises(jobs.JobError, match='two inputs'):
        jobs.make(tmp_path / 'bad', 'sync', ['behaviour.csv'], data_root,
                  params={'mouse': 1, 'day': 2, 'session_type': 'OF1'})
    rel = clean['recording'].relative_to(data_root)
    job = jobs.make(tmp_path / 'sync-1', 'sync', ['behaviour.csv', str(rel)], data_root,
                    resources={'gpu': False}, params={'mouse': 1, 'day': 2,
                                                      'session_type': 'OF1'})
    jobs.run(job, data_root)
    folder = job / 'outputs' / 'M1' / 'D2' / 'OF1'
    assert (folder / 'sub-M1_ses-D2_typ-OF1_beh.csv').exists()
    assert (folder / 'qc' / 'sub-M1_ses-D2_typ-OF1_sync.png').exists()
    assert (folder / 'provenance' / 'sync-1.json').exists()


def test_command_on_a_simulated_session(tmp_path, capsys):
    from raphael.scripts.sync import main
    assert main(['--simulate', str(tmp_path)]) == 0
    assert 'sync PASSED' in capsys.readouterr().out
    assert (tmp_path / 'processed' / 'M1' / 'D1' / 'OF1' / 'sub-M1_ses-D1_typ-OF1_beh.csv').exists()
