"""Imaging output template: synthetic planes in, publishable NWB out."""

import numpy as np
import pandas as pd
import pynapple as nap
import pynwb
import pytest

from raphael.imaging import nwb
from raphael.imaging.layout import Session, load_settings
from raphael.imaging.traces import baseline, dff

RATE = 10.0


def make_plane(n_frames=600, n_rois=5, shape=(32, 48), seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / RATE
    masks = []
    for k in range(n_rois):
        y, x = np.mgrid[0:3, 0:3]
        masks.append(np.column_stack([x.ravel() + 5 * k, y.ravel() + 2,
                                      np.full(9, 1 / 9)]).astype(np.float32))
    F = 100 + rng.normal(0, 1, (n_frames, n_rois)).astype(np.float32)
    F[300:310, 0] += 50  # one transient
    return nwb.Plane(
        timestamps=t, rate=RATE, masks=masks, image_shape=shape, F=F,
        grid_spacing_um=(2.0, 2.0), depth_um=150.0,
        dff=dff(F, RATE), shifts=rng.normal(0, 0.5, (n_frames, 2)),
        images={'mean': rng.random(shape), 'correlation': rng.random(shape)},
        rois=pd.DataFrame({'accepted': [True, True, False, True, True],
                           'mask_type': ['soma'] * 5}),
    )


def test_baseline_tracks_slow_drift():
    t = np.arange(6000) / RATE
    F = (100 + 0.05 * t)[:, None].astype(np.float32)
    F0 = baseline(F, RATE, percentile=8, window_s=60)
    assert np.max(np.abs(F0 - F)) < 2.0
    assert np.nanmax(np.abs(dff(F, RATE))) < 0.02


def test_session_paths_follow_clark2025():
    s = Session('17797', '7', 'scan4')
    assert s.beh('/r').as_posix() == '/r/M17797/D7/scan4/sub-M17797_ses-D7_typ-scan4_beh.nwb'
    assert s.ophys('/r', 'suite2p').name == 'sub-M17797_ses-D7_typ-scan4_src-suite2p_ophys.nwb'


@pytest.fixture
def written(tmp_path):
    settings = load_settings()
    session = Session('1', '1', 'test')
    planes = [make_plane(seed=0), make_plane(seed=1)]
    subject = {'sex': 'M', 'age': 'P90D'}
    f = nwb.new_nwbfile(settings, session, 'synthetic', subject=subject, notes={'a': 1})
    nwb.build_ophys_nwb(f, planes, settings, indicator='GCaMP8m')
    ophys = nwb.write(f, session.ophys(tmp_path, 'synthetic'))
    speed = np.r_[np.zeros(300), np.full(300, 10.0)]
    t = np.arange(600) / RATE
    stim = pd.DataFrame({'start_time': [1.0, 20.0], 'stop_time': [5.0, 25.0],
                         'stimulus_type': ['a', 'b']})
    b = nwb.new_nwbfile(settings, session, 'synthetic', subject=subject)
    nwb.build_beh_nwb(b, {'S': (speed, 'cm/s', t, 'speed')}, {'Stim': (stim, 'stimuli')},
                      settings)
    beh = nwb.write(b, session.beh(tmp_path))
    rois = nwb.export_tables(planes, session.traces(tmp_path, 'synthetic'),
                             session.rois(tmp_path, 'synthetic'))
    return session, planes, ophys, beh, rois, tmp_path


def test_ophys_roundtrip(written):
    _, planes, ophys, _, _, _ = written
    with pynwb.NWBHDF5IO(str(ophys), 'r') as io:
        f = io.read()
        mod = f.processing['ophys']
        seg = mod['ImageSegmentation']['PlaneSegmentation2']
        assert len(seg) == 5
        assert list(seg['accepted'][:]) == [True, True, False, True, True]
        assert seg['pixel_mask'][0].shape == (9,)
        F = mod['Fluorescence']['RoiResponseSeries1']
        np.testing.assert_allclose(F.data[:], planes[0].F)
        dff_series = mod['DfOverF']['RoiResponseSeries1']
        np.testing.assert_allclose(dff_series.timestamps[:], planes[0].timestamps)
        assert f.imaging_planes['ImagingPlane1'].indicator == 'GCaMP8m'
        assert f.imaging_planes['ImagingPlane1'].imaging_rate == RATE
        assert set(mod['SummaryImages1'].images) == {'mean', 'correlation'}
        assert 'xy_shifts2' in mod.data_interfaces
        assert f.subject.subject_id == 'M1'


def test_behaviour_and_tables(written):
    session, _, _, beh, rois, tmp = written
    with pynwb.NWBHDF5IO(str(beh), 'r') as io:
        f = io.read()
        assert f.acquisition['S'].unit == 'cm/s'
        moving = f.intervals['moving'].to_dataframe()
        assert len(moving) == 1 and moving.start_time.iloc[0] == pytest.approx(30.0, abs=0.1)
        assert list(f.intervals['Stim'].to_dataframe().stimulus_type) == ['a', 'b']
    assert list(rois['plane']) == [1] * 5 + [2] * 5
    assert list(rois['roi']) == list(range(10))
    traces = nap.load_file(str(session.traces(tmp, 'synthetic')))
    assert traces.shape == (600, 10)


def test_files_pass_nwbinspector_critical_checks(written):
    nwbinspector = pytest.importorskip('nwbinspector')
    _, _, ophys, beh, _, _ = written
    for path in (ophys, beh):
        critical = [m for m in nwbinspector.inspect_nwbfile(nwbfile_path=str(path))
                    if m.importance.name in ('CRITICAL', 'PYNWB_VALIDATION', 'ERROR')]
        assert not critical, [m.message for m in critical]
