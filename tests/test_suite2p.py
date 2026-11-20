"""suite2p output -> Plane, on a fake plane folder (suite2p itself not run)."""

import numpy as np
import pytest

from raphael.imaging import suite2p as s2p
from raphael.imaging.layout import DEFAULT_CONFIG, load_settings

RATE = 10.0


@pytest.fixture
def plane_folder(tmp_path):
    rng = np.random.default_rng(0)
    n_rois, n_frames, shape = 4, 300, (20, 30)
    F = 100 + rng.random((n_rois, n_frames)).astype(np.float32)
    stat = np.array([{'ypix': np.array([i, i + 1]), 'xpix': np.array([2 * i, 2 * i]),
                      'lam': np.array([0.6, 0.4]), 'med': [i, 2 * i], 'npix': 2,
                      'radius': 1.0, 'compact': 1.0, 'aspect_ratio': 1.0,
                      'snr': 1.0, 'skew': 0.5, 'std': 3.0} for i in range(n_rois)])
    reg = {'meanImg': rng.random(shape), 'meanImgE': rng.random(shape),
           'xoff': rng.normal(size=n_frames), 'yoff': rng.normal(size=n_frames),
           'yrange': [2, 18], 'xrange': [3, 27]}
    det = {'max_proj': rng.random((16, 24)), 'Vcorr': rng.random((16, 24))}
    for name, value in {'F': F, 'Fneu': F * 0.2, 'spks': F * 0, 'stat': stat,
                        'iscell': np.array([[1, 0.9], [0, 0.2], [1, 0.7], [0, 0.4]]),
                        'reg_outputs': reg, 'detect_outputs': det}.items():
        np.save(tmp_path / f'{name}.npy', value, allow_pickle=True)
    return tmp_path


def test_to_plane(plane_folder):
    plane = s2p.to_plane(plane_folder, RATE, load_settings(DEFAULT_CONFIG))
    assert plane.F.shape == (300, 4) and plane.dff.shape == (300, 4)  # frames x ROIs
    assert plane.rois['accepted'].tolist() == [True, False, True, False]
    np.testing.assert_allclose(plane.masks[1], [[2, 1, 0.6], [2, 2, 0.4]], rtol=1e-6)  # x, y, w
    assert plane.images['max_projection'].shape == (20, 30)  # padded to full frame
    assert plane.images['max_projection'][:2].sum() == 0
    assert plane.shifts.shape == (300, 2) and plane.timestamps[-1] == pytest.approx(29.9)


def test_settings_merge_keeps_sibling_defaults():
    pytest.importorskip('suite2p')
    settings = s2p.settings_for(6.3, 'cpu', {'registration': {'nonrigid': False}})
    assert settings['registration']['nonrigid'] is False
    assert 'maxregshift' in settings['registration']  # suite2p's own merge drops this
    assert settings['fs'] == 6.3 and settings['torch_device'] == 'cpu'


def test_benchmark_matching():
    from raphael.imaging import benchmark

    def square(x, y, w=1.0):
        xs, ys = np.meshgrid(range(x, x + 3), range(y, y + 3))
        return np.column_stack([xs.ravel(), ys.ravel(), np.full(9, w)]).astype(np.float32)

    reference = [square(0, 0), square(10, 10), square(20, 0)]
    candidate = [square(10, 10), square(1, 0), square(30, 10)]  # exact, 6/12 overlap, miss
    summary, pairs = benchmark.match(candidate, reference, (40, 40), min_iou=0.3)
    assert summary['matched'] == 2 and summary['recall'] == pytest.approx(2 / 3)
    assert summary['precision'] == pytest.approx(2 / 3)
    assert sorted(round(p[2], 2) for p in pairs) == [0.5, 1.0]


def test_blank_frames_are_found_and_blanked(tmp_path, plane_folder):
    import h5py

    movie = np.full((50, 8, 8), 800, np.int16)
    movie[[3, 40, 41, 42]] = -5                       # shutter closed
    with h5py.File(tmp_path / 'movie.h5', 'w') as f:
        f['data'] = movie
    bad = s2p.blank_frames(tmp_path / 'movie.h5', chunk=7)
    assert bad.tolist() == [3, 40, 41, 42]

    plane = s2p.to_plane(plane_folder, RATE, load_settings(DEFAULT_CONFIG), bad_frames=[5, 6])
    assert np.isnan(plane.dff[[5, 6]]).all() and not np.isnan(plane.dff[[4, 7]]).any()
    assert not np.isnan(plane.F).any()                # raw traces are kept as recorded


def test_frame_ranges():
    from raphael.cluster.jobs import _ranges
    assert _ranges([3, 40, 41, 42]) == [[3, 3], [40, 42]] and _ranges([]) == []


def test_session_metadata():
    from datetime import datetime

    from raphael.cluster import jobs
    from raphael.imaging import nwb

    t = nwb.parse_start_time('2026-11-03T14:05', 'Europe/Vienna')
    assert t.utcoffset().total_seconds() == 3600                      # CET in November
    assert nwb.parse_start_time('2026-11-03T14:05+00:00', 'Europe/Vienna').utcoffset().seconds == 0
    assert nwb.parse_start_time(datetime(2026, 7, 1, 9), 'Europe/Vienna').utcoffset().seconds == 7200
    sub = nwb.subject_fields({'age': 'P84D', 'sex': 'F', 'date_of_birth': '2026-08-11'})
    assert sub['date_of_birth'].tzinfo is not None
    for bad in ({'age': '84 days'}, {'sex': 'female'}, {'colour': 'black'}):
        with pytest.raises(ValueError):
            nwb.subject_fields(bad)

    with pytest.raises(jobs.JobError, match='session_start'):
        jobs.CHECKS['suite2p']({'fs': 6.3})
    with pytest.raises(jobs.JobError, match='ISO 8601'):
        jobs.CHECKS['suite2p']({'fs': 6.3, 'session_start': '2026-11-03',
                                'subject': {'age': '12 weeks'}})
    start, _ = jobs.CHECKS['suite2p']({'fs': 6.3, 'session_start': '2026-11-03T14:05'})
    assert start.tzinfo is not None
