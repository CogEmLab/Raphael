"""One-photon imaging: the Inscopix reader on a tiny hand-built .isxd, the
session table, CNMF-E output into a Plane, pose into the behaviour file,
and the cnmfe job checks. CaImAn itself is not run here (it runs in its
own environment); the real-data check is the candidate recording."""

import json
import struct
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pynwb
import pytest
from scipy import sparse

from raphael.cluster import jobs
from raphael.imaging import cnmfe, inscopix, nwb
from raphael.imaging.layout import Session, load_settings

START_MS = 1717607590614                      # 2024-06-05 17:13:10.614 Vienna


def write_isxd(path, frames, period=(16670, 1000000), header_footer=True, dropped=(),
               animal='8472', start_ms=START_MS):
    """An .isxd as nVista writes one: frames (with 2 header and 2 footer
    rows each), NUL, JSON, uint64 JSON length."""
    t, h, w = frames.shape
    meta = {
        'type': 0, 'dataType': 0, 'fileVersion': 1, 'hasFrameHeaderFooter': header_footer,
        'spacingInfo': {'numPixels': {'x': w, 'y': h},
                        'pixelSize': {'x': {'num': 3, 'den': 1}, 'y': {'num': 3, 'den': 1}}},
        'timingInfo': {'numTimes': t, 'period': {'num': period[0], 'den': period[1]},
                       'start': {'secsSinceEpoch': {'num': start_ms, 'den': 1000},
                                 'utcOffset': 0},
                       'dropped': list(dropped), 'cropped': []},
        'extraProperties': {'microscope': {'type': 'NVista3', 'gain': 8, 'exp': 16.57,
                                           'led': {'exPower': 1.8},
                                           'customMetadata': {'Animal ID': animal}},
                            'probe': {'name': 'ProView Integrated Lens 1.0 mm x 4 mm'}},
    }
    with open(path, 'wb') as f:
        for frame in frames.astype(np.uint16):
            if header_footer:
                f.write(np.full((2, w), 7, np.uint16).tobytes())
            f.write(frame.tobytes())
            if header_footer:
                f.write(np.full((2, w), 9, np.uint16).tobytes())
        text = json.dumps(meta).encode()
        f.write(b'\x00' + text + struct.pack('<Q', len(text)))
    return path


@pytest.fixture
def movie():
    rng = np.random.default_rng(0)
    return rng.integers(0, 4000, (10, 6, 8)).astype(np.uint16)


def test_isxd_frames_and_timing(tmp_path, movie):
    rec = inscopix.open_isxd(write_isxd(tmp_path / 'a.isxd', movie, dropped=[3]))
    np.testing.assert_array_equal(rec.frames, movie)          # header/footer rows left out
    assert rec.rate == pytest.approx(1 / 0.01667)
    # nVista stores the local wall clock with utcOffset 0: 17:13 in Vienna, 15:13 UTC.
    assert rec.start == datetime(2024, 6, 5, 15, 13, 10, 614000, tzinfo=UTC)
    assert rec.start.isoformat() == '2024-06-05T17:13:10.614000+02:00'
    np.testing.assert_allclose(rec.timestamps[:3], [0, 0.01667, 0.03334])
    assert list(rec.dropped) == [3]
    info = rec.info()
    assert info['animal_id'] == '8472' and info['frames'] == 10 and info['gain'] == 8
    flat = inscopix.open_isxd(write_isxd(tmp_path / 'b.isxd', movie, header_footer=False))
    np.testing.assert_array_equal(flat.frames, movie)


def test_isxd_without_description_is_refused(tmp_path, movie):
    path = tmp_path / 'broken.isxd'
    path.write_bytes(movie.tobytes() + b'\x00' * 64)          # a recording never closed
    with pytest.raises(inscopix.InscopixError, match='JSON description'):
        inscopix.open_isxd(path)
    write_isxd(path, movie)
    path.write_bytes(path.read_bytes()[:-200] + path.read_bytes()[-199:])   # a byte short
    with pytest.raises(inscopix.InscopixError):
        _ = inscopix.open_isxd(path).frames


def test_downsample_bins_space_and_time(tmp_path, movie):
    import tifffile

    rec = inscopix.open_isxd(write_isxd(tmp_path / 'a.isxd', movie, dropped=[3]))
    rate, times = inscopix.downsample(rec, tmp_path / 'out.tif', spatial=2, temporal=2,
                                      chunk_frames=4)
    out = tifffile.imread(tmp_path / 'out.tif')
    filled = movie.astype(float)
    filled[3] = filled[2]                                     # dropped frame: previous one
    expected = filled.reshape(5, 2, 3, 2, 4, 2).mean(axis=(1, 3, 5))
    np.testing.assert_array_equal(out, np.rint(expected).astype(np.uint16))
    assert rate == pytest.approx(rec.rate / 2)
    np.testing.assert_allclose(times, (np.arange(5) * 2 + 0.5) * rec.period)


def test_session_table_numbers_days_by_recording_date(tmp_path, movie):
    day, hour = 86400000, 3600000
    for folder, animal, offset in [('hab', '1', 0), ('hbt1', '1', day),
                                   ('hbt1b', '1', day + hour), ('hab', '2', 0)]:
        d = tmp_path / animal / folder
        d.mkdir(parents=True)
        write_isxd(d / 'm.isxd', movie, animal=animal, start_ms=START_MS + offset)
        (d / 'session.json').write_text('{}')
    other = tmp_path / '1' / 'exported'                       # no session.json: not raw
    other.mkdir()
    write_isxd(other / 'm-PP.isxd', movie, animal='1')
    table = inscopix.sessions(tmp_path)
    assert len(table) == 4 and len(inscopix.sessions(tmp_path, raw_only=False)) == 5
    table['day'] = inscopix.day_numbers(table)
    days = {(r.mouse, r.folder): r.day for r in table.itertuples()}
    assert days == {('1', 'hab'): 1, ('1', 'hbt1'): 2, ('1', 'hbt1b'): 2, ('2', 'hab'): 1}
    assert table.start_local.iloc[0] == '2024-06-05T17:13:10+02:00'


def test_unclosed_recording_falls_back_to_session_json(tmp_path, movie):
    d = tmp_path / 'hab2'
    d.mkdir()
    (d / 'm.isxd').write_bytes(movie.tobytes() + b'\x00' * 64)
    (d / 'session.json').write_text(json.dumps({
        'date': '2024-05-31T13:02:10.000Z', 'name': 'HAB2',
        'microscope': {'fps': {'fps': 60}, 'gain': 8, 'customMetadata': {'Animal ID': '8476'}}}))
    row = inscopix.sessions(tmp_path).iloc[0]
    assert row['mouse'] == '8476' and row['date'] == '2024-05-31'
    assert 'JSON description' in row['error'] and pd.isna(row['duration_s'])


def cnmfe_result(dims=(6, 8), n=3, frames=20, seed=0):
    rng = np.random.default_rng(seed)
    A = np.zeros((dims[0] * dims[1], n), np.float32)
    footprints = [[(1, 2), (2, 2)], [(4, 0)], [(5, 7), (5, 6), (4, 7)]]   # (y, x)
    for k, pixels in enumerate(footprints[:n]):
        for y, x in pixels:
            A[y + x * dims[0], k] = 1 + k                     # CaImAn: column-major
    C = rng.random((n, frames)).astype(np.float32)
    return {'A': sparse.csc_matrix(A), 'dims': np.asarray(dims), 'C': C,
            'YrA': np.full_like(C, 0.5), 'S': C * 2, 'detrended': C - 1,
            'accepted': np.array([True, False, True]), 'snr': np.array([4.0, 1.0, 5.0]),
            'r_value': np.array([0.9, 0.2, 0.95]), 'shifts': np.zeros((frames, 2), np.float32),
            'mean': rng.random(dims), 'correlation': rng.random(dims), 'pnr': rng.random(dims)}


def test_masks_use_caimans_pixel_order():
    masks = cnmfe.masks(cnmfe_result()['A'], (6, 8))
    np.testing.assert_array_equal(masks[0][:, :2], [[2, 1], [2, 2]])      # x, y
    np.testing.assert_array_equal(masks[1], [[0, 4, 2]])
    assert sorted(map(tuple, masks[2][:, :2].tolist())) == [(6, 5), (7, 4), (7, 5)]


def test_to_plane_and_tables(tmp_path):
    result = cnmfe_result()
    times = np.arange(20) / 30
    plane = cnmfe.to_plane(result, times, 30.0)
    np.testing.assert_allclose(plane.F, (result['C'] + 0.5).T)
    np.testing.assert_allclose(plane.denoised, result['C'].T)
    assert plane.dff is None and plane.image_shape == (6, 8)
    assert list(plane.rois['accepted']) == [True, False, True]
    settings = load_settings(jobs.provenance.PACKAGE_ROOT / 'configs' / 'imaging_1p.yaml')
    session = Session('8472', '6', 'HBTD3')
    f = nwb.new_nwbfile(settings, session, '1p test')
    nwb.build_ophys_nwb(f, [plane], settings, indicator='GCaMP6f',
                        description='Processed one-photon (miniscope) imaging data.')
    path = nwb.write(f, session.ophys(tmp_path, 'cnmfe'))
    with pynwb.NWBHDF5IO(str(path), 'r') as io:
        mod = io.read().processing['ophys']
        assert mod.description.startswith('Processed one-photon')
        np.testing.assert_allclose(mod['Denoised']['RoiResponseSeries1'].data[:],
                                   result['C'].T)
        assert 'not dF/F' in mod['Detrended']['RoiResponseSeries1'].description
        assert 'DfOverF' not in mod.data_interfaces
        assert set(mod['SummaryImages1'].images) == {'mean', 'correlation', 'peak_to_noise'}
    rois = nwb.export_tables([plane], session.traces(tmp_path, 'cnmfe'),
                             session.rois(tmp_path, 'cnmfe'), trace='denoised')
    import pynapple as nap
    traces = nap.load_file(str(session.traces(tmp_path, 'cnmfe')))
    np.testing.assert_allclose(traces.values, result['C'].T)
    assert list(rois.columns[:3]) == ['roi', 'plane', 'plane_roi']
    with pytest.raises(ValueError, match='dff'):
        nwb.export_tables([plane], tmp_path / 't.npz', tmp_path / 'r.csv', trace='dff')


def test_temporal_factor_and_settings():
    assert cnmfe.temporal_factor(59.988, {'temporal_downsample': {'target_hz': 30}}) == 2
    assert cnmfe.temporal_factor(20.0, {'temporal_downsample': {'target_hz': 30}}) == 1
    with pytest.raises(cnmfe.CnmfeError, match='whole multiple'):
        cnmfe.temporal_factor(40.0, {'temporal_downsample': {'target_hz': 30}})
    assert cnmfe.temporal_factor(40.0, {'temporal_downsample': 2}) == 2
    s = cnmfe.settings_with_defaults({'quality': {'min_SNR': 2.5}})
    assert s['quality']['min_SNR'] == 2.5 and s['quality']['rval_thr'] == 0.85
    settings = load_settings(jobs.provenance.PACKAGE_ROOT / 'configs' / 'imaging_1p.yaml')
    assert cnmfe.settings_with_defaults(settings['cnmfe'])['cnmfe'] == cnmfe.DEFAULTS['cnmfe']


def test_cnmfe_job_checks():
    jobs.CHECKS['cnmfe']({}, ['raw/m.isxd'])
    with pytest.raises(jobs.JobError, match='fs'):
        jobs.CHECKS['cnmfe']({'session_start': '2024-06-05T17:13'}, ['movie.tif'])
    with pytest.raises(jobs.JobError, match='one movie'):
        jobs.CHECKS['cnmfe']({}, ['a.isxd', 'b.isxd'])
    with pytest.raises(jobs.JobError, match='unknown cnmfe'):
        jobs.CHECKS['cnmfe']({'cnmfe': {'qualty': {}}}, ['m.isxd'])
    start, _ = jobs.CHECKS['cnmfe']({'fs': 30, 'session_start': '2024-06-05T17:13'}, ['m.tif'])
    assert start.utcoffset().total_seconds() == 7200


def test_pose_goes_into_the_behaviour_file(tmp_path):
    from ndx_pose import PoseEstimation  # noqa: F401  (registers the extension)

    t = np.arange(50) / 30
    columns = pd.MultiIndex.from_tuples(
        [(i, bp, c) for i in ('helper', 'victim') for bp in ('Nose', 'BodyMid')
         for c in ('x', 'y', 'likelihood')], names=['individual', 'bodypart', 'coord'])
    pose = pd.DataFrame(np.random.default_rng(0).random((50, 12)), index=t, columns=columns)
    pose.attrs = {'fps': 30, 'arena': {'units': 'cm'}}
    settings = load_settings(jobs.provenance.PACKAGE_ROOT / 'configs' / 'imaging_1p.yaml')
    session = Session('8472', '6', 'HBTD3')
    f = nwb.new_nwbfile(settings, session, 'behaviour')
    nwb.build_beh_nwb(f, {'S': (np.full(50, 10.0), 'cm/s', t, 'speed')}, settings=settings,
                      pose=pose, scorer='DLC_HrnetW32', video='bottom.mp4')
    path = nwb.write(f, session.beh(tmp_path))
    with pynwb.NWBHDF5IO(str(path), 'r') as io:
        behavior = io.read().processing['behavior']
        helper = behavior['PoseEstimation_helper']
        nose = helper.pose_estimation_series['Nose']
        np.testing.assert_allclose(nose.data[:], pose[('helper', 'Nose')][['x', 'y']])
        np.testing.assert_allclose(nose.confidence[:], pose[('helper', 'Nose', 'likelihood')])
        assert nose.unit == 'cm' and helper.scorer == 'DLC_HrnetW32'
        assert set(behavior['Skeletons'].skeletons) == {'helper_skeleton', 'victim_skeleton'}
        assert 'PoseEstimation_victim' in behavior.data_interfaces
        assert helper.source_video.external_file[:].tolist() == ['bottom.mp4']


def test_memory_check_refuses_a_movie_too_big(tmp_path):
    movie = tmp_path / 'm.tif'
    movie.write_bytes(b'\0' * 1000)
    assert cnmfe.memory_needed(movie) == 8000
    with pytest.raises(cnmfe.CnmfeError, match='cluster'):
        cnmfe.check_memory(movie, available=7000)
    assert cnmfe.check_memory(movie, available=9000) == 8000


def test_from_caiman_hdf5_returns_every_component(tmp_path):
    import h5py

    result = cnmfe_result()
    path = tmp_path / 'model.hdf5'
    with h5py.File(path, 'w') as f:
        f['dims'] = np.array([6, 8])
        for name, k in (('estimates', [0, 2]), ('estimates/discarded_components', [1])):
            g = f.require_group(name)
            A = result['A'][:, k].tocsc()
            for key, value in (('data', A.data), ('indices', A.indices), ('indptr', A.indptr),
                               ('shape', np.array(A.shape))):
                g[f'A/{key}'] = value
            for key in ('C', 'YrA', 'S'):
                g[key] = result[key][k]
            g['SNR_comp'], g['r_values'] = result['snr'][k], result['r_value'][k]
            if name == 'estimates':
                g['F_dff'] = result['detrended'][k]
                g['idx_components'] = h5py.Empty('f')   # what select_components leaves
    out = cnmfe.from_caiman_hdf5(path)
    assert list(out['accepted']) == [True, True, False]          # model's own, then discarded
    np.testing.assert_allclose(out['C'], result['C'][[0, 2, 1]])
    assert np.isnan(out['detrended'][2]).all() and not np.isnan(out['detrended'][:2]).any()
    plane = cnmfe.to_plane(out, np.arange(20) / 30, 30.0)
    assert len(plane.masks) == 3 and plane.images == {} and plane.shifts is None


def test_helping_behaviour_port_helpers():
    from raphael.imaging import helping_behaviour as hb

    known = {'8000', '848', '8014'}
    assert hb._mouse_id('B6J-8000', 'x', 'y', known=known) == '8000'
    assert hb._mouse_id(None, 'Session-20211025-143540_0848', 'v.isxd', known=known) == '848'
    assert hb._mouse_id(None, 'Moises-dHPC-HBT-240112-B6J-8014-f-HABD1', '', known=known) == '8014'
    assert hb.session_type('RECNvlCtx') == 'RECNvlCtx' and hb.session_type('HBT D1') == 'HBTD1'
    trials = hb.trials_table({'timevictiminsertion(s)': 68.0, 'timeonsetdooropening (s)': '75',
                              'timevictimheadcrossing(s)': None}, stop=212.0)
    assert trials.loc[0, 'victim_insertion_time'] == 68.0
    assert trials.loc[0, 'door_opening_time'] == 75.0
    assert np.isnan(trials.loc[0, 'victim_head_crossing_time']) and trials.loc[0, 'stop_time'] == 212
    assert hb.local_path('/a', hb.SERVER_PREFIX + 'dlc/x.csv').as_posix() == '/a/dlc/x.csv'


def test_cnmfe_batch_plan_files_and_jobs(tmp_path, movie):
    """The batch plan: rates, memory and batches from the movies themselves;
    the files to copy; one job per session with its params."""
    import yaml

    from raphael.imaging import cnmfe_batch

    raw = tmp_path / 'raw'
    for cohort, name, period in (('C1', 'a', (16670, 1000000)), ('C1', 'b', (25000, 1000000)),
                                 ('C2', 'c', (16670, 1000000))):
        folder = raw / cohort / name
        folder.mkdir(parents=True)
        write_isxd(folder / f'{name}_video.isxd', movie, period=period)
        (folder / f'{name}_video.gpio').write_bytes(b'g')
        (folder / 'session.json').write_text('{}')
    sessions = pd.DataFrame({
        'cohort': ['C1', 'C1', 'C2', 'C3'], 'mouse': ['1', '1', '2', '3'], 'day': [1, 2, 1, 1],
        'type': ['HAB1', 'HBTD1', 'HAB1', 'HAB1'],
        'raw_recording': ['a/a_video.isxd', 'b/b_video.isxd', 'c/c_video.isxd', ''],
        'sex': ['M', 'M', 'F', 'F']})
    subjects = pd.DataFrame({'mouse': ['1', '2'], 'date_of_birth': ['2024-01-23', None],
                             'strain': ['C57BL/6J', 'C57BL/6J']})
    settings = load_settings('src/raphael/configs/imaging_1p.yaml')
    plan, skipped = cnmfe_batch.plan(sessions, raw, settings, subjects, batch_size=2)

    assert skipped == [('3', 1, 'HAB1', 'no raw movie')]
    assert list(plan['job_id']) == ['cnmfe-b01-M1-D1-HAB1', 'cnmfe-b01-M1-D2-HBTD1',
                                    'cnmfe-b02-M2-D1-HAB1']
    a, b, c = plan.to_dict('records')
    assert a['temporal_downsample'] == 2 and a['binned_frames'] == 5 and a['cnmfe'] == ''
    assert b['temporal_downsample'] == 1 and json.loads(b['cnmfe']) == {'temporal_downsample': 1}
    assert '40.00 Hz' in b['note']                       # 40 Hz can't reach 30: kept as is
    assert a['date_of_birth'] == '2024-01-23' and c['date_of_birth'] == '' and c['sex'] == 'F'
    assert a['mem_gb'] == 16 and a['hours'] == 2         # tiny movie: the 8 GB floor, rounded
    by_gb, _ = cnmfe_batch.plan(sessions, raw, settings, batch_size=12, batch_gb=0)
    assert list(by_gb['batch']) == [1, 2, 3]             # one movie over the cap: its own batch

    path = tmp_path / 'plan.csv'
    plan.to_csv(path, index=False)
    plan = cnmfe_batch.read_plan(path)
    assert cnmfe_batch.batch_files(plan, 2) == ['C2/c/c_video.isxd']
    assert cnmfe_batch.batch_files(plan, 2, raw) == [
        'C2/c/c_video.isxd', 'C2/c/c_video.gpio', 'C2/c/session.json']

    site = yaml.safe_load(jobs.SITE_CONFIG.read_text())
    site['paths'].update(repo='/cluster/repo', data='/cluster/data', jobs='/cluster/jobs')
    (tmp_path / 'cluster.yaml').write_text(yaml.safe_dump(site))
    config = tmp_path / 'imaging_1p.yaml'
    config.write_text('nwb: {}\n')
    made, existing = cnmfe_batch.make_jobs(plan, 1, raw, tmp_path / 'jobs', config,
                                           hash_inputs=False, site=tmp_path / 'cluster.yaml')
    assert [j.name for j in made] == ['cnmfe-b01-M1-D1-HAB1', 'cnmfe-b01-M1-D2-HBTD1']
    spec = jobs.load(made[1])
    assert spec['params'] == {'mouse': '1', 'day': 2, 'session_type': 'HBTD1',
                              'subject': {'sex': 'M', 'date_of_birth': '2024-01-23',
                                          'strain': 'C57BL/6J'},
                              'cnmfe': {'temporal_downsample': 1}}
    assert spec['resources']['gpu'] is False and spec['resources']['mem_gb'] == 16
    assert (made[0] / 'config' / 'imaging_1p.yaml').exists()
    again, existing = cnmfe_batch.make_jobs(plan, 1, raw, tmp_path / 'jobs', config,
                                            hash_inputs=False, site=tmp_path / 'cluster.yaml')
    assert again == [] and len(existing) == 2              # made once only
