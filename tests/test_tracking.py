"""Behavioural tracking without a GPU: arena mapping, video splits, reading
DeepLabCut output, pose -> Raphael behaviour, the processed-tree pose file,
the day-held-out label split, and VAME's settings check and export.
(Multi-peak, sweep and identity tests are in test_tracking_*.py.)
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from raphael.behavioural_tracking import arena as A
from raphael.behavioural_tracking import io as IO
from raphael.behavioural_tracking import vame, vame_runner, zoo
from raphael.behavioural_tracking import videos as VID
from raphael.behavioural_tracking.dlc import split_indices
from raphael.ephys.layout import Session

ARENA = (100.0, 50.0)
LONG, SHORT = max(ARENA), min(ARENA)


# -------------------------------------------------------------------- arena

def _corners(door):
    """A chamber 600x400px at (100, 50) in the image, door on one wall."""
    medians = {'CornerTL': (100, 50), 'CornerTR': (700, 50),
               'CornerBR': (700, 450), 'CornerBL': (100, 450)}
    medians['Door'] = {'left': (100, 250), 'right': (700, 250)}[door]
    return medians


@pytest.mark.parametrize('door', ['left', 'right'])
def test_fit_arena_puts_door_at_x0(door):
    matrix, info = A.fit_arena(_corners(door), ARENA)
    assert info['door_wall'] == door
    door_x, door_y = A.apply(matrix, *_corners(door)['Door'])
    assert door_x == pytest.approx(0, abs=1e-6)
    assert door_y == pytest.approx(SHORT / 2, abs=1e-6)
    far = (700, 250) if door == 'left' else (100, 250)
    assert A.apply(matrix, *far)[0] == pytest.approx(LONG, abs=1e-6)


def test_fit_arena_handles_rotated_chamber_and_no_door():
    medians = {'CornerTL': (0, 0), 'CornerTR': (400, 0), 'CornerBR': (400, 600),
               'CornerBL': (0, 600), 'Door': (200, 0)}
    matrix, info = A.fit_arena(medians, ARENA)
    assert info['door_wall'] == 'top'
    assert A.apply(matrix, *medians['Door']) == pytest.approx((0, SHORT / 2), abs=1e-6)
    matrix, info = A.fit_arena(medians, ARENA, door=None)
    assert info['door_wall'] == 'unknown'


def test_fit_arena_refuses_missing_corner_and_ignores_low_likelihood():
    medians = _corners('left')
    del medians['CornerBR']
    with pytest.raises(ValueError):
        A.fit_arena(medians, ARENA)
    x = np.array([10.0, 10.0, 999.0])
    out = A.landmark_medians({'CornerTL': (x, x, np.array([0.95, 0.99, 0.1]))}, pcutoff=0.9)
    assert out['CornerTL'] == (10.0, 10.0)


# ------------------------------------------------------------------- videos

def test_parse_name_with_patterns():
    name = 'videos/Basler__bottom__20240315_7964-7963_f_cropped.mp4'
    assert VID.parse_name(name) == {'video_id': 'Basler__bottom__20240315_7964-7963_f_cropped',
                                    'rig': 'all', 'date': '20240315'}
    assert VID.parse_name(name, rig_pattern=r'__(\w+?)__')['rig'] == 'bottom'
    assert VID.parse_name(name, rig_pattern={'b': '__bottom__', 't': '__top__'})['rig'] == 'b'


def test_split_never_divides_a_day():
    inventory = pd.DataFrame({'rig': ['a'] * 6 + ['b'] * 4,
                              'date': ['1', '1', '2', '3', '4', '5', '6', '6', '7', '8']})
    split = VID.assign_split(inventory, test_fraction=0.3, seed=1)
    for _, group in inventory.assign(split=split).groupby('date'):
        assert group['split'].nunique() == 1
    assert (split == 'test').any() and (split == 'train').any()


def test_output_geometry_is_even_and_bounded():
    width, height = VID.output_geometry((0, 0, 1001, 333), max_side=800)
    assert width <= 800 and width % 2 == 0 and height % 2 == 0


def test_split_indices_by_video():
    frames = pd.DataFrame({'video_id': ['a', 'a', 'b', 'c'], 'image': list('wxyz')})
    train, test = split_indices(frames, {'a': 'train', 'b': 'test', 'c': 'train'})
    assert train == [0, 1, 3] and test == [2]
    with pytest.raises(ValueError):
        split_indices(frames, {'a': 'train'})


def test_choose_split_holds_out_whole_days_with_two_animal_frames():
    summary = pd.DataFrame({'video_id': [f'v{i}' for i in range(12)],
                            'rig': ['a'] * 6 + ['b'] * 6,
                            'date': [str(d) for d in (1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10)],
                            'frames': [20] * 12,
                            'two_animal_frames': [5, 0, 5, 0, 5, 0, 5, 5, 0, 5, 0, 5]})
    split = zoo.choose_split(summary, 0.25, seed=0, min_two_animal=10)
    days = summary.assign(split=split).groupby(['rig', 'date'])['split'].nunique()
    assert (days == 1).all()
    assert summary.loc[split == 'test', 'two_animal_frames'].sum() >= 10
    for rig in 'ab':
        assert set(split[summary['rig'] == rig]) == {'train', 'test'}


# ----------------------------------------------------- DLC output and pose

def _dlc_table(n=300, fps=30.0, individual=None, landmarks=True):
    """A DLC-style table: one animal walking a circle, plus corner landmarks."""
    t = np.arange(n) / fps
    cx, cy = 400 + 100 * np.cos(t), 250 + 100 * np.sin(t)
    heading = t + np.pi / 2                                  # walking round anticlockwise
    parts = {'Nose': 20, 'BodyMid': 0, 'TailBase': -20}
    cols, data = [], []
    level = [individual] if individual else []
    for part, d in parts.items():
        for c, v in (('x', cx + d * np.cos(heading)), ('y', cy + d * np.sin(heading)),
                     ('likelihood', np.full(n, 0.95))):
            cols.append(('scorer', *level, part, c))
            data.append(v)
    if landmarks:
        for name, (x, y) in {'CornerTL': (100, 50), 'CornerTR': (700, 50),
                             'CornerBR': (700, 450), 'CornerBL': (100, 450)}.items():
            for c, v in (('x', x), ('y', y), ('likelihood', 0.99)):
                cols.append(('scorer', *(['single'] if individual else []), name, c))
                data.append(np.full(n, v, float))
    names = (['scorer', 'individuals', 'bodyparts', 'coords'] if individual
             else ['scorer', 'bodyparts', 'coords'])
    return pd.DataFrame(np.column_stack(data), columns=pd.MultiIndex.from_tuples(cols, names=names))


@pytest.mark.parametrize('individual', [None, 'mouse'])
def test_read_dlc_csv_and_h5_single_and_multi_animal(tmp_path, individual):
    table = _dlc_table(individual=individual)
    table.to_csv(tmp_path / 'p.csv')
    table.to_hdf(tmp_path / 'p.h5', key='df')
    for path in (tmp_path / 'p.csv', tmp_path / 'p.h5'):
        raw = IO.read_dlc_h5(path)
        assert raw.columns.names == ['individual', 'bodypart', 'coord']
        assert (individual or 'animal') in raw.columns.get_level_values('individual')
        assert len(raw) == 300


def test_dlc_to_arena_maps_to_cm_and_drops_landmarks():
    table = _dlc_table()
    table.columns = pd.MultiIndex.from_tuples([('animal', bp, c) for _, bp, c in table.columns],
                                              names=['individual', 'bodypart', 'coord'])
    pose = IO.dlc_to_arena(table, fps=30, arena_cm=(120, 80), door=None)
    assert 'CornerTL' not in pose.columns.get_level_values('bodypart')
    assert pose.attrs['arena']['units'] == 'cm' and pose.index[1] == pytest.approx(1 / 30)
    x = pose[('animal', 'BodyMid', 'x')]
    # the 600 px wide floor is 120 cm: the 200 px circle spans 40 cm
    assert x.max() - x.min() == pytest.approx(40, abs=0.5)
    pixels = IO.dlc_to_arena(table, fps=30)
    assert pixels.attrs['arena']['units'] == 'px'


def test_to_behaviour_gives_position_heading_and_speed():
    table = _dlc_table(n=600)
    table.columns = pd.MultiIndex.from_tuples([('animal', bp, c) for _, bp, c in table.columns],
                                              names=['individual', 'bodypart', 'coord'])
    pose = IO.dlc_to_arena(table, fps=30, door=None)          # pixels
    beh = IO.to_behaviour(pose)
    assert set(beh) == {'P_x', 'P_y', 'H', 'S', 'moving'}
    t = beh['H'].t
    expected = (t + np.pi / 2) % (2 * np.pi)
    err = np.angle(np.exp(1j * (beh['H'].values - expected)))
    assert np.abs(err).max() < 0.05
    assert np.median(beh['S'].values) == pytest.approx(100, rel=0.05)   # 100 px/s
    assert beh['moving'].tot_length() > 0.9 * t[-1]


def test_pose_round_trips_through_the_processed_tree(tmp_path):
    table = _dlc_table()
    table.columns = pd.MultiIndex.from_tuples([('animal', bp, c) for _, bp, c in table.columns],
                                              names=['individual', 'bodypart', 'coord'])
    pose = IO.dlc_to_arena(table, fps=30, arena_cm=(120, 80), door=None)
    session = Session('1', '2', 'OF1')
    path = IO.write_pose(pose, session, tmp_path)
    assert path.name == 'sub-M1_day-02_ses-OF1_src-dlc_pose.h5'
    back = IO.read_pose(path)
    pd.testing.assert_frame_equal(back, pose, check_freq=False)
    assert back.attrs['fps'] == 30 and back.attrs['arena']['units'] == 'cm'


# --------------------------------------------------------------------- VAME

def test_vame_refuses_settings_that_would_save_nothing():
    with pytest.raises(vame.VameError, match='kl_start'):
        vame.check(vame.settings_with_defaults({'fps': 30, 'config': {'max_epochs': 5}}))
    with pytest.raises(vame.VameError, match='fps'):
        vame.check(vame.settings_with_defaults({}))
    vame.check(vame.settings_with_defaults({'fps': 30, 'config': {'max_epochs': 50}}))
    merged = vame.settings_with_defaults({'config': {'n_clusters': 7}})
    assert merged['config']['n_clusters'] == 7 and merged['config']['pose_confidence'] == 0.6


def test_vame_environment_is_locked_and_shipped():
    assert (vame.ENV_PROJECT / 'pyproject.toml').exists()
    assert (vame.ENV_PROJECT / 'uv.lock').exists() and len(vame.lock_sha256()) == 64
    assert 'vame-py==0.13.0' in (vame.ENV_PROJECT / 'pyproject.toml').read_text()


def test_vame_export_offsets_frames_by_half_the_time_window(tmp_path):
    project = tmp_path / 'proj'
    folder = project / 'results' / 's1' / 'VAME' / 'kmeans-3'
    (folder / 'community').mkdir(parents=True)
    np.save(folder / '3_kmeans_label_s1.npy', np.array([0, 1, 2, 2]))
    np.save(folder / 'community' / 'cohort_community_label_s1.npy', np.array([0, 0, 1, 1]))
    config = {'project_path': str(project), 'model_name': 'VAME', 'n_clusters': 3, 'fps': 30.0,
              'time_window': 15, 'session_names': ['s1'], 'segmentation_algorithms': ['kmeans']}
    written = vame_runner.export(config, tmp_path / 'motifs')
    table = pd.read_csv(written[0]['path'])
    assert table['frame'].tolist() == [7, 8, 9, 10]
    assert table['time'].iloc[0] == pytest.approx(7 / 30, abs=1e-6)
    assert table['community'].tolist() == [0, 0, 1, 1]
    session = Session('1', '2', 'OF1')
    out = vame.write_motifs(written[0]['path'], session, tmp_path / 'processed', 'kmeans-3')
    assert out.name == 'sub-M1_day-02_ses-OF1_src-vame-kmeans-3_motifs.csv'


@pytest.mark.skipif(not os.environ.get('RAPHAEL_TEST_VAME'),
                    reason='runs real VAME in its own environment (minutes); set RAPHAEL_TEST_VAME=1')
def test_vame_runs_end_to_end(tmp_path):
    from raphael import toy

    parts = {'Nose': (4, 0), 'BodyMid': (0, 0), 'TailBase': (-4, 0), 'FrontPawL': (2, 1.5),
             'FrontPawR': (2, -1.5)}
    rng = np.random.default_rng(0)
    for day in (1, 2):                       # a varied walk, so there are motifs to find
        b = toy.session(duration=70, seed=day)['behaviour']
        t = b['P_x'].t[::3]
        x, y, h = (np.interp(t, b[k].t, b[k].values) for k in ('P_x', 'P_y', 'H'))
        cols, data = [], []
        for part, (dx, dy) in parts.items():
            px = 5 * (x + dx * np.cos(h) - dy * np.sin(h)) + rng.normal(0, 0.5, len(t))
            py = 5 * (y + dx * np.sin(h) + dy * np.cos(h)) + rng.normal(0, 0.5, len(t))
            for c, v in (('x', px), ('y', py), ('likelihood', np.full(len(t), 0.95))):
                cols.append(('DLC', part, c))
                data.append(v)
        pd.DataFrame(np.column_stack(data), columns=pd.MultiIndex.from_tuples(
            cols, names=['scorer', 'bodyparts', 'coords'])).to_csv(tmp_path / f'session{day}DLC.csv')
    summary = vame.run(sorted(tmp_path.glob('*.csv')), tmp_path / 'work',
                       {'fps': 33.3, 'config': {'max_epochs': 5, 'kl_start': 1, 'annealtime': 1,
                                                'n_clusters': 4, 'time_window': 15, 'zdims': 8,
                                                'segmentation_algorithms': ['kmeans']}},
                       log=tmp_path / 'vame.log')
    assert len(summary['motif_files']) == 2 and len(summary['vame_env_lock_sha256']) == 64
    table = pd.read_csv(summary['motif_files'][0]['path'])
    assert table['frame'].iloc[0] == 7 and table['motif'].nunique() > 1
    assert json.loads((tmp_path / 'work' / 'vame_summary.json').read_text())['steps']
