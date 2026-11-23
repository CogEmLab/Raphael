"""Pose in one canonical table, whatever produced it, and from there into the
behaviour the rest of Raphael reads.

The canonical table is a DataFrame indexed by time in seconds, with columns
(individual, bodypart, coord), coord being x / y (arena cm, or pixels if no
arena was fitted) or likelihood. `table.attrs['fps']` carries the frame rate.

    raw = read_dlc_h5('video...DLC_..._el.h5')               # pixels
    pose = dlc_to_arena(raw, fps=30, arena_cm=(100, 100))    # cm, landmarks dropped
    behaviour = to_behaviour(pose)                           # P_x, P_y, H, S, moving
    write_pose(pose, session, processed_root)                # sub-..._src-dlc_pose.h5
"""

from pathlib import Path

import numpy as np
import pandas as pd

from raphael.behavioural_tracking.arena import (
    CORNERS,
    DOOR,
    apply,
    fit_arena,
    landmark_medians,
)

COORDS = ('x', 'y', 'likelihood')


def read_dlc_h5(path, single_animal='animal'):
    """Any DLC output (h5 or csv) as (individual, bodypart, coord) columns.

    Single-animal outputs have no individuals level: their bodyparts are put
    under `single_animal`. Multi-animal unique bodyparts (landmarks) are
    under the individual `single`, as DLC writes them."""
    path = Path(path)
    if path.suffix == '.csv':
        # DLC labels its header rows in the first column: scorer,
        # [individuals,] bodyparts, coords.
        with open(path) as handle:
            first = [handle.readline().split(',')[0].strip() for _ in range(4)]
        rows = 4 if 'individuals' in first else 3
        table = pd.read_csv(path, header=list(range(rows)), index_col=0)
    else:
        table = pd.read_hdf(path)
    columns = table.columns
    if columns.nlevels == 4:
        table.columns = columns.droplevel(0)
    elif columns.nlevels == 3:
        table.columns = pd.MultiIndex.from_tuples(
            [(single_animal, bp, c) for _, bp, c in columns])
    else:
        raise ValueError(f'unexpected DLC column levels in {path}')
    table.columns.names = ['individual', 'bodypart', 'coord']
    return table


def dlc_to_arena(raw, fps, arena_cm=None, pcutoff=0.9, corners=CORNERS, door=DOOR):
    """A raw DLC table (pixels) as the canonical table, timed in seconds.

    With `arena_cm`, the corner landmarks fit a per-video homography into
    arena cm (`arena.fit_arena`), and the landmarks are then dropped;
    `attrs['arena']` records the fit, so a bad session can be traced.
    Without it, coordinates stay in pixels."""
    landmark_names = set(corners) | ({door} if door else set())
    animals = raw.drop(columns=[c for c in raw.columns if c[1] in landmark_names]).copy()
    info = {'units': 'px'}
    if arena_cm is not None:
        landmarks = {}
        for (individual, bodypart), _ in raw.T.groupby(level=[0, 1]):
            if bodypart in landmark_names:
                block = raw[(individual, bodypart)]
                landmarks[bodypart] = (block['x'].to_numpy(), block['y'].to_numpy(),
                                       block['likelihood'].to_numpy())
        matrix, info = fit_arena(landmark_medians(landmarks, pcutoff), arena_cm, corners, door)
        info['units'] = 'cm'
        for individual, bodypart in {c[:2] for c in animals.columns}:
            x, y = apply(matrix, animals[(individual, bodypart, 'x')].to_numpy(),
                         animals[(individual, bodypart, 'y')].to_numpy())
            animals[(individual, bodypart, 'x')] = x
            animals[(individual, bodypart, 'y')] = y
    animals.index = pd.Index(np.arange(len(animals)) / fps, name='time')
    animals = animals.sort_index(axis=1)
    animals.attrs['fps'] = fps
    animals.attrs['arena'] = info
    return animals


def load_dlc(path, fps, arena_cm=None, pcutoff=0.9, **kwargs):
    return dlc_to_arena(read_dlc_h5(path), fps, arena_cm, pcutoff, **kwargs)


def to_behaviour(pose, individual=None, centre='BodyMid', front='Nose', back='TailBase',
                 pcutoff=0.6, moving_cm_s=2.0, smooth_s=0.2):
    """One animal's pose as Raphael's behaviour: pynapple Tsd `P_x`, `P_y`
    (its `centre` bodypart), `H` (heading, rad, from `back` to `front`), `S`
    (speed, units/s) and the `moving` epoch, as the tuning scores, plots and
    ml modules expect. Points below `pcutoff` are dropped and bridged by
    linear interpolation; position is smoothed over `smooth_s` before speed
    is taken."""
    import pynapple as nap

    individuals = list(dict.fromkeys(pose.columns.get_level_values('individual')))
    individual = individual or next(i for i in individuals if i != 'single')
    block = pose[individual]
    t = pose.index.to_numpy(float)

    def part(name):
        x = block[(name, 'x')].to_numpy(float).copy()
        y = block[(name, 'y')].to_numpy(float).copy()
        if (name, 'likelihood') in block:
            low = block[(name, 'likelihood')].to_numpy(float) < pcutoff
            x[low] = y[low] = np.nan
        return pd.Series(x).interpolate(limit_direction='both').to_numpy(), \
            pd.Series(y).interpolate(limit_direction='both').to_numpy()

    x, y = part(centre)
    fx, fy = part(front)
    bx, by = part(back)
    heading = np.arctan2(fy - by, fx - bx) % (2 * np.pi)
    fps = pose.attrs.get('fps') or 1 / np.median(np.diff(t))
    width = max(round(smooth_s * fps), 1)
    kernel = np.ones(width) / width
    sx = np.convolve(np.pad(x, (width // 2, width - 1 - width // 2), mode='edge'), kernel, 'valid')
    sy = np.convolve(np.pad(y, (width // 2, width - 1 - width // 2), mode='edge'), kernel, 'valid')
    speed = np.hypot(np.gradient(sx, t), np.gradient(sy, t))
    s = nap.Tsd(t=t, d=speed)
    return {'P_x': nap.Tsd(t=t, d=x), 'P_y': nap.Tsd(t=t, d=y), 'H': nap.Tsd(t=t, d=heading),
            'S': s, 'moving': s.threshold(moving_cm_s).time_support}


def pose_path(session, root, source='dlc'):
    """Where a session's pose lives in the processed tree."""
    return session.folder(root) / f'{session.stem}_src-{source}_pose.h5'


def write_pose(pose, session, root, source='dlc'):
    """Save the canonical table in the processed tree; returns the path."""
    path = pose_path(session, root, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pose.copy()
    out.attrs = {}
    out.to_hdf(path, key='pose', mode='w')
    pd.Series({k: str(v) for k, v in pose.attrs.items()}).to_hdf(path, key='attrs')
    return path


def read_pose(path):
    """A pose table written by `write_pose`, with its attrs."""
    import ast

    pose = pd.read_hdf(path, key='pose')
    attrs = pd.read_hdf(path, key='attrs')
    for key, value in attrs.items():
        try:
            pose.attrs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pose.attrs[key] = value
    return pose
