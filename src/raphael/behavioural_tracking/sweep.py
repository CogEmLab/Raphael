"""Flexible likelihood thresholding: re-track a capture at many cutoffs.

DLC applies one `pcutoff` inside tracking and keeps only the result. Here the
model runs once (`multipeak.capture_video`, permissive cutoffs, everything
saved), and tracking is redone from that capture at any grid of keypoint
likelihood and detector-box thresholds, with DLC's own tracker
(`build_tracklets`) and stitcher (`TrackletStitcher`). No video is needed.

Each threshold pair gets metrics aimed at the two-mouse confusion problem:
how much pose survives, how often both animals are present, how fragmented
the tracks are, and how often identities look swapped.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from raphael.behavioural_tracking.multipeak import PAD, load_capture


def from_full_pickle(path, bodyparts, individuals):
    """A DLC `_full.pickle` as a capture-like dict (no peaks).

    For models `capture_video` cannot run (conditional top-down), analyse
    with deeplabcut.analyze_videos and sweep its full pickle instead.
    """
    import pickle

    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    meta = data.pop('metadata')
    keys = sorted(data, key=lambda k: int(k.replace('frame', '')))
    n_bp = len(bodyparts)
    n_det = max(len(data[k]['confidence'][0]) for k in keys) if keys else 0
    poses = np.full((len(keys), n_det, n_bp, 3), np.nan, dtype=np.float32)
    poses[..., 2] = PAD
    bbox_scores = None
    for i, key in enumerate(keys):
        frame = data[key]
        coords = np.stack(frame['coordinates'][0][:n_bp])   # (K, n, 2)
        scores = np.stack(frame['confidence'][:n_bp])[..., 0]  # (K, n)
        poses[i, : coords.shape[1], :, :2] = coords.transpose(1, 0, 2)
        poses[i, : coords.shape[1], :, 2] = scores.T
        if 'bbox_scores' in frame:
            if bbox_scores is None:
                bbox_scores = np.full((len(keys), n_det), PAD, dtype=np.float32)
            bbox_scores[i, : len(frame['bbox_scores'])] = frame['bbox_scores'][:n_det]
    raw = {
        'bodyparts': poses,
        'meta': {
            'bodyparts': list(bodyparts),
            'individuals': list(individuals),
            'n_frames': meta['nframes'],
            'source': str(path),
        },
    }
    if bbox_scores is not None:
        raw['bbox_scores'] = bbox_scores
    return raw


# --------------------------------------------------------------- filtering


def detections(raw, bbox_threshold=0.0):
    """Per-frame candidate animals after the detector-box threshold.

    Returns (poses, keep): poses (F, M, K, 3) and a boolean (F, M) mask of
    detections kept. Bottom-up captures have no boxes; every assembly is
    kept and only the keypoint cutoff applies.
    """
    poses = raw['bodyparts']
    present = ~np.all(poses[..., 2] < 0, axis=-1)
    if 'bbox_scores' in raw:
        present &= raw['bbox_scores'] >= bbox_threshold
    return poses, present


def thresholded_pose(raw, pcutoff, bbox_threshold=0.0):
    """Untracked poses with low-likelihood keypoints set to NaN."""
    poses, keep = detections(raw, bbox_threshold)
    out = poses.copy()
    low = (out[..., 2] < pcutoff) | ~keep[..., None]
    out[..., 0][low] = np.nan
    out[..., 1][low] = np.nan
    return out


def thresholded_peaks(raw, pcutoff, bbox_threshold=0.0):
    """All heatmap peaks at or above pcutoff (others NaN), per detection."""
    if 'peaks' not in raw:
        raise KeyError('this capture has no peaks (made from a DLC full pickle?)')
    peaks = raw['peaks'].copy()
    if 'bbox_scores' in raw:
        keep = raw['bbox_scores'][..., : peaks.shape[1]] >= bbox_threshold
        peaks[~keep] = np.nan
    low = ~(peaks[..., 2] >= pcutoff)
    peaks[..., 0][low] = np.nan
    peaks[..., 1][low] = np.nan
    return peaks


def assemblies(raw, bbox_threshold=0.0):
    """DLC assemblies dict {frame: (n, K, 4)} from a capture.

    Column 3 is the predicted identity per keypoint (argmax of the identity
    head) or -1, exactly as DLC's `_generate_assemblies_file` builds it.
    """
    poses, keep = detections(raw, bbox_threshold)
    ids = None
    if 'identity_scores' in raw:
        ids = np.argmax(raw['identity_scores'], axis=-1).astype(np.float32)
    out = {}
    for frame in range(len(poses)):
        rows = np.flatnonzero(keep[frame])
        if len(rows) == 0:
            continue
        assembly = np.concatenate(
            [
                poses[frame, rows],
                (ids[frame, rows, :, None] if ids is not None
                 else -np.ones((len(rows), poses.shape[2], 1), np.float32)),
            ],
            axis=-1,
        )
        # DLC's tracker expects -1 for missing keypoints, not NaN.
        assembly[np.isnan(assembly)] = -1
        out[frame] = assembly
    return out


# ---------------------------------------------------------------- tracking


def track(raw, pcutoff, bbox_threshold=0.0, params=None):
    """DLC tracking + stitching at one threshold pair -> tracked DataFrame.

    Columns are (scorer, individuals, bodyparts, coords) as in DLC's `_el.h5`;
    rows are every frame of the capture.
    """
    from deeplabcut.pose_estimation_pytorch.apis.tracklets import (
        build_tracklets,
    )
    from deeplabcut.refine_training_dataset.stitch import TrackletStitcher

    params = params or {}
    meta = raw['meta']
    bodyparts, individuals = meta['bodyparts'], meta['individuals']
    n_frames = int(meta['n_frames'])
    inference_cfg = {
        'pcutoff': pcutoff,
        'topktoretain': len(individuals),
        'boundingboxslack': 0,
        'max_age': params.get('max_age', 1),
        'min_hits': params.get('min_hits', 1),
        'iou_threshold': params.get('iou_threshold', 0.6),
    }
    scorer = meta.get('model') or 'capture'
    tracklets = build_tracklets(
        assemblies_data=assemblies(raw, bbox_threshold),
        track_method=params.get('track_method', 'ellipse'),
        inference_cfg=inference_cfg,
        joints=bodyparts,
        scorer=scorer,
        num_frames=n_frames,
    )
    n_tracklets = len([k for k in tracklets if k not in ('header', 'single')])
    columns = pd.MultiIndex.from_product(
        [[scorer], individuals, bodyparts, ['x', 'y', 'likelihood']],
        names=['scorer', 'individuals', 'bodyparts', 'coords'],
    )
    empty = pd.DataFrame(np.nan, index=range(n_frames), columns=columns)
    if n_tracklets == 0:
        return empty, {'n_tracklets': 0, 'stitched': False}
    try:
        stitcher = TrackletStitcher.from_dict_of_dict(
            tracklets,
            n_tracks=len(individuals),
            # DLC's stitcher rejects tracklets shorter than 3 frames outright.
            min_length=max(3, int(params.get('min_tracklet_length', 10))),
        )
        weight = None
        if any(t.identity != -1 for t in stitcher):
            # As DLC does: strongly prefer joining tracklets of one identity.
            def weight(t1, t2, stitcher=stitcher):
                w = 0.01 if t1.identity == t2.identity else 1
                return w * stitcher.calculate_edge_weight(t1, t2)
        stitcher.build_graph(max_gap=params.get('max_gap'), weight_func=weight)
        stitcher.stitch()
        df = stitcher.format_df(animal_names=individuals)
    except (ValueError, IndexError, KeyError, OSError) as error:
        # Too few or too short tracklets for the stitcher at strict cutoffs
        # (DeepLabCut 3.0.2 raises OSError 'Tracklets are empty').
        return empty, {'n_tracklets': n_tracklets, 'stitched': False, 'error': repr(error)}
    return df.reindex(range(n_frames)), {'n_tracklets': n_tracklets, 'stitched': True}


# ----------------------------------------------------------------- metrics


def centroids(df, individuals):
    """(F, n_individuals, 2): mean of the visible keypoints per animal."""
    out = []
    for ind in individuals:
        block = df.xs(ind, axis=1, level='individuals')
        x = block.xs('x', axis=1, level='coords').to_numpy(float)
        y = block.xs('y', axis=1, level='coords').to_numpy(float)
        with np.errstate(all='ignore'):
            out.append(np.stack([np.nanmean(x, axis=1), np.nanmean(y, axis=1)], -1))
    return np.stack(out, axis=1)


def swap_events(cent, margin_px=5.0):
    """Frames where the two identities fit the previous frame better swapped.

    For two animals A, B: a swap is flagged when |A_t - B_t-1| + |B_t - A_t-1|
    is smaller than |A_t - A_t-1| + |B_t - B_t-1| by more than margin_px.
    A cheap, label-free proxy for identity switches.
    """
    if cent.shape[1] != 2:
        return np.zeros(len(cent), dtype=bool)
    a, b = cent[:, 0], cent[:, 1]
    stay = np.linalg.norm(a[1:] - a[:-1], axis=-1) + np.linalg.norm(b[1:] - b[:-1], axis=-1)
    swap = np.linalg.norm(a[1:] - b[:-1], axis=-1) + np.linalg.norm(b[1:] - a[:-1], axis=-1)
    flagged = np.zeros(len(cent), dtype=bool)
    with np.errstate(invalid='ignore'):
        flagged[1:] = swap + margin_px < stay
    return flagged


def track_metrics(df, individuals, jump_px=40.0):
    """Summary of one tracked result, per animal and overall."""
    cent = centroids(df, individuals)
    present = np.isfinite(cent).all(-1)
    out = {
        'frames': len(df),
        'frames_both_present': int(present.all(1).sum()),
        'frames_none_present': int((~present).all(1).sum()),
        'swap_events': int(swap_events(cent).sum()),
    }
    for i, ind in enumerate(individuals):
        block = df.xs(ind, axis=1, level='individuals')
        lik = block.xs('likelihood', axis=1, level='coords').to_numpy(float)
        out[f'{ind}_present_frac'] = float(present[:, i].mean())
        out[f'{ind}_keypoint_frac'] = float(np.isfinite(lik).mean())
        step = np.linalg.norm(np.diff(cent[:, i], axis=0), axis=-1)
        with np.errstate(invalid='ignore'):
            out[f'{ind}_jumps'] = int((step > jump_px).sum())
    out['jumps_per_1000_frames'] = 1000 * sum(
        out[f'{ind}_jumps'] for ind in individuals
    ) / max(1, len(df))
    out['swaps_per_1000_frames'] = 1000 * out['swap_events'] / max(1, len(df))
    return out


def untracked_metrics(raw, pcutoff, bbox_threshold):
    """What survives the thresholds before any tracking."""
    pose = thresholded_pose(raw, pcutoff, bbox_threshold)
    kept = np.isfinite(pose[..., 0])
    animals = kept.any(-1).sum(-1)  # per frame
    return {
        'keypoints_kept_per_frame': float(kept.sum((1, 2)).mean()),
        'frames_0_animals': int((animals == 0).sum()),
        'frames_1_animal': int((animals == 1).sum()),
        'frames_2_animals': int((animals == 2).sum()),
        'frames_3plus_animals': int((animals >= 3).sum()),
    }


def sweep(raw, pcutoffs, bbox_thresholds=(0.0,), params=None, save_dir=None, save=()):
    """Track at every (pcutoff, bbox_threshold); return a metrics table.

    `save` lists (pcutoff, bbox_threshold) pairs whose tracked poses are
    written to `save_dir` as h5; everything else is reproducible on demand
    with `track`.
    """
    params = params or {}
    individuals = raw['meta']['individuals']
    has_boxes = 'bbox_scores' in raw
    rows = []
    for pcutoff, bbox in itertools.product(pcutoffs, bbox_thresholds if has_boxes else (0.0,)):
        df, info = track(raw, pcutoff, bbox, params)
        row = {
            'pcutoff': pcutoff,
            'bbox_threshold': bbox if has_boxes else np.nan,
            **info,
            **untracked_metrics(raw, pcutoff, bbox),
            **track_metrics(df, individuals, params.get('jump_px', 40.0)),
        }
        rows.append(row)
        if save_dir and (pcutoff, bbox) in {tuple(s) for s in save}:
            out = Path(save_dir) / f'tracked_p{pcutoff:.2f}_b{bbox:.2f}.h5'
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_hdf(out, key='tracks', mode='w')
    return pd.DataFrame(rows)


def sweep_capture(path, settings, save_dir=None, save=()):
    """Convenience wrapper: load a capture, sweep with the settings' `sweep` section."""
    spec = settings['sweep']
    raw = load_capture(path)
    table = sweep(
        raw,
        spec['pcutoffs'],
        spec['bbox_thresholds'],
        params=spec,
        save_dir=save_dir,
        save=save,
    )
    table.insert(0, 'video', raw['meta'].get('video', ''))
    table.insert(0, 'model', raw['meta'].get('model', ''))
    return table
