"""Multi-peak extraction and threshold sweep, on tiny synthetic inputs.

Tests touching DeepLabCut classes are skipped without the tracking extra.
Moved from social-manifolds with the code.
"""

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from raphael.behavioural_tracking import multipeak as MP
from raphael.behavioural_tracking import sweep as S


def _two_blob_heatmap(h=32, w=32):
    """One bodypart, two Gaussian blobs: a strong one and a weaker one."""
    yy, xx = np.mgrid[0:h, 0:w]
    hm = 0.9 * np.exp(-((xx - 8) ** 2 + (yy - 10) ** 2) / 4.0)
    hm += 0.5 * np.exp(-((xx - 24) ** 2 + (yy - 20) ** 2) / 4.0)
    return torch.tensor(hm, dtype=torch.float32)[None, None]  # (1, 1, H, W)


def test_heatmap_peaks_finds_both_blobs_in_order():
    peaks = MP.heatmap_peaks(_two_blob_heatmap(), None, stride=4, k=3, nms_radius=2)
    (x0, y0, s0), (x1, y1, s1), (_, _, s2) = peaks[0, 0].tolist()
    assert (x0, y0) == (8 * 4 + 2, 10 * 4 + 2) and s0 == pytest.approx(0.9, abs=1e-3)
    assert (x1, y1) == (24 * 4 + 2, 20 * 4 + 2) and s1 == pytest.approx(0.5, abs=1e-3)
    assert s2 < s1  # a third peak, if any, is weaker


def test_heatmap_peaks_applies_locref_and_marks_missing():
    hm = torch.zeros(1, 1, 8, 8)
    hm[0, 0, 3, 5] = 0.8
    locref = torch.zeros(1, 1, 2, 8, 8)
    locref[0, 0, 0, 3, 5], locref[0, 0, 1, 3, 5] = 1.5, -2.0
    peaks = MP.heatmap_peaks(hm, locref, stride=8, k=2, min_score=0.1)
    assert peaks[0, 0, 0].tolist() == pytest.approx([5 * 8 + 4 + 1.5, 3 * 8 + 4 - 2.0, 0.8])
    assert torch.isnan(peaks[0, 0, 1, 0]) and peaks[0, 0, 1, 2] == MP.PAD


def test_peak0_equals_dlc_heatmap_predictor():
    pytest.importorskip('deeplabcut')
    from deeplabcut.pose_estimation_pytorch.models.predictors.single_predictor import (
        HeatmapPredictor,
    )

    torch.manual_seed(0)
    outputs = {'heatmap': torch.rand(2, 3, 16, 16), 'locref': torch.randn(2, 6, 16, 16)}
    inner = HeatmapPredictor(apply_sigmoid=False, clip_scores=True)
    wrapped = MP.MultiPeakPredictor(inner, num_bodyparts=3, k=4, min_score=0.0)
    result = wrapped(4, outputs)
    poses = result['poses'][:, 0]                          # (B, K, 3)
    peaks = result['peaks'].reshape(2, 3, 4, 3)[:, :, 0]   # peak 0
    assert torch.allclose(poses, peaks, atol=1e-5)


def test_postprocessor_rescales_peaks_per_crop():
    pytest.importorskip('deeplabcut')
    peaks = np.zeros((1, 2 * 1, 3))
    peaks[0, :, :] = [[10, 20, 0.9], [30, 40, 0.5]]  # K=1, k=2 for one crop
    predictions = [{'bodypart': {'poses': np.array([[[10, 20, 0.9]]]), 'peaks': peaks}}]
    context = {'scales': [(2.0, 2.0)], 'offsets': [(100.0, 50.0)]}

    def base(preds, ctx):
        return {'bodyparts': np.array([[[0.0, 0.0, 0.9]]])}, ctx

    post = MP.PeaksPostprocessor(base, top_down=True, num_bodyparts=1, k=2, max_individuals=2)
    out, _ = post(predictions, context)
    assert out['peaks'].shape == (2, 1, 2, 3)
    assert out['peaks'][0, 0, 0].tolist() == [120.0, 90.0, 0.9]
    assert out['peaks'][0, 0, 1].tolist() == [160.0, 130.0, 0.5]
    assert np.isnan(out['peaks'][1]).all(axis=None) or (out['peaks'][1, ..., 2] == MP.PAD).all()


# ------------------------------------------------------------------- sweep


def _synthetic_capture(n=120):
    """Two animals walking in parallel lines, top-down layout.

    Four bodyparts in a body-like (non-collinear) shape: DLC's default
    ellipse tracker fits an ellipse per animal and needs at least three
    non-collinear points.
    """
    t = np.arange(n)
    shape = np.array([[10, 0], [0, 4], [0, -4], [-10, 0]], dtype=np.float32)
    poses = np.full((n, 3, 4, 3), np.nan, dtype=np.float32)
    poses[..., 2] = MP.PAD
    for slot, y in enumerate([50.0, 250.0]):
        poses[:, slot, :, 0] = 100 + 2 * t[:, None] + shape[:, 0]
        poses[:, slot, :, 1] = y + shape[:, 1]
        poses[:, slot, :, 2] = 0.9
    # A low-confidence spurious third detection in some frames.
    poses[::10, 2, :, :2] = 400 + shape
    poses[::10, 2, :, 2] = 0.2
    bbox_scores = np.full((n, 3), MP.PAD, dtype=np.float32)
    bbox_scores[:, :2] = 0.95
    bbox_scores[::10, 2] = 0.2
    return {
        'bodyparts': poses,
        'bbox_scores': bbox_scores,
        'meta': {'bodyparts': ['Nose', 'PawR', 'PawL', 'TailBase'],
                 'individuals': ['Helper', 'Victim'], 'n_frames': n, 'model': 'synthetic'},
    }


def test_thresholds_drop_low_confidence_detections():
    raw = _synthetic_capture()
    loose = S.untracked_metrics(raw, pcutoff=0.1, bbox_threshold=0.1)
    strict = S.untracked_metrics(raw, pcutoff=0.5, bbox_threshold=0.5)
    assert loose['frames_3plus_animals'] == 12 and strict['frames_3plus_animals'] == 0
    assert strict['frames_2_animals'] == 120


def test_swap_events_flags_an_identity_switch():
    cent = np.zeros((6, 2, 2))
    cent[:, 0] = [[0, 0], [1, 0], [2, 0], [3, 100], [4, 100], [5, 100]]
    cent[:, 1] = [[0, 100], [1, 100], [2, 100], [3, 0], [4, 0], [5, 0]]
    assert S.swap_events(cent).tolist() == [False, False, False, True, False, False]


def test_track_and_sweep_with_dlc_tracker():
    pytest.importorskip('deeplabcut')
    raw = _synthetic_capture()
    # The box tracker: on these perfectly regular synthetic animals,
    # DeepLabCut 3.0.2's ellipse tracker starts a new tracklet every frame.
    params = {'min_tracklet_length': 5, 'track_method': 'box'}
    df, info = S.track(raw, pcutoff=0.5, bbox_threshold=0.5, params=params)
    assert info['stitched'] and df.shape[0] == 120
    metrics = S.track_metrics(df, ['Helper', 'Victim'])
    assert metrics['frames_both_present'] >= 110
    assert metrics['swap_events'] == 0
    table = S.sweep(raw, [0.1, 0.5], [0.1, 0.5], params=params)
    ellipse, info = S.track(raw, 0.5, 0.5, {'min_tracklet_length': 5})   # reported, not raised
    assert len(ellipse) == 120 and 'n_tracklets' in info
    assert len(table) == 4 and {'swaps_per_1000_frames', 'frames_2_animals'} <= set(table.columns)


def test_clip_fails_loudly_and_writes_requested_frames(tmp_path):
    cv2 = pytest.importorskip('cv2')
    src = tmp_path / 'src.mp4'
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 48))
    for i in range(20):
        writer.write(np.full((48, 64, 3), i * 10, np.uint8))
    writer.release()
    with pytest.raises(OSError):
        MP._clip(src, tmp_path / 'missing' / 'clip.mp4', 5)
    clip = MP._clip(src, tmp_path / 'clip.mp4', 5)
    assert int(cv2.VideoCapture(str(clip)).get(cv2.CAP_PROP_FRAME_COUNT)) == 5


def test_peak0_matches_dlc_when_heatmaps_exceed_one():
    # Real models trained with MSE emit values above 1. DLC takes the argmax
    # of the RAW heatmap and clips only the reported score; clipping the
    # heatmap first would create plateaus at 1.0 and move the peak.
    pytest.importorskip('deeplabcut')
    from deeplabcut.pose_estimation_pytorch.models.predictors.single_predictor import (
        HeatmapPredictor,
    )

    torch.manual_seed(1)
    outputs = {'heatmap': 1.5 * torch.rand(4, 8, 32, 32), 'locref': torch.randn(4, 16, 32, 32)}
    inner = HeatmapPredictor(apply_sigmoid=False, clip_scores=True)
    result = MP.MultiPeakPredictor(inner, num_bodyparts=8, k=3, min_score=0.0)(4, outputs)
    poses = result['poses'][:, 0]
    peaks = result['peaks'].reshape(4, 8, 3, 3)
    assert torch.allclose(poses, peaks[:, :, 0], atol=1e-5)
    assert float(peaks[..., 2].max()) <= 1.0  # scores still clipped like DLC's


def test_peak0_kept_even_below_min_score():
    # DLC reports the argmax however weak; peak 0 must too, so it can be
    # judged by the threshold sweep rather than silently dropped.
    pytest.importorskip('deeplabcut')
    from deeplabcut.pose_estimation_pytorch.models.predictors.single_predictor import (
        HeatmapPredictor,
    )

    torch.manual_seed(2)
    outputs = {'heatmap': 0.005 * torch.rand(2, 4, 16, 16) - 0.002, 'locref': torch.randn(2, 8, 16, 16)}
    inner = HeatmapPredictor(apply_sigmoid=False, clip_scores=True)
    result = MP.MultiPeakPredictor(inner, num_bodyparts=4, k=3, min_score=0.01)(4, outputs)
    peaks = result['peaks'].reshape(2, 4, 3, 3)
    assert torch.allclose(result['poses'][:, 0], peaks[:, :, 0], atol=1e-5)
    assert torch.isnan(peaks[:, :, 1:, 0]).all()  # extra peaks below min_score dropped
