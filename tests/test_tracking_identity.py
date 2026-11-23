"""Skeleton model and multi-peak identity correction, on synthetic mice.
Moved from social-manifolds with the code."""

import numpy as np
import pytest

from raphael.behavioural_tracking import identity as I
from raphael.behavioural_tracking import skeleton as S

# A mouse pointing right, BodyMid-TailBase = 1 (the scale).
TEMPLATE = np.array([
    [1.6, 0.0],    # Nose
    [1.0, 0.35],   # FrontPawR
    [1.0, -0.35],  # FrontPawL
    [0.5, 0.0],    # BodyMid
    [-0.2, 0.4],   # HindPawR
    [-0.2, -0.4],  # HindPawL
    [-0.5, 0.0],   # TailBase
    [-1.6, 0.1],   # TailTip
])


def _mouse(rng, scale, centre, angle):
    pts = TEMPLATE * rng.normal(1, 0.06, (8, 1)) + rng.normal(0, 0.04, (8, 2))
    c, s = np.cos(angle), np.sin(angle)
    return pts @ np.array([[c, s], [-s, c]]) * scale + centre


def _labels(n=300, seed=0):
    rng = np.random.default_rng(seed)
    points = np.empty((n, 2, 8, 2))
    videos = np.repeat(np.arange(10), n // 10)
    for f in range(n):
        scale = 40 + 8 * videos[f]
        for a in range(2):
            points[f, a] = _mouse(rng, scale, rng.uniform(0, 500, 2), rng.uniform(0, 2 * np.pi))
    return points, videos


@pytest.fixture(scope='module')
def model():
    points, videos = _labels()
    return S.SkeletonModel.fit(points, videos)


def test_scale_is_bodymid_tailbase():
    pts = TEMPLATE[None] * 50
    assert S.scale_of(pts) == pytest.approx(50)


def test_model_scores_mixed_skeletons_worse(model, tmp_path):
    rng = np.random.default_rng(1)
    a = _mouse(rng, 60, np.array([100, 100]), 0.3)
    b = _mouse(rng, 60, np.array([180, 130]), 2.0)
    mixed = a.copy()
    mixed[0] = b[0]                         # the other mouse's nose
    (real, df), (bad, _) = model.chi2(a, 60), model.chi2(mixed, 60)
    assert df == 8 and bad > 3 * real
    assert model.pvalue(mixed, 60) < 1e-3 < model.pvalue(a, 60)
    assert model.violations(mixed, 60).any()
    # missing parts: fewer pairs, still finite
    partial = a.copy()
    partial[[1, 2]] = np.nan
    assert model.chi2(partial, 60)[1] == 6
    assert abs(model.excess(partial, 60)) < abs(model.excess(mixed, 60))
    model.save(tmp_path / 'm.json')
    again = S.SkeletonModel.load(tmp_path / 'm.json')
    assert again.chi2(a, 60)[0] == pytest.approx(real)


def _capture(frames, swap_part=None, flip_order=()):
    """Two mice walking side by side; peaks: 0 = DLC's, 1 = runner-up.
    With `swap_part`, mouse 0's argmax for that part is mouse 1's, and the
    true point is its second peak. In `flip_order` frames the detector lists
    the mice the other way round."""
    rng = np.random.default_rng(2)
    peaks = np.full((frames, 4, 8, 3, 3), np.nan)
    peaks[..., 2] = -1
    truth = np.empty((frames, 2, 8, 2))
    for f in range(frames):
        a = _mouse(rng, 50, np.array([100 + 3 * f, 100]), 0.0)
        b = _mouse(rng, 50, np.array([100 + 3 * f, 160]), 0.0)
        truth[f] = a, b
        for slot, (own, other) in enumerate(((a, b), (b, a))):
            peaks[f, slot, :, 0, :2], peaks[f, slot, :, 0, 2] = own, 0.9
            peaks[f, slot, :, 1, :2], peaks[f, slot, :, 1, 2] = other, 0.3
        if swap_part is not None:
            peaks[f, 0, swap_part, 0] = [*b[swap_part], 0.8]
            peaks[f, 0, swap_part, 1] = [*a[swap_part], 0.6]
        if f in flip_order:
            peaks[f, :2] = peaks[f, 1::-1].copy()
    bodyparts = peaks[:, :, :, 0].copy()
    bbox = np.zeros((frames, 4))
    bbox[:, :2] = 0.95
    return {'peaks': peaks, 'bodyparts': bodyparts, 'bbox_scores': bbox}, truth


def test_correction_repairs_a_part_taken_from_the_other_mouse(model):
    raw, truth = _capture(20, swap_part=0)
    out = I.correct(raw, model)
    scales = np.full(20, 50.0)
    _, before = I.evaluate_frames(out['dlc'], truth, scales)
    _, after = I.evaluate_frames(out['poses'], truth, scales)
    assert before['all']['mixed_fraction'] > 0.05
    assert after['all']['mixed_fraction'] == 0
    assert after['all']['correct_fraction'] == 1
    assert (out['parts_changed'] > 0).all()


def test_tracks_stay_continuous_when_detection_order_flips(model):
    raw, truth = _capture(30, flip_order=set(range(10, 20)))
    out = I.correct(raw, model)
    for f in (5, 15, 25):
        assert np.allclose(out['poses'][f, 0, :, :2], truth[f, 0])
        assert np.allclose(out['poses'][f, 1, :, :2], truth[f, 1])


def test_implausible_skeleton_is_rejected_not_guessed(model):
    raw, truth = _capture(1)
    raw['peaks'][0, 0, 0, :2] = [[900, 900, 0.95], [905, 905, 0.9]]   # nose far away
    raw['peaks'][0, 1, 0] = np.nan
    raw['peaks'][0, 1, 0, :, 2] = -1
    out = I.correct(raw, model, {'p_missing': 0.01})
    assert np.isnan(out['poses'][0, 0, 0, 0])
    assert out['parts_rejected'][0] >= 1
    assert np.allclose(out['poses'][0, 0, 1:, :2], truth[0, 0, 1:])


def test_candidates_merge_duplicates_and_drop_weak_boxes():
    peaks = np.full((3, 1, 2, 3), np.nan)
    peaks[..., 2] = -1
    peaks[0, 0, 0] = [10, 10, 0.9]
    peaks[1, 0, 0] = [10.5, 10, 0.7]       # same point from the other crop
    peaks[1, 0, 1] = [50, 50, 0.4]
    peaks[2, 0, 0] = [80, 80, 0.9]         # weak box
    settings = I.Settings(bbox_threshold=0.3)
    pools, dlc = I.candidates(peaks, np.array([0.9, 0.8, 0.1]), settings, scale=50)
    assert pools[0][:, :2].tolist() == [[10, 10], [50, 50]]
    assert dlc.tolist() == [[0], [0]]      # both detections' argmax is candidate 0
