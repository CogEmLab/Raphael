"""Fix bodyparts that DLC gives to the wrong mouse, using the other heatmap
peaks.

DLC's top-down model reports, for each detected mouse, the argmax of each
bodypart heatmap. When the two mice touch, the argmax of one mouse's crop is
often the other mouse's part, and the true part is the second or third peak.
The multi-peak capture (`multipeak.py`) keeps the top-k peaks of every
heatmap for every detection, so each frame offers, per bodypart, a small
pool of candidate positions from both crops.

Per frame, `correct_frame` chooses for each of two mice one candidate per
bodypart (or none), each candidate used once, maximising

    sum over chosen parts of log(heatmap score)
    + keep_dlc for every part where the choice is DLC's own prediction
    + log(p_missing) for every part left empty
                                    (log(p_absent) each if a mouse is empty:
                                    one mouse split into two half-skeletons
                                    then costs more than one mouse alone)
    - skeleton_weight / 2 * excess            per mouse, skeleton.SkeletonModel
    - temporal term                           distance to the same part of
                                              the same track in the last frame

starting from DLC's own assignment and changing one bodypart (for both mice
at once) at a time until nothing improves, plus a move that exchanges the
two mice whole. A mouse whose skeleton is still implausible
(chi2 p < `reject_p`) loses its worst parts one at a time until it is
plausible, and a mouse left with fewer than `min_parts` parts is dropped:
rejected rather than guessed.

`correct` runs this frame by frame over a capture, carrying the two tracks
forward, so identities stay continuous; `evaluate_frames` scores any
predictions against labelled frames.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from raphael.behavioural_tracking.skeleton import scale_of


@dataclass
class Settings:
    k: int = 3                      # peaks per bodypart per detection
    bbox_threshold: float = 0.3     # detections below this are ignored
    max_detections: int = 3
    min_peak_score: float = 0.02
    merge: float = 0.08             # candidates closer than this (x scale) merge
    p_missing: float = 0.02
    p_absent: float = 0.15          # per part, for a mouse with no parts at all
    skeleton_weight: float = 0.5
    keep_dlc: float = 1.5           # per part where the choice is DLC's own
    temporal_sigma: float = 0.25    # x scale, per frame since the track was seen
    temporal_cap: float = 9.0       # max squared z per part
    reject_p: float = 1e-5
    min_parts: int = 3
    max_sweeps: int = 5
    # Restraint (tuned by leave-one-fold-out; see docs/behavioural_tracking.md):
    gate: bool = False              # only correct suspect frames, keep DLC elsewhere
    gate_p: float = 0.01            # suspect: a DLC skeleton below this p ...
    gate_close: float = 0.5         # ... or the mice's nearest parts within this x scale
    max_move: float | None = None   # a part may only move this far (x scale) from
                                    # one of DLC's two points for it
    keep_confident: float | None = None   # DLC parts at least this likely are never emptied
    min_gain: float = 0.0           # keep DLC's answer unless the correction scores
                                    # at least this much better
    skeleton_cap: float | None = None     # cap on one skeleton's excess, so one stray
                                          # part cannot dominate
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__}
        return cls(**known, extra=d)


# ----------------------------------------------------------------- one frame


def candidates(peaks, bbox_scores, settings, scale):
    """Per bodypart, the pooled candidates of a frame's detections:
    (n, 3) arrays of x, y, score, best first, with near-duplicates (the same
    point seen in both crops) merged, keeping the higher score.

    peaks: (slots, parts, k, 3); bbox_scores: (slots,) or None.
    Also returns, per bodypart, the index of each kept detection's peak 0
    (DLC's prediction) in that pool, -1 if it was dropped.
    """
    slots = np.arange(len(peaks))
    if bbox_scores is not None:
        slots = slots[np.asarray(bbox_scores) >= settings.bbox_threshold]
        slots = slots[np.argsort(-np.asarray(bbox_scores)[slots], kind='stable')]
    slots = slots[: settings.max_detections]
    radius = settings.merge * scale
    pools, dlc = [], []
    for p in range(peaks.shape[1]):
        pts = []
        origin = []
        for rank, s in enumerate(slots):
            for j in range(min(settings.k, peaks.shape[2])):
                x, y, score = peaks[s, p, j]
                if np.isfinite(x) and score >= (0 if j == 0 else settings.min_peak_score):
                    pts.append((x, y, max(score, 1e-6)))
                    origin.append((rank, j))
        pts = np.array(pts, float).reshape(-1, 3)
        order = np.argsort(-pts[:, 2], kind='stable')
        kept, index_of = [], {}
        for i in order:
            near = [n for n, q in enumerate(kept)
                    if np.hypot(*(pts[i, :2] - q[:2])) < radius]
            if near:
                index_of[i] = near[0]
            else:
                index_of[i] = len(kept)
                kept.append(pts[i])
        pools.append(np.array(kept, float).reshape(-1, 3))
        first = [-1] * len(slots)
        for i, (rank, j) in enumerate(origin):
            if j == 0:
                first[rank] = index_of[i]
        dlc.append(first)
    return pools, np.array(dlc, int).reshape(peaks.shape[1], len(slots)).T


def _points(pools, choice):
    out = np.full((len(pools), 2), np.nan)
    for p, c in enumerate(choice):
        if c >= 0:
            out[p] = pools[p][c, :2]
    return out


def _part_terms(pools, choice, settings, prev, prev_age, scale):
    """Per part: log score (or log p_missing) plus the temporal term; a mouse
    with no parts at all costs log p_absent per part instead."""
    out = np.empty(len(pools))
    if (np.asarray(choice) < 0).all():
        out[:] = np.log(settings.p_absent)
        return out
    for p, c in enumerate(choice):
        if c < 0:
            out[p] = np.log(settings.p_missing)
            continue
        out[p] = np.log(pools[p][c, 2])
        if prev is not None and np.isfinite(prev[p]).all():
            sigma = settings.temporal_sigma * scale * prev_age[p]
            z2 = np.sum((pools[p][c, :2] - prev[p]) ** 2) / sigma ** 2
            out[p] -= 0.5 * min(z2, settings.temporal_cap)
    return out


def _skeleton_term(model, points, scale, settings):
    excess = model.excess(points, scale)
    if settings.skeleton_cap is not None:
        excess = np.minimum(excess, settings.skeleton_cap)
    return -0.5 * settings.skeleton_weight * excess


def correct_frame(pools, init, model, scale, settings, prev=None, prev_age=None):
    """Choose candidates for two mice. `init` (2, parts) candidate indices
    (-1 = none) is the starting assignment; `prev` (2, parts, 2) and
    `prev_age` (2, parts) the tracks' last positions and frames since.
    Returns the choice (2, parts) and the total score."""
    n_parts = len(pools)
    dlc = np.array(init, int)
    choice = _resolve_clash(pools, dlc, model, scale)
    prev = prev if prev is not None else [None, None]
    ages = prev_age if prev_age is not None else np.ones((2, n_parts))
    allowed = [_allowed(pools[p], dlc[:, p], scale, settings) for p in range(n_parts)]

    def protected(ref, a, p):
        c = ref[a, p]
        return (settings.keep_confident is not None and c >= 0
                and pools[p][c, 2] >= settings.keep_confident)

    def reference(ch):
        # DLC's detections in the orientation that matches `ch` best: the
        # keep_dlc prior rewards agreeing with DLC, not a mouse order
        same = ((ch == dlc) & (ch >= 0)).sum()
        cross = ((ch == dlc[::-1]) & (ch >= 0)).sum()
        return dlc if same >= cross else dlc[::-1]

    def total(ch):
        s = settings.keep_dlc * ((ch == reference(ch)) & (ch >= 0)).sum()
        for a in range(2):
            s += _part_terms(pools, ch[a], settings, prev[a], ages[a], scale).sum()
            s += _skeleton_term(model, _points(pools, ch[a]), scale, settings)
        return s

    start = choice.copy()
    best = baseline = total(choice)
    for _ in range(settings.max_sweeps):
        improved = False
        swapped = choice[::-1].copy()          # exchange the two mice whole
        if (score := total(swapped)) > best + 1e-9:
            choice, best, improved = swapped, score, True
        for p in range(n_parts):
            options = allowed[p]
            if len(options) == 1:
                continue
            gain = np.empty((2, len(options)))
            ref = reference(choice)
            for a in range(2):
                trial = np.repeat(choice[a][None], len(options), axis=0)
                trial[:, p] = options
                pts = np.stack([_points(pools, t) for t in trial])
                skel = _skeleton_term(model, pts, np.full(len(options), scale), settings)
                part = [_part_terms(pools, t, settings, prev[a], ages[a], scale)[p]
                        for t in trial]
                keep = settings.keep_dlc * ((options == ref[a, p]) & (options >= 0))
                gain[a] = skel + np.array(part) + keep
                if protected(ref, a, p):
                    gain[a][options < 0] = -np.inf
            # best (a, b) with a != b unless both empty
            pair = gain[0][:, None] + gain[1][None, :]
            clash = (options[:, None] == options[None, :]) & (options[:, None] >= 0)
            pair[clash] = -np.inf
            ia, ib = np.unravel_index(np.argmax(pair), pair.shape)
            new = choice.copy()
            new[0, p], new[1, p] = options[ia], options[ib]
            if (new != choice).any() and (score := total(new)) > best + 1e-9:
                choice, best, improved = new, score, True
        if not improved:
            break
    if best - baseline < settings.min_gain:
        return start, baseline
    return choice, best


def _resolve_clash(pools, init, model, scale):
    """A starting point where no point belongs to both mice. When DLC's two
    detections share half their parts or more, the second is a duplicate box
    on the same mouse: the first keeps everything and the second starts
    empty. Otherwise each shared point starts with the mouse whose skeleton
    it fits better."""
    choice = np.array(init, int).copy()
    clash = np.flatnonzero((choice[0] == choice[1]) & (choice[0] >= 0))
    if not len(clash):
        return choice
    present = (choice >= 0).sum(axis=1)
    if len(clash) >= 0.5 * max(min(present), 1):
        choice[1] = -1
        return choice
    for p in clash:
        fit = []
        for a in range(2):
            without = choice[a].copy()
            without[p] = -1
            fit.append(float(model.excess(_points(pools, choice[a]), scale)
                             - model.excess(_points(pools, without), scale)))
        choice[int(np.argmax(fit)), p] = -1     # the worse fit gives it up
    return choice


def _allowed(pool, dlc_choices, scale, settings):
    """Candidate indices a part may take (-1 = empty): with `max_move`, only
    those near one of DLC's two points for the part."""
    index = np.arange(len(pool))
    anchors = [pool[c, :2] for c in dlc_choices if c >= 0]
    if settings.max_move is not None and anchors:
        d = np.min([np.linalg.norm(pool[:, :2] - a, axis=1) for a in anchors], axis=0)
        index = index[d <= settings.max_move * scale]
    return np.concatenate([[-1], index]).astype(int)


def suspect(pools, init, model, scale, settings):
    """Whether DLC's own answer for a frame needs a second look: both
    detections claim one point, a skeleton is implausible, or the two mice
    are close."""
    if ((init[0] == init[1]) & (init[0] >= 0)).any():
        return True
    pts = [_points(pools, init[a]) for a in range(2)]
    for a in range(2):
        if (settings.gate_p > 0 and np.isfinite(pts[a][:, 0]).sum() >= 2
                and model.pvalue(pts[a], scale) < settings.gate_p):
            return True
    d = np.linalg.norm(pts[0][:, None] - pts[1][None], axis=-1)
    return bool(np.isfinite(d).any() and np.nanmin(d) < settings.gate_close * scale)


def reject(pools, choice, model, scale, settings):
    """Drop parts until each mouse's skeleton is plausible, worst first.
    Returns the new choice and the number of parts dropped per mouse."""
    choice = choice.copy()
    dropped = np.zeros(2, int)
    for a in range(2):
        while True:
            present = np.flatnonzero(choice[a] >= 0)
            if len(present) < settings.min_parts:
                dropped[a] += len(present)
                choice[a] = -1
                break
            if model.pvalue(_points(pools, choice[a]), scale) >= settings.reject_p:
                break
            trials = []
            droppable = [p for p in present if settings.keep_confident is None
                         or pools[p][choice[a, p], 2] < settings.keep_confident]
            if not droppable:
                break
            present = np.array(droppable)
            for p in present:
                t = choice[a].copy()
                t[p] = -1
                trials.append(float(model.pvalue(_points(pools, t), scale)))
            choice[a, present[int(np.argmax(trials))]] = -1
            dropped[a] += 1
    return choice, dropped


# ----------------------------------------------------------------- a capture


def correct(raw, model, settings=None, scale=None, track=True):
    """Correct a capture (`multipeak.load_capture`). With `track=False`
    frames are independent (labelled frames), with no temporal term and no
    track continuity. Returns a dict with
    `poses` (frames, 2, parts, 3) x, y, score, NaN where missing, as two
    continuous tracks; `dlc` the same layout from DLC's own top-2
    detections, matched to the same tracks; and per-frame diagnostics."""
    settings = settings if isinstance(settings, Settings) else Settings.from_dict(settings)
    peaks = raw['peaks']
    bbox = raw.get('bbox_scores')
    n_frames, _, n_parts = peaks.shape[:3]
    if scale is None:
        scale = video_scale(raw, settings, model.bodyparts)
    poses = np.full((n_frames, 2, n_parts, 3), np.nan)
    dlc = np.full((n_frames, 2, n_parts, 3), np.nan)
    info = {key: np.zeros(n_frames, int) for key in
            ('parts_changed', 'parts_rejected', 'parts_failed_check', 'mice', 'swapped',
             'suspect')}
    info['chi2_p'] = np.full((n_frames, 2), np.nan)
    prev = np.full((2, n_parts, 2), np.nan)
    age = np.ones((2, n_parts))
    for f in range(n_frames):
        pools, firsts = candidates(peaks[f], None if bbox is None else bbox[f], settings, scale)
        init = np.full((2, n_parts), -1)
        init[: len(firsts)] = firsts[:2]
        init = _match_tracks(pools, init, prev)
        if settings.gate and not suspect(pools, init, model, scale, settings):
            choice, dropped = init.copy(), np.zeros(2, int)
        else:
            info['suspect'][f] = 1
            choice, _ = correct_frame(pools, init, model, scale, settings, prev, age)
            choice, dropped = reject(pools, choice, model, scale, settings)
        for a in range(2):
            for p in range(n_parts):
                if choice[a, p] >= 0:
                    poses[f, a, p] = pools[p][choice[a, p]]
                if init[a, p] >= 0:
                    dlc[f, a, p] = pools[p][init[a, p]]
            if np.isfinite(poses[f, a, :, 0]).sum() >= 2:
                info['chi2_p'][f, a] = model.pvalue(poses[f, a, :, :2], scale)
        info['parts_changed'][f] = int(((choice != init) & (choice >= 0)).sum())
        # DLC proposed a point and nothing was kept, by the optimiser or the check
        info['parts_rejected'][f] = int(((init >= 0) & (choice < 0)).sum())
        info['parts_failed_check'][f] = int(dropped.sum())
        info['mice'][f] = int((choice >= 0).any(axis=1).sum())
        info['swapped'][f] = int(_is_swap(choice, init))
        seen = np.isfinite(poses[f, :, :, 0])
        prev = np.where(seen[..., None], poses[f, :, :, :2], prev)
        age = np.where(seen, 1, age + 1)
        if not track:
            prev[:] = np.nan
    return {'poses': poses, 'dlc': dlc, 'scale': scale, **info}


def video_scale(raw, settings, bodyparts):
    """The capture's own scale, from DLC's confident predictions."""
    bp = raw['bodyparts']
    bbox = raw.get('bbox_scores')
    if bbox is not None:
        bp = np.where((bbox >= settings.bbox_threshold)[..., None, None], bp, np.nan)
    return scale_of(bp[..., :2], bodyparts, likelihood=bp[..., 2])


def _centre(pools, choice):
    pts = _points(pools, choice)
    return np.nanmean(pts, axis=0) if np.isfinite(pts).any() else np.full(2, np.nan)


def _match_tracks(pools, init, prev):
    """Order DLC's two detections to follow the two tracks."""
    if not np.isfinite(prev).any():
        return init
    cost = np.zeros((2, 2))
    for a in range(2):
        pts = _points(pools, init[a])
        for t in range(2):
            d = np.linalg.norm(pts - prev[t], axis=1)
            cost[a, t] = np.nanmean(d) if np.isfinite(d).any() else 1e6
    rows, cols = linear_sum_assignment(cost)
    out = init.copy()
    out[cols] = init[rows]
    return out


def _is_swap(choice, init):
    """The corrected mice took more parts from the other's DLC detection."""
    same = ((choice == init) & (choice >= 0)).sum()
    cross = ((choice == init[::-1]) & (choice >= 0)).sum()
    return cross > same


# ----------------------------------------------------------------- scoring


def evaluate_frames(pred, truth, scales, tolerance=0.3):
    """Score predictions (frames, mice, parts, >=2) against labels (frames,
    individuals, parts, 2), ignoring identity: each frame's predicted mice
    are matched to the labelled mice by mean part distance. A labelled part
    is then

        correct  within `tolerance` x scale of its own mouse's part
        mixed    otherwise within tolerance of the other mouse's same part
        wrong    otherwise
        missing  not predicted

    and a predicted mouse is `mixed` if it holds parts of both mice.
    Returns a per-part table (pandas) and summary counts."""
    import pandas as pd

    rows = []
    for f in range(len(truth)):
        gt = truth[f]
        pr = pred[f, :, :, :2]
        n_gt, n_pr = len(gt), len(pr)
        cost = np.full((n_pr, n_gt), 1e6)
        for i in range(n_pr):
            for j in range(n_gt):
                d = np.linalg.norm(pr[i] - gt[j], axis=1)
                if np.isfinite(d).any():
                    cost[i, j] = np.nanmean(d)
        r, c = linear_sum_assignment(cost)
        owner = {j: i for i, j in zip(r, c, strict=True) if cost[i, j] < 1e6}
        tol = tolerance * scales[f]
        for j in range(n_gt):
            for p in range(gt.shape[1]):
                if not np.isfinite(gt[j, p]).all():
                    continue
                i = owner.get(j)
                point = pr[i, p] if i is not None else np.full(2, np.nan)
                err = float(np.linalg.norm(point - gt[j, p]))
                others = [np.linalg.norm(point - gt[o, p]) for o in range(n_gt)
                          if o != j and np.isfinite(gt[o, p]).all()]
                if not np.isfinite(point).all():
                    outcome = 'missing'
                elif err <= tol:
                    outcome = 'correct'
                elif others and min(others) <= tol:
                    outcome = 'mixed'
                else:
                    outcome = 'wrong'
                rows.append({'frame': f, 'mouse': j, 'part': p, 'outcome': outcome,
                             'error_px': err, 'error_scaled': err / scales[f],
                             'two_mice': int(np.isfinite(gt).all(axis=-1).any(axis=-1).sum() == 2)})
    table = pd.DataFrame(rows)
    return table, summarise(table)


def summarise(table):
    out = {}
    for name, sub in (('all', table), ('two_mice', table[table.two_mice == 1])):
        n = len(sub)
        counts = sub.outcome.value_counts()
        out[name] = {
            'parts': n,
            **{f'{k}_fraction': round(float(counts.get(k, 0)) / max(n, 1), 4)
               for k in ('correct', 'mixed', 'wrong', 'missing')},
            'mixed_mice': int(_mixed_mice(sub)),
            'median_error_scaled': round(float(sub.loc[sub.outcome != 'missing', 'error_scaled'].median()), 4),
            'rmse_px_found': round(float(np.sqrt(np.mean(sub.loc[sub.outcome != 'missing', 'error_px'] ** 2))), 2),
        }
    return out


def _mixed_mice(sub):
    """Labelled mice whose matched prediction holds a part of the other mouse."""
    return sub[sub.outcome == 'mixed'].groupby(['frame', 'mouse']).ngroups


# ----------------------------------------------------------------- videos


def poses_from_table(df, individuals, bodyparts):
    """A DLC pose table (e.g. another model's `_el.csv`/h5) -> (frames, mice, parts, 3)."""
    out = np.full((len(df), len(individuals), len(bodyparts), 3), np.nan)
    for a, ind in enumerate(individuals):
        for p, part in enumerate(bodyparts):
            for c, coord in enumerate(('x', 'y', 'likelihood')):
                key = [col for col in df.columns if col[-3:] == (ind, part, coord)]
                if key:
                    out[:, a, p, c] = df[key[0]].to_numpy(float)
    return out


def video_metrics(poses, model, scale, jump=1.0, reject_p=1e-3):
    """Label-free checks of tracked poses (frames, 2, parts, >=2):
    implausible skeletons (`SkeletonModel.pvalue` < reject_p) and skeletons with any pair
    outside the label range, per 1000 mouse-frames; whole-mouse swaps
    (`sweep.swap_events`) and body-centre jumps over `jump` x scale, per
    1000 frames; and presence."""
    import warnings

    from raphael.behavioural_tracking.sweep import swap_events

    pts = poses[..., :2]
    n = len(pts)
    present = np.isfinite(pts).all(axis=-1).sum(axis=-1) >= 2
    p = np.where(present, model.pvalue(pts, scale), np.nan)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        cent = np.nanmean(pts, axis=2)
        step = np.linalg.norm(np.diff(cent, axis=0), axis=-1)
    mouse_frames = max(int(present.sum()), 1)
    return {
        'frames': n,
        'both_present_frac': round(float(present.all(axis=1).mean()), 4),
        'parts_per_mouse': round(float(np.isfinite(pts[..., 0]).sum() / mouse_frames), 3),
        'implausible_per_1000': round(1000 * float(np.nansum(p < reject_p)) / mouse_frames, 2),
        'pair_violation_per_1000': round(
            1000 * float(model.violations(pts, scale).any(-1).sum()) / mouse_frames, 2),
        'swaps_per_1000': round(1000 * float(swap_events(cent, 0.1 * scale).sum()) / n, 2),
        'jumps_per_1000': round(1000 * float(np.nansum(step > jump * scale)) / n, 2),
    }


def correct_capture(npz, skeleton_json, settings=None, benchmark=None):
    """Correct one video capture; write `<capture>.identity.npz` beside it
    and return a table of `video_metrics` for DLC's top-2 detections, the
    corrected poses and, if given (frames, 2, parts, 3), a benchmark."""
    import json
    from pathlib import Path

    import pandas as pd

    from raphael.behavioural_tracking.multipeak import load_capture
    from raphael.behavioural_tracking.skeleton import SkeletonModel

    model = SkeletonModel.load(skeleton_json)
    raw = load_capture(npz)
    out = correct(raw, model, settings)
    scale = out['scale']
    np.savez_compressed(
        Path(npz).with_suffix('.identity.npz'),
        meta=json.dumps({**raw['meta'], 'scale': scale, 'identity': dict(settings or {})}),
        **{k: v for k, v in out.items() if k != 'scale'})
    rows = [('dlc_top2', out['dlc']), ('corrected', out['poses'])]
    if benchmark is not None:
        start = raw['meta'].get('start_frame', 0)
        rows.append(('benchmark', benchmark[start: start + len(out['poses'])]))
    table = pd.DataFrame([{'video': raw['meta']['video'], 'method': name,
                           **video_metrics(poses, model, scale)} for name, poses in rows])
    found = max(int(np.isfinite(out['dlc'][..., 0]).sum()), 1)
    table.loc[table.method == 'corrected', 'parts_changed_frac'] = round(
        out['parts_changed'].sum() / found, 4)
    table.loc[table.method == 'corrected', 'parts_rejected_frac'] = round(
        out['parts_rejected'].sum() / found, 4)
    return table


def qc_frames(video, identity_npz, out_png, n=8, start_frame=None):
    """The `n` frames where the correction moved most parts, DLC's top-2
    (left) beside the corrected poses (right), for checking by eye."""
    import json
    import warnings

    import cv2
    import matplotlib.pyplot as plt

    data = np.load(identity_npz, allow_pickle=False)
    meta = json.loads(str(data['meta']))
    start = meta.get('start_frame', 0) if start_frame is None else start_frame
    poses, dlc = data['poses'], data['dlc']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        moved = np.nansum(np.linalg.norm(poses[..., :2] - dlc[..., :2], axis=-1) > 1, axis=(1, 2))
    moved += (np.isfinite(dlc[..., 0]) & ~np.isfinite(poses[..., 0])).sum(axis=(1, 2))
    frames = np.sort(np.argsort(-moved, kind='stable')[:n])
    frames = frames[moved[frames] > 0]
    reader = cv2.VideoCapture(str(video))
    fig, axes = plt.subplots(len(frames), 2, figsize=(6, 2.4 * len(frames)), squeeze=False)
    colours = ('tab:orange', 'tab:blue')
    for row, f in zip(axes, frames, strict=True):
        reader.set(cv2.CAP_PROP_POS_FRAMES, start + int(f))
        ok, image = reader.read()
        for ax, (name, p) in zip(row, (('DLC', dlc[f]), ('corrected', poses[f])), strict=True):
            if ok:
                ax.imshow(image[..., ::-1])
            for a in range(2):
                ax.plot(p[a, :, 0], p[a, :, 1], 'o-', ms=3, lw=0.8, color=colours[a])
            pts = np.concatenate([dlc[f, ..., :2], poses[f, ..., :2]]).reshape(-1, 2)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts):
                pad = 0.6 * meta.get('scale', 50)
                ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
                ax.set_ylim(pts[:, 1].max() + pad, pts[:, 1].min() - pad)
            ax.set_title(f'{name}, frame {start + int(f)}', fontsize=8)
            ax.axis('off')
    reader.release()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
