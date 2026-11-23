"""Multi-peak pose extraction from DeepLabCut heatmaps.

DLC reduces each bodypart heatmap to a single point (its argmax, for top-down
models) before anything is saved. When two mice overlap, the second-best
peak in a heatmap is often the *other* mouse's bodypart, which is exactly the
evidence needed to diagnose identity confusion. This module keeps the top-k
local maxima of every bodypart heatmap, for every detected animal, alongside
DLC's normal output.

It works by wrapping, not replacing, DLC's own pieces:

- `MultiPeakPredictor` wraps a head's predictor: DLC's prediction is returned
  untouched as `poses`, and an extra `peaks` array is added.
- `PeaksPostprocessor` wraps the runner's postprocessor and maps `peaks` from
  model-input pixels back to video pixels with DLC's own `RescaleAndOffset`,
  using the same per-crop scales and offsets as the poses.

So peak 0 of each bodypart equals DLC's own prediction (the global maximum
always survives non-maximum suppression), which is checked in the tests.

Supported: any head that emits per-bodypart heatmaps, i.e. top-down and
conditional top-down heatmap heads, bottom-up PAF heads, and DEKR (whose
extra centre channel is ignored). Heads without heatmaps (SimCC/RTMPose)
fall back to DLC's single prediction; their `peaks` are empty.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PAD = -1.0


def heatmap_peaks(heatmaps, locref, stride, k, nms_radius=2, min_score=0.01):
    """Top-k local maxima per heatmap.

    heatmaps: (B, K, H, W), activated like DLC's (sigmoid if the model uses it).
    locref:   (B, K, 2, H, W) offsets in pixels, or None.
    Returns (B, K, k, 3) of x, y, score in model-input pixels; missing peaks
    have NaN coordinates and score PAD.

    Peak 0 is always the heatmap's argmax, computed exactly as DLC computes
    it, however low its score: it IS DLC's prediction, and judging it is the
    threshold sweep's job. `min_score` and non-maximum suppression apply only
    to the additional peaks 1..k-1.
    """
    batch, joints, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch, joints, -1)
    top = flat.argmax(dim=-1, keepdim=True)                      # (B, K, 1)
    size = 2 * nms_radius + 1
    pooled = F.max_pool2d(heatmaps, size, stride=1, padding=nms_radius)
    keep = ((heatmaps == pooled) & (heatmaps >= min_score)).reshape(batch, joints, -1)
    keep.scatter_(-1, top, False)                                # peak 0 is handled above
    others = torch.where(keep, flat, torch.full_like(flat, -np.inf))
    k = min(k, height * width)
    other_values, other_index = others.topk(k - 1, dim=-1) if k > 1 else (
        flat[..., :0], top[..., :0]
    )
    index = torch.cat([top, other_index], dim=-1)
    values = torch.cat([flat.gather(-1, top), other_values], dim=-1)
    y = torch.div(index, width, rounding_mode='floor').float()
    x = (index % width).float()
    # Same convention as DLC's HeatmapPredictor: cell centre, then locref.
    x = x * stride + 0.5 * stride
    y = y * stride + 0.5 * stride
    if locref is not None:
        offsets = locref.reshape(batch, joints, 2, -1)
        x = x + offsets[:, :, 0].gather(-1, index)
        y = y + offsets[:, :, 1].gather(-1, index)
    valid = torch.isfinite(values)
    out = torch.stack(
        [
            torch.where(valid, x, torch.full_like(x, float('nan'))),
            torch.where(valid, y, torch.full_like(y, float('nan'))),
            torch.where(valid, values, torch.full_like(values, PAD)),
        ],
        dim=-1,
    )
    return out


class MultiPeakPredictor(torch.nn.Module):
    """Wraps a DLC predictor; adds `peaks` of shape (B, 1, K*k, 3)."""

    def __init__(self, inner, num_bodyparts, k=5, nms_radius=2, min_score=0.01):
        super().__init__()
        self.inner = inner
        self.num_bodyparts = num_bodyparts
        self.k = k
        self.nms_radius = nms_radius
        self.min_score = min_score

    def forward(self, stride, outputs):
        result = dict(self.inner(stride, outputs))
        heatmaps = outputs.get('heatmap')
        if heatmaps is None:
            return result
        n = self.num_bodyparts
        heatmaps = heatmaps[:, :n]  # DEKR appends a centre channel
        if getattr(self.inner, 'apply_sigmoid', False):
            heatmaps = torch.sigmoid(heatmaps)
        # Not clipped here: DLC takes the argmax of the raw heatmap and clips
        # only the reported score. Clipping first would flatten every value
        # above 1 into a plateau and move the peak off DLC's.
        clip = getattr(self.inner, 'clip_scores', False)
        locref = None
        use_locref = getattr(self.inner, 'location_refinement', False)
        if use_locref and outputs.get('locref') is not None:
            std = getattr(self.inner, 'locref_std', None) or getattr(
                self.inner, 'locref_stdev', 1.0
            )
            batch, _, height, width = heatmaps.shape
            locref = outputs['locref'][:, : 2 * n].reshape(batch, n, 2, height, width) * std
        peaks = heatmap_peaks(
            heatmaps, locref, stride, self.k, self.nms_radius, self.min_score
        )
        if clip:
            found = ~torch.isnan(peaks[..., 0])
            peaks[..., 2] = torch.where(found, peaks[..., 2].clamp(0, 1), peaks[..., 2])
        batch = peaks.shape[0]
        result['peaks'] = peaks.reshape(batch, 1, n * self.k, 3)
        return result


class PeaksPostprocessor:
    """Runs DLC's postprocessor, then maps `peaks` into video pixels.

    Output `peaks` has shape (max_individuals, K, k, 3) for top-down models
    (one candidate set per detection, in the same order as `bodyparts`), and
    (1, K, k, 3) for bottom-up models, where peaks cover the whole image.
    """

    def __init__(self, base, top_down, num_bodyparts, k, max_individuals):
        self.base = base
        self.top_down = top_down
        self.num_bodyparts = num_bodyparts
        self.k = k
        self.max_individuals = max_individuals

    def __call__(self, predictions, context):
        from deeplabcut.pose_estimation_pytorch.data.postprocessor import (
            RescaleAndOffset,
        )

        outputs, context = self.base(predictions, context)
        n, k = self.num_bodyparts, self.k
        rows = [
            p['bodypart']['peaks'] for p in predictions
            if 'peaks' in p.get('bodypart', {})
        ]
        if rows:
            peaks = np.concatenate(rows)  # (detections, K*k, 3)
            mode = (
                RescaleAndOffset.Mode.KEYPOINT_TD if self.top_down
                else RescaleAndOffset.Mode.KEYPOINT
            )
            peaks = RescaleAndOffset(['peaks'], mode)({'peaks': peaks}, context)[0]['peaks']
            peaks = peaks.reshape(len(peaks), n, k, 3)
        else:
            peaks = np.zeros((0, n, k, 3))
        slots = self.max_individuals if self.top_down else 1
        padded = np.full((slots, n, k, 3), np.nan)
        padded[..., 2] = PAD
        padded[: min(len(peaks), slots)] = peaks[:slots]
        outputs['peaks'] = padded
        return outputs, context


def install(runner, num_bodyparts, top_down, k, nms_radius, min_score, max_individuals):
    """Patch a DLC PoseInferenceRunner in place to also return peaks."""
    head = runner.model.heads['bodypart']
    if not isinstance(head.predictor, MultiPeakPredictor):
        head.predictor = MultiPeakPredictor(
            head.predictor, num_bodyparts, k, nms_radius, min_score
        )
    if not isinstance(runner.postprocessor, PeaksPostprocessor):
        runner.postprocessor = PeaksPostprocessor(
            runner.postprocessor, top_down, num_bodyparts, k, max_individuals
        )
    return runner


# ------------------------------------------------------------------ capture


def load_model_config(model_dir):
    from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict

    return read_config_as_dict(Path(model_dir) / 'train' / 'pytorch_config.yaml')


def build_runners(model_dir, snapshot, detector_snapshot=None, capture=None, device=None):
    """Pose (and detector) runners for any trained model folder.

    The folder needs only train/pytorch_config.yaml and the snapshots, so a
    model copied from elsewhere works without its DLC project.
    """
    from raphael.behavioural_tracking.dlc import _dlc

    _dlc()
    from deeplabcut.pose_estimation_pytorch.apis import utils

    capture = capture or {}
    model_dir = Path(model_dir)
    cfg = load_model_config(model_dir)
    method = cfg['method'].lower()
    if method == 'ctd':
        raise NotImplementedError(
            'conditional top-down models need bottom-up conditions; analyse '
            'them with deeplabcut.analyze_videos and sweep the _full.pickle'
        )
    top_down = method == 'td'
    max_ind = int(capture.get('max_individuals', 4))
    pose_runner = utils.get_pose_inference_runner(
        model_config=cfg,
        snapshot_path=str(model_dir / 'train' / snapshot),
        batch_size=int(capture.get('batch_size', 8)),
        max_individuals=max_ind,
        device=device,
    )
    install(
        pose_runner,
        num_bodyparts=len(cfg['metadata']['bodyparts']),
        top_down=top_down,
        k=int(capture.get('peaks_per_bodypart', 5)),
        nms_radius=int(capture.get('nms_radius', 2)),
        min_score=float(capture.get('min_peak_score', 0.01)),
        max_individuals=max_ind,
    )
    detector_runner = None
    if top_down:
        if detector_snapshot is None:
            raise ValueError('top-down models need a detector snapshot')
        detector_runner = utils.get_detector_inference_runner(
            model_config=cfg,
            snapshot_path=str(model_dir / 'train' / detector_snapshot),
            batch_size=int(capture.get('batch_size', 8)),
            max_individuals=max_ind,
            min_bbox_score=float(capture.get('min_bbox_score', 0.01)),
            device=device,
        )
    return cfg, pose_runner, detector_runner


def stack_predictions(predictions, max_individuals, num_bodyparts, k):
    """List of per-frame dicts -> fixed-shape arrays (frames first)."""
    n_frames = len(predictions)

    def gather(key, shape, fill):
        out = np.full((n_frames, *shape), fill, dtype=np.float32)
        for i, frame in enumerate(predictions):
            value = frame.get(key)
            if value is None or len(value) == 0:
                continue
            value = np.asarray(value)[: shape[0]]
            out[i, : len(value)] = value
        return out

    arrays = {
        'bodyparts': gather('bodyparts', (max_individuals, num_bodyparts, 3), PAD),
        'peaks': gather('peaks', (max_individuals, num_bodyparts, k, 3), PAD),
    }
    if any('bboxes' in f for f in predictions):
        arrays['bboxes'] = gather('bboxes', (max_individuals, 4), PAD)
        arrays['bbox_scores'] = gather('bbox_scores', (max_individuals,), PAD)
    identity = [f['identity_scores'] for f in predictions if 'identity_scores' in f]
    if identity:
        n_ids = np.asarray(identity[0]).shape[-1]
        arrays['identity_scores'] = gather(
            'identity_scores', (max_individuals, num_bodyparts, n_ids), PAD
        )
    # DLC pads missing detections with -1 in x/y too; make those NaN so they
    # can never be mistaken for a point in the image corner.
    for key in ('bodyparts', 'peaks'):
        missing = arrays[key][..., 2] < 0
        arrays[key][..., 0][missing] = np.nan
        arrays[key][..., 1][missing] = np.nan
    return arrays


def capture_video(video, out_path, model_dir, snapshot, detector_snapshot=None,
                  capture=None, runners=None, name='', max_frames=None, start_frame=0):
    """Run once with permissive cutoffs and save everything for the sweep.
    With `max_frames` (and `start_frame`), only that stretch of the video."""
    import cv2
    from deeplabcut.pose_estimation_pytorch.apis.videos import (
        VideoIterator,
        video_inference,
    )

    capture = capture or {}
    if runners is None:
        runners = build_runners(model_dir, snapshot, detector_snapshot, capture)
    cfg, pose_runner, detector_runner = runners

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = Path(video)
    if max_frames is not None:
        source = _clip(video, out_path.with_suffix('.clip.mp4'), max_frames, start_frame)
    probe = cv2.VideoCapture(str(source))
    fps = probe.get(cv2.CAP_PROP_FPS)
    size = (int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)), int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    probe.release()

    predictions = video_inference(
        video=VideoIterator(str(source)),
        pose_runner=pose_runner,
        detector_runner=detector_runner,
    )
    bodyparts = list(cfg['metadata']['bodyparts'])
    arrays = stack_predictions(
        predictions,
        int(capture.get('max_individuals', 4)),
        len(bodyparts),
        int(capture.get('peaks_per_bodypart', 5)),
    )
    meta = {
        'model': name,
        'model_dir': str(model_dir),
        'snapshot': snapshot,
        'detector_snapshot': detector_snapshot,
        'method': cfg['method'],
        'bodyparts': bodyparts,
        'individuals': list(cfg['metadata']['individuals']),
        'with_identity': bool(cfg['metadata'].get('with_identity')),
        'video': Path(video).name,
        'fps': fps,
        'frame_size': size,
        'n_frames': len(predictions),
        'start_frame': start_frame if max_frames is not None else 0,
        'capture': capture,
    }
    np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
    if source != Path(video):
        source.unlink()
    return out_path


def capture_images(images, out_path, model_dir, snapshot, detector_snapshot=None,
                   capture=None, runners=None, name=''):
    """As `capture_video`, for a list of image files (e.g. labelled frames);
    frame i of the capture is `images[i]`."""
    capture = capture or {}
    if runners is None:
        # labelled frames come from videos of different sizes, which the
        # detector cannot stack into one batch
        runners = build_runners(model_dir, snapshot, detector_snapshot,
                                {**capture, 'batch_size': 1})
    cfg, pose_runner, detector_runner = runners
    images = [str(p) for p in images]
    inputs = images
    if detector_runner is not None:
        inputs = list(zip(images, detector_runner.inference(images=images), strict=True))
    predictions = pose_runner.inference(inputs)
    bodyparts = list(cfg['metadata']['bodyparts'])
    arrays = stack_predictions(
        predictions,
        int(capture.get('max_individuals', 4)),
        len(bodyparts),
        int(capture.get('peaks_per_bodypart', 5)),
    )
    meta = {
        'model': name, 'model_dir': str(model_dir), 'snapshot': snapshot,
        'detector_snapshot': detector_snapshot, 'method': cfg['method'],
        'bodyparts': bodyparts, 'individuals': list(cfg['metadata']['individuals']),
        'with_identity': bool(cfg['metadata'].get('with_identity')),
        'images': images, 'n_frames': len(predictions), 'capture': capture,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
    return out_path


def _clip(video, out, max_frames, start_frame=0):
    """`max_frames` frames of a video from `start_frame`, for quick tests."""
    import cv2

    reader = cv2.VideoCapture(str(video))
    reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = reader.get(cv2.CAP_PROP_FPS)
    size = (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    # VideoWriter fails silently (e.g. missing folder); fail loudly instead.
    if not reader.isOpened() or not writer.isOpened():
        raise OSError(f'cannot clip {video} to {out}')
    for _ in range(max_frames):
        ok, frame = reader.read()
        if not ok:
            break
        writer.write(frame)
    reader.release()
    writer.release()
    return Path(out)


def load_capture(path):
    with np.load(path, allow_pickle=False) as data:
        raw = {key: data[key] for key in data.files if key != 'meta'}
        raw['meta'] = json.loads(str(data['meta']))
    return raw
