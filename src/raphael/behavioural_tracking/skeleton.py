"""What a mouse's skeleton looks like, learned from the labelled frames.

Mice differ in size between videos (camera height, crop, zoom: the median
BodyMid-TailBase distance ranged 32-148 px over one lab's 66 videos), so
every distance is divided by its video's scale, the median distance of the
`scale_pair` (default BodyMid-TailBase). Within a video the 28 distances between the 8 bodyparts are
strongly correlated (a rearing mouse shortens all of them), so one pair
alone says little, but all of them together say a lot.

The model holds, for the log of each scaled pair distance:

- a Gaussian over all pairs (mean and shrunk covariance), giving a
  plausibility score for a whole skeleton with any parts missing:
  `chi2(points, scale)` is the squared Mahalanobis distance over the pairs
  present. Its degrees of freedom are not the number of pairs: the 28
  distances of 8 points in a plane lie on a curved surface of about 13
  dimensions (2n - 3 for n points), and the surface's curvature gives a
  Gaussian thin tails. So the model is calibrated on the training
  skeletons themselves: for each number of parts present, n = 2..8, chi2 of
  every subset of n parts is matched by its moments to a scaled
  c * chi2(nu). `excess` (chi2 / c - nu, zero on average for a real
  skeleton whatever n is) and `pvalue` use that calibration, so skeletons
  with different parts missing are comparable;
- per-pair percentile bounds, the plain rule: a pair outside its
  [`lo`, `hi`] range in the labels is a violation (`violations`).

Fitted on training-split labels only, so test frames stay unseen.
"""

import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist

BODYPARTS = ['Nose', 'FrontPawR', 'FrontPawL', 'BodyMid', 'HindPawR', 'HindPawL',
             'TailBase', 'TailTip']
SCALE_PAIR = ('BodyMid', 'TailBase')


def read_labels(csv):
    """DLC CollectedData csv -> (points (frames, individuals, parts, 2), index
    frame table with `video` and `image`, individuals, bodyparts)."""
    raw = pd.read_csv(csv, header=[0, 1, 2, 3], index_col=[0, 1, 2])
    raw.columns = raw.columns.droplevel(0)
    individuals = list(dict.fromkeys(raw.columns.get_level_values(0)))
    bodyparts = list(dict.fromkeys(raw.columns.get_level_values(1)))
    points = np.full((len(raw), len(individuals), len(bodyparts), 2), np.nan)
    for i, ind in enumerate(individuals):
        for j, part in enumerate(bodyparts):
            for c, coord in enumerate('xy'):
                points[:, i, j, c] = raw[(ind, part, coord)].to_numpy(float)
    frames = pd.DataFrame({'video': raw.index.get_level_values(1),
                           'image': raw.index.get_level_values(2)})
    return points, frames, individuals, bodyparts


def pairs(n_parts):
    return list(itertools.combinations(range(n_parts), 2))


def pair_distances(points):
    """(..., parts, 2) -> (..., pairs); NaN where either part is missing."""
    i, j = np.array(pairs(points.shape[-2])).T
    return np.linalg.norm(points[..., i, :] - points[..., j, :], axis=-1)


def scale_of(points, bodyparts=BODYPARTS, likelihood=None, min_likelihood=0.6,
             scale_pair=SCALE_PAIR):
    """A video's scale: the median BodyMid-TailBase distance over every animal
    and frame in `points` (..., parts, 2) where both parts are present (and,
    for predictions, both have `likelihood` >= `min_likelihood`)."""
    a, b = (list(bodyparts).index(p) for p in scale_pair)
    d = np.linalg.norm(points[..., a, :] - points[..., b, :], axis=-1)
    if likelihood is not None:
        d = np.where((likelihood[..., a] >= min_likelihood)
                     & (likelihood[..., b] >= min_likelihood), d, np.nan)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if len(d) else np.nan


class SkeletonModel:
    def __init__(self, mean, cov, lo, hi, bodyparts, percentiles, n_skeletons,
                 calibration=None):
        self.mean = np.asarray(mean, float)
        self.cov = np.asarray(cov, float)
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.bodyparts = list(bodyparts)
        self.percentiles = tuple(percentiles)
        self.n_skeletons = int(n_skeletons)
        self.pairs = pairs(len(self.bodyparts))
        self._precision = lru_cache(maxsize=512)(self._precision_for)
        # parts present -> (c, nu) of c * chi2(nu); uncalibrated: 1, pairs
        n = len(self.bodyparts)
        self.calibration = {k: (1.0, k * (k - 1) / 2) for k in range(2, n + 1)}
        self.calibration.update({int(k): tuple(v) for k, v in (calibration or {}).items()})

    # ------------------------------------------------------------- fitting

    @classmethod
    def fit(cls, points, videos, bodyparts=BODYPARTS, percentiles=(0.5, 99.5), shrinkage=0.5):
        """`points` (frames, individuals, parts, 2), `videos` one id per frame.
        Gaussian on complete skeletons; bounds on every pair observed;
        calibration on every subset of parts of the complete skeletons."""
        videos = np.asarray(videos)
        logd, skeletons, scales = [], [], []
        for video in np.unique(videos):
            sel = points[videos == video].reshape(-1, *points.shape[-2:])
            scale = scale_of(sel, bodyparts)
            if not np.isfinite(scale):
                continue
            with np.errstate(divide='ignore'):
                logd.append(np.log(pair_distances(sel) / scale))
            complete = np.isfinite(sel).all(axis=(1, 2))
            skeletons.append(sel[complete])
            scales.append(np.full(complete.sum(), scale))
        logd = np.concatenate(logd)
        logd[~np.isfinite(logd)] = np.nan
        complete = logd[np.isfinite(logd).all(axis=1)]
        mean = complete.mean(axis=0)
        sample = np.cov(complete, rowvar=False)
        cov = (1 - shrinkage) * sample + shrinkage * np.diag(np.diag(sample))
        lo, hi = np.nanpercentile(logd, percentiles, axis=0)
        model = cls(mean, cov, lo, hi, bodyparts, percentiles, len(complete))
        skeletons, scales = np.concatenate(skeletons), np.concatenate(scales)
        for k in range(2, len(bodyparts) + 1):
            values = []
            for keep in itertools.combinations(range(len(bodyparts)), k):
                pts = np.full_like(skeletons, np.nan)
                pts[:, keep] = skeletons[:, keep]
                values.append(model.chi2(pts, scales)[0])
            values = np.concatenate(values)
            m, v = values.mean(), values.var()
            model.calibration[k] = (float(v / (2 * m)), float(2 * m ** 2 / v))
        return model

    # ------------------------------------------------------------- scoring

    def log_distances(self, points, scale):
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.log(pair_distances(points) / np.asarray(scale)[..., None])
        d[~np.isfinite(d)] = np.nan
        return d

    def _precision_for(self, mask_bytes):
        mask = np.frombuffer(mask_bytes, dtype=bool)
        return np.linalg.inv(self.cov[np.ix_(mask, mask)])

    def chi2(self, points, scale):
        """Squared Mahalanobis distance of each skeleton (..., parts, 2) over
        its present pairs (uncalibrated), and the number of parts present."""
        logd = self.log_distances(points, scale)
        flat = logd.reshape(-1, logd.shape[-1])
        out = np.zeros(len(flat))
        masks = np.isfinite(flat)
        for mask in np.unique(masks, axis=0):
            if not mask.any():
                continue
            rows = (masks == mask).all(axis=1)
            r = flat[rows][:, mask] - self.mean[mask]
            out[rows] = np.einsum('ni,ij,nj->n', r, self._precision(mask.tobytes()), r)
        n = np.isfinite(points).all(axis=-1).sum(axis=-1)
        return out.reshape(logd.shape[:-1]), n

    def _cal(self, n):
        n = np.asarray(n)
        c = np.ones(n.shape)
        nu = np.zeros(n.shape)
        for k, (ck, nuk) in self.calibration.items():
            c = np.where(n == k, ck, c)
            nu = np.where(n == k, nuk, nu)
        return c, nu

    def excess(self, points, scale):
        """Calibrated chi2 minus its expectation: ~0 for real skeletons,
        large for mixed ones; 0 with fewer than 2 parts."""
        chi2, n = self.chi2(points, scale)
        c, nu = self._cal(n)
        return np.where(n >= 2, chi2 / c - nu, 0.0)

    def pvalue(self, points, scale):
        """Probability of a real skeleton scoring worse; 1 below 2 parts."""
        chi2, n = self.chi2(points, scale)
        c, nu = self._cal(n)
        with np.errstate(all='ignore'):
            return np.where(n >= 2, chi2_dist.sf(chi2 / c, np.maximum(nu, 1e-9)), 1.0)

    def violations(self, points, scale):
        """Boolean (..., pairs): the pair distance is outside the label range."""
        logd = self.log_distances(points, scale)
        return np.isfinite(logd) & ((logd < self.lo) | (logd > self.hi))

    def pair_names(self):
        return [f'{self.bodyparts[i]}-{self.bodyparts[j]}' for i, j in self.pairs]

    # ------------------------------------------------------------- storage

    def save(self, path):
        Path(path).write_text(json.dumps({
            'bodyparts': self.bodyparts, 'percentiles': self.percentiles,
            'n_skeletons': self.n_skeletons, 'calibration': self.calibration,
            'mean': self.mean.tolist(), 'cov': self.cov.tolist(),
            'lo': self.lo.tolist(), 'hi': self.hi.tolist(),
        }, indent=1))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
