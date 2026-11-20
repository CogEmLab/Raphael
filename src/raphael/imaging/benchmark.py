"""Score one set of ROIs against a reference set (e.g. a published
segmentation of the same plane).

Masks are `Plane.masks`: per ROI an (n_pixels, 3) array of x, y, weight.
ROIs are paired one-to-one by mask overlap (intersection over union, IoU)
with the Hungarian algorithm; a pair counts as a match at IoU >= `min_iou`.
recall = matched / reference ROIs, precision = matched / candidate ROIs.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix


def binary_masks(masks, shape, core=0.0):
    """ROIs x pixels sparse 0/1 matrix. `core` drops pixels whose weight is
    below that fraction of the ROI's peak weight (reference masks often have
    faint tails)."""
    rows, cols = [], []
    for i, m in enumerate(masks):
        keep = m[:, 2] > core * m[:, 2].max() if core else slice(None)
        x, y = m[keep, 0].astype(int), m[keep, 1].astype(int)
        cols.append(y * shape[1] + x)
        rows.append(np.full(len(x), i))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                      shape=(len(masks), shape[0] * shape[1]))


def iou_matrix(a, b):
    inter = (a @ b.T).toarray()
    size_a = np.asarray(a.sum(axis=1)).ravel()
    size_b = np.asarray(b.sum(axis=1)).ravel()
    return inter / (size_a[:, None] + size_b[None, :] - inter)


def match(candidate, reference, shape, min_iou=0.3, reference_core=0.0):
    """Pair candidate ROIs with reference ROIs; returns a summary dict and
    the matched (candidate, reference, iou) triples."""
    a = binary_masks(candidate, shape)
    b = binary_masks(reference, shape, core=reference_core)
    iou = iou_matrix(a, b)
    rows, cols = linear_sum_assignment(-iou)
    keep = iou[rows, cols] >= min_iou
    pairs = list(zip(rows[keep], cols[keep], iou[rows, cols][keep], strict=True))
    n = len(pairs)
    return {
        'candidate': len(candidate),
        'reference': len(reference),
        'matched': n,
        'recall': n / len(reference) if len(reference) else float('nan'),
        'precision': n / len(candidate) if len(candidate) else float('nan'),
        'median_iou': float(np.median([p[2] for p in pairs])) if pairs else float('nan'),
    }, pairs
