"""Quality-control figures, saved next to the data like clark2025's sync/."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from raphael.style import styled


@styled
def plane_figure(plane, index, out_dir, n_traces=30):
    """Mean image with ROI centroids, and a sample of traces (dF/F, else
    denoised, else F)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    image = next(iter(plane.images.values()), None)
    accepted = (
        plane.rois['accepted'].to_numpy() if plane.rois is not None
        else np.ones(len(plane.masks), bool)
    )
    centroids = np.array([
        np.average(np.asarray(m)[:, :2], axis=0, weights=np.asarray(m)[:, 2])
        for m in plane.masks
    ]).reshape(-1, 2)
    name, traces = next((n, getattr(plane, a)) for n, a in (
        ('dF/F', 'dff'), ('denoised (C)', 'denoised'), ('F', 'F'))
        if getattr(plane, a) is not None)

    fig, (ax_img, ax_tr) = plt.subplots(1, 2, figsize=(10, 3.5),
                                        gridspec_kw={'width_ratios': [1.4, 1]})
    if image is not None:
        lo, hi = np.nanpercentile(image, [1, 99.5])
        ax_img.imshow(image, cmap='gray', vmin=lo, vmax=hi)
    ax_img.scatter(*centroids[accepted].T, s=2, c='tab:green', label='accepted')
    ax_img.scatter(*centroids[~accepted].T, s=2, c='tab:red', label='rejected')
    ax_img.set_title(f'plane {index}: {accepted.sum()} accepted / {len(accepted)} ROIs')
    ax_img.legend(loc='lower right', fontsize=7, markerscale=4)
    ax_img.axis('off')

    t = np.asarray(plane.timestamps)
    show = np.flatnonzero(accepted)[:n_traces]
    window = slice(0, min(len(t), int(plane.rate * 120)))
    for k, roi in enumerate(show):
        y = np.asarray(traces[window, roi], dtype=float)
        scale = np.nanmax(np.abs(y)) or 1.0
        ax_tr.plot(t[window] - t[0], k + 0.9 * y / scale, lw=0.5, c='k')
    ax_tr.set_xlabel('time (s)')
    ax_tr.set_ylabel('ROI')
    ax_tr.set_title(f'{name}, first 120 s')
    fig.tight_layout()
    path = out_dir / f'plane{index}_rois.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
