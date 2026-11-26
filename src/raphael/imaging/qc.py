"""Quality-control figures, saved next to the data like clark2025's sync/."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from raphael.style import styled

# The mean image shows the field of view (vessels, lens edge); the correlation
# image shows where active cells are. Both are drawn, so a session's ROIs can be
# checked against each and two pipelines compared by eye.
IMAGES = ('mean', 'correlation')


@styled
def plane_figure(plane, index, out_dir, n_traces=30):
    """The mean and correlation images with ROI centroids, and a sample of
    traces (dF/F, else denoised, else F). A missing image leaves its panel
    blank with a note; a plane with neither shows the first image it has."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(IMAGES)
    if not any(n in plane.images for n in names) and plane.images:
        names = [next(iter(plane.images))]
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

    height, width = plane.image_shape
    image_w = 3.6 * width / height
    fig, axes = plt.subplots(
        1, len(names) + 1, figsize=(image_w * len(names) + 4.5, 3.9),
        gridspec_kw={'width_ratios': [image_w] * len(names) + [4.5]})
    for ax, image_name in zip(axes, names, strict=False):
        image = plane.images.get(image_name)
        if image is not None:
            lo, hi = np.nanpercentile(image, [1, 99.5])
            ax.imshow(image, cmap='gray', vmin=lo, vmax=hi)
        else:
            ax.imshow(np.zeros(plane.image_shape), cmap='gray', vmin=0, vmax=1)
            ax.text(width / 2, height / 2, f'no {image_name} image', color='w',
                    ha='center', va='center', fontsize=8)
        ax.scatter(*centroids[accepted].T, s=2, c='tab:green', label='accepted')
        ax.scatter(*centroids[~accepted].T, s=2, c='tab:red', label='rejected')
        ax.set_title(f'{image_name.replace("_", " ")} image', fontsize=9)
        ax.axis('off')
    axes[0].legend(loc='lower right', fontsize=7, markerscale=4, frameon=True,
                   facecolor='w', framealpha=0.85)
    fig.suptitle(f'plane {index}: {accepted.sum()} accepted / {len(accepted)} ROIs',
                 fontsize=10)

    ax_tr = axes[-1]
    t = np.asarray(plane.timestamps)
    show = np.flatnonzero(accepted)[:n_traces]
    window = slice(0, min(len(t), int(plane.rate * 120)))
    for k, roi in enumerate(show):
        y = np.asarray(traces[window, roi], dtype=float)
        scale = np.nanmax(np.abs(y)) or 1.0
        ax_tr.plot(t[window] - t[0], k + 0.9 * y / scale, lw=0.5, c='k')
    ax_tr.set_xlabel('time (s)')
    ax_tr.set_ylabel('ROI')
    ax_tr.set_title(f'{name}, first 120 s', fontsize=9)
    fig.tight_layout()
    path = out_dir / f'plane{index}_rois.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
