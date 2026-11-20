"""suite2p as a pipeline stage: raw movie -> registered, segmented `Plane`.

suite2p (torch, so it runs on a GPU) does motion correction, ROI detection,
extraction, neuropil estimation, deconvolution and cell classification.
dF/F is computed by `traces`, not suite2p, so every pipeline baselines the
same way. Needs the `gpu` extra (`uv sync --frozen --extra gpu`).

suite2p merges its settings one level deep only: passing
{'registration': {'nonrigid': False}} would drop every other registration
setting. `settings_for` therefore merges overrides recursively into the
defaults.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from raphael.imaging import traces
from raphael.imaging.nwb import Plane

ROI_COLUMNS = ['npix', 'radius', 'compact', 'aspect_ratio', 'snr', 'skew', 'std']


def _deep_update(base, new):
    out = copy.deepcopy(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def settings_for(rate, device, overrides=None):
    """suite2p's defaults, with the frame rate, device and any overrides."""
    from suite2p.parameters import default_settings

    settings = default_settings()
    settings.update(fs=float(rate), torch_device=device)
    return _deep_update(settings, overrides or {})


def blank_frames(movie, h5_key='data', fraction=0.3, chunk=1000):
    """Frames whose mean intensity is below `fraction` of the movie's median
    frame mean: shutter closed, laser off or acquisition gaps. Registration
    cannot align them (suite2p pushes them to its maximum shift), and left in
    they degrade detection. HDF5 movies only; returns indices."""
    import h5py

    with h5py.File(movie, 'r') as f:
        data = f[h5_key]
        means = np.concatenate([data[i:i + chunk].mean(axis=(1, 2))
                                for i in range(0, len(data), chunk)])
    return np.flatnonzero(means < fraction * np.median(means))


def run(movie, work, rate, device, overrides=None, h5_key='data', bad_frames=None):
    """Run suite2p on one single-plane movie; returns the plane folder.

    movie: HDF5 (frames x height x width under `h5_key`) or TIFF.
    bad_frames: frame indices suite2p leaves out of registration cropping and
    detection (it reads them from bad_frames.npy in its save folder); the
    frames stay in the timeline.
    """
    from suite2p import run_s2p

    movie, work = Path(movie), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    bad_path = work / 'bad_frames.npy'
    if bad_frames is not None and len(bad_frames):
        np.save(bad_path, np.asarray(bad_frames, dtype=int))
    elif bad_path.exists():
        bad_path.unlink()
    db = {
        'data_path': [str(movie.parent)],
        'file_list': [str(movie)],
        'input_format': 'h5' if movie.suffix in ('.h5', '.hdf5') else 'tif',
        'h5py_key': h5_key,
        'save_path0': str(work),
        'fast_disk': str(work),
        'nplanes': 1,
        'nchannels': 1,
        'keep_movie_raw': False,
    }
    run_s2p(db=db, settings=settings_for(rate, device, overrides))
    return work / 'suite2p' / 'plane0'


def _full_frame(image, reg, shape):
    """suite2p crops some images to the valid registration range; pad back."""
    if image.shape == tuple(shape):
        return image
    full = np.zeros(shape, dtype=np.float32)
    y0, y1 = reg['yrange']
    x0, x1 = reg['xrange']
    full[y0:y1, x0:x1] = image
    return full


def to_plane(folder, rate, settings, description='suite2p output.', bad_frames=None):
    """Read a suite2p plane folder into a `Plane`. `settings` is the imaging
    config (configs/imaging.yaml), which sets the neuropil and dF/F
    parameters. dF/F is NaN at `bad_frames`, and they are left out of the
    baseline."""
    folder = Path(folder)

    def load(name):
        return np.load(folder / f'{name}.npy', allow_pickle=True)

    F, Fneu, spks = load('F').T, load('Fneu').T, load('spks').T   # frames x ROIs
    iscell, stat = load('iscell'), load('stat')
    reg, det = load('reg_outputs').item(), load('detect_outputs').item()
    shape = reg['meanImg'].shape

    proc = settings['processing']
    corrected = traces.neuropil_corrected(F, Fneu, proc['neuropil_coefficient'])
    bad = np.zeros(len(F), bool)
    if bad_frames is not None:
        bad[np.asarray(bad_frames, dtype=int)] = True
    if bad.any():  # interpolate across them for the baseline, then blank
        good = np.flatnonzero(~bad)
        corrected = corrected.copy()
        for i in range(corrected.shape[1]):
            corrected[bad, i] = np.interp(np.flatnonzero(bad), good, corrected[good, i])
    rois = pd.DataFrame({
        'accepted': iscell[:, 0] > 0.5,
        'cell_probability': iscell[:, 1],
        'centre_y': [s['med'][0] for s in stat],
        'centre_x': [s['med'][1] for s in stat],
        **{c: [s[c] for s in stat] for c in ROI_COLUMNS if c in stat[0]},
    })
    images = {'mean': reg['meanImg'], 'mean_enhanced': reg['meanImgE']}
    for name, key in (('max_projection', 'max_proj'), ('correlation', 'Vcorr')):
        if key in det:
            images[name] = _full_frame(det[key], reg, shape)
    dff = traces.dff(corrected, rate, proc['dff']['percentile'], proc['dff']['window_s'])
    dff[bad] = np.nan
    return Plane(
        timestamps=np.arange(len(F)) / rate,
        rate=float(rate),
        masks=[np.column_stack([s['xpix'], s['ypix'], s['lam']]).astype(np.float32)
               for s in stat],
        image_shape=shape,
        F=F,
        neuropil=Fneu,
        dff=dff,
        deconvolved=spks,
        shifts=np.column_stack([reg['xoff'], reg['yoff']]).astype(np.float32),
        images=images,
        rois=rois,
        description=description,
    )
