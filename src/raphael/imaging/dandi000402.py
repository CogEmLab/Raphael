"""Read the MICrONS functional data (DANDI 000402) as pipeline input.

Each scan file holds 8 two-photon planes (raw int16 movies, 40,000 frames
of 248 x 440 px at ~6.3 Hz), the authors' segmentation (pixel-weighted
masks labelled soma/artifact), raw fluorescence per ROI, summary images,
and treadmill, pupil and stimulus tables. It is our test bed: re-exporting
its processed data exercises the output template end to end, and its raw
movies are the input for our own motion correction, denoising and
segmentation.

Dataset: https://dandiarchive.org/dandiset/000402
Paper: MICrONS Consortium, doi:10.1101/2021.07.28.454025
"""

import re
from datetime import datetime

import h5py
import numpy as np
import pandas as pd

from raphael.imaging.layout import Session
from raphael.imaging.nwb import Plane

ACQ = 'acquisition'
OPHYS = 'processing/ophys'


def session_for(path):
    """sub-17797_ses-7-scan-4_... -> Session('17797', '7', 'scan4')."""
    match = re.search(r'sub-(\d+)_ses-(\d+)-scan-(\d+)', str(path))
    if not match:
        raise ValueError(f'not a DANDI 000402 file name: {path}')
    mouse, day, scan = match.groups()
    return Session(mouse, day, f'scan{scan}')


def plane_names(f):
    return sorted(
        (k for k in f[ACQ] if k.startswith('TwoPhotonSeries')),
        key=lambda k: int(k.removeprefix('TwoPhotonSeries')),
    )


def metadata(f):
    general = f['general']
    text = lambda p: _text(f[p][()])
    plane = general['optophysiology/ImagingPlane1']
    return {
        'session_description': text('session_description'),
        'session_start_time': datetime.fromisoformat(text('session_start_time')),
        'experiment_description': text('general/experiment_description'),
        'related_publications': [_text(p) for p in general['related_publications'][()]],
        'subject': {
            'subject_id': text('general/subject/subject_id'),
            'age': text('general/subject/age'),
            'sex': text('general/subject/sex'),
            'species': text('general/subject/species'),
        },
        'indicator': _text(plane['indicator'][()]),
        'location': _text(plane['location'][()]),
        'microscope': _text(general['devices/Microscope'].attrs.get('description', '')),
    }


def raw_movie(f, plane=1):
    """The raw movie as an h5py dataset (frames x height x width, int16).

    Lazy: slicing reads only what is asked for. Chunks are 2254 frames x
    14 x 14 px, so reading whole frames is slow; read blocks of frames.
    """
    return f[f'{ACQ}/TwoPhotonSeries{plane}/data']


def read_plane(f, plane, max_frames=None):
    """The authors' processed output for one plane, as a `Plane`."""
    series = f[f'{ACQ}/TwoPhotonSeries{plane}']
    frames = slice(None, max_frames)
    timestamps = series['timestamps'][frames]
    fov_m = series['field_of_view'][()]          # (height, width) in metres
    width, height = series['dimension'][()]      # stored as (width, height)
    seg = f[f'{OPHYS}/ImageSegmentation/PlaneSegmentation{plane}']
    image_mask = seg['image_mask']
    masks = []
    for start in range(0, image_mask.shape[0], 48):  # one chunk of ROIs at a time
        for dense in image_mask[start:start + 48]:
            y, x = np.nonzero(dense)
            masks.append(np.column_stack([x, y, dense[y, x]]).astype(np.float32))
    mask_type = np.array([_text(t) for t in seg['mask_type'][()]])
    rois = pd.DataFrame({
        'source_id': seg['id'][()],
        'mask_type': mask_type,
        'accepted': mask_type == 'soma',
    })
    images_group = f[f'{OPHYS}/SegmentationImages{plane}']
    return Plane(
        timestamps=timestamps,
        rate=float(1 / np.median(np.diff(timestamps))),
        masks=masks,
        image_shape=(int(height), int(width)),
        F=f[f'{OPHYS}/Fluorescence/RoiResponseSeries{plane}/data'][frames],
        grid_spacing_um=(fov_m[0] / height * 1e6, fov_m[1] / width * 1e6),
        depth_um=_depth_um(series),
        description=_text(series.attrs.get('description', '')),
        images={name: images_group[name][()] for name in images_group},
        rois=rois,
    )


def behaviour(f, max_time=None):
    """Treadmill speed (as clark2025's `S`, cm/s), pupil and eye position."""
    def clip(group):
        t = group['timestamps'][()]
        keep = slice(None) if max_time is None else t <= max_time
        return group['data'][()][keep] * float(group['data'].attrs.get('conversion', 1.0)), t[keep]

    speed, t_speed = clip(f[f'{ACQ}/treadmill_velocity'])
    series = {
        # Stored as m/s with conversion 0.01, i.e. the raw values are cm/s.
        # Speed is unsigned in clark2025; velocity keeps the sign.
        'S': (np.abs(speed) * 100, 'cm/s', t_speed, 'Treadmill speed.'),
        'treadmill_velocity': (speed * 100, 'cm/s', t_speed,
                               'Treadmill velocity, rostral-caudal, signed.'),
    }
    for name in ('pupil_major_radius', 'pupil_minor_radius'):
        data, t = clip(f[f'{ACQ}/PupilTracking/{name}'])
        series[name] = (data, 'px', t, name.replace('_', ' ') + ' of the fitted ellipse.')
    eye, t_eye = clip(f[f'{ACQ}/EyeTracking/eye_position'])
    series['eye_x'] = (eye[:, 0], 'px', t_eye, 'Pupil centre, x.')
    series['eye_y'] = (eye[:, 1], 'px', t_eye, 'Pupil centre, y.')
    return series


def stimulus_intervals(f, max_time=None):
    out = {}
    for name in f['intervals']:
        group = f[f'intervals/{name}']
        columns = [k for k in group if k != 'id' and group[k].ndim == 1
                   and len(group[k]) == len(group['id'])]
        table = pd.DataFrame({
            k: [_text(v) for v in group[k][()]] if group[k].dtype.kind == 'O' else group[k][()]
            for k in columns
        })
        if max_time is not None:
            table = table[table['start_time'] <= max_time]
        out[name] = (table, _text(group.attrs.get('description', name)))
    return out


def open_file(path):
    return h5py.File(path, 'r')


def _depth_um(series):
    match = re.search(r'([\d.]+) meters depth', _text(series.attrs.get('description', '')))
    return float(match.group(1)) * 1e6 if match else float('nan')


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)
