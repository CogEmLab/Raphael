"""Write processed imaging and behaviour as publishable NWB, clark2025-style.

Every pipeline (suite2p, CaImAn, a denoiser in front of either, or a
re-export of someone else's processed data) fills in `Plane` objects; the
writers here are the only code that knows the NWB layout. One file per
session holds every plane, so a DANDI deposit is one file per session.

Processed ophys file:

    general/        subject, lab, procedures, keywords, publications
    devices/        Microscope
    optophysiology/ ImagingPlane{i}
    processing/ophys/
        SummaryImages{i}     mean / correlation / max projection
        ImageSegmentation/PlaneSegmentation{i}
                             pixel masks + one column per ROI property
        Fluorescence/RoiResponseSeries{i}    raw F
        Neuropil/RoiResponseSeries{i}        if the pipeline gives it
        DfOverF/RoiResponseSeries{i}
        Deconvolved/RoiResponseSeries{i}     if spike inference was run
        MotionCorrection/xy_shifts{i}        rigid shifts in pixels
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pynapple as nap
import pynwb
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb.base import Images
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from pynwb.image import GrayscaleImage
from pynwb.ophys import (
    DfOverF,
    Fluorescence,
    ImageSegmentation,
    OpticalChannel,
)

ROI_RESPONSES = [
    ('Fluorescence', 'F', Fluorescence, 'Raw fluorescence, mean over the ROI mask.'),
    ('Neuropil', 'neuropil', Fluorescence, 'Neuropil fluorescence around each ROI.'),
    ('DfOverF', 'dff', DfOverF, '(F - F0) / F0, F0 a running low percentile.'),
    ('Deconvolved', 'deconvolved', Fluorescence, 'Deconvolved activity (spike inference).'),
]


@dataclass
class Plane:
    """One imaging plane's processed output. Arrays are frames x ROIs."""

    timestamps: np.ndarray
    rate: float
    masks: list                       # per ROI: (n_pixels, 3) array of x, y, weight
    image_shape: tuple                # (height, width)
    F: np.ndarray
    grid_spacing_um: tuple = (1.0, 1.0)
    depth_um: float = float('nan')
    description: str = ''
    neuropil: np.ndarray | None = None
    dff: np.ndarray | None = None
    deconvolved: np.ndarray | None = None
    shifts: np.ndarray | None = None  # frames x 2, pixels
    images: dict = field(default_factory=dict)   # name -> 2D array
    rois: pd.DataFrame | None = None  # one row per ROI; must hold `accepted`


def _compressed(data):
    return H5DataIO(np.asarray(data), compression='gzip', chunks=True)


SUBJECT_FIELDS = {'subject_id', 'age', 'date_of_birth', 'sex', 'strain', 'genotype',
                  'weight', 'description', 'species'}


def parse_start_time(value, timezone='UTC'):
    """ISO 8601 date or date-time (or a datetime/date) -> timezone-aware
    datetime. Values without a UTC offset are read as local time in
    `timezone`."""
    if not isinstance(value, datetime):
        value = datetime.fromisoformat(str(value))
    return value if value.tzinfo else value.replace(tzinfo=ZoneInfo(timezone))


def subject_fields(subject, timezone='UTC'):
    """Check per-mouse subject metadata (as passed in job params) and convert
    it for pynwb. age is an ISO 8601 duration, e.g. P84D for 84 days."""
    subject = dict(subject or {})
    unknown = set(subject) - SUBJECT_FIELDS
    if unknown:
        raise ValueError(f'unknown subject fields {sorted(unknown)}; allowed: '
                         f'{sorted(SUBJECT_FIELDS)}')
    if 'age' in subject and not str(subject['age']).startswith('P'):
        raise ValueError(f'age must be an ISO 8601 duration such as P84D, got {subject["age"]!r}')
    if 'sex' in subject and subject['sex'] not in ('M', 'F', 'U', 'O'):
        raise ValueError(f'sex must be M, F, U or O, got {subject["sex"]!r}')
    if 'date_of_birth' in subject:
        subject['date_of_birth'] = parse_start_time(subject['date_of_birth'], timezone)
    return subject


def new_nwbfile(settings, session, session_description, session_start_time=None,
                subject=None, notes=None):
    """NWBFile carrying the publication metadata from configs/imaging.yaml."""
    meta = settings['nwb']
    subject_defaults = dict(settings['subject'])
    subject_defaults.update(subject or {})
    subject_defaults.setdefault('subject_id', f'M{session.mouse}')
    subject_defaults.setdefault('description', f'Mouse {subject_defaults["subject_id"]}.')
    return pynwb.NWBFile(
        session_description=session_description,
        identifier=str(uuid4()),
        session_start_time=session_start_time or datetime.now(UTC),
        session_id=session.stem.removeprefix('sub-'),
        experimenter=meta.get('experimenter'),
        lab=meta.get('lab'),
        institution=meta.get('institution'),
        keywords=meta.get('keywords') or None,
        related_publications=meta.get('related_publications') or None,
        experiment_description=meta.get('experiment_description'),
        surgery=meta.get('surgery'),
        virus=meta.get('virus'),
        pharmacology=meta.get('pharmacology'),
        data_collection=meta.get('data_collection'),
        notes=json.dumps(notes) if isinstance(notes, dict) else notes,
        subject=Subject(**subject_defaults),
    )


def build_ophys_nwb(nwbfile, planes, settings, indicator=None, location=None):
    scope = settings['microscope']
    plane_meta = settings['imaging_plane']
    device = nwbfile.create_device(
        name=scope['name'],
        description='; '.join(
            str(scope[k]) for k in ('description', 'manufacturer') if scope.get(k)
        ) or None,
    )
    ophys = nwbfile.create_processing_module('ophys', 'Processed two-photon data.')
    segmentation = ImageSegmentation()
    ophys.add(segmentation)
    containers = {}

    for i, plane in enumerate(planes, start=1):
        imaging_plane = nwbfile.create_imaging_plane(
            name=f'ImagingPlane{i}',
            optical_channel=OpticalChannel(
                name='OpticalChannel',
                description='green channel',
                emission_lambda=plane_meta['emission_lambda'],
            ),
            description=plane.description or f'Imaging plane {i}.',
            device=device,
            excitation_lambda=plane_meta['excitation_lambda'],
            imaging_rate=float(plane.rate),
            indicator=indicator or plane_meta['indicator'],
            location=location or plane_meta['location'],
            grid_spacing=[s * 1e-6 for s in plane.grid_spacing_um],
            grid_spacing_unit='meters',
            origin_coords=[0.0, 0.0, plane.depth_um * 1e-6],
            origin_coords_unit='meters',
        )

        if plane.images:
            ophys.add(Images(
                name=f'SummaryImages{i}',
                description=f'Summary images for plane {i}.',
                images=[
                    GrayscaleImage(name=name, data=np.asarray(image, dtype=np.float32),
                                   description=f'{name} image of plane {i}.')
                    for name, image in plane.images.items()
                ],
            ))

        seg = segmentation.create_plane_segmentation(
            name=f'PlaneSegmentation{i}',
            description=f'ROIs in plane {i}.',
            imaging_plane=imaging_plane,
        )
        rois = plane.rois if plane.rois is not None else pd.DataFrame(
            {'accepted': np.ones(len(plane.masks), bool)}
        )
        for column in rois.columns:
            seg.add_column(name=column, description=_describe(column))
        for row, mask in zip(rois.itertuples(index=False), plane.masks, strict=True):
            seg.add_roi(
                pixel_mask=[tuple(p) for p in np.asarray(mask).tolist()],
                **{c: _scalar(v) for c, v in zip(rois.columns, row, strict=True)},
            )
        region = seg.create_roi_table_region(
            region=list(range(len(plane.masks))), description=f'all ROIs in plane {i}'
        )

        first = None
        for container_name, attr, cls, description in ROI_RESPONSES:
            data = getattr(plane, attr)
            if data is None:
                continue
            if container_name not in containers:
                containers[container_name] = cls(name=container_name)
                ophys.add(containers[container_name])
            series = containers[container_name].create_roi_response_series(
                name=f'RoiResponseSeries{i}',
                data=_compressed(np.asarray(data, dtype=np.float32)),
                rois=region,
                unit='n.a.',
                description=description,
                # Share one timestamps dataset per plane.
                timestamps=first if first is not None else np.asarray(plane.timestamps),
            )
            if first is None:
                first = series

        if plane.shifts is not None:
            ophys.add(pynwb.TimeSeries(
                name=f'xy_shifts{i}',
                description=f'Rigid motion-correction shifts (x, y) for plane {i}.',
                data=_compressed(plane.shifts),
                unit='pixels',
                timestamps=first if first is not None else np.asarray(plane.timestamps),
            ))
    return nwbfile


def build_beh_nwb(nwbfile, series, intervals=None, settings=None):
    """Behaviour file: one TimeSeries per variable, plus interval tables.

    `series`: name -> (data, unit, timestamps, description).
    `intervals`: name -> (DataFrame with start_time, stop_time, ..., description).
    Adds clark2025's `moving`/`stationary` epochs when `S` (cm/s) is present.
    """
    for name, (data, unit, timestamps, description) in series.items():
        nwbfile.add_acquisition(pynwb.TimeSeries(
            name=name, data=_compressed(data), unit=unit,
            timestamps=_compressed(timestamps), description=description,
        ))
    for name, (table, description) in (intervals or {}).items():
        _add_intervals(nwbfile, name, table, description)

    if 'S' in series and settings is not None:
        data, _, timestamps, _ = series['S']
        speed = nap.Tsd(t=np.asarray(timestamps), d=np.asarray(data))
        threshold = settings['behaviour']['moving_threshold_cm_s']
        for name, method in [('moving', 'above'), ('stationary', 'below')]:
            support = speed.threshold(threshold, method=method).time_support
            support = support.drop_short_intervals(settings['behaviour']['min_epoch_s'])
            table = pd.DataFrame({'start_time': support.start, 'stop_time': support.end - 1e-5})
            _add_intervals(nwbfile, name, table,
                           f'Epochs when the mouse is {name}, {threshold} cm/s threshold.')
    return nwbfile


def write(nwbfile, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pynwb.NWBHDF5IO(str(path), 'w') as io:
        io.write(nwbfile)
    return path


def export_tables(planes, traces_path, rois_path):
    """dF/F of every plane as one TsdFrame, and the ROI table as CSV.

    Columns are global ROI indices (plane order, then ROI order), matching
    the `roi` column of the CSV. Planes are sampled at slightly different
    times in multi-plane scans; the TsdFrame uses plane 1's timestamps,
    and `rois.csv` keeps each ROI's plane so exact times can be recovered
    from the NWB file.
    """
    tables, traces = [], []
    start = 0
    for i, plane in enumerate(planes, start=1):
        n = len(plane.masks)
        table = plane.rois.copy() if plane.rois is not None else pd.DataFrame(index=range(n))
        centroids = np.array([
            np.average(np.asarray(m)[:, :2], axis=0, weights=np.asarray(m)[:, 2])
            for m in plane.masks
        ]).reshape(-1, 2)
        table.insert(0, 'roi', np.arange(start, start + n))
        table.insert(1, 'plane', i)
        table.insert(2, 'plane_roi', np.arange(n))
        table['centroid_x_px'] = centroids[:, 0]
        table['centroid_y_px'] = centroids[:, 1]
        tables.append(table)
        traces.append(plane.dff if plane.dff is not None else plane.F)
        start += n
    rois = pd.concat(tables, ignore_index=True)
    rois_path.parent.mkdir(parents=True, exist_ok=True)
    rois.to_csv(rois_path, index=False)
    length = min(len(t) for t in traces)
    nap.TsdFrame(
        t=np.asarray(planes[0].timestamps)[:length],
        d=np.concatenate([np.asarray(t[:length], dtype=np.float32) for t in traces], axis=1),
        columns=rois['roi'].to_numpy(),
    ).save(str(traces_path))
    return rois


def _add_intervals(nwbfile, name, table, description):
    # NWB cannot store an empty table (e.g. a stimulus never shown in a
    # clipped recording, or no movement at all).
    if not len(table):
        return
    epochs = TimeIntervals(name=name, description=description)
    extra = [c for c in table.columns if c not in ('start_time', 'stop_time')]
    for column in extra:
        epochs.add_column(name=column, description=_describe(column))
    for row in table.itertuples(index=False):
        values = row._asdict()
        epochs.add_row(**{k: _scalar(v) for k, v in values.items()})
    nwbfile.add_time_intervals(epochs)


def _describe(column):
    return {
        'accepted': 'True if the ROI passed curation and is used in analysis.',
        'mask_type': 'Classification of the mask (e.g. soma, artifact).',
    }.get(column, column.replace('_', ' '))


def _scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode()
    return value
