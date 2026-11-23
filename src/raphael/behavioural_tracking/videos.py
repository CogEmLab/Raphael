"""Inventory, split and standardise raw videos for DeepLabCut.

Videos can stay inside a zip: nothing here extracts a whole archive. Each
member is extracted to a scratch file, read or transcoded, and deleted
before the next one, so free disk only needs to hold the largest video.

The train/test split holds out whole recording days (per rig), because
frames of one day share lighting and camera placement: a finer split would
leak appearance from train into test.
"""

import math
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

INVENTORY = 'inventory.csv'
ROIS = 'rois.csv'
SPLIT = 'split.csv'


def video_id(name):
    """A video's id is its filename stem; DLC names label folders by it."""
    return Path(name).stem


def parse_name(name, day_pattern=r'(20\d{6})', rig_pattern=None):
    """A video's id, recording day and rig from its file name.

    `day_pattern`: a regex whose first group is the day (default: an
    8-digit date such as 20240315). `rig_pattern`: a regex whose first group
    names the rig/camera, or a dict {rig: substring}; without it every
    video is rig 'all'. Projects with richer names add their own fields."""
    stem = video_id(name)
    day = re.search(day_pattern, stem) if day_pattern else None
    if isinstance(rig_pattern, dict):
        rig = next((r for r, s in rig_pattern.items() if s in stem), 'unknown')
    elif rig_pattern:
        m = re.search(rig_pattern, stem)
        rig = m.group(1) if m else 'unknown'
    else:
        rig = 'all'
    return {'video_id': stem, 'rig': rig, 'date': day.group(1) if day else ''}


def list_members(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        return [
            info
            for info in archive.infolist()
            if info.filename.lower().endswith(('.mp4', '.avi'))
            and '/._' not in info.filename
            and '__MACOSX' not in info.filename
        ]


def build_inventory(zip_path, day_pattern=r'(20\d{6})', rig_pattern=None):
    """One row per video in a zip: member name, size, id, day and rig."""
    members = list_members(zip_path)
    return pd.DataFrame([{'member': info.filename, 'bytes': info.file_size,
                          **parse_name(info.filename, day_pattern, rig_pattern)}
                         for info in members])


def extract_member(zip_path, member, scratch):
    """Extract one member to `scratch` and return its path."""
    with zipfile.ZipFile(zip_path) as archive:
        target = Path(scratch) / Path(member).name
        with archive.open(member) as src, open(target, 'wb') as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    return target


def probe(path, thumbnail=None):
    """Width, height, fps and frame count; optionally save the middle frame."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    info = {
        'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': float(capture.get(cv2.CAP_PROP_FPS)),
        'frames': int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info['minutes'] = (
        info['frames'] / info['fps'] / 60 if info['fps'] else np.nan
    )
    if thumbnail is not None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, info['frames'] // 2)
        ok, frame = capture.read()
        if ok:
            cv2.imwrite(str(thumbnail), frame)
    capture.release()
    return info


def assign_split(inventory, test_fraction=0.2, seed=0):
    """Hold out whole recording days, stratified by rig.

    Adjacent frames of one video, and videos of one day, share lighting and
    camera placement; splitting any finer than a day would leak appearance
    from train into test and overstate how well the model transfers to the
    unlabelled session videos.
    """
    rng = np.random.default_rng(seed)
    split = pd.Series('train', index=inventory.index)
    for _, group in inventory.groupby('rig'):
        days = np.array(sorted(group['date'].unique()))
        rng.shuffle(days)
        n_test = math.ceil(test_fraction * len(days))
        # A rig with a single day cannot contribute to both sides.
        if len(days) < 2:
            n_test = 0
        held_out = set(days[:n_test])
        split[group.index[group['date'].isin(held_out)]] = 'test'
    return split


def ffmpeg_binary():
    found = shutil.which('ffmpeg')
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError(
            'ffmpeg not found: install it (macOS: brew install ffmpeg; Windows: '
            'winget install ffmpeg), or install Raphael with the tracking extra, '
            'whose imageio-ffmpeg ships one.'
        ) from error


def output_geometry(roi, max_side):
    """Even output width/height for a crop scaled to fit max_side."""
    _, _, width, height = roi
    scale = min(1.0, max_side / max(width, height))
    return (
        int(width * scale) // 2 * 2,
        int(height * scale) // 2 * 2,
    )


def transcode(src, dst, roi, native_fps, max_fps, max_side, crf):
    """Crop, drop to at most max_fps, scale, and write H.264."""
    x, y, width, height = (int(v) for v in roi)
    out_w, out_h = output_geometry((x, y, width, height), max_side)
    fps = min(native_fps, max_fps) if native_fps else max_fps
    filters = f'crop={width}:{height}:{x}:{y},fps={fps},scale={out_w}:{out_h}'
    command = [
        ffmpeg_binary(),
        '-hide_banner',
        '-loglevel',
        'error',
        '-y',
        '-i',
        str(src),
        '-vf',
        filters,
        '-c:v',
        'libx264',
        '-crf',
        str(crf),
        '-pix_fmt',
        'yuv420p',
        '-an',
        str(dst),
    ]
    subprocess.run(command, check=True)
    return {'out_width': out_w, 'out_height': out_h, 'out_fps': fps}


def scratch_dir(root):
    path = Path(root) / '_scratch'
    path.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(dir=path)


def load_rois(root):
    path = Path(root) / ROIS
    if not path.exists():
        return pd.DataFrame(columns=['video_id', 'x', 'y', 'w', 'h'])
    return pd.read_csv(path)


def roi_for(rois, row):
    """The saved crop for a video, or the full frame if none was drawn."""
    hit = rois[rois['video_id'] == row['video_id']]
    if len(hit):
        r = hit.iloc[-1]
        return (int(r['x']), int(r['y']), int(r['w']), int(r['h']))
    return (0, 0, int(row['width']), int(row['height']))
