"""Inscopix nVista/nVoke recordings (`.isxd` movies), read without Inscopix's
software.

An `.isxd` movie is the frames, then a JSON description, then the JSON's
length as a little-endian uint64 (and one NUL byte before the JSON). Frames
are uint16, row-major; with `hasFrameHeaderFooter` each frame carries two
extra rows before and two after the image (acquisition counters, not
pixels), which `Recording.frames` leaves out. The JSON gives the size, the
frame period, the start time (UTC), the dropped and cropped frames, and,
under `extraProperties`, everything the acquisition software recorded:
microscope gain, LED power, focus, lens, the animal fields typed in at
acquisition, and the GPIO routing.

    rec = inscopix.open_isxd('..._video.isxd')
    rec.frames.shape            # (12701, 800, 1280), a read-only memmap
    rec.rate, rec.start         # 59.99 Hz, aware datetime (local time zone)
    rec.timestamps              # s from the first frame
    inscopix.downsample(rec, 'movie.tif', spatial=2, temporal=2)

Checked against Inscopix's own TIFF export of the same recording (equal
pixel for pixel). The start time: nVista writes the acquisition computer's
wall-clock time as `secsSinceEpoch`, with `utcOffset` 0 (the file name's
time, and minutes after session.json's true-UTC `date`), so it is read as
local time in `timezone` (the imaging config's), not as UTC. The `.gpio` file next to the movie (sync lines, camera
triggers) is a different, packetised format, not read here.
"""

import json
import struct
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from functools import cached_property
from pathlib import Path

import numpy as np

HEADER_ROWS = 2
FOOTER_ROWS = 2


class InscopixError(ValueError):
    pass


def read_metadata(path):
    """The JSON description at the end of an `.isxd` file, and the number of
    bytes before it (the frame data)."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, 'rb') as f:
        f.seek(size - 8)
        n = struct.unpack('<Q', f.read(8))[0]
        if n <= 0 or n + 9 > size:
            raise InscopixError(f'{path} does not end in an isxd JSON description')
        f.seek(size - 8 - n - 1)
        text = f.read(n + 1)
    try:
        meta = json.loads(text.strip(b'\x00'))
    except json.JSONDecodeError as e:
        raise InscopixError(f'{path}: unreadable isxd description ({e})') from e
    return meta, size - 8 - n - 1


def _fraction(value):
    return Fraction(int(value['num']), int(value['den']))


@dataclass(frozen=True)
class Recording:
    """One `.isxd` movie. `frames` is (time, y, x) uint16 on disk."""
    path: Path
    meta: dict
    data_bytes: int
    timezone: str = 'Europe/Vienna'

    @property
    def width(self):
        return int(self.meta['spacingInfo']['numPixels']['x'])

    @property
    def height(self):
        return int(self.meta['spacingInfo']['numPixels']['y'])

    @property
    def n_frames(self):
        return int(self.meta['timingInfo']['numTimes'])

    @property
    def period(self):
        """Frame period in s, exactly as recorded."""
        return float(_fraction(self.meta['timingInfo']['period']))

    @property
    def rate(self):
        return 1 / self.period

    @property
    def start(self):
        """Time of the first frame, timezone-aware. With `utcOffset` 0 (what
        nVista writes) the stored seconds are the acquisition computer's
        wall clock, read in `timezone`; otherwise they are taken as UTC."""
        from zoneinfo import ZoneInfo

        start = self.meta['timingInfo']['start']
        clock = datetime(1970, 1, 1) + timedelta(seconds=float(_fraction(start['secsSinceEpoch'])))
        if not start.get('utcOffset'):
            return clock.replace(tzinfo=ZoneInfo(self.timezone))
        return clock.replace(tzinfo=UTC).astimezone(ZoneInfo(self.timezone))

    @property
    def timestamps(self):
        """Frame times in s from the first frame."""
        return np.arange(self.n_frames) * self.period

    @property
    def dropped(self):
        """Indices of frames the acquisition dropped (their pixels are not
        real data)."""
        return np.asarray(self.meta['timingInfo'].get('dropped') or [], dtype=int)

    @property
    def cropped(self):
        """[[first, last], ...] frame ranges cut out after acquisition."""
        return [list(map(int, r)) for r in self.meta['timingInfo'].get('cropped') or []]

    @property
    def extra(self):
        return self.meta.get('extraProperties') or {}

    @cached_property
    def frames(self):
        if self.meta.get('dataType', 0) != 0:
            raise InscopixError(f'{self.path}: only uint16 movies are supported '
                                f'(dataType {self.meta.get("dataType")})')
        rows = self.height + (HEADER_ROWS + FOOTER_ROWS
                              if self.meta.get('hasFrameHeaderFooter') else 0)
        expected = self.n_frames * rows * self.width * 2
        if expected != self.data_bytes:
            raise InscopixError(f'{self.path}: {self.data_bytes} bytes of frames, expected '
                                f'{expected} for {self.n_frames} x {rows} x {self.width} uint16')
        raw = np.memmap(self.path, np.uint16, 'r', shape=(self.n_frames, rows, self.width))
        if rows == self.height:
            return raw
        return raw[:, HEADER_ROWS:HEADER_ROWS + self.height]

    def info(self):
        """What the acquisition recorded, as flat fields for a session table
        or an NWB file's notes."""
        scope = self.extra.get('microscope') or {}
        custom = scope.get('customMetadata') or {}
        probe = self.extra.get('probe') or {}
        led = scope.get('led') or {}
        pixel = self.meta['spacingInfo'].get('pixelSize', {}).get('x')
        return {
            'file': self.path.name,
            'start': self.start.isoformat(),
            'frames': self.n_frames,
            'rate_hz': self.rate,
            'width': self.width,
            'height': self.height,
            'dropped_frames': len(self.dropped),
            'animal_id': custom.get('Animal ID') or (self.extra.get('animal') or {}).get('id'),
            'animal_sex': custom.get('Animal Sex') or (self.extra.get('animal') or {}).get('sex'),
            'animal_dob': custom.get('DOB'),
            'personnel': custom.get('Personnel'),
            'session_name': self.extra.get('name'),
            'microscope': scope.get('type'),
            'microscope_serial': scope.get('serial'),
            'gain': scope.get('gain'),
            'exposure_ms': scope.get('exp'),
            'led_power_mw_mm2': led.get('exPower'),
            'focus': scope.get('focus'),
            'probe': probe.get('name'),
            'probe_id': probe.get('id'),
            'sensor_pixel_size': float(_fraction(pixel)) if pixel else None,
        }


def open_isxd(path, timezone='Europe/Vienna'):
    path = Path(path)
    meta, data_bytes = read_metadata(path)
    return Recording(path, meta, data_bytes, timezone)


def downsample(recording, out, spatial=2, temporal=1, chunk_frames=600, dtype=np.uint16,
               max_frames=None):
    """Bin a recording `spatial` x `spatial` in space and `temporal` frames in
    time (block means; for factor 2 this equals CaImAn's area resize), and
    write it as a BigTIFF CaImAn reads. Edge pixels and trailing frames that
    don't fill a block are dropped. Dropped frames are replaced by the frame
    before them first, so they don't darken a bin. Returns the output rate
    and the time of each output frame (s from the first raw frame: the
    middle of its bin). `max_frames` stops after that many raw frames."""
    import tifffile

    frames = recording.frames
    if chunk_frames % temporal:
        chunk_frames -= chunk_frames % temporal
    n_raw = min(recording.n_frames, max_frames or recording.n_frames)
    n_out = n_raw // temporal
    h = (recording.height // spatial) * spatial
    w = (recording.width // spatial) * spatial
    dropped = set(recording.dropped.tolist())
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(out, bigtiff=True) as tif:
        for start in range(0, n_out * temporal, chunk_frames):
            stop = min(start + chunk_frames, n_out * temporal)
            block = np.asarray(frames[start:stop, :h, :w], dtype=np.float32)
            for i in sorted(d for d in dropped if start <= d < stop):
                if i > 0:
                    block[i - start] = (block[i - start - 1] if i > start
                                        else np.asarray(frames[i - 1, :h, :w], np.float32))
            t = (stop - start) // temporal
            block = block.reshape(t, temporal, h // spatial, spatial, w // spatial, spatial)
            block = block.mean(axis=(1, 3, 5))
            if np.issubdtype(dtype, np.integer):
                block = np.clip(np.rint(block), 0, np.iinfo(dtype).max)
            for frame in block.astype(dtype):
                tif.write(frame, contiguous=True)
    times = (np.arange(n_out) * temporal + (temporal - 1) / 2) * recording.period
    return recording.rate / temporal, times


def sessions(root, timezone='Europe/Vienna', raw_only=True):
    """Every `.isxd` movie under `root` as one row: path (relative to root),
    the acquisition's animal ID, local start time and date, frames, rate,
    and the acquisition fields of `Recording.info`. `raw_only` keeps movies
    straight from the microscope (a `session.json` beside them), not files
    Inscopix's software wrote after processing. Only the end of each file is
    read, so this is quick on a network share."""
    from zoneinfo import ZoneInfo

    import pandas as pd

    root = Path(root)
    rows = []
    for path in sorted(root.rglob('*.isxd')):
        if path.name.startswith('.') or (raw_only and not (path.parent / 'session.json').exists()):
            continue
        row = {'path': path.relative_to(root).as_posix(), 'folder': path.parent.name}
        try:
            rec = open_isxd(path, timezone)
            info, start, duration = rec.info(), rec.start, round(rec.n_frames * rec.period, 2)
            error = None
        except InscopixError as e:          # e.g. a recording that was never closed
            info, start, duration = _session_json_info(path), None, None
            error = str(e).replace(str(path), path.name)
            if info.get('start'):
                start = datetime.fromisoformat(info['start'])
        local = start.astimezone(ZoneInfo(timezone)) if start else None
        rows.append({**row, 'mouse': str(info.get('animal_id') or ''),
                     'date': local.date().isoformat() if local else None,
                     'start_local': local.isoformat(timespec='seconds') if local else None,
                     'duration_s': duration, **info, 'error': error})
    return pd.DataFrame(rows)


def _session_json_info(path):
    """What the acquisition's session.json says, for a movie whose own
    description is missing. Its date is when the session was opened, which
    can be minutes before the recording started."""
    session = json.loads((path.parent / 'session.json').read_text())
    scope = session.get('microscope') or {}
    custom = scope.get('customMetadata') or {}
    return {'file': path.name, 'start': session.get('date'),
            'rate_hz': (scope.get('fps') or {}).get('fps'),
            'animal_id': custom.get('Animal ID'), 'animal_sex': custom.get('Animal Sex'),
            'animal_dob': custom.get('DOB'), 'personnel': custom.get('Personnel'),
            'session_name': session.get('name'), 'microscope': scope.get('type'),
            'gain': scope.get('gain'), 'led_power_mw_mm2': (scope.get('led') or {}).get('exPower')}


def day_numbers(table, mouse='mouse', date='date'):
    """Day of each recording per mouse: 1 for the mouse's first recording
    date, counting every date it was recorded on (habituation too), so the
    same session keeps its day however the table is filtered later."""
    return table.groupby(mouse)[date].rank(method='dense').astype(int)
