"""What a raw recording is, from its files alone.

Raphael picks the processing for a recording from what is on disk, so
nobody has to name the modality:

| on disk | modality | default stages |
|---|---|---|
| a folder with Open Ephys `Record Node …/` or SpikeGLX `*.ap.meta` | neuropixels | kilosort4, lfp |
| ScanImage TIFF (or a TIFF / HDF5 movie in an `_ophys/` folder) | two_photon | suite2p |
| an Inscopix `.isxd` movie (with nVista's `session.json` beside it) | one_photon | cnmfe |
| a Doric `.doric` file | photometry | photometry |

Each recording also gets what its files say about it: the start time and,
for movies, the frame rate. Under the raw layout, the describing files at
each level above a recording (`describing`) add what the recording files
don't hold: the session's start time and notes, the mouse's sex, date of
birth and strain, its procedures, the cohort. Under the raw layout of docs/data_server.md
(`raw/{cohort}/M{mouse}/D{day}/{type}/sub-M…_ses-D…_typ-…_{ephys,ophys,photometry}`)
the path gives the cohort, mouse, day and session type too; elsewhere they
come from a session table (`raphael-jobs plan --sessions`).

    from raphael import raw
    raw.scan('/path/to/raw')          # every recording under a folder
    raw.detect('/path/to/one/recording')
"""

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MODALITIES = {
    # modality: (default stages, in order; the pipeline's page in docs/)
    'neuropixels': (('kilosort4', 'lfp'), 'ephys_pipeline.md'),
    'two_photon': (('suite2p',), 'imaging_pipeline.md'),
    'one_photon': (('cnmfe',), 'imaging_1p_pipeline.md'),
    'photometry': (('photometry',), 'photometry_pipeline.md'),
}
#: modality -> the stage config it reads (src/raphael/configs/).
CONFIGS = {'neuropixels': 'ephys.yaml', 'two_photon': 'imaging.yaml',
           'one_photon': 'imaging_1p.yaml', 'photometry': 'photometry.yaml'}
LAYOUT = re.compile(r'(?:^|/)M([^/]+)/D(\d+)/([^/]+)/sub-M\1_ses-D\2_typ-\3_[^/]+$')
#: The cohort: the folder above M{mouse}, unless that is the `raw` folder itself.
COHORT = re.compile(r'([^/]+)/M([^/]+)/D(\d+)/([^/]+)/sub-M\2_ses-D\3_typ-\4_[^/]+$')
COPY = re.compile(r'_(recovered\d*|fixed)$')        # Inscopix copies made after acquisition
MOVIE_SUFFIXES = ('.tif', '.tiff', '.h5', '.hdf5')


@dataclass
class Recording:
    path: str                     # relative to the scanned root, '/'-separated
    modality: str
    format: str                   # openephys, spikeglx, scanimage, tiff, hdf5, inscopix, doric
    cohort: str | None = None     # from the raw layout (the folder above M{mouse})
    mouse: str | None = None      # from the raw layout, when the path follows it
    day: int | None = None
    session_type: str | None = None
    start: str | None = None      # ISO 8601; no offset = the lab's local time
    rate_hz: float | None = None  # movies
    bytes: int = 0
    note: str = ''                # anything that stops or changes processing

    @property
    def stages(self):
        return MODALITIES[self.modality][0]

    def as_dict(self):
        return asdict(self)


# ------------------------------------------------------------ start times

def _openephys_start(folder, timezone):
    """Open Ephys: sync_messages.txt's software time (UTC, version >= 0.6),
    else settings.xml's DATE (local)."""
    for sync in sorted(Path(folder).rglob('sync_messages.txt')):
        m = re.search(r'Software Time[^:]*:\s*(\d+)', sync.read_text(errors='replace'))
        if m:
            utc = datetime.fromtimestamp(int(m.group(1)) / 1000, UTC)
            return utc.astimezone(ZoneInfo(timezone)).isoformat()
    for settings in sorted(Path(folder).rglob('settings.xml')):
        m = re.search(r'<DATE>([^<]+)</DATE>', settings.read_text(errors='replace'))
        if m:
            try:
                return datetime.strptime(m.group(1).strip(), '%d %b %Y %H:%M:%S').isoformat()
            except ValueError:
                pass
    return None


def _spikeglx_start(folder):
    for meta in sorted(Path(folder).rglob('*.ap.meta')):
        m = re.search(r'^fileCreateTime=(\S+)', meta.read_text(errors="replace"), re.MULTILINE)
        if m:
            return m.group(1)
    return None


def _tiff_info(path, any_movie=False):
    """(format, rate_hz, start) of a TIFF movie: ScanImage's frame rate and
    epoch, or (with `any_movie`) ImageJ's fps; (None, ...) otherwise. Only
    the file's header and first page are read: counting a TIFF's pages
    reads through the whole file."""
    import tifffile

    with tifffile.TiffFile(path, _multifile=False) as tif:
        if tif.is_scanimage:
            meta = tif.scanimage_metadata or {}
            frame = meta.get('FrameData', {})
            rate = frame.get('SI.hRoiManager.scanFrameRate')
            start = None
            text = tif.pages[0].tags.get('ImageDescription')
            m = re.search(r'epoch\s*=\s*\[([^\]]+)\]', str(text.value if text else ''))
            if m:
                y, mo, d, h, mi, s = (float(v) for v in re.split(r'[\s,]+', m.group(1).strip()))
                start = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s),
                                 round((s % 1) * 1e6)).isoformat()
            return 'scanimage', float(rate) if rate else None, start
        if not any_movie:
            return None, None, None
        imagej = (tif.imagej_metadata or {}) if tif.is_imagej else {}
        rate = imagej.get('fps')
        return 'tiff', float(rate) if rate else None, None


def _size(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


# --------------------------------------------------------------- detection

def _is_openephys(folder):
    return any(c.is_dir() and c.name.startswith('Record Node') for c in folder.iterdir())


def _is_spikeglx(folder):
    """A SpikeGLX run folder (`name_g0`, holding `*.ap.meta` directly or in
    its probe folders), or a raw-layout `_ephys` folder wrapping one."""
    if re.search(r'_g\d+$', folder.name):
        return any(folder.glob('*.ap.meta')) or any(folder.glob('*/*.ap.meta'))
    return folder.name.endswith('_ephys') and any(folder.glob('*/*/*.ap.meta')) or (
        folder.name.endswith('_ephys') and any(folder.glob('*/*.ap.meta')))


def _movie_in(folder):
    """The one movie in an `_ophys`-style folder: (path, note)."""
    isxd = sorted(f for f in folder.glob('*.isxd') if not COPY.search(f.stem))
    if isxd:
        return isxd[0], ''
    movies = sorted(f for f in folder.iterdir() if f.suffix.lower() in MOVIE_SUFFIXES)
    if len(movies) > 1:
        return movies[0], (f'{len(movies)} movie files; the suite2p stage takes one movie, '
                           'so only the first is listed')
    return (movies[0], '') if movies else (None, 'no movie found')


def detect(path, root=None, timezone='Europe/Vienna'):
    """The Recording at `path` (a file or folder), or None if it is not one.
    `root` makes `path` relative in the result (default: its parent)."""
    path = Path(path)
    root = Path(root) if root is not None else path.parent
    rec = None
    if path.is_dir():
        if path.name.endswith('_ophys'):
            movie, note = _movie_in(path)
            if movie is None:                       # named from its path below, like the rest
                rec = Recording(_rel(path, root), 'two_photon', 'unknown', note=note)
            else:
                rec = detect(movie, root, timezone)
                if rec and note:
                    rec.note = '; '.join(filter(None, [rec.note, note]))
        elif _is_openephys(path):
            rec = Recording(_rel(path, root), 'neuropixels', 'openephys',
                            start=_openephys_start(path, timezone))
        elif _is_spikeglx(path):
            rec = Recording(_rel(path, root), 'neuropixels', 'spikeglx',
                            start=_spikeglx_start(path))
        elif any(f.suffix == '.env' for f in path.iterdir() if f.is_file()):
            rec = Recording(_rel(path, root), 'two_photon', 'bruker',
                            note='Bruker (Prairie View) folders are not read by the '
                                 'suite2p stage yet')
    elif path.suffix.lower() == '.isxd' and not COPY.search(path.stem) and (
            (path.parent / 'session.json').exists() or path.parent.name.endswith('_ophys')):
        # Raw movies only: nVista writes session.json beside each; files made
        # afterwards (conversions, IMU/GPIO as .isxd, processed movies) have none.
        from raphael.imaging import inscopix

        # An unclosed original (no description) may have a readable copy
        # beside it, written by Inscopix's recovery tool: use the first.
        copies = sorted(f for f in path.parent.glob(f'{path.stem}_*.isxd') if COPY.search(f.stem))
        errors = []
        for candidate in [path, *copies]:
            try:
                movie = inscopix.open_isxd(candidate, timezone)
                start, rate = movie.start.isoformat(), round(movie.rate, 4)
            except (OSError, ValueError, KeyError) as e:   # KeyError: no timing (not a movie)
                errors.append(f'{candidate.name}: {str(e).replace(str(candidate), "")[:100]}')
                continue
            path = candidate
            rec = Recording(_rel(path, root), 'one_photon', 'inscopix', start=start,
                            rate_hz=rate, note=f'using {path.name}; {errors[0]}' if errors else '')
            break
        else:
            rec = Recording(_rel(path, root), 'one_photon', 'inscopix',
                            note='unreadable: ' + '; '.join(errors))
    elif path.suffix.lower() == '.doric':
        import h5py

        from raphael.photometry import doric

        rec = Recording(_rel(path, root), 'photometry', 'doric')
        try:
            with h5py.File(path, 'r') as f:
                console = next(iter(f['DataAcquisition']))
                start = doric._start_time(f, console)
            rec.start = start.isoformat() if start else None
        except (OSError, KeyError, StopIteration, ValueError) as e:
            rec.note = f'unreadable: {str(e)[:150]}'
    elif path.suffix.lower() in ('.tif', '.tiff') and not (
            path.with_suffix('.isxd').exists()):            # Inscopix's own TIFF export
        try:
            fmt, rate, start = _tiff_info(path, any_movie=path.parent.name.endswith('_ophys'))
        except Exception as e:  # noqa: BLE001 - any unreadable TIFF is just not a movie
            fmt, rate, start = None, None, None
            if path.parent.name.endswith('_ophys'):
                return Recording(_rel(path, root), 'two_photon', 'tiff',
                                 note=f'unreadable: {str(e)[:150]}')
        if fmt == 'scanimage' or (fmt and path.parent.name.endswith('_ophys')):
            rec = Recording(_rel(path, root), 'two_photon', fmt, start=start, rate_hz=rate)
    elif path.suffix.lower() in ('.h5', '.hdf5') and path.parent.name.endswith('_ophys'):
        rec = Recording(_rel(path, root), 'two_photon', 'hdf5')
    if rec is None:
        return None
    rec.bytes = _size(path)
    named = path if path.is_dir() or not path.parent.name.endswith('_ophys') else path.parent
    m = LAYOUT.search(_rel(named, root))
    if m:
        rec.mouse, rec.day, rec.session_type = m.group(1), int(m.group(2)), m.group(3)
        c = COHORT.search(Path(named).resolve().as_posix())
        rec.cohort = c.group(1) if c and c.group(1) != 'raw' else None
    if rec.modality == 'two_photon' and rec.format in ('tiff', 'hdf5', 'scanimage') \
            and rec.rate_hz is None:
        rec.note = '; '.join(filter(None, [rec.note, 'frame rate not in the file: give fs']))
    return rec


def _rel(path, root):
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix()


def scan(root, timezone='Europe/Vienna'):
    """Every recording under `root`, in path order. A recording folder (an
    Open Ephys or SpikeGLX run, an `_ophys` folder) is not searched inside.
    TIFFs count as two-photon movies only with ScanImage metadata or inside
    an `_ophys` folder (so images and Inscopix TIFF exports are left out)."""
    root = Path(root)
    found = []
    for folder, dirs, files in os.walk(root):
        folder = Path(folder)
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        if folder != root:
            rec = detect(folder, root, timezone)
            if rec is not None:
                found.append(rec)
                dirs[:] = []
                continue
        for name in sorted(files):
            if name.startswith('.') or Path(name).suffix.lower() not in (
                    '.isxd', '.doric', '.tif', '.tiff'):
                continue
            rec = detect(folder / name, root, timezone)
            if rec is not None:
                found.append(rec)
    return found


def _yaml(path):
    import yaml

    if not path.is_file():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def describing(path, recording=None):
    """The describing files above a recording in the raw layout, as dicts
    (None where a file is missing):

        session      {type}/sub-M…_ses-D…_typ-…_session.yaml
        subject      M{mouse}/sub-M{mouse}_subject.yaml
        procedures   M{mouse}/sub-M{mouse}_procedures.yaml (a list)
        cohort       {cohort}/cohort.yaml

    plus `paths`, the files read. `path` is the recording (file or folder);
    `recording` its Recording (detected if not given). Empty if the path
    doesn't follow the layout."""
    path = Path(path).resolve()
    rec = recording or detect(path)
    if rec is None or rec.mouse is None:
        return {}
    mouse, day, kind = rec.mouse, rec.day, rec.session_type
    session_dir = next((p for p in path.parents if p.name == kind and p.parent.name == f'D{day}'
                        and p.parent.parent.name == f'M{mouse}'), None)
    if session_dir is None:
        return {}
    mouse_dir = session_dir.parent.parent
    files = {
        'session': session_dir / f'sub-M{mouse}_ses-D{day}_typ-{kind}_session.yaml',
        'subject': mouse_dir / f'sub-M{mouse}_subject.yaml',
        'procedures': mouse_dir / f'sub-M{mouse}_procedures.yaml',
        'cohort': mouse_dir.parent / 'cohort.yaml',
    }
    out = {name: _yaml(f) for name, f in files.items()}
    out['paths'] = {name: str(f) for name, f in files.items() if f.is_file()}
    return out


def _iso(value):
    """A YAML date or time (parsed to date/datetime) as ISO 8601 text."""
    return value.isoformat() if hasattr(value, 'isoformat') else (
        None if value is None else str(value))


def to_table(recordings):
    import pandas as pd

    rows = [{**r.as_dict(), 'stages': ' '.join(r.stages)} for r in recordings]
    columns = list(Recording.__dataclass_fields__) + ['stages']
    return pd.DataFrame(rows, columns=columns)


def describe(recordings):
    """One line per recording, for the terminal."""
    lines = []
    for r in recordings:
        who = (f'{r.cohort + " " if r.cohort else ""}M{r.mouse} D{r.day} {r.session_type}'
               if r.mouse is not None else '(no mouse/day/type in the path)')
        rate = f', {r.rate_hz:g} Hz' if r.rate_hz else ''
        lines.append(f'{r.modality:<12} {r.format:<10} {who:<28} {r.bytes / 1e9:8.1f} GB'
                     f'{rate}  {r.path}' + (f'\n{"":13}! {r.note}' if r.note else ''))
    return '\n'.join(lines)


def to_json(recordings):
    return json.dumps([r.as_dict() for r in recordings], indent=2, default=str)
