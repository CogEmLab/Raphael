"""The data server's `data/` tree: incoming/, raw/, processed/, metadata/.

`catalogue(root)` writes `metadata/catalogue.csv`: one row per session in
`raw/`, saying what it is, what was recorded, and whether and when each of
its default processing stages has run. It is generated (by ingest, by
`raphael-jobs collect`, or by `raphael-data catalogue`), never edited, so it
can't disagree with the data. See docs/data_server.md.
"""

import csv
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from raphael import raw
from raphael.imaging.layout import Session

CATALOGUE = 'catalogue.csv'
#: Every default stage, in the catalogue's column order.
STAGES = ('kilosort4', 'lfp', 'suite2p', 'cnmfe', 'photometry')
#: A session folder under raw/ (or processed/), current or legacy names,
#: with or without a cohort level above the mouse.
SESSION_DIRS = (
    (re.compile(r'^(?:(?P<cohort>[^/]+)/)?sub-M(?P<mouse>[^/]+)/day-(?P<day>\d+)/'
                r'ses-(?P<type>[^/]+)$'), False),
    (re.compile(r'^(?:(?P<cohort>[^/]+)/)?M(?P<mouse>[^/]+)/D(?P<day>\d+)/(?P<type>[^/]+)$'),
     True),
)
VIDEO = ('.mp4', '.avi', '.mov', '.mkv')
#: Each stage's main software: the package whose version the catalogue shows.
SOFTWARE = {'kilosort4': 'kilosort', 'lfp': 'spikeinterface', 'suite2p': 'suite2p',
            'cnmfe': 'CaImAn', 'photometry': 'raphael'}
COLUMNS = ['cohort', 'mouse', 'day', 'session', 'start', 'task', 'project', 'experimenter',
           'modalities', 'raw_gb', 'ingested', 'status', 'processed_on', 'software', 'raphael',
           'outdated', 'quality', 'notes', 'raw_path']


def sessions(raw_root):
    """(cohort, Session, folder) for every session folder under `raw_root`."""
    raw_root = Path(raw_root)
    for folder, dirs, _ in os.walk(raw_root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(('.', '_')))
        rel = Path(folder).relative_to(raw_root).as_posix()
        for pattern, legacy in SESSION_DIRS:
            m = pattern.match(rel)
            if m:
                yield (m['cohort'], Session(m['mouse'], str(int(m['day'])), m['type'],
                                            legacy=legacy), Path(folder))
                dirs[:] = []                  # don't look inside a session's recordings
                break


def _yaml(path):
    import yaml

    try:
        return yaml.safe_load(path.read_text()) if path.is_file() else None
    except (OSError, yaml.YAMLError):
        return None


def _text(value):
    if value is None:
        return ''
    if isinstance(value, list):
        return ' '.join(map(str, value))
    return value.isoformat() if hasattr(value, 'isoformat') else str(value).strip()


def _size(folder):
    return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())


def _modalities(folder):
    """What the session recorded: the modalities detected, plus video and
    behaviour files."""
    found = {r.modality for r in raw.scan(folder)}
    for f in Path(folder).iterdir():
        name = f.name.lower()
        if f.suffix.lower() in VIDEO or '_video' in name:
            found.add('video')
        elif '_beh' in name:
            found.add('behaviour')
    return sorted(found)


def _raphael(record):
    """The Raphael that made a run: its release (`0.5.0`), or, when it ran
    from a code checkout, the version and commit (`0.5.0+3d947b6`, with
    `-dirty` for uncommitted changes)."""
    info = record.get('raphael') or {}
    version, git = info.get('version') or '?', info.get('git') or {}
    if git.get('tag') or not git.get('commit'):
        return version
    return f"{version}+{git['commit'][:7]}" + ('-dirty' if git.get('dirty') else '')


def _software(stage, record, folder):
    """'kilosort 4.0.16': a run's main software and its version."""
    name = SOFTWARE.get(stage, stage)
    if stage == 'photometry':
        return 'Raphael photometry'
    if stage == 'cnmfe':                 # CaImAn runs in its own environment: its summary
        summary = _json(Path(folder) / 'cnmfe_summary.json')
        version = (summary or {}).get('caiman')
    else:
        version = ((record.get('environment') or {}).get('packages') or {}).get(name)
    label = f'{name} {version}' if version else f'{name} (version not recorded)'
    return f'{label} ({stage})' if stage == 'lfp' else label


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _runs(processed_folder):
    """{stage: {'finished', 'raphael', 'software'}} of the latest run of each
    stage, from a processed session folder's provenance records."""
    runs, folder = {}, Path(processed_folder, 'provenance')
    for path in sorted(folder.glob('*.json')):
        p = _json(path)
        if not p or not p.get('stage') or not p.get('finished'):
            continue                      # a stage's summary, not a provenance record
        stage = p['stage']
        if p['finished'] >= runs.get(stage, {}).get('finished', ''):
            runs[stage] = {'finished': p['finished'], 'raphael': _raphael(p),
                           'software': _software(stage, p, folder)}
    return runs


def _unique(values):
    return '; '.join(dict.fromkeys(v for v in values if v))


def _row(root, cohort, session, folder):
    root = Path(root)
    info = raw.describing(folder / '_', raw.Recording(           # any path inside the session
        '', 'unknown', '', cohort, session.mouse, int(session.day), session.session_type))
    described = info.get('session') or {}
    modalities = _modalities(folder)
    expected = {s for m in modalities if m in raw.MODALITIES for s in raw.MODALITIES[m][0]}
    processed_root = root / 'processed' / cohort if cohort else root / 'processed'
    done = _runs(Session.find(processed_root, session.mouse, session.day,
                              session.session_type).folder(processed_root))
    manifest = folder / f'{session.stem}_manifest.tsv'
    ingested = _text(described.get('ingested')) or (
        datetime.fromtimestamp(manifest.stat().st_mtime).date().isoformat()
        if manifest.is_file() else '')
    ran = [s for s in STAGES if s in done]
    if not expected:
        status = 'n/a'
    elif expected <= set(done):
        status = 'processed'
    else:
        status = 'partly' if expected & set(done) else 'to do'
    last = max((done[s]['finished'] for s in ran), default='')
    row = {
        'cohort': cohort or '', 'mouse': session.mouse, 'day': int(session.day),
        'session': session.session_type,
        'start': _text(described.get('start')).replace('T', ' ')[:16],
        'task': _text(described.get('task')), 'project': _text(described.get('project')),
        'experimenter': _text(described.get('experimenter')),
        'modalities': ' '.join(modalities), 'raw_gb': round(_size(folder) / 1e9, 2),
        'ingested': ingested, 'status': status, 'processed_on': last[:10],
        'software': _unique(done[s]['software'] for s in ran),
        'raphael': _unique(done[s]['raphael'] for s in ran),
        'quality': _text(described.get('quality')),
        'notes': _text(described.get('notes')).replace('\n', ' '),
        'raw_path': folder.relative_to(root).as_posix(),
    }
    if last:
        last_run = datetime.fromisoformat(last)
        changed = [datetime.fromtimestamp(Path(p).stat().st_mtime, UTC)
                   for p in info.get('paths', {}).values()]
        row['outdated'] = 'yes' if any(c > last_run for c in changed) else ''
    else:
        row['outdated'] = ''
    return row


def catalogue(root, out=None):
    """Write `{root}/metadata/catalogue.csv` (or `out`): one row per session
    in `{root}/raw/`. `root` is the `data/` folder. Returns the rows."""
    root = Path(root)
    rows = [_row(root, cohort, session, folder)
            for cohort, session, folder in sessions(root / 'raw')]
    rows.sort(key=lambda r: (r['cohort'], r['mouse'].zfill(12), r['day'], r['session']))
    out = Path(out) if out else root / 'metadata' / CATALOGUE
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        f.write(f'# Generated by raphael-data catalogue on '
                f'{datetime.now().astimezone().isoformat(timespec="minutes")} from raw/ '
                f'and processed/. Do not edit: it is rebuilt.\n')
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def data_root(path):
    """The `data/` folder holding `path` (it has raw/ and metadata/), or None."""
    for parent in [Path(path).resolve(), *Path(path).resolve().parents]:
        if (parent / 'raw').is_dir() and (parent / 'metadata').is_dir():
            return parent
    return None
