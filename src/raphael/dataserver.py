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
COLUMNS = (['cohort', 'mouse', 'day', 'session', 'task', 'start', 'experimenter', 'project',
            'rig', 'quality', 'modalities', 'raw_gb', 'ingested']
           + [c for s in STAGES for c in (s, f'{s}_version')]
           + ['needs_processing', 'outdated', 'raw_path', 'notes'])


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


def _runs(processed_folder):
    """{stage: (finished, raphael version)} of the latest run of each stage,
    from a processed session folder's provenance records."""
    runs = {}
    for record in sorted(Path(processed_folder, 'provenance').glob('*.json')):
        try:
            p = json.loads(record.read_text())
        except (OSError, ValueError):
            continue
        stage, finished = p.get('stage'), p.get('finished') or ''
        if stage and finished >= runs.get(stage, ('', ''))[0]:
            runs[stage] = (finished, (p.get('raphael') or {}).get('version') or '')
    return runs


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
    row = {
        'cohort': cohort or '', 'mouse': session.mouse, 'day': int(session.day),
        'session': session.session_type, 'task': _text(described.get('task')),
        'start': _text(described.get('start')),
        'experimenter': _text(described.get('experimenter')),
        'project': _text(described.get('project')), 'rig': _text(described.get('rig')),
        'quality': _text(described.get('quality')), 'modalities': ' '.join(modalities),
        'raw_gb': round(_size(folder) / 1e9, 2), 'ingested': ingested,
        'raw_path': folder.relative_to(root).as_posix(),
        'notes': _text(described.get('notes')).replace('\n', ' '),
    }
    for stage in STAGES:
        finished, version = done.get(stage, ('', ''))
        row[stage] = finished[:10] if finished else ('to do' if stage in expected else '')
        row[f'{stage}_version'] = version
    row['needs_processing'] = 'yes' if any(row[s] == 'to do' for s in STAGES) else ''
    last = max((f for f, _ in done.values()), default='')
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
