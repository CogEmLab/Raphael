"""Ingest: from people's hand-ins in `data/incoming/` into `data/raw/`.

A hand-in is flat (docs/data_server.md; templates/data/incoming/_example):
`incoming/{user}/` holds the cohort file (`{cohort}_cohort.yaml`), each
mouse's subject and procedures files (`sub-M{m}_subject.yaml`,
`sub-M{m}_procedures.yaml`) and one folder per session named
`sub-M{m}_day-{dd}_ses-{type}_{YYYY-MM-DD}_{HH-MM-SS}` holding its
`sub-M{m}_day-{dd}_ses-{type}_session.yaml` and what the rig wrote, not
renamed.

For each session, ingest

1. checks it (nothing that fails is filed): the describing files are there
   with their required fields; the subject's cohort matches the cohort file
   and any earlier sessions of the mouse; the session code, project, rig
   and people are registered in `metadata/`; the start time in the folder
   name agrees with the recordings' own start times and the session file;
   the day number fits the mouse's earlier days and the session isn't taken;
   every recording is recognised and named unambiguously;
2. files it: copies it into `raw/{cohort}/sub-M{m}/day-{dd}/ses-{type}/`
   with each recording renamed for what it is (acquisition folders wrapped,
   their insides untouched), writes the manifest (size, sha256, original
   name), verifies every copy, fills blank fields of the session file, locks
   the recordings and only then removes the session from `incoming/`;
3. adds or updates the cohort, subject and procedures files (old versions
   kept in `.history/`; changes to identity fields are left for the data
   manager);

then writes one processing plan per person who had something filed
(`metadata/plans/plan_{date}_{time}_{user}.csv`), rebuilds the catalogue,
and reports to each person: `incoming/{user}/PROBLEMS.md`, and email when
`metadata/lab.yaml` has mail settings. `dry_run=True` (and `check`) report
what would happen and change nothing.
"""

import getpass
import hashlib
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from raphael import raw
from raphael.imaging.layout import Session, day_label

#: A session folder in a hand-in.
SESSION_FOLDER = re.compile(
    r'^sub-M(?P<mouse>[A-Za-z0-9]+)_day-(?P<day>\d+)_ses-(?P<type>[A-Za-z0-9]+)_'
    r'(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})$')
SUBJECT_FILE = re.compile(r'^sub-M(?P<mouse>[A-Za-z0-9]+)_subject\.yaml$')
PROCEDURES_FILE = re.compile(r'^sub-M(?P<mouse>[A-Za-z0-9]+)_procedures\.yaml$')
COHORT_FILE = re.compile(r'^(?P<cohort>.+)_cohort\.yaml$')
COHORT_NAME = re.compile(r'^\d{4}-\d{2}_[A-Za-z0-9-]+$')
SESSION_TYPE = re.compile(r'^(?P<code>[A-Za-z]+)(?P<run>\d+)$')
#: How far apart (s) the folder name's time and a recording's own start may be.
TOLERANCE_S = 300
JUNK = ('.DS_Store', 'Thumbs.db', 'desktop.ini', 'PROBLEMS.md')
VIDEO = ('.mp4', '.avi', '.mov', '.mkv')
INSCOPIX_SIDECARS = ('.gpio', '.imu')
SUBJECT_REQUIRED = ('mouse', 'cohort', 'sex', 'date_of_birth', 'species')
SUBJECT_IDENTITY = ('mouse', 'cohort', 'sex', 'date_of_birth')


@dataclass
class Report:
    """What ingest did (or would do) for one person's hand-in."""
    user: str
    filed: list = field(default_factory=list)       # raw/ paths of sessions filed
    problems: list = field(default_factory=list)    # (item, problem, fix)
    notes: list = field(default_factory=list)       # (item, note)
    plan: str = ''

    def text(self, dry_run=False):
        verb = 'would be added' if dry_run else 'added'
        lines = [f'# Hand-in from {self.user}', '']
        if self.filed:
            lines += [f'## {len(self.filed)} session(s) {verb}', '']
            lines += [f'- {p}' for p in self.filed] + ['']
        if self.problems:
            lines += [f'## {len(self.problems)} problem(s): not added', '']
            for item, problem, fix in self.problems:
                lines.append(f'- **{item}**: {problem}' + (f'  \n  Fix: {fix}' if fix else ''))
            lines.append('')
        if self.notes:
            lines += ['## Notes', '']
            lines += [f'- {item}: {note}' for item, note in self.notes] + ['']
        if self.plan:
            lines += [f'Processing plan: `{self.plan}`', '']
        if not (self.filed or self.problems or self.notes):
            lines += ['Nothing to add.', '']
        return '\n'.join(lines)


# ----------------------------------------------------------------- reading

def _yaml_file(path):
    """(content, error) of a YAML file."""
    try:
        return yaml.safe_load(Path(path).read_text()), None
    except (OSError, yaml.YAMLError) as e:
        return None, str(e).splitlines()[0][:150]


def _text(value):
    if value is None:
        return ''
    return value.isoformat() if hasattr(value, 'isoformat') else str(value).strip()


def lab(root):
    """The registered people, session codes, projects and rigs, and lab.yaml."""
    meta = Path(root) / 'metadata'
    lists = {}
    for name in ('people', 'sessions', 'projects'):
        content, _ = _yaml_file(meta / f'{name}.yaml')
        lists[name] = content if isinstance(content, dict) else {}
    rigs = set()
    for f in sorted((meta / 'rigs').glob('*.yaml')):
        content, _ = _yaml_file(f)
        rigs.add(f.stem)
        if isinstance(content, dict) and content.get('rig'):
            rigs.add(str(content['rig']))
    lists['rigs'] = rigs
    content, _ = _yaml_file(meta / 'lab.yaml')
    lists['lab'] = content if isinstance(content, dict) else {}
    return lists


def _local(value, timezone):
    """A start time (ISO text or datetime) as naive local time, or None."""
    if not value:
        return None
    try:
        t = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if t.tzinfo is not None:
        t = t.astimezone(ZoneInfo(timezone)).replace(tzinfo=None)
    return t


def mouse_history(raw_root, mouse):
    """(cohorts, {day: date}, sessions) of a mouse already in raw/: the
    cohort folders it is in, the date of each of its days, and its session
    files (day, type, content), newest last."""
    cohorts, days, sessions = set(), {}, []
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        return cohorts, days, sessions
    for cohort_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()
                             and not p.name.startswith('.')):
        for mouse_dir in (cohort_dir / f'sub-M{mouse}', cohort_dir / f'M{mouse}'):
            if not mouse_dir.is_dir():
                continue
            cohorts.add(cohort_dir.name)
            legacy = mouse_dir.name.startswith('M')
            for day_dir in sorted(mouse_dir.glob('D*' if legacy else 'day-*')):
                m = re.match(r'^(?:D|day-)(\d+)$', day_dir.name)
                if not m:
                    continue
                day = int(m[1])
                ses_dirs = sorted(p for p in day_dir.iterdir() if p.is_dir()
                                  and not p.name.startswith('.'))
                if ses_dirs:                              # an empty day folder isn't a day
                    days.setdefault(day, None)
                for ses_dir in ses_dirs:
                    kind = ses_dir.name.removeprefix('ses-')
                    s = Session(str(mouse), str(day), kind, legacy=legacy)
                    content, _ = _yaml_file(ses_dir / f'{s.stem}_session.yaml')
                    content = content if isinstance(content, dict) else {}
                    start = _text(content.get('start'))
                    if start:
                        days[day] = days[day] or start[:10]
                    sessions.append((day, kind, content))
    sessions.sort(key=lambda s: (s[0], _text(s[2].get('start'))))
    return cohorts, days, sessions


def expected_day(days, date):
    """The day number a session on `date` (YYYY-MM-DD) should have, given
    the mouse's {day: date}, or (None, reason)."""
    for day, d in days.items():
        if d == date:
            return day, ''
    known = [d for d in days.values() if d]
    if known and date < max(known):
        return None, (f'{date} is before the mouse’s latest recording day ({max(known)}): '
                      'its days would need renumbering')
    return (max(days) + 1 if days else 1), ''


# ---------------------------------------------------------- the recordings

def _clean(text):
    return re.sub(r'[^A-Za-z0-9]', '', text) or 'x'


def _match_label(name, labels):
    """The one label contained in `name` (case-insensitive), or None."""
    hits = [label for label in labels if label and label.lower() in name.lower()]
    return hits[0] if len(hits) == 1 else None


def recordings(folder, session_file, procedures, rig, timezone='Europe/Vienna'):
    """What the rig wrote in a hand-in session folder, and what each becomes:
    ([{'sources', 'name', 'kind', 'start', 'dir'}], problems). `name` is the
    suffix after the session's stem (`_ephys`, `_photometry-dHPC.doric`, …)."""
    folder = Path(folder)
    problems, items = [], []
    labels = [str(p.get('label')) for p in (procedures or []) if isinstance(p, dict)
              and p.get('label')]
    cameras = [str(c.get('label')) for c in ((rig or {}).get('cameras') or [])
               if isinstance(c, dict) and c.get('label')]
    children = sorted(p for p in folder.iterdir() if p.name not in JUNK
                      and not p.name.startswith('._') and p.name != session_file)
    used = set()
    # The session folder may itself be the acquisition folder: Open Ephys's
    # (Record Node … inside, when it is set to name its own folder) or a
    # SpikeGLX run's (probe folders and .bin/.meta files inside). Those parts
    # are the session's one ephys recording, kept as recorded inside it.
    nodes = [p for p in children if p.is_dir() and p.name.startswith('Record Node')]
    glx = [p for p in children if (p.is_dir() and re.search(r'_imec\d+$', p.name))
           or re.search(r'\.(ap|lf|nidq)\.(bin|meta)$', p.name)]
    for parts, start in ((nodes, raw._openephys_start(folder, timezone) if nodes else None),
                         (glx, raw._spikeglx_start(folder) if glx else None)):
        if parts:
            used.update(parts)
            items.append({'sources': parts, 'kind': 'ephys', 'start': start, 'dir': True,
                          'label_from': folder.name, 'keep_names': True})
    loose_ophys = {'inscopix': [], 'scanimage': []}
    for child in children:
        if child in used:
            continue
        if child.is_dir():
            rec = raw.detect(child, timezone=timezone)
            if rec is not None and rec.modality == 'neuropixels':
                items.append({'sources': [child], 'kind': 'ephys', 'start': rec.start,
                              'dir': True, 'label_from': child.name})
                continue
            inner = [r for r in raw.scan(child, timezone) if r.modality in
                     ('one_photon', 'two_photon')]
            if inner:
                items.append({'sources': [child], 'kind': 'ophys', 'start': inner[0].start,
                              'dir': True, 'label_from': child.name})
                continue
            problems.append((child.name, 'not a recording Raphael recognises',
                             'only what the rig wrote goes in the session folder'))
            continue
        suffix = child.suffix.lower()
        if suffix == '.isxd':
            rec = raw.detect(child, timezone=timezone)
            if rec is None or rec.note.startswith('unreadable'):
                problems.append((child.name, 'an Inscopix movie that can’t be read'
                                 + ('' if (folder / 'session.json').exists()
                                    else ' (no session.json beside it)'),
                                 'hand in the movie with the session.json nVista wrote'))
                continue
            group = [child] + [p for p in children if p.stem == child.stem
                               and p.suffix.lower() in INSCOPIX_SIDECARS]
            for side in ('session.json', 'apply_session.json'):
                if (folder / side).exists():
                    group.append(folder / side)
            used.update(group)
            loose_ophys['inscopix'].append((group, rec.start))
        elif suffix in ('.tif', '.tiff'):
            rec = raw.detect(child, timezone=timezone)
            used.add(child)
            loose_ophys['scanimage'].append(([child], rec.start if rec else None))
        elif suffix == '.doric':
            rec = raw.detect(child, timezone=timezone)
            items.append({'sources': [child], 'kind': 'photometry', 'ext': '.doric',
                          'start': rec.start if rec else None, 'dir': False,
                          'label_from': child.stem})
        elif suffix in VIDEO:
            items.append({'sources': [child], 'kind': 'video', 'ext': suffix, 'start': None,
                          'dir': False, 'label_from': child.stem})
        elif child.name in ('session.json', 'apply_session.json'):
            continue                                      # taken with an .isxd, or stray
        else:
            items.append({'sources': [child], 'kind': 'beh', 'ext': suffix, 'start': None,
                          'dir': False, 'label_from': child.stem})
    for kind, groups in loose_ophys.items():
        for group, start in groups:
            if kind == 'scanimage' and len(groups) > 1:   # ScanImage's split files: one movie
                continue
            items.append({'sources': group, 'kind': 'ophys', 'start': start, 'dir': False,
                          'label_from': group[0].stem, 'loose': True})
        if kind == 'scanimage' and len(groups) > 1:
            files = [g[0][0] for g in groups]
            items.append({'sources': files, 'kind': 'ophys', 'start': groups[0][1],
                          'dir': False, 'label_from': files[0].stem, 'loose': True})
    # Names: plain when one of a kind; labelled when several.
    by_kind = {}
    for item in items:
        by_kind.setdefault(item['kind'], []).append(item)
    for kind, group in by_kind.items():
        taken = set()
        for item in group:
            ext = item.get('ext', '')
            if len(group) == 1:
                item['name'] = f'_{kind}{ext}'
                continue
            if kind in ('ephys', 'ophys', 'photometry'):
                label = _match_label(item['label_from'], labels)
                if label is None:
                    problems.append((item['sources'][0].name,
                                     (f'{len(group)} {kind} recordings, and this one’s name '
                                     'doesn’t contain exactly one procedure label'),
                                     ('put its label (from the procedures file, e.g. dHPC) '
                                     'in its name')))
                    item['name'] = None
                    continue
            elif kind == 'video':
                label = _match_label(item['label_from'], cameras) or _clean(item['label_from'])
            else:
                label = _clean(item['label_from'])
            name = f'_{kind}-{label}{ext}'
            while name in taken:
                name = f'_{kind}-{label}{len(taken)}{ext}'
            taken.add(name)
            item['name'] = name
    if not items and not problems:
        problems.append((folder.name, 'no recordings in the session folder', ''))
    return [i for i in items if i.get('name')], problems


# ------------------------------------------------------------- the checks

@dataclass
class HandIn:
    user: str
    folder: Path
    sessions: list = field(default_factory=list)      # session folders
    subjects: dict = field(default_factory=dict)      # mouse -> path
    procedures: dict = field(default_factory=dict)    # mouse -> path
    cohorts: dict = field(default_factory=dict)       # cohort -> path
    other: list = field(default_factory=list)


def read_handin(folder, user=None):
    folder = Path(folder)
    h = HandIn(user or folder.name, folder)
    for p in sorted(folder.iterdir()):
        if p.name in JUNK or p.name.startswith(('.', '_')):
            continue
        if p.is_dir() and SESSION_FOLDER.match(p.name):
            h.sessions.append(p)
        elif m := SUBJECT_FILE.match(p.name):
            h.subjects[m['mouse']] = p
        elif m := PROCEDURES_FILE.match(p.name):
            h.procedures[m['mouse']] = p
        elif m := COHORT_FILE.match(p.name):
            h.cohorts[m['cohort']] = p
        else:
            h.other.append(p)
    return h


def _check_subject(path, mouse, report):
    content, error = _yaml_file(path)
    if error or not isinstance(content, dict):
        report.problems.append((path.name, f'can’t be read: {error or "not key: value lines"}',
                                'copy metadata/templates/subject.yaml and fill it in'))
        return None
    missing = [k for k in SUBJECT_REQUIRED if not _text(content.get(k))]
    if missing:
        report.problems.append((path.name, f'missing {", ".join(missing)}',
                                'fill them in (they go in every NWB file of this mouse)'))
        return None
    if _text(content['mouse']) != str(mouse):
        report.problems.append((path.name, f'says mouse {content["mouse"]}, its name says {mouse}',
                                'make them agree'))
        return None
    if _text(content['sex']).upper() not in ('M', 'F', 'U', 'O'):
        report.problems.append((path.name, f'sex is {content["sex"]}', 'use M, F, U or O'))
        return None
    return content


def _check_cohort(path, cohort, report, new):
    content, error = _yaml_file(path)
    if error or not isinstance(content, dict):
        report.problems.append((path.name, f'can’t be read: {error or "not key: value lines"}',
                                'copy metadata/templates/cohort.yaml and fill it in'))
        return None
    if _text(content.get('cohort')) != cohort:
        report.problems.append((path.name, (f'says cohort {content.get("cohort")}, its name '
                                f'says {cohort}'), 'make them agree'))
        return None
    if new and not COHORT_NAME.match(cohort):
        report.problems.append((path.name, f'{cohort} isn’t named YYYY-MM_label',
                                ('name new cohorts by the month they started, e.g. '
                                '2026-03_dHPC')))
        return None
    return content


def _check_procedures(path, report):
    content, error = _yaml_file(path)
    if error or not (content is None or isinstance(content, list)):
        report.problems.append((path.name, f'can’t be read as a list: {error or ""}'.rstrip(': '),
                                'before any surgery the file is just: []'))
        return None
    return content or []


def plan_session(folder, handin, root, lists, report, pending):
    """Everything about one hand-in session, or None if it fails a check
    (the problems are added to `report`). `pending` holds the {day: date}
    of sessions filed earlier in this run, per mouse."""
    root = Path(root)
    timezone = (lists['lab'] or {}).get('timezone') or 'Europe/Vienna'
    m = SESSION_FOLDER.match(folder.name)
    mouse, day, kind = m['mouse'], int(m['day']), m['type']
    item = folder.name
    start = datetime.strptime(f'{m["date"]}_{m["time"]}', '%Y-%m-%d_%H-%M-%S')
    before = len(report.problems)
    problem = lambda what, fix='': report.problems.append((item, what, fix))

    if m['day'] != day_label(day).removeprefix('day-'):
        problem(f'day-{m["day"]} should be written {day_label(day)}', 'two digits')
    code = SESSION_TYPE.match(kind)
    if not code:
        problem(f'ses-{kind} isn’t a session code and run number', 'e.g. ses-OF1, ses-OF2')
        return None
    session = Session(str(mouse), str(day), kind)
    session_file = f'{session.stem}_session.yaml'

    # Describing files.
    subject_path = handin.subjects.get(mouse)
    procedures_path = handin.procedures.get(mouse)
    if subject_path is None:
        problem(f'no sub-M{mouse}_subject.yaml in the hand-in',
                'hand in the mouse’s subject file with every session')
    if procedures_path is None:
        problem(f'no sub-M{mouse}_procedures.yaml in the hand-in',
                'hand it in with every session; before any surgery it is just []')
    subject = _check_subject(subject_path, mouse, report) if subject_path else None
    procedures = _check_procedures(procedures_path, report) if procedures_path else None
    cohorts_before, days, history = mouse_history(root / 'raw', mouse)
    cohort = _text((subject or {}).get('cohort'))
    cohort_doc = None
    if subject:
        if cohort not in handin.cohorts:
            problem(f'no {cohort}_cohort.yaml in the hand-in',
                    'hand in the cohort’s file with every session')
        else:
            cohort_doc = _check_cohort(handin.cohorts[cohort], cohort, report,
                                       new=not (root / 'raw' / cohort).is_dir())
        if cohorts_before and cohort not in cohorts_before:
            problem(f'the subject file says cohort {cohort}, but mouse {mouse} is already in '
                    f'{", ".join(sorted(cohorts_before))}',
                    'a mouse’s cohort can’t change: ask the data manager if it is wrong')

    # The session file, filled in where it can be.
    content, error = (None, None)
    if (folder / session_file).exists():
        content, error = _yaml_file(folder / session_file)
    else:
        named = [p.name for p in folder.glob('*_session.yaml')]
        problem(f'no {session_file}' + (f' (found {", ".join(named)})' if named else ''),
                'copy metadata/templates/session.yaml into the folder, named like the folder '
                'without the time')
    if error:
        problem(f'{session_file} can’t be read: {error}', '')
    described = dict(content) if isinstance(content, dict) else {}
    last = history[-1][2] if history else {}
    filled = {
        'task': _text(described.get('task')) or code['code'],
        'experimenter': described.get('experimenter') or [handin.user],
        'project': _text(described.get('project')) or _text(last.get('project'))
        or _text((cohort_doc or {}).get('project')),
        'rig': _text(described.get('rig')) or _text(last.get('rig')),
    }
    if filled['task'] != code['code']:
        problem(f'the session file says task {filled["task"]}, the folder says {code["code"]}',
                'make them agree')
    people, codes = lists['people'], lists['sessions']
    if handin.user not in people:
        problem(f'{handin.user} is not in metadata/people.yaml', 'add yourself there first')
    if code['code'] not in codes:
        problem(f'session code {code["code"]} is not in metadata/sessions.yaml',
                'ask its owner or the data manager to add it')
    if not filled['project']:
        problem('no project', 'give project in the session file (or the cohort file)')
    elif filled['project'] not in lists['projects']:
        problem(f'project {filled["project"]} is not in metadata/projects.yaml',
                'add it, or correct the session file')
    if filled['rig'] and filled['rig'] not in lists['rigs']:
        problem(f'rig {filled["rig"]} is not in metadata/rigs/', 'add it, or correct the session file')
    experimenters = filled['experimenter'] if isinstance(filled['experimenter'], list) \
        else [filled['experimenter']]
    for person in experimenters:
        if str(person) not in people:
            problem(f'experimenter {person} is not in metadata/people.yaml', 'add them there')
    if [str(p) for p in experimenters] != [handin.user] and handin.user not in map(str,
                                                                                experimenters):
        report.notes.append((item, (f'handed in by {handin.user}, recorded by '
                                   f'{", ".join(map(str, experimenters))}')))

    # The day and the session.
    known = {**days, **pending.get(mouse, {})}
    want, why = expected_day(known, m['date'])
    if want is None:
        problem(why, 'ask the data manager')
    elif want != day:
        problem(f'mouse {mouse} on {m["date"]} is {day_label(want)}, not {day_label(day)}'
                + (f' ({day_label(day)} is {known[day]})' if known.get(day) else ''),
                f'rename the folder sub-M{mouse}_{day_label(want)}_ses-{kind}_…')
    if cohort:
        target = session.folder(root / 'raw' / cohort)
        if target.exists():
            problem(f'{target.relative_to(root).as_posix()} is already in raw/',
                    f'a second {code["code"]} that day is ses-{code["code"]}'
                    f'{int(code["run"]) + 1}')

    # The recordings, and their start times against the folder name.
    rig_doc = None
    if filled['rig']:
        rig_doc, _ = _yaml_file(root / 'metadata' / 'rigs' / f'{filled["rig"]}.yaml')
    items, rec_problems = recordings(folder, session_file, procedures, rig_doc
                                     if isinstance(rig_doc, dict) else None, timezone)
    report.problems += [(f'{item}/{name}', what, fix) for name, what, fix in rec_problems]
    for rec in items:
        t = _local(rec['start'], timezone)
        if t and abs((t - start).total_seconds()) > TOLERANCE_S:
            problem(f'{rec["sources"][0].name} started at {t:%Y-%m-%d %H:%M:%S}, the folder '
                    f'name says {start:%Y-%m-%d %H:%M:%S}',
                    'name the folder with the recording’s start time')
    given = _local(described.get('start'), timezone)
    if given and abs((given - start).total_seconds()) > TOLERANCE_S:
        problem(f'the session file’s start is {given}, the folder name says {start}',
                'make them agree, or leave start blank')

    # Who owns the files (POSIX only): a file in someone's folder that
    # another account wrote is worth a look.
    try:
        import pwd
        pwd.getpwnam(handin.user)                 # only where usernames are accounts
        owners = {pwd.getpwuid(p.stat().st_uid).pw_name for p in folder.rglob('*')}
        others = owners - {handin.user}
        if others:
            report.notes.append((item, (f'files owned by {", ".join(sorted(others))}, not '
                                       f'{handin.user}')))
    except (ImportError, KeyError, OSError):
        pass

    if len(report.problems) > before:
        return None
    described.update({k: v for k, v in filled.items() if v})
    described['start'] = _text(given) or start.isoformat()
    return {'folder': folder, 'mouse': mouse, 'day': day, 'kind': kind, 'cohort': cohort,
            'session': session, 'session_file': session_file, 'described': described,
            'items': items, 'date': m['date']}


# ------------------------------------------------------------------ filing

def _copy(src, dst):
    """Copy a file, returning (bytes, sha256 of what was read)."""
    h, size = hashlib.sha256(), 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, 'rb') as fi, open(dst, 'wb') as fo:
        while block := fi.read(1 << 24):
            h.update(block)
            fo.write(block)
            size += len(block)
    shutil.copystat(src, dst)
    return size, h.hexdigest()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def file_session(plan, root, runner, now):
    """Copy a checked session into raw/, verify it, then remove it from
    incoming/. Returns its raw/ folder. Raises if anything fails, leaving
    the hand-in as it was and nothing half-made in raw/."""
    root = Path(root)
    session = plan['session']
    target = session.folder(root / 'raw' / plan['cohort'])
    if target.exists():
        raise FileExistsError(target)
    # Copied into a hidden staging folder and moved into place only when
    # complete, so a failure leaves nothing in the mouse's tree.
    staging = root / 'raw' / '.ingesting' / session.stem
    if staging.exists():
        shutil.rmtree(staging)
    rows = []
    try:
        for item in plan['items']:
            name = f'{session.stem}{item["name"]}'
            for src in item['sources']:
                if item.get('keep_names'):                 # parts of an acquisition folder
                    files = sorted(f for f in src.rglob('*') if f.is_file()) if src.is_dir() \
                        else [src]
                    pairs = [(f, staging / name / f.relative_to(src.parent)) for f in files]
                elif src.is_dir():
                    files = sorted(f for f in src.rglob('*') if f.is_file())
                    pairs = [(f, staging / name / f.relative_to(src)) for f in files]
                elif item.get('loose'):                    # loose movie files: into a folder
                    pairs = [(src, staging / name / src.name)]
                else:
                    pairs = [(src, staging / name)]
                for f, dst in pairs:
                    size, digest = _copy(f, dst)
                    rows.append((dst.relative_to(staging).as_posix(), size, digest,
                                 f.relative_to(plan['folder']).as_posix()))
        for rel, size, digest, _ in rows:                  # verify every copy
            dst = staging / rel
            if dst.stat().st_size != size or _sha256(dst) != digest:
                raise OSError(f'copy of {rel} does not match the original')
        described = dict(plan['described'])
        described['ingested'] = now.isoformat(timespec='seconds')
        described['ingested_by'] = runner
        (staging / plan['session_file']).write_text(
            '# Filed by raphael-data ingest; describing fields may be corrected later.\n'
            + yaml.safe_dump(described, sort_keys=False, allow_unicode=True))
        manifest = staging / f'{session.stem}_manifest.tsv'
        manifest.write_text('path\tbytes\tsha256\toriginal_name\n' + ''.join(
            f'{r}\t{s}\t{d}\t{o}\n' for r, s, d, o in rows))
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        try:
            (root / 'raw' / '.ingesting').rmdir()            # only if empty
        except OSError:
            pass
    for rel, *_ in rows:                                    # lock the recordings
        try:
            os.chmod(target / rel, 0o444)
        except OSError:
            pass
    shutil.rmtree(plan['folder'])
    return target


def _update(src, target, identity, report, stamp, dry_run):
    """Add or update a describing file in raw/ from a hand-in. Identity
    fields that differ are left for the data manager. Returns True when
    the hand-in's copy is in raw/ (added, unchanged or updated)."""
    new, _ = _yaml_file(src)
    old, _ = _yaml_file(target) if target.exists() else (None, None)
    rel = target.as_posix().split('/raw/', 1)[-1]
    if old is None:
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        report.notes.append((src.name, f'{"would be " if dry_run else ""}added as raw/{rel}'))
        return True
    if old == new:
        return True
    if isinstance(old, dict) and isinstance(new, dict):
        changed = [k for k in identity if _text(old.get(k)) != _text(new.get(k))]
        if changed:
            report.notes.append((src.name, (f'not applied: {", ".join(changed)} differ from '
                                 f'raw/{rel}; the data manager must confirm identity changes')))
            return False
    if not dry_run:
        history = target.parent / '.history'
        history.mkdir(exist_ok=True)
        shutil.copy2(target, history / f'{target.stem}.{stamp}{target.suffix}')
        shutil.copy2(src, target)
    report.notes.append((src.name, (f'{"would update" if dry_run else "updated"} raw/{rel} '
                                   '(old version kept in .history/)')))
    return True


def update_describing(handin, root, report, stamp, dry_run):
    """Put each mouse's cohort, subject and procedures files into raw/ (when
    all are valid and the cohort agrees with raw/), and remove them from the
    hand-in once applied and no session left there still needs them."""
    root = Path(root)
    quiet = Report(handin.user)                       # checks already reported per session
    left = {SESSION_FOLDER.match(p.name)['mouse'] for p in handin.sessions
            if dry_run or p.exists()}
    cohort_needed = {}                                # cohort -> still needed in incoming?
    for mouse, path in handin.subjects.items():
        subject = _check_subject(path, mouse, quiet if mouse in left else report)
        if subject is None:
            continue
        cohort = _text(subject['cohort'])
        cohort_path = handin.cohorts.get(cohort)
        proc_path = handin.procedures.get(mouse)
        before, _, _ = mouse_history(root / 'raw', mouse)
        problems = []
        if cohort_path is None:
            problems.append((path.name, f'no {cohort}_cohort.yaml beside it',
                             'hand in the cohort’s file with the subject file'))
        elif _check_cohort(cohort_path, cohort, quiet,
                           new=not (root / 'raw' / cohort).is_dir()) is None:
            problems.append((cohort_path.name, 'invalid cohort file', ''))
        if before and cohort not in before:
            problems.append((path.name, (f'mouse {mouse} is already in '
                             f'{", ".join(sorted(before))}, not {cohort}'),
                             'a mouse’s cohort can’t change: ask the data manager'))
        if proc_path is not None and _check_procedures(proc_path, quiet) is None:
            problems.append((proc_path.name, 'invalid procedures file', ''))
        if problems:
            if mouse not in left:                     # sessions report their own problems
                report.problems += problems
            cohort_needed[cohort] = True
            continue
        mouse_dir = root / 'raw' / cohort / f'sub-M{mouse}'
        ok = _update(cohort_path, root / 'raw' / cohort / 'cohort.yaml', ('cohort',),
                     report, stamp, dry_run)
        ok &= _update(path, mouse_dir / path.name, SUBJECT_IDENTITY, report, stamp, dry_run)
        if proc_path is not None:
            ok &= _update(proc_path, mouse_dir / proc_path.name, (), report, stamp, dry_run)
        done = ok and mouse not in left
        cohort_needed[cohort] = cohort_needed.get(cohort, False) or not done
        if done and not dry_run:
            path.unlink()
            if proc_path is not None:
                proc_path.unlink()
    for cohort, path in handin.cohorts.items():
        if not dry_run and cohort in cohort_needed and not cohort_needed[cohort]:
            path.unlink()
    for mouse, path in handin.procedures.items():
        if mouse not in handin.subjects:
            report.problems.append((path.name, 'handed in without its subject file',
                                    f'hand in sub-M{mouse}_subject.yaml with it'))


# --------------------------------------------------------------- the run

def _plan_file(root, user, stamp, folders):
    """Write metadata/plans/plan_{stamp}_{user}.csv for the sessions filed."""
    from dataclasses import replace

    from raphael.cluster import batch

    raw_root = Path(root) / 'raw'
    recs = []
    for folder in folders:
        prefix = folder.relative_to(raw_root).as_posix()
        recs += [replace(r, path=f'{prefix}/{r.path}') for r in raw.scan(folder)]
    table, _ = batch.plan(recs, raw_root)
    if table.empty:
        return ''
    out = Path(root) / 'metadata' / 'plans' / f'plan_{stamp}_{user}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return out.relative_to(root).as_posix()


def _email(report, lists, dry_run, body):
    """Send the report to the person and the data managers, when lab.yaml
    has mail settings. Returns a note on what happened."""
    labinfo = lists['lab'] or {}
    mail = labinfo.get('mail') or {}
    if dry_run or not mail.get('host'):
        return ''
    people = lists['people']
    to = [people.get(report.user, {}).get('email')]
    to += [people.get(u, {}).get('email') for u in labinfo.get('data_managers') or []]
    to = [a for a in dict.fromkeys(to) if a]
    if not to:
        return 'no email addresses in people.yaml'
    import smtplib
    from email.message import EmailMessage
    msg = EmailMessage()
    msg['Subject'] = (f'Data hand-in from {report.user}: {len(report.filed)} added, '
                      f'{len(report.problems)} problem(s)')
    msg['From'] = mail.get('from') or f'raphael-data@{mail["host"]}'
    msg['To'] = ', '.join(to)
    msg.set_content(body)
    try:
        with smtplib.SMTP(mail['host'], int(mail.get('port', 25)), timeout=30) as s:
            if mail.get('starttls'):
                s.starttls()
            s.send_message(msg)
        return f'emailed {", ".join(to)}'
    except (OSError, smtplib.SMTPException) as e:
        return f'email failed: {e}'


def ingest(root, users=None, dry_run=False, now=None, runner=None):
    """Ingest every hand-in under `{root}/incoming/` (or only `users`').
    Returns [Report], one per person with a hand-in."""
    root = Path(root)
    now = now or datetime.now()
    runner = runner or getpass.getuser()
    stamp = now.strftime('%Y-%m-%d_%H-%M-%S')
    lists = lab(root)
    incoming = root / 'incoming'
    folders = sorted(p for p in incoming.iterdir() if p.is_dir()
                     and not p.name.startswith(('.', '_'))) if incoming.is_dir() else []
    if users:
        folders = [p for p in folders if p.name in users]
    reports, pending = [], {}
    for folder in folders:
        handin = read_handin(folder)
        if not (handin.sessions or handin.subjects or handin.procedures or handin.cohorts
                or handin.other):
            continue
        report = Report(handin.user)
        for p in handin.other:
            report.problems.append((p.name, 'not a session folder or describing file',
                                    ('session folders are named '
                                    'sub-M{mouse}_day-{dd}_ses-{type}_{date}_{time}')))
        # Sessions in time order, so two on one day are numbered against each other.
        order = sorted(handin.sessions, key=lambda p: SESSION_FOLDER.match(p.name)['date']
                       + SESSION_FOLDER.match(p.name)['time'])
        filed_folders = []
        for session_folder in order:
            plan = plan_session(session_folder, handin, root, lists, report, pending)
            if plan is None:
                continue
            pending.setdefault(plan['mouse'], {})[plan['day']] = plan['date']
            target = plan['session'].folder(root / 'raw' / plan['cohort'])
            if not dry_run:
                try:
                    target = file_session(plan, root, runner, now)
                except (OSError, FileExistsError) as e:
                    report.problems.append((session_folder.name, f'filing failed: {e}',
                                            'nothing was changed; run ingest again'))
                    continue
                filed_folders.append(target)
            report.filed.append(target.relative_to(root).as_posix())
        update_describing(handin, root, report, stamp, dry_run)
        if filed_folders:
            report.plan = _plan_file(root, handin.user, stamp, filed_folders)
        reports.append(report)
        body = report.text(dry_run)
        problems_md = folder / 'PROBLEMS.md'
        if not dry_run:
            if report.problems:
                problems_md.write_text(body)
            elif problems_md.exists():
                problems_md.unlink()
            sent = _email(report, lists, dry_run, body)
            if sent:
                report.notes.append(('email', sent))
    if not dry_run and any(r.filed for r in reports):
        from raphael import dataserver
        dataserver.catalogue(root)
    return reports
