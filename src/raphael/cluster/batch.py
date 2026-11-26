"""From raw recordings to job bundles, for one session or a whole project.

`raphael.raw` says what each recording is (modality, start time, frame
rate, and the mouse, day and session type when the path follows the raw
layout). This module turns recordings into jobs for each modality's default
stages, with the inputs, parameters and resources each needs, and groups
them into batches that fit the cluster's disk:

1. `plan`: one row per job (CSV). Checked and, if needed, edited by hand.
2. `batch_files`: the raw files one batch needs, for `rsync --files-from`.
3. `make_jobs`: that batch's job bundles (`raphael-jobs make-batch`).

docs/processing.md walks through it.
"""

import json
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd

from raphael import raw
from raphael.cluster import provenance

#: What each stage's job asks for unless the plan says otherwise. cnmfe's
#: memory and time follow from the movie (see _cnmfe_resources).
STAGE_RESOURCES = {
    'kilosort4': {'gpu': True, 'gpu_type': 'a16', 'cpus': 8, 'mem_gb': 64, 'hours': 8},
    'lfp': {'gpu': False, 'gpu_type': '', 'cpus': 8, 'mem_gb': 32, 'hours': 4},
    'suite2p': {'gpu': True, 'gpu_type': 'a16', 'cpus': 8, 'mem_gb': 32, 'hours': 2},
    'cnmfe': {'gpu': False, 'gpu_type': '', 'cpus': 16, 'mem_gb': 32, 'hours': 2},
    'photometry': {'gpu': False, 'gpu_type': '', 'cpus': 2, 'mem_gb': 8, 'hours': 1},
}
#: Stages that take all of a mouse's recordings of one day in one job
#: (Kilosort sorts them together, so a unit keeps its id across the day).
PER_DAY = ('kilosort4', 'lfp')
#: Parameters a stage can't run without, and where they come from.
REQUIRED = {'kilosort4': ('session_start',), 'lfp': ('session_start',),
            'suite2p': ('session_start', 'fs')}
PLAN_COLUMNS = [
    'job_id', 'batch', 'stage', 'modality', 'inputs', 'cohort', 'mouse', 'day', 'session_type',
    'session_start', 'fs', 'sex', 'date_of_birth', 'strain', 'raw_gb', 'gpu', 'gpu_type',
    'cpus', 'mem_gb', 'hours', 'params', 'ready', 'note',
]
SEP = ';'                                       # several inputs in one plan cell
#: Small files recorded beside an Inscopix movie, copied with it.
INSCOPIX_SIDECARS = ('{stem}.gpio', '{stem}.imu', 'session.json', 'apply_session.json')
#: cnmfe timing, measured on MSC (2026-11-25): 7 min 45 s for 6,350 binned
#: frames of 400 x 640 on 16 CPUs. Jobs ask for 3x that, at least 2 hours.
CNMFE_MINUTES_PER_MEGAPIXEL_FRAME = 7.75 / (6350 * 400 * 640 / 1e6)


def _text(value):
    return '' if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)


# ------------------------------------------------------------------- names

def _with_input(sessions):
    """The table if it can name recordings by path (an `input` column, or
    the port's `cohort` + `raw_recording`), else None."""
    if sessions is None or 'input' in sessions or {'cohort', 'raw_recording'} <= set(
            sessions.columns):
        return sessions
    return None


def _session_lookup(sessions):
    """input path -> {mouse, day, session_type, session_start, fs} from a
    session table. Columns: `input` (or the port's `cohort` + `raw_recording`),
    `mouse`, `day`, `session_type` (or `type`); optional `session_start`
    (or `start`), `fs`."""
    if sessions is None:
        return {}
    t = sessions.copy()
    if 'input' not in t and {'cohort', 'raw_recording'} <= set(t.columns):
        t['input'] = [f'{c}/{r}' if _text(r) else '' for c, r in
                      zip(t['cohort'], t['raw_recording'], strict=True)]
    if 'session_type' not in t and 'type' in t:
        t['session_type'] = t['type']
    t['session_start'] = _start_column(t)
    missing = {'input', 'mouse', 'day', 'session_type'} - set(t.columns)
    if missing:
        raise ValueError(f'the session table needs columns {sorted(missing)}')
    keep = [c for c in ('cohort', 'mouse', 'day', 'session_type', 'session_start', 'fs')
            if c in t]
    return {_text(r['input']): r for r in t[['input', *keep]].to_dict('records')
            if _text(r['input'])}


def _start_column(t):
    """A session table's start time as one column: `session_start`, `start`,
    or `date` + `time`."""
    for name in ('session_start', 'start'):
        if name in t:
            return t[name].map(_text)
    if {'date', 'time'} <= set(t.columns):
        return (t['date'].map(_text) + 'T' + t['time'].map(_text)).where(
            t['date'].map(bool) & t['time'].map(bool), '')
    return pd.Series([''] * len(t), index=t.index)


def _session_times(sessions):
    """(mouse, day, session type) -> {'session_start', 'fs'} from a session
    table such as the raw layout's metadata/sessions.csv (mouse, day, type,
    start, ...)."""
    if sessions is None or not {'mouse', 'day'} <= set(sessions.columns):
        return {}
    kind = 'session_type' if 'session_type' in sessions else 'type'
    if kind not in sessions:
        return {}
    starts = _start_column(sessions)
    out = {}
    for i, r in sessions.iterrows():
        try:
            key = (str(r['mouse']), int(r['day']), str(r[kind]))
        except (TypeError, ValueError):
            continue
        out[key] = {'session_start': starts[i], 'fs': _text(r.get('fs'))}
    return out


def metadata_tables(path):
    """(sessions, subjects) from the raw layout's metadata/ folder: the
    `metadata` beside the `raw` folder that holds `path`; (None, None) if
    there is none."""
    for parent in [Path(path), *Path(path).parents]:
        if parent.name == 'raw':
            folder = parent.parent / 'metadata'
            tables = [folder / 'sessions.csv', folder / 'subjects.csv']
            return tuple(pd.read_csv(t, dtype={'mouse': str}) if t.exists() else None
                         for t in tables)
    return None, None


def _subjects(subjects):
    if subjects is None:
        return {}
    s = subjects.assign(mouse=subjects['mouse'].astype(str)).drop_duplicates('mouse')
    return {r['mouse']: r for r in s.to_dict('records')}


# --------------------------------------------------------------- resources

def _cnmfe_resources(path, settings):
    """Memory and hours for one Inscopix movie, from its description, and
    the cnmfe overrides it needs (a rate that can't be binned to the target
    is kept as recorded)."""
    from raphael.imaging import cnmfe, inscopix

    s = cnmfe.settings_with_defaults(settings.get('cnmfe') or {})
    spatial = int(s['spatial_downsample'])
    movie = inscopix.open_isxd(path)
    overrides, note = {}, ''
    try:
        factor = cnmfe.temporal_factor(movie.rate, s)
    except cnmfe.CnmfeError:
        factor, overrides['temporal_downsample'] = 1, 1
        note = f'{movie.rate:.2f} Hz kept at its own rate'
    binned = movie.n_frames // factor
    pixels = (movie.height // spatial) * (movie.width // spatial)
    need_gb = binned * pixels * 4 * cnmfe.MEMORY_FACTOR / 1e9 + 8
    minutes = CNMFE_MINUTES_PER_MEGAPIXEL_FRAME * binned * pixels / 1e6
    return ({'mem_gb': int(8 * math.ceil(need_gb / 8)),
             'hours': max(2, math.ceil(3 * minutes / 60))}, overrides, note)


def _settings(modality, configs=None):
    from raphael.imaging.layout import load_settings

    path = config_path(modality, configs)
    return load_settings(path) if path.suffix == '.yaml' and path.exists() else {}


def config_path(modality, configs=None):
    """The stage config a job of this modality gets: the project's copy in
    `configs` if it has one, else Raphael's default."""
    name = raw.CONFIGS[modality]
    if configs is not None and (Path(configs) / name).exists():
        return Path(configs) / name
    return provenance.PACKAGE_ROOT / 'configs' / name


# -------------------------------------------------------------------- jobs

def job_row(stage, recordings, data_root, configs=None, subject=None):
    """One job (a plan row, without job_id and batch) for `stage` on
    `recordings` (one, or a day's for PER_DAY stages), all with names."""
    recs = sorted(recordings, key=lambda r: (r.start or '~', r.session_type or ''))
    first = recs[0]
    modality = first.modality
    res = dict(STAGE_RESOURCES[stage])
    params, notes = {}, [r.note for r in recs if r.note]
    if stage == 'cnmfe':
        try:
            sized, overrides, note = _cnmfe_resources(Path(data_root) / first.path,
                                                      _settings(modality, configs))
            res.update(sized)
            if overrides:
                params['cnmfe'] = overrides
            notes += [note] if note else []
        except (OSError, ValueError) as e:
            notes.append(f'cannot size the job: {str(e)[:100]}')
    multi = stage in PER_DAY
    starts = [r.start or '' for r in recs]
    row = {
        'stage': stage, 'modality': modality,
        'inputs': SEP.join(r.path for r in recs),
        'cohort': first.cohort or '', 'mouse': first.mouse, 'day': first.day,
        'session_type': (SEP.join(_text(r.session_type) for r in recs) if multi
                         else first.session_type),
        'session_start': SEP.join(starts) if multi else starts[0],
        'fs': first.rate_hz if stage == 'suite2p' and first.rate_hz else '',
        'sex': _text((subject or {}).get('sex')),
        'date_of_birth': _text((subject or {}).get('date_of_birth')),
        'strain': _text((subject or {}).get('strain')),
        'raw_gb': round(sum(r.bytes for r in recs) / 1e9, 1),
        **res, 'params': json.dumps(params) if params else '',
    }
    missing = [k for k in REQUIRED.get(stage, ()) if not all(_text(v) for v in (
        row[k].split(SEP) if isinstance(row[k], str) else [row[k]]))]
    if missing:
        notes.append(f'needs {", ".join(missing)}: fill in the plan')
    if any(r.note.startswith('unreadable') for r in recs):
        missing.append('a readable recording')
    row['ready'] = not missing
    row['note'] = '; '.join(dict.fromkeys(n for n in notes if n))
    return row


def plan(recordings, data_root, sessions=None, subjects=None, batch_size=12, batch_gb=400,
         stages=None, configs=None):
    """The plan: one row per job, in batches. Returns (table, skipped).

    Names (mouse, day, session type) come from the raw layout, else from
    `sessions`, which can also supply start times and frame rates the files
    lack; recordings with no name are skipped. A batch holds at most
    `batch_size` jobs' worth of recordings and `batch_gb` of raw data (one
    larger recording gets a batch of its own); a day's ephys recordings stay
    together. `stages` limits the plan to some stages."""
    names, subj = _session_lookup(_with_input(sessions)), _subjects(subjects)
    times = _session_times(sessions)
    named, skipped = [], []
    for r in map(replace, recordings):            # copies: names are filled in below
        info = names.get(r.path)
        if info is not None:
            r.mouse, r.day = str(info['mouse']), int(info['day'])
            r.cohort = r.cohort or (_text(info.get('cohort')) or None)
            r.session_type = str(info['session_type'])
            r.start = r.start or _text(info.get('session_start')) or None
            r.rate_hz = r.rate_hz or (float(info['fs']) if _text(info.get('fs')) else None)
        if r.mouse is None:
            skipped.append((r.path, 'no mouse/day/type (not in the raw layout or --sessions)'))
            continue
        fill = times.get((str(r.mouse), int(r.day), str(r.session_type)), {})
        r.start = r.start or fill.get('session_start') or None
        r.rate_hz = r.rate_hz or (float(fill['fs']) if fill.get('fs') else None)
        named.append(r)

    # Units of copying: a day's ephys recordings together, else one recording.
    units = {}
    for r in named:
        key = (r.cohort, r.mouse, r.day, 'ephys') if r.modality == 'neuropixels' else (r.path,)
        units.setdefault(key, []).append(r)
    modalities = list(raw.MODALITIES)
    order = sorted(units.values(), key=lambda rs: (rs[0].cohort or '', str(rs[0].mouse), rs[0].day,
                                                   rs[0].session_type or '',
                                                   modalities.index(rs[0].modality)))
    rows, batch, n, gb = [], 1, 0, 0.0
    for recs in order:
        size = sum(r.bytes for r in recs) / 1e9
        if n and (n == batch_size or gb + size > batch_gb):
            batch, n, gb = batch + 1, 0, 0.0
        n, gb = n + 1, gb + size
        modality = recs[0].modality
        for stage in raw.MODALITIES[modality][0]:
            if stages and stage not in stages:
                continue
            groups = [recs] if stage in PER_DAY else [[r] for r in recs]
            for group in groups:
                row = job_row(stage, group, data_root, configs, subj.get(str(group[0].mouse)))
                row['batch'] = batch
                suffix = '' if stage in PER_DAY else f'-{group[0].session_type}'
                row['job_id'] = f'{stage}-b{batch:02d}-M{group[0].mouse}-D{group[0].day}{suffix}'
                rows.append(row)
    table = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    return table, skipped


def read_plan(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _inputs(row):
    return [p for p in str(row['inputs']).split(SEP) if p]


def batch_files(plan_table, batch, data_root=None):
    """Paths (relative to the data root) to copy for one batch: every file
    of each recording folder; each movie or file, with an Inscopix movie's
    small companions. Without `data_root`, the recordings' paths only."""
    files = []
    for row in plan_table[plan_table['batch'].astype(int) == int(batch)].to_dict('records'):
        for rel in _inputs(row):
            if data_root is None:
                files.append(rel)
                continue
            path = Path(data_root) / rel
            if path.is_dir():
                files += [f.relative_to(data_root).as_posix()
                          for f in sorted(path.rglob('*')) if f.is_file()]
                continue
            files.append(rel)
            if path.suffix.lower() == '.isxd':
                stem = raw.COPY.sub('', path.stem)
                for pattern in INSCOPIX_SIDECARS:
                    side = path.parent / pattern.format(stem=stem)
                    if side.is_file():
                        files.append(side.relative_to(data_root).as_posix())
    return list(dict.fromkeys(files))


def collect_commands(plan_table, batch, source, processed):
    """The `rsync` commands that copy one batch's outputs into the processed
    tree: each job's `outputs/M*` (its sessions' folders) into
    `processed/{cohort}/` (`processed/` when the plan has no cohort).
    `source` is the jobs folder, local or remote (`user@host:jobs`)."""
    commands = []
    for row in plan_table[plan_table['batch'].astype(int) == int(batch)].to_dict('records'):
        dest = Path(processed) / _text(row.get('cohort')) if _text(row.get('cohort')) \
            else Path(processed)
        if ':' in str(source) and not Path(source).exists():   # remote; its shell expands M*
            sources = [f'{str(source).rstrip("/")}/{row["job_id"]}/outputs/M*']
        else:                                  # local ('/' paths: rsync's form everywhere)
            outputs = Path(source) / row['job_id'] / 'outputs'
            sources = [p.as_posix() for p in sorted(outputs.glob('M*'))]
        commands.append((row['job_id'], dest, ['rsync', '-a', *sources, f'{dest.as_posix()}/']))
    return commands


def row_params(row):
    """A plan row's job parameters."""
    multi = row['stage'] in PER_DAY
    params = {}
    if _text(row.get('mouse')) and _text(row.get('mouse')) != 'None':
        params['mouse'] = str(row['mouse'])
    if _text(row.get('day')) and _text(row.get('day')) != 'None':
        params['day'] = int(row['day'])
    types = [t for t in _text(row.get('session_type')).split(SEP) if t and t != 'None']
    if types:
        params['session_type'] = types if multi and len(types) > 1 else types[0]
    starts = [s for s in str(row['session_start']).split(SEP)]
    if all(starts):
        params['session_start'] = starts if multi and len(starts) > 1 else starts[0]
    if _text(row.get('fs')):
        params['fs'] = float(row['fs'])
    subject = {k: str(row[k]) for k in ('sex', 'date_of_birth', 'strain') if _text(row.get(k))}
    if subject:
        params['subject'] = subject
    if _text(row.get('params')):
        params.update(json.loads(row['params']))
    return params


def row_resources(row):
    gpu = str(row['gpu']).lower() in ('true', '1')
    return {'gpu': gpu, 'gpu_type': _text(row.get('gpu_type')) or None if gpu else None,
            'cpus': int(row['cpus']), 'mem_gb': int(row['mem_gb']), 'hours': int(row['hours'])}


def make_jobs(plan_table, batch, data_root, jobs_root, configs=None, hash_inputs=True,
              site=None):
    """The job bundles of one batch, in `jobs_root`. Jobs already made and
    rows not ready (missing a start time or frame rate) are left out.
    Returns (made, existing, not_ready)."""
    from raphael.cluster import jobs

    made, existing, not_ready = [], [], []
    rows = plan_table[plan_table['batch'].astype(int) == int(batch)].to_dict('records')
    for row in rows:
        job_dir = Path(jobs_root) / row['job_id']
        if str(row['ready']).lower() not in ('true', '1'):
            not_ready.append((row['job_id'], row['note']))
            continue
        if job_dir.exists() and any(job_dir.iterdir()):
            existing.append(job_dir)
            continue
        kwargs = {'site': site} if site is not None else {}
        made.append(jobs.make(job_dir, row['stage'], _inputs(row), data_root, row_params(row),
                              row_resources(row), [config_path(row['modality'], configs)],
                              hash_inputs=hash_inputs, **kwargs))
    return made, existing, not_ready


def auto_job(inputs, data_root, stage=None, configs=None, timezone='Europe/Vienna'):
    """What `raphael-jobs make` fills in when it can read the inputs:
    (stage, params, resources, config) for the recordings at `inputs`
    (relative to `data_root`), or None if they aren't recordings Raphael
    recognises. With no `stage`, the modality's first default stage."""
    recs = [raw.detect(Path(data_root) / i, data_root, timezone) for i in inputs]
    if not recs or any(r is None for r in recs):
        return None
    modalities = {r.modality for r in recs}
    if len(modalities) > 1:
        raise ValueError(f'the inputs are of different modalities: {sorted(modalities)}')
    modality = modalities.pop()
    stages = raw.MODALITIES[modality][0]
    if stage is None:
        stage = stages[0]
    elif stage not in stages:
        return None
    sessions, subjects = metadata_tables(Path(data_root) / inputs[0])
    times = _session_times(sessions)
    for r in recs:
        if r.mouse is not None:
            fill = times.get((str(r.mouse), int(r.day), str(r.session_type)), {})
            r.start = r.start or fill.get('session_start') or None
            r.rate_hz = r.rate_hz or (float(fill['fs']) if fill.get('fs') else None)
    subject = _subjects(subjects).get(str(recs[0].mouse)) if recs[0].mouse else None
    row = job_row(stage, recs, data_root, configs, subject)
    return stage, row_params(row), row_resources(row), config_path(modality, configs), row
