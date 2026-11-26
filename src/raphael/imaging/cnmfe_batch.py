"""Many Inscopix sessions through the `cnmfe` stage, in batches.

A cluster's home disk rarely holds a whole project's raw movies, so the
sessions go through in batches: copy a batch's movies, run its jobs, bring
the outputs back, delete the movies, then the next batch. Three steps, each
a `raphael-imaging` command (docs/msc_cluster.md section 11):

1. `cnmfe-plan` (where the raw data is mounted): one row per session, with
   its job name, batch, input, output names, subject, and the memory and
   time its job needs, worked out from each movie's own description. The
   plan is a CSV: check it, and edit it if a session needs something else.
2. `cnmfe-files PLAN --batch N`: the files to copy for a batch (the movie,
   and the small files recorded beside it), for `rsync --files-from`.
3. `cnmfe-jobs PLAN --batch N` (on the cluster, where the copied movies are):
   one job bundle per session, ready for `sbatch`.
"""

import json
import math
from pathlib import Path

import pandas as pd

from raphael.imaging import cnmfe, inscopix

PLAN_COLUMNS = [
    'job_id', 'batch', 'input', 'mouse', 'day', 'session_type', 'sex', 'date_of_birth',
    'strain', 'frames', 'rate_hz', 'width', 'height', 'temporal_downsample', 'binned_frames',
    'raw_gb', 'mem_gb', 'hours', 'cnmfe', 'note',
]
#: Files recorded beside a movie, worth copying with it (small).
SIDECARS = ('{stem}.gpio', '{stem}.imu', 'session.json', 'apply_session.json')
#: Measured on MSC (2026-11-25): 7 min 45 s for 6,350 binned frames of
#: 400 x 640 with 16 CPUs. Jobs ask for 3x the time this predicts, and at
#: least 2 hours.
MINUTES_PER_MEGAPIXEL_FRAME = 7.75 / (6350 * 400 * 640 / 1e6)


def _text(value):
    return '' if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)


def _sessions(table):
    """`input`, `mouse`, `day`, `session_type` (+ subject columns) from a
    session table. Accepts the port's metadata/sessions.csv too: `cohort` +
    `raw_recording` for input, `type` for session_type."""
    t = table.copy()
    if 'input' not in t and {'cohort', 'raw_recording'} <= set(t.columns):
        t['input'] = [f'{c}/{r}' if _text(r) else '' for c, r in
                      zip(t['cohort'], t['raw_recording'], strict=True)]
    if 'session_type' not in t and 'type' in t:
        t['session_type'] = t['type']
    missing = {'input', 'mouse', 'day', 'session_type'} - set(t.columns)
    if missing:
        raise ValueError(f'the session table needs columns {sorted(missing)}')
    return t


def plan(sessions, data_root, settings, subjects=None, batch_size=12, batch_gb=400, cpus=16,
         prefix='cnmfe'):
    """The plan: one row per session with a raw movie under `data_root`.
    Sessions without one are left out and named in `skipped`. A batch holds
    at most `batch_size` sessions and `batch_gb` of movies (a single larger
    movie gets a batch of its own), so the cluster disk never holds more.

    Memory follows the stage's own check (MEMORY_FACTOR x the float32 binned
    movie, plus 8 GB, in steps of 8 GB). A rate the configured
    `temporal_downsample` can't reach (40 Hz with a 30 Hz target) is kept at
    its own rate (`cnmfe` override `{"temporal_downsample": 1}`), not binned
    to an uneven rate."""
    s = cnmfe.settings_with_defaults(settings.get('cnmfe') or {})
    spatial = int(s['spatial_downsample'])
    t = _sessions(sessions)
    if subjects is not None:                   # the subject table wins where it has a value
        subj = subjects.assign(mouse=subjects['mouse'].astype(str)).drop_duplicates('mouse')
        subj = subj.set_index('mouse')
        for column in ('sex', 'date_of_birth', 'strain'):
            if column in subj:
                value = t['mouse'].astype(str).map(subj[column])
                t[column] = value.fillna(t[column]) if column in t else value
    rows, skipped = [], []
    for r in t.sort_values(['input']).to_dict('records'):
        rel = _text(r['input'])
        if not rel:
            skipped.append((r['mouse'], r['day'], r['session_type'], 'no raw movie'))
            continue
        path = Path(data_root) / rel
        try:
            rec = inscopix.open_isxd(path)
        except (OSError, inscopix.InscopixError) as e:
            skipped.append((r['mouse'], r['day'], r['session_type'], str(e)[:120]))
            continue
        overrides, note = {}, ''
        try:
            factor = cnmfe.temporal_factor(rec.rate, s)
        except cnmfe.CnmfeError:
            factor, overrides['temporal_downsample'] = 1, 1
            note = f'{rec.rate:.2f} Hz kept at its own rate'
        binned = rec.n_frames // factor
        pixels = (rec.height // spatial) * (rec.width // spatial)
        need_gb = binned * pixels * 4 * cnmfe.MEMORY_FACTOR / 1e9 + 8
        minutes = MINUTES_PER_MEGAPIXEL_FRAME * binned * pixels / 1e6 * 16 / cpus
        mouse, day, kind = str(r['mouse']), int(r['day']), str(r['session_type'])
        rows.append({
            'input': rel, 'mouse': mouse, 'day': day, 'session_type': kind,
            'sex': _text(r.get('sex')), 'date_of_birth': _text(r.get('date_of_birth')),
            'strain': _text(r.get('strain')), 'frames': rec.n_frames,
            'rate_hz': round(rec.rate, 3), 'width': rec.width, 'height': rec.height,
            'temporal_downsample': factor, 'binned_frames': binned,
            'raw_gb': round(path.stat().st_size / 1e9, 1), '_bytes': path.stat().st_size,
            'mem_gb': int(8 * math.ceil(need_gb / 8)),
            'hours': max(2, math.ceil(3 * minutes / 60)),
            'cnmfe': json.dumps(overrides) if overrides else '', 'note': note,
        })
    columns = [c for c in PLAN_COLUMNS if c not in ('job_id', 'batch')] + ['_bytes']
    out = pd.DataFrame(rows, columns=columns).sort_values(['input']).reset_index(drop=True)
    batches, batch, n, gb = [], 1, 0, 0.0
    for size in out['_bytes'] / 1e9:           # a new batch at batch_size or batch_gb
        if n and (n == batch_size or gb + size > batch_gb):
            batch, n, gb = batch + 1, 0, 0.0
        batches.append(batch)
        n, gb = n + 1, gb + size
    out.insert(0, 'batch', batches)
    out.insert(0, 'job_id', [f'{prefix}-b{b:02d}-M{m}-D{d}-{k}' for b, m, d, k in
                             zip(out['batch'], out['mouse'], out['day'], out['session_type'],
                                 strict=True)])
    return out[PLAN_COLUMNS], skipped


def read_plan(path):
    return pd.read_csv(path, dtype={'mouse': str, 'sex': str, 'date_of_birth': str,
                                    'strain': str, 'cnmfe': str, 'note': str},
                       keep_default_na=False)


def batch_files(plan_table, batch, data_root=None):
    """Paths (relative to the data root) to copy for one batch: each movie,
    and, when `data_root` is given, the small files beside it that exist."""
    files = []
    for rel in plan_table.loc[plan_table['batch'] == batch, 'input']:
        files.append(rel)
        if data_root is None:
            continue
        movie = Path(rel)
        stem = movie.stem.split('_recovered')[0].removesuffix('_fixed')
        for pattern in SIDECARS:
            side = movie.parent / pattern.format(stem=stem)
            if (Path(data_root) / side).is_file():
                files.append(side.as_posix())
    return files


def make_jobs(plan_table, batch, data_root, jobs_root, config, cpus=16, hash_inputs=True,
              site=None):
    """One `cnmfe` job bundle per session of the batch, in `jobs_root`. Jobs
    already made are left as they are. Returns (made, existing)."""
    from raphael.cluster import jobs

    made, existing = [], []
    for r in plan_table[plan_table['batch'] == batch].to_dict('records'):
        job_dir = Path(jobs_root) / r['job_id']
        if job_dir.exists() and any(job_dir.iterdir()):
            existing.append(job_dir)
            continue
        params = {'mouse': r['mouse'], 'day': int(r['day']), 'session_type': r['session_type']}
        subject = {k: str(r[k]) for k in ('sex', 'date_of_birth', 'strain') if _text(r.get(k))}
        if subject:
            params['subject'] = subject
        if _text(r.get('cnmfe')):
            params['cnmfe'] = json.loads(r['cnmfe'])
        resources = {'gpu': False, 'cpus': cpus, 'mem_gb': int(r['mem_gb']),
                     'hours': int(r['hours'])}
        kwargs = {'site': site} if site is not None else {}
        made.append(jobs.make(job_dir, 'cnmfe', [r['input']], data_root, params, resources,
                              [config], hash_inputs=hash_inputs, **kwargs))
    return made, existing
