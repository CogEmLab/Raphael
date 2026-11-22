"""Self-contained job bundles: prepare on a laptop, run on any machine.

A bundle is a small folder that can be copied to a cluster, a GPU
workstation or Colab and run there with `raphael-jobs run`:

    {jobs}/{job_id}/
        job.yaml            what to run: stage, inputs, params, resources, and
                            the uv.lock hash and git commit it was made with
        config/             snapshot of the configs the stage reads
        submit.slurm        scheduler scripts rendered from configs/cluster.yaml
        submit.sge
        status.json         PENDING -> RUNNING -> DONE | FAILED
        logs/               env.json, gpu.json, scheduler output, traceback
        outputs/            results, in the clark2025 tree for imaging stages

Data is not copied into the bundle. Inputs are files or folders (an ephys
recording is a folder) given as paths relative to a data
root that each machine supplies at run time (`--data-root`), so the same
bundle runs wherever a copy of the data lives; sizes and checksums recorded
at creation catch a stale or partial copy before any GPU time is spent.

Stages. Raphael defines the shared ones (`smoke`, `suite2p`, `kilosort4`,
`lfp`). A project adds its own by decorating functions with `stage` in a
module it names under the `raphael.stages` entry-point group in its
pyproject.toml, e.g. in social-manifolds:

    [project.entry-points."raphael.stages"]
    pose = "social_manifolds.pose.stages"

`raphael-jobs` imports every such module before it makes or runs a job, so
`raphael-jobs make --stage dlc_capture` works from the project's checkout.
"""

import json
import os
import shutil
import socket
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import entry_points
from pathlib import Path
from string import Template

import numpy as np
import yaml

from raphael.cluster import gpu, provenance

TEMPLATES = Path(__file__).resolve().parent / 'templates'
DEFAULT_SITE_CONFIG = provenance.PACKAGE_ROOT / 'configs' / 'cluster.yaml'
SCHEDULERS = {'slurm': 'submit.slurm', 'sge': 'submit.sge'}
DEFAULT_RESOURCES = {'gpu': True, 'gpu_backend': 'cuda', 'gpu_type': None,
                     'min_gpu_memory_gb': 0, 'cpus': 4, 'mem_gb': 16, 'hours': 4}
STAGES = {}
CHECKS = {}  # stage name -> f(params, inputs), run by `make` so mistakes surface early
ENTRY_POINT_GROUP = 'raphael.stages'


class JobError(RuntimeError):
    pass


def site_config():
    """The project's configs/cluster.yaml if it has one, else Raphael's."""
    root = provenance.project_root()
    own = root / 'configs' / 'cluster.yaml' if root else None
    return own if own and own.exists() else DEFAULT_SITE_CONFIG


SITE_CONFIG = site_config()

# Per-person settings: the cluster login, and anything else to override in
# the shared site config (deep-merged over it), e.g. paths.data. Never in git.
USER_CONFIG = Path(os.environ.get('RAPHAEL_USER_CONFIG',
                                  Path.home() / '.config' / 'raphael' / 'cluster.yaml'))
USER_ENV = 'RAPHAEL_CLUSTER_USER'


def _merge(base, new):
    out = dict(base)
    for key, value in new.items():
        out[key] = (_merge(out[key], value)
                    if isinstance(value, dict) and isinstance(out.get(key), dict) else value)
    return out


def _fill(value, tokens):
    if isinstance(value, str):
        for token, filled in tokens.items():
            value = value.replace(f'{{{token}}}', filled)
        return value
    if isinstance(value, dict):
        return {k: _fill(v, tokens) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, tokens) for v in value]
    return value


def read_user_config(path=None):
    path = Path(path or USER_CONFIG)
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}


def set_cluster_user(user, path=None):
    """Save this person's cluster login in their user config."""
    path = Path(path or USER_CONFIG)
    cfg = {**read_user_config(path), 'user': user}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def load_site(site=SITE_CONFIG, user_config=None):
    """The site config, with this person's overrides merged in and
    `{user}` (their cluster login) and `{project}` (the project checkout's
    folder name) filled in everywhere."""
    with open(site) as f:
        cfg = yaml.safe_load(f)
    mine = read_user_config(user_config)
    user = os.environ.get(USER_ENV) or mine.pop('user', None)
    mine.pop('user', None)
    cfg = _merge(cfg, mine)
    tokens = {}
    if user:
        tokens['user'] = str(user)
    root = provenance.project_root()
    if root:
        tokens['project'] = root.name
    cfg = _fill(cfg, tokens)
    if '{user}' in json.dumps(cfg):
        raise JobError(f'{site} needs your cluster login: run `raphael-jobs cluster-user '
                       f'YOUR_LOGIN` (saved in {Path(user_config or USER_CONFIG)}), or set '
                       f'{USER_ENV}')
    if '{project}' in json.dumps(cfg):
        raise JobError('{project} in the site config, but no project checkout (a folder '
                       'with uv.lock) at or above here: run from the project, or set '
                       'RAPHAEL_PROJECT_ROOT')
    return cfg


def load_project_stages():
    """Import every module registered under `raphael.stages`, so the stages
    they define join STAGES. Returns the entry-point names loaded."""
    loaded = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        ep.load()
        loaded.append(ep.name)
    return loaded


@dataclass
class Context:
    """What a stage gets: resolved paths, its params and the checked device."""
    job_dir: Path
    inputs: list
    work: Path
    outputs: Path
    config: Path
    params: dict
    device: str


def stage(name, check=None):
    """Register a function `f(ctx)` as a job stage. `check(params, inputs)`,
    if given, runs when the job is made and raises JobError on bad params."""
    def register(f):
        STAGES[name] = f
        if check:
            CHECKS[name] = check
        return f
    return register


def _now():
    return datetime.now(UTC).isoformat(timespec='seconds')


def _write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + '\n')


def set_status(job_dir, state, **extra):
    _write_json(Path(job_dir) / 'status.json', {'state': state, 'time': _now(), **extra})


def read_status(job_dir):
    path = Path(job_dir) / 'status.json'
    return json.loads(path.read_text()) if path.exists() else {'state': 'UNKNOWN'}


def load(job_dir):
    with open(Path(job_dir) / 'job.yaml') as f:
        return yaml.safe_load(f)


def _folder_files(folder):
    """Every file under `folder` except hidden ones (.DS_Store, ._*)."""
    return sorted(p for p in Path(folder).rglob('*') if p.is_file()
                  and not any(part.startswith('.') for part in p.relative_to(folder).parts))


def _record(path, rel, hash_inputs):
    return {'path': str(rel), 'bytes': path.stat().st_size,
            'sha256': provenance.sha256(path) if hash_inputs else None}


def make(job_dir, stage_name, inputs=(), data_root=None, params=None, resources=None,
         configs=(), hash_inputs=True, site=SITE_CONFIG):
    """Create a bundle. `inputs` are files or folders, relative to `data_root`;
    a folder is recorded file by file."""
    if stage_name not in STAGES:
        raise JobError(f'unknown stage {stage_name!r}; known: {", ".join(sorted(STAGES))}')
    if stage_name in CHECKS:
        CHECKS[stage_name](params or {}, list(inputs))
    job_dir = Path(job_dir)
    if job_dir.exists() and any(job_dir.iterdir()):
        raise JobError(f'{job_dir} already exists and is not empty')
    records = []
    for rel in inputs:
        if Path(rel).is_absolute():
            raise JobError(f'inputs must be relative to the data root, got {rel}')
        path = Path(data_root) / rel
        if path.is_file():
            records.append(_record(path, rel, hash_inputs))
        elif path.is_dir():
            files = [_record(f, f.relative_to(path), hash_inputs) for f in _folder_files(path)]
            if not files:
                raise JobError(f'input folder is empty: {path}')
            records.append({'path': str(rel), 'bytes': sum(f['bytes'] for f in files),
                            'files': files})
        else:
            raise JobError(f'input not found: {path}')

    for sub in ('config', 'logs', 'outputs'):
        (job_dir / sub).mkdir(parents=True, exist_ok=True)
    for c in configs:
        shutil.copy2(c, job_dir / 'config' / Path(c).name)
    spec = {
        'job_id': job_dir.name,
        'stage': stage_name,
        'created': _now(),
        'created_on': socket.gethostname(),
        'git': provenance.git_state(),
        'uv_lock_sha256': provenance.lock_hash(),
        'inputs': records,
        'params': params or {},
        'resources': {**DEFAULT_RESOURCES, **(resources or {})},
    }
    (job_dir / 'job.yaml').write_text(yaml.safe_dump(spec, sort_keys=False))
    if spec['git'] and spec['git']['dirty']:
        print('warning: the checkout has uncommitted changes, so this job cannot be '
              'traced to a commit. Commit, then remake the job for anything you keep.')
    set_status(job_dir, 'PENDING')
    try:
        render(job_dir, site)
    except JobError as e:
        print(f'note: scheduler scripts not written ({e}). Fill in {site}, then run '
              f'`raphael-jobs render {job_dir}`.')
    return job_dir


# ---------------------------------------------------------------- scheduler

class _Script(Template):
    delimiter = '@'  # bash owns `$`


def _unfilled(value):
    return value is None or 'TODO' in str(value)


def _gpu_directives(sched, res):
    """The site's GPU request lines, with `{gpu_type}` filled from the job
    (`make --gpu-type`) or the site's default_gpu_type, and `{gres_type}`
    from the site's gres_types map (for types whose GRES name differs)."""
    request = sched.get('gpu_request') or []
    request = [request] if isinstance(request, str) else request
    gpu_type = res.get('gpu_type') or sched.get('default_gpu_type')
    if gpu_type is None and any('_type}' in r for r in request):
        raise JobError('the site requests a GPU type: pass --gpu-type or set default_gpu_type')
    gres_type = (sched.get('gres_types') or {}).get(gpu_type, gpu_type)
    return [r.replace('{gpu_type}', str(gpu_type)).replace('{gres_type}', str(gres_type))
            for r in request]


def render(job_dir, site=SITE_CONFIG):
    """Write submit.slurm and submit.sge for this job from the site config."""
    job_dir = Path(job_dir)
    spec = load(job_dir)
    cfg = load_site(site)
    paths = cfg['paths']
    missing = [k for k in ('repo', 'data', 'jobs') if _unfilled(paths.get(k))]
    if missing:
        raise JobError(f'paths.{", paths.".join(missing)} not set in {site}')
    res = spec['resources']
    remote_job = f'{paths["jobs"].rstrip("/")}/{spec["job_id"]}'
    common = {
        'job_id': spec['job_id'], 'job_dir': remote_job, 'repo': paths['repo'],
        'data': paths['data'],
        'work': (f'{paths["scratch"].rstrip("/")}/{spec["job_id"]}' if paths.get('scratch')
                 else f'{remote_job}/work'),
        'cpus': res['cpus'], 'mem_gb': res['mem_gb'], 'hours': res['hours'],
        'setup': '\n'.join(cfg.get('setup') or []),
        'env': '\n'.join(f'export {k}="{v}"' for k, v in (cfg.get('env') or {}).items()
                         if not _unfilled(v)),
    }
    written = []
    for name, filename in SCHEDULERS.items():
        sched = cfg.get(name) or {}
        if res['gpu']:
            lines = _gpu_directives(sched, res)
            lines += [sched.get(k) for k in ('queue', 'account')
                      if not _unfilled(sched.get(k))]
        else:   # a CPU job: the site's CPU request replaces the GPU queue
            cpu = sched.get('cpu_request') or ([sched['queue']] if not _unfilled(
                sched.get('queue')) else [])
            lines = [cpu] if isinstance(cpu, str) else list(cpu)
            lines += [sched['account']] if not _unfilled(sched.get('account')) else []
        lines += sched.get('extra') or []
        prefix = '#SBATCH' if name == 'slurm' else '#$'
        fields = {**common, 'directives': '\n'.join(f'{prefix} {x}' for x in lines if x),
                  'pe': sched.get('parallel_environment', 'sharedmem'),
                  'mem_per_cpu_gb': max(1, -(-res['mem_gb'] // res['cpus']))}
        text = _Script((TEMPLATES / f'{name}.sh').read_text()).substitute(fields)
        (job_dir / filename).write_text(text)
        written.append(job_dir / filename)
    return written


# --------------------------------------------------------------------- run

def _verify_file(path, rec, hash_inputs):
    if not path.is_file():
        raise JobError(f'input missing: {path}')
    size = path.stat().st_size
    if size != rec['bytes']:
        raise JobError(f'{path} is {size} bytes, expected {rec["bytes"]} '
                       '(partial or different copy?)')
    if hash_inputs and rec.get('sha256') and provenance.sha256(path) != rec['sha256']:
        raise JobError(f'{path} checksum differs from the one recorded at creation')


def verify_inputs(spec, data_root, hash_inputs=True):
    paths = []
    for rec in spec['inputs']:
        path = Path(data_root) / rec['path']
        if 'files' in rec:  # a folder: every recorded file, and nothing extra
            if not path.is_dir():
                raise JobError(f'input folder missing: {path}')
            for f in rec['files']:
                _verify_file(path / f['path'], f, hash_inputs)
            extra = {str(f.relative_to(path)) for f in _folder_files(path)} - {
                f['path'] for f in rec['files']}
            if extra:
                raise JobError(f'{path} has files not there when the job was made: '
                               f'{sorted(extra)[:5]}')
        else:
            _verify_file(path, rec, hash_inputs)
        paths.append(path)
    return paths


def run(job_dir, data_root, work=None, hash_inputs=True, allow_env_mismatch=False):
    """Run a bundle here. Every check happens before the stage starts."""
    job_dir = Path(job_dir).resolve()
    spec = load(job_dir)
    logs = job_dir / 'logs'
    logs.mkdir(exist_ok=True)
    set_status(job_dir, 'RUNNING', host=socket.gethostname())
    try:
        env = provenance.environment()
        _write_json(logs / 'env.json', env)
        made, here = spec.get('uv_lock_sha256'), env['uv_lock_sha256']
        if made != here and not allow_env_mismatch:
            raise JobError('uv.lock here differs from the one the job was made with '
                           f'({here} vs {made}): check out the same commit and '
                           '`uv sync --frozen`, or pass --allow-env-mismatch.')
        inputs = verify_inputs(spec, data_root, hash_inputs)

        res = spec['resources']
        backend = res['gpu_backend'] if res['gpu'] else 'cpu'
        device, gpu_info = gpu.check(backend, res['min_gpu_memory_gb'])
        _write_json(logs / 'gpu.json', gpu_info)

        work = Path(work) if work else job_dir / 'work'
        work.mkdir(parents=True, exist_ok=True)
        ctx = Context(job_dir=job_dir, inputs=inputs, work=work,
                      outputs=job_dir / 'outputs', config=job_dir / 'config',
                      params=spec['params'], device=device)
        started = _now()
        STAGES[spec['stage']](ctx)
        record_provenance(job_dir, spec, env, gpu_info, device, started)
    except BaseException as e:
        (logs / 'traceback.txt').write_text(traceback.format_exc())
        set_status(job_dir, 'FAILED', host=socket.gethostname(),
                   error=f'{type(e).__name__}: {e}')
        raise
    set_status(job_dir, 'DONE', host=socket.gethostname(), device=device)


def record_provenance(job_dir, spec, env, gpu_info, device, started):
    """After a stage succeeds: write provenance/{job_id}.json into every
    session folder it wrote (so the record travels with the processed
    files), and the same record without per-folder outputs to
    logs/provenance.json."""
    job_dir = Path(job_dir)
    configs = {p.name: {'sha256': provenance.sha256(p), 'text': p.read_text()}
               for p in sorted((job_dir / 'config').glob('*')) if p.is_file()}
    record = {
        'job_id': spec['job_id'],
        'stage': spec['stage'],
        'started': started,
        'finished': _now(),
        'host': socket.gethostname(),
        'device': device,
        'raphael': env['raphael'],
        'project': {'root': env['project_root'], 'git': env['git'],
                    'uv_lock_sha256': env['uv_lock_sha256']},
        'job_made': {k: spec.get(k) for k in ('created', 'created_on', 'git',
                                              'uv_lock_sha256')},
        'params': spec['params'],
        'resources': spec['resources'],
        'configs': configs,
        'inputs': spec['inputs'],
        'environment': {k: env[k] for k in ('python', 'platform', 'scheduler', 'packages')},
        'gpu': gpu_info,
    }
    written = provenance.write_records(job_dir / 'outputs', record, spec['job_id'])
    _write_json(job_dir / 'logs' / 'provenance.json',
                {'format': provenance.RECORD_FORMAT, **record,
                 'session_records': [str(p.relative_to(job_dir)) for p in written.values()]})
    return written


def summary(root):
    """(job_id, stage, state, detail) for every bundle under `root`."""
    rows = []
    for spec_file in sorted(Path(root).glob('*/job.yaml')):
        job_dir = spec_file.parent
        status = read_status(job_dir)
        rows.append((job_dir.name, load(job_dir)['stage'], status['state'],
                     status.get('error') or status.get('time', '')))
    return rows


# ------------------------------------------------------------------ stages

@stage('smoke')
def smoke(ctx):
    """End-to-end test of a machine: inputs readable, device computes, outputs
    come back. Run this first on any new cluster."""
    import time

    read = {}
    for path in ctx.inputs:
        start = time.perf_counter()
        with open(path, 'rb') as f:
            n = len(f.read(64 << 20))
        read[path.name] = {'mb': n / 1e6, 'mb_per_s': n / 1e6 / (time.perf_counter() - start)}
    result = {'device': ctx.device, 'inputs_read': read}
    if ctx.device != 'cpu':
        _, info = gpu.check('mps' if ctx.device == 'mps' else 'cuda',
                            size=ctx.params.get('size', 8192))
        result['matmul'] = info['test']
    ctx.outputs.mkdir(exist_ok=True)
    _write_json(ctx.outputs / 'smoke.json', result)


def _ranges(indices):
    """[[start, stop], ...] runs of consecutive frame indices (stop inclusive)."""
    out = []
    for i in map(int, indices):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def _suite2p_metadata(params, inputs=None, timezone='Europe/Vienna'):
    """Recording start time and subject fields from suite2p job params."""
    from raphael.imaging import nwb

    if 'fs' not in params:
        raise JobError('suite2p needs the frame rate: make --param fs=...')
    if 'session_start' not in params:
        raise JobError('suite2p needs the recording start time, which NWB files '
                       'carry: make --param session_start=2026-11-03T14:05 '
                       '(local time) or with a UTC offset (...T14:05+01:00)')
    try:
        start = nwb.parse_start_time(params['session_start'], timezone)
        subject = nwb.subject_fields(params.get('subject'), timezone)
    except ValueError as e:
        raise JobError(str(e)) from e
    return start, subject


@stage('suite2p', check=_suite2p_metadata)
def suite2p_stage(ctx):
    """suite2p on one movie -> processed NWB, traces, ROI table and QC in
    ctx.outputs (clark2025 tree, `src-suite2p`).

    params: fs (frame rate, required); session_start (recording start, ISO
    8601, required; no offset = imaging.yaml's timezone); subject (dict:
    age as P84D or date_of_birth, sex, strain, genotype); mouse, day,
    session_type (output names, default test/0/test); um_per_pixel (pixel
    size, one number or [y, x]); depth_um (imaging depth); indicator, location (else imaging.yaml);
    h5_key (default `data`); suite2p (dict of suite2p setting overrides);
    keep_binary (keep the registered movie, default false: it is ~the size
    of the raw movie); blank_fraction (frames darker than this fraction of
    the median frame are excluded as blank, default 0.3; null to disable).
    """
    import time

    from raphael.imaging import nwb, qc
    from raphael.imaging import suite2p as s2p
    from raphael.imaging.layout import (
        DEFAULT_CONFIG,
        Session,
        load_settings,
    )

    if len(ctx.inputs) != 1:
        raise JobError(f'suite2p takes one movie, got {len(ctx.inputs)} inputs')
    p = ctx.params
    config = ctx.config / 'imaging.yaml'
    settings = load_settings(config if config.exists() else DEFAULT_CONFIG)
    start_time, subject = _suite2p_metadata(p, timezone=settings['nwb'].get('timezone', 'UTC'))
    session = Session(str(p.get('mouse', 'test')), str(p.get('day', '0')),
                      str(p.get('session_type', 'test')))

    timing = {}
    if ctx.device == 'cuda':
        import torch
        torch.cuda.reset_peak_memory_stats()
    h5_key = p.get('h5_key', 'data')
    blank = p.get('blank_fraction', 0.3)
    bad = (s2p.blank_frames(ctx.inputs[0], h5_key, blank)
           if blank and ctx.inputs[0].suffix in ('.h5', '.hdf5') else np.array([], int))
    start = time.perf_counter()
    folder = s2p.run(ctx.inputs[0], ctx.work, p['fs'], ctx.device,
                     p.get('suite2p'), h5_key, bad_frames=bad)
    timing['suite2p_s'] = round(time.perf_counter() - start, 1)

    start = time.perf_counter()
    plane = s2p.to_plane(folder, p['fs'], settings,
                         description=f'suite2p on {ctx.inputs[0].name}.', bad_frames=bad)
    if 'um_per_pixel' in p:
        um = p['um_per_pixel']
        plane.grid_spacing_um = tuple(map(float, um)) if isinstance(um, list) else (float(um),) * 2
    if 'depth_um' in p:
        plane.depth_um = float(p['depth_um'])
    notes = {'pipeline': 'suite2p', 'input': ctx.inputs[0].name, 'job': ctx.job_dir.name,
             'suite2p_overrides': p.get('suite2p') or {},
             'blank_frames': _ranges(bad)}
    ophys = nwb.new_nwbfile(settings, session,
                            f'Processed two-photon data for {session.stem}.',
                            session_start_time=start_time, subject=subject, notes=notes)
    nwb.build_ophys_nwb(ophys, [plane], settings, indicator=p.get('indicator'),
                        location=p.get('location'))
    nwb.write(ophys, session.ophys(ctx.outputs, 'suite2p'))
    nwb.export_tables([plane], session.traces(ctx.outputs, 'suite2p'),
                      session.rois(ctx.outputs, 'suite2p'))
    qc.plane_figure(plane, 1, session.qc(ctx.outputs))
    timing['write_s'] = round(time.perf_counter() - start, 1)

    # suite2p's own files go to outputs, for re-curation in its GUI. The work
    # folder (registered movie, ~the raw movie's size, plus copies of those
    # files) is then deleted unless keep_binary.
    raw = ctx.outputs / 'suite2p' / 'plane0'
    raw.mkdir(parents=True, exist_ok=True)
    for f in folder.glob('*.npy'):
        shutil.copy2(f, raw / f.name)
    if not p.get('keep_binary', False):
        shutil.rmtree(folder.parent)

    summary = {'device': ctx.device, 'frames': len(plane.timestamps),
               'image_shape': list(plane.image_shape), 'rois': len(plane.masks),
               'accepted': int(plane.rois['accepted'].sum()),
               'blank_frames': len(bad), **timing}
    if ctx.device == 'cuda':
        summary['peak_gpu_memory_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    _write_json(ctx.outputs / 'suite2p_summary.json', summary)


def _ephys_metadata(params, inputs=(), timezone='Europe/Vienna'):
    """[(session_type, start time), ...] in input order, and subject fields,
    from kilosort4 job params. One input is one recording (session)."""
    from raphael.imaging import nwb

    n = len(inputs)
    if n == 0:
        raise JobError('kilosort4 needs at least one recording folder: make --input ...')

    def per_input(key, default=None):
        value = params.get(key, default)
        values = value if isinstance(value, list) else [value] * (1 if n == 1 else 0)
        if len(values) != n:
            raise JobError(f'{key} needs one value per input recording ({n}), in input '
                           f"order: make --param '{key}=[..., ...]'")
        return values

    if 'session_start' not in params:
        raise JobError('kilosort4 needs each recording\'s start time, which NWB files '
                       'carry: make --param session_start=2026-11-03T14:05 (local time; '
                       "several recordings: 'session_start=[..., ...]')")
    types = per_input('session_type', 'test' if n == 1 else None)
    if len(set(map(str, types))) != n:
        raise JobError(f'session_type must name each recording differently, got {types}')
    if params.get('format') not in (None, 'spikeglx', 'openephys', 'spikeinterface'):
        raise JobError('format must be spikeglx, openephys or spikeinterface '
                       '(or left out to detect it)')
    try:
        starts = [nwb.parse_start_time(s, timezone) for s in per_input('session_start')]
        subject = nwb.subject_fields(params.get('subject'), timezone)
    except ValueError as e:
        raise JobError(str(e)) from e
    return list(zip(map(str, types), starts, strict=True)), subject


@stage('kilosort4', check=_ephys_metadata)
def kilosort4_stage(ctx):
    """Kilosort 4.0.16 on a day's Neuropixels recordings, as in the Methods
    (see ephys/sorting.py) -> per session: processed NWB, spikes (TsGroup),
    units table, sorting analyser, bombcell curation and QC, in ctx.outputs
    (clark2025 tree, `srt-kilosort4`).

    inputs: one recording folder per session, in recording order (SpikeGLX
    run folder, Open Ephys record node, or a spikeinterface folder).
    params: session_start (each recording's start, ISO 8601, required; a
    list for several inputs; no offset = ephys.yaml's timezone);
    session_type (one per input; default `test` for one input); mouse,
    day (output names, default test/0); subject (dict: age as P84D or
    date_of_birth, sex, strain, genotype); format and stream (else
    detected); location (brain region, else ephys.yaml); kilosort4 (dict
    of Kilosort parameter overrides); preprocessing, postprocessing,
    curation (dicts overriding ephys.yaml's, e.g.
    postprocessing={keep_analyzer: true}).

    The processed tree gets, per session: _ecephys.nwb (units with spike
    times, waveforms, metrics, curation; electrodes), _clusters.npz,
    _units.csv and qc/. Sorting analysers and bombcell curation files stay
    out of it: deleted, or kept in outputs/kilosort4/analyzers/.
    """
    import os
    import time

    from raphael.ephys import nwb as ephys_nwb
    from raphael.ephys import qc
    from raphael.ephys import sorting as srt
    from raphael.ephys.layout import (
        DEFAULT_CONFIG,
        Session,
        load_settings,
    )
    from raphael.imaging import nwb

    p = ctx.params
    config = ctx.config / 'ephys.yaml'
    settings = load_settings(config if config.exists() else DEFAULT_CONFIG)
    sessions, subject = _ephys_metadata(p, ctx.inputs,
                                        timezone=settings['nwb'].get('timezone', 'UTC'))
    sorter = settings['sorter']['name']
    overrides = {**(settings['sorter'].get('params') or {}), **(p.get('kilosort4') or {})}
    preprocessing = {**settings['preprocessing'], **(p.get('preprocessing') or {})}
    curation = {**settings['curation'], **(p.get('curation') or {})}
    post = {**settings['postprocessing'], **(p.get('postprocessing') or {})}
    probe = {**settings['probe'], **({'location': p['location']} if 'location' in p else {})}
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK') or os.cpu_count() or 1)

    # 1-2. Bad channels, phase shift, Kilosort.
    from raphael.ephys import sort_step

    sorted_dir = ctx.work / 'sorted'
    sort_summary = sort_step.run({
        'inputs': [str(i) for i in ctx.inputs], 'format': p.get('format'),
        'stream': p.get('stream'), 'preprocessing': preprocessing,
        'kilosort': overrides, 'device': ctx.device, 'n_jobs': n_jobs,
        'out': str(sorted_dir)})

    # 3-4. Analysers, curation and files.
    import pandas as pd
    import spikeinterface.full as si
    si.set_global_job_kwargs(n_jobs=n_jobs, chunk_duration='1s', progress_bar=False)
    start = time.perf_counter()
    recordings = [srt.read(path, p.get('format'), p.get('stream')) for path in ctx.inputs]
    channels = pd.read_csv(sorted_dir / 'channels.csv', dtype={'channel_id': str})
    joined = srt.read_sorting(sorted_dir)
    lengths = [r.get_num_samples() for r in recordings]
    parts = srt.split(joined, srt.analyzer_recording(recordings, channels, preprocessing | post),
                      lengths)
    # Analysers and curation files are working files: kept outside the
    # processed tree, and only if asked for.
    working = (ctx.outputs / sorter / 'analyzers' if post.get('keep_analyzer')
               else ctx.work / 'analyzers')
    analyzers, tables = {}, {}
    for (session_type, _), (part, recording) in zip(sessions, parts, strict=True):
        session = Session(str(p.get('mouse', 'test')), str(p.get('day', '0')), session_type)
        analyzers[session_type] = srt.analyze(part, recording,
                                              session.analyzer(working, sorter), post)
        labels = srt.label(analyzers[session_type])
        srt.write_curation(labels, session.curation(working, sorter))
        tables[session_type] = srt.units_table(analyzers[session_type], labels, part,
                                                channels)
    srt.curate(tables, curation)

    per_session = []
    for (session_type, start_time), source, n in zip(sessions, ctx.inputs, lengths, strict=True):
        session = Session(str(p.get('mouse', 'test')), str(p.get('day', '0')), session_type)
        analyzer, units = analyzers[session_type], tables[session_type]
        spikes = srt.spike_times(analyzer.sorting)
        rate = analyzer.sampling_frequency
        notes = {'pipeline': f'kilosort {sort_summary["versions"]["kilosort"]}, '
                             f'spikeinterface {sort_summary["versions"]["spikeinterface"]}',
                 'input': source.name, 'sorted_with': [i.name for i in ctx.inputs],
                 'job': ctx.job_dir.name, 'kilosort_params': overrides,
                 'preprocessing': preprocessing, 'curation': curation,
                 'bad_channels': channels.loc[~channels['used'], 'channel_id'].tolist()}
        waveforms = srt.unit_waveforms(analyzer, post['waveform_channels'])
        ephys_nwb.export_clusters(
            units, spikes, n / rate, session.clusters(ctx.outputs, sorter), waveforms,
            channels, rate, analyzer.get_extension('templates').nbefore)
        if post.get('write_ecephys_nwb'):
            ephys = nwb.new_nwbfile(settings, session,
                                    f'Spike-sorted Neuropixels data for {session.stem}.',
                                    session_start_time=start_time, subject=subject,
                                    notes=notes)
            ephys_nwb.build_ecephys_nwb(ephys, units, spikes, waveforms,
                                        channels, n / rate, rate, probe)
            nwb.write(ephys, session.ecephys(ctx.outputs, sorter))
        qc.session_figure(units, channels, analyzer, session.qc(ctx.outputs),
                          title=f'{session.stem}_srt-{sorter}')
        per_session.append({
            'session_type': session_type, 'input': source.name,
            'duration_s': round(n / rate, 1), 'units': len(units),
            'labels': units['bombcell_label'].value_counts().to_dict(),
            'isi_excluded': int(units['isi_excluded'].sum()),
            'accepted': int(units['accepted'].sum())})
    postprocess_s = round(time.perf_counter() - start, 1)

    # Kilosort's own files go to outputs, for phy; the work folder is deleted.
    for f in (sorted_dir / 'kilosort4').rglob('*'):
        if f.is_file() and f.suffix != '.dat':
            dest = ctx.outputs / sorter / f.relative_to(sorted_dir / 'kilosort4')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
    for name in ('sort_summary.json', 'channels.csv'):
        shutil.copy2(sorted_dir / name, ctx.outputs / sorter / name)
    shutil.rmtree(sorted_dir)
    if not post.get('keep_analyzer'):
        shutil.rmtree(working, ignore_errors=True)

    summary = {'device': ctx.device, 'n_jobs': n_jobs,
               'versions': sort_summary['versions'],
               'channels': len(channels), 'channels_used': int(channels['used'].sum()),
               'bad_channels': channels.loc[~channels['used'], 'label'].value_counts().to_dict(),
               'units_sorted': len(joined.unit_ids), 'units_per_shank': sort_summary['shanks'],
               'sessions': per_session, 'kilosort_params': overrides,
               'preprocessing': preprocessing,
               'prepare_s': sort_summary['prepare_s'], 'kilosort_s': sort_summary['kilosort_s'],
               'postprocess_s': postprocess_s,
               'peak_gpu_memory_gb': sort_summary['peak_gpu_memory_gb']}
    _write_json(ctx.outputs / f'{sorter}_summary.json', summary)


def _lfp_settings(settings, overrides):
    from raphael.imaging.suite2p import _deep_update

    lfp = _deep_update(settings['lfp'], overrides or {})
    if lfp['band_hz'][1] >= lfp['rate_hz'] / 2:
        raise JobError(f'lfp.band_hz upper edge ({lfp["band_hz"][1]} Hz) must be below half '
                       f'of rate_hz ({lfp["rate_hz"]} Hz)')
    if lfp['dtype'] not in ('float32', 'float16'):
        raise JobError(f'lfp.dtype must be float32 or float16, got {lfp["dtype"]!r}')
    if lfp['channels']['block_size'] < 1 or lfp['channels']['every'] < 1:
        raise JobError('lfp.channels.block_size and every must be at least 1')
    return lfp


def _lfp_check(params, inputs=(), timezone='Europe/Vienna'):
    from raphael.ephys.layout import DEFAULT_CONFIG, load_settings

    _lfp_settings(load_settings(DEFAULT_CONFIG), params.get('lfp'))
    return _ephys_metadata(params, inputs, timezone)


@stage('lfp', check=_lfp_check)
def lfp_stage(ctx):
    """LFP and theta phase of each Neuropixels recording -> one _lfp.nwb per
    session in ctx.outputs (clark2025 tree). A CPU job: make --cpu.

    inputs: one recording folder per session (as for kilosort4).
    params: session_start, session_type, mouse, day, subject, format,
    stream, location (as for kilosort4); lfp (dict overriding ephys.yaml's
    `lfp`: rate, band, channel blocks, dtype, compression, theta).
    """
    import os
    import time

    from raphael.ephys import lfp as ephys_lfp
    from raphael.ephys import nwb as ephys_nwb
    from raphael.ephys import sorting as srt
    from raphael.ephys.layout import (
        DEFAULT_CONFIG,
        Session,
        load_settings,
    )
    from raphael.imaging import nwb

    p = ctx.params
    config = ctx.config / 'ephys.yaml'
    settings = load_settings(config if config.exists() else DEFAULT_CONFIG)
    sessions, subject = _ephys_metadata(p, ctx.inputs,
                                        timezone=settings['nwb'].get('timezone', 'UTC'))
    lfp = _lfp_settings(settings, p.get('lfp'))
    probe = {**settings['probe'], **({'location': p['location']} if 'location' in p else {})}

    import spikeinterface.full as si
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK') or os.cpu_count() or 1)
    si.set_global_job_kwargs(n_jobs=n_jobs, chunk_duration='1s', progress_bar=False)

    rows = []
    for (session_type, start_time), source in zip(sessions, ctx.inputs, strict=True):
        start = time.perf_counter()
        session = Session(str(p.get('mouse', 'test')), str(p.get('day', '0')), session_type)
        recording = srt.read(source, p.get('format'), p.get('stream'))
        cache = ctx.work / 'lfp' / session_type
        data, rate, channels = ephys_lfp.extract(recording, cache, lfp)
        phase = (ephys_lfp.theta_phase(data, rate, lfp['theta']['band_hz'],
                                       lfp['theta']['min_peak_distance_s'])
                 if lfp['theta']['enabled'] else None)
        notes = {'pipeline': f'spikeinterface {si.__version__}', 'input': source.name,
                 'job': ctx.job_dir.name, 'lfp': lfp}
        nwbfile = nwb.new_nwbfile(settings, session, f'Local field potential for {session.stem}.',
                                  session_start_time=start_time, subject=subject, notes=notes)
        ephys_nwb.build_lfp_nwb(nwbfile, data, rate, channels, probe, lfp, phase)
        path = nwb.write(nwbfile, session.lfp(ctx.outputs))
        shutil.rmtree(cache)
        rows.append({'session_type': session_type, 'input': source.name,
                     'duration_s': round(len(data) / rate, 1), 'rate_hz': rate,
                     'lfp_channels': data.shape[1], 'theta': phase is not None,
                     'file_mb': round(path.stat().st_size / 1e6, 1),
                     'seconds': round(time.perf_counter() - start, 1)})
    _write_json(ctx.outputs / 'lfp_summary.json', {'n_jobs': n_jobs, 'lfp': lfp,
                                                   'sessions': rows})



def _photometry_check(params, inputs=()):
    n = len(inputs)
    if n == 0:
        raise JobError('photometry needs at least one .doric file: make --input ...')
    if any(not str(i).endswith('.doric') for i in inputs):
        raise JobError('photometry inputs must be .doric files, one per session')
    types = params.get('session_type', 'test' if n == 1 else None)
    types = types if isinstance(types, list) else [types] * (1 if n == 1 else 0)
    if len(types) != n or len(set(map(str, types))) != n:
        raise JobError(f'session_type needs one distinct value per input ({n}), in input order: '
                       "make --param 'session_type=[..., ...]'")
    starts = params.get('session_start')
    if starts is not None and (not isinstance(starts, list) or len(starts) != n) and n > 1:
        raise JobError('session_start, if given, needs one value per input')
    return list(map(str, types))


@stage('photometry', check=_photometry_check)
def photometry_stage(ctx):
    """Fibre photometry (Doric) -> per session: _photometry.nwb, .npz, .csv
    and QC, in ctx.outputs (processed tree). A CPU job: make --cpu.

    inputs: one .doric file per session. params: session_type (one per
    input), mouse, day (default test/0), subject (dict), session_start
    (optional; else read from the file), channels and processing (dicts
    overriding photometry.yaml's)."""
    from raphael.imaging.suite2p import _deep_update
    from raphael.photometry.layout import (
        DEFAULT_CONFIG,
        Session,
        load_settings,
    )
    from raphael.photometry.run import process_file

    p = ctx.params
    config = ctx.config / 'photometry.yaml'
    settings = load_settings(config if config.exists() else DEFAULT_CONFIG)
    settings = _deep_update(settings, {k: p[k] for k in ('channels', 'processing') if k in p})
    types = _photometry_check(p, ctx.inputs)
    starts = p.get('session_start')
    starts = starts if isinstance(starts, list) else [starts] * len(ctx.inputs)
    rows = []
    for path, session_type, start in zip(ctx.inputs, types, starts, strict=True):
        session = Session(str(p.get('mouse', 'test')), str(p.get('day', '0')), session_type)
        rows.append(process_file(path, ctx.outputs, session, settings, subject=p.get('subject'),
                                 start_time=start, notes={'job': ctx.job_dir.name}))
    _write_json(ctx.outputs / 'photometry_summary.json', {'sessions': rows})
