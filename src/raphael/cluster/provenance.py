"""What code and environment produced a result: recorded with every job run.

Two checkouts matter. Raphael is a dependency, so the environment is locked
by the *project* that installs it (e.g. social-manifolds): its `uv.lock`
pins every package including Raphael, and its git commit says which
project code ran. Raphael's own version, and its commit when it is
installed from a checkout (editable or a git URL), are recorded beside it.

The project root is `$RAPHAEL_PROJECT_ROOT` if set, otherwise the nearest
folder at or above the working directory that holds a `uv.lock`. Jobs run
from the project checkout (the rendered submit scripts `cd` there first),
so the default finds it.

What a run did travels with its results in two ways:

- every NWB file names the Raphael that wrote it (`source_script`), and
- every processed session folder a job writes gets
  `provenance/{job_id}.json`: the stage, its settings and configs, the
  inputs with their checksums, the Raphael version, tag and commit, the
  project's commit and uv.lock, every package version, the machine and GPU,
  the timings, and the checksum of each file the job wrote there.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]      # src/raphael
SCHEDULER_VARS = ['SLURM_JOB_ID', 'SLURM_JOB_NODELIST', 'SLURM_ARRAY_TASK_ID',
                  'JOB_ID', 'SGE_TASK_ID', 'HOSTNAME', 'CUDA_VISIBLE_DEVICES']


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def project_root(start=None):
    """The checkout whose uv.lock defines this environment, or None."""
    if os.environ.get('RAPHAEL_PROJECT_ROOT'):
        return Path(os.environ['RAPHAEL_PROJECT_ROOT']).expanduser().resolve()
    here = Path(start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / 'uv.lock').exists():
            return folder
    return None


def lock_hash(root=None):
    """sha256 of the project's uv.lock, or None outside a checkout."""
    root = root or project_root()
    lock = root / 'uv.lock' if root else None
    return sha256(lock) if lock and lock.exists() else None


def git_state(folder=None):
    """{'commit', 'dirty'} of the checkout holding `folder`, or None."""
    folder = folder or project_root()
    if folder is None:
        return None

    def git(*args):
        return subprocess.run(['git', '-C', str(folder), *args], capture_output=True,
                              text=True, timeout=30, check=True).stdout.strip()
    try:
        return {'commit': git('rev-parse', 'HEAD'),
                'dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}
    except (subprocess.SubprocessError, OSError):
        return None


def raphael_state():
    """Raphael's version, and its commit if it runs from a git checkout or
    was installed from a git URL (pip records that in direct_url.json)."""
    try:
        dist = importlib.metadata.distribution('raphael')
    except importlib.metadata.PackageNotFoundError:
        return {'version': None, 'git': git_state(PACKAGE_ROOT)}
    state = {'version': dist.version, 'git': None}
    direct = dist.read_text('direct_url.json')
    if direct:
        info = json.loads(direct)
        vcs = info.get('vcs_info')
        if vcs:
            state['git'] = {'commit': vcs.get('commit_id'), 'url': info.get('url'),
                            'tag': vcs.get('requested_revision'), 'dirty': False}
        elif info.get('dir_info', {}).get('editable'):
            state['git'] = git_state(PACKAGE_ROOT)
    return state


def environment():
    root = project_root()
    return {
        'python': sys.version.split()[0],
        'executable': sys.executable,
        'platform': platform.platform(),
        'project_root': str(root) if root else None,
        'git': git_state(root),
        'uv_lock_sha256': lock_hash(root),
        'raphael': raphael_state(),
        'scheduler': {k: os.environ[k] for k in SCHEDULER_VARS if k in os.environ},
        'packages': dict(sorted((d.metadata['Name'].lower(), d.version)
                                for d in importlib.metadata.distributions())),
    }


# ------------------------------------------------- provenance beside results

RECORD_FORMAT = 'raphael-provenance/1'
RECORD_DIR = 'provenance'


def source_script():
    """(source_script, source_script_file_name) for NWBFile: the Raphael
    that wrote the file, e.g. 'raphael 0.3.0 (v0.3.0, commit 27595f7...)
    https://github.com/CogEmLab/Raphael'."""
    state = raphael_state()
    git = state.get('git') or {}
    detail = ', '.join(x for x in (git.get('tag'),
                                   git.get('commit') and f"commit {git['commit']}",
                                   git.get('dirty') and 'uncommitted changes') if x)
    text = f"raphael {state.get('version') or 'unknown'}" + (f' ({detail})' if detail else '')
    url = git.get('url') or 'https://github.com/CogEmLab/Raphael'
    return f'{text} {url}', 'raphael'


def session_folders(outputs):
    """Folders under `outputs` that directly hold processed files (named
    `sub-...`), i.e. the session folders of the processed tree."""
    outputs = Path(outputs)
    return sorted({p.parent for p in outputs.rglob('sub-*')
                   if p.is_file() and RECORD_DIR not in p.relative_to(outputs).parts})


def _files(folder):
    return sorted(p for p in Path(folder).rglob('*') if p.is_file()
                  and p.relative_to(folder).parts[0] != RECORD_DIR
                  and not any(part.startswith('.') for part in p.relative_to(folder).parts))


def write_records(outputs, record, name):
    """Write `record` (a dict) plus the checksum of every file in each
    session folder under `outputs` to `{folder}/provenance/{name}.json`.
    Returns {session folder: record path}."""
    written = {}
    for folder in session_folders(outputs):
        files = [{'path': f.relative_to(folder).as_posix(), 'bytes': f.stat().st_size,
                  'sha256': sha256(f)} for f in _files(folder)]
        target = folder / RECORD_DIR / f'{name}.json'
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps({'format': RECORD_FORMAT, **record,
                                      'session_folder': folder.relative_to(outputs).as_posix(),
                                      'outputs': files}, indent=2, default=str) + '\n')
        written[folder] = target
    return written
