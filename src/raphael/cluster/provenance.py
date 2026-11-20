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
                            'dirty': False}
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
