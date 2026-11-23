"""VAME behavioural motifs from pose, run in VAME's own environment.

Every VAME release pins an exact, older pynwb than Raphael's NWB checks
need, so VAME can't be installed alongside Raphael. Instead it runs in a
small environment of its own, locked in vame_env/uv.lock (VAME 0.13) and
created on first use in ~/.cache/raphael/vame-env ($RAPHAEL_VAME_ENV to
move it). Raphael writes a spec, runs vame_runner.py there with
`uv run --frozen`, and reads back the results. Nothing from VAME is
imported into Raphael's process. uv must be on PATH (it is wherever Raphael
was installed with uv).

    from raphael.behavioural_tracking import vame
    summary = vame.run(['s1DLC_..._el.h5', 's2DLC_..._el.h5'], work_dir, settings)
    summary['motif_files']        # one CSV per session: frame, time, motif, community

The steps, in order: init (a VAME project from the pose files), preprocess
(drop low-confidence points, egocentric alignment, outliers, smoothing),
trainset, train (the recurrent VAE), evaluate, segment (motifs per frame:
HMM and/or k-means), community (motifs grouped into communities across
sessions), export (the CSVs). Re-running with the same work_dir reuses the
project, so later steps can be run on their own (`steps=[...]`).
"""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ENV_PROJECT = Path(__file__).with_name('vame_env')
RUNNER = Path(__file__).with_name('vame_runner.py')
STEPS = ('init', 'preprocess', 'trainset', 'train', 'evaluate', 'segment', 'community', 'export')

#: Settings (the `vame` section of the tracking settings), with VAME's own
#: names inside `config` (any key of VAME's config.yaml).
DEFAULTS = {
    'project_name': 'vame',
    'source_software': 'DeepLabCut',
    'fps': None,                     # needed unless videos are given
    'device': 'auto',                # auto | cpu | cuda | mps
    'config': {
        'pose_confidence': 0.6,      # VAME's default, 0.99, discards most DLC points
        'n_clusters': 15,
        'segmentation_algorithms': ['hmm', 'kmeans'],
    },
    'preprocessing': {
        'centered_reference_keypoint': 'BodyMid',
        'orientation_reference_keypoint': 'Nose',
    },
    'trainset': {'test_fraction': 0.1},
    'community': {'cut_tree': 2},
}


class VameError(RuntimeError):
    pass


def env_path():
    return Path(os.environ.get('RAPHAEL_VAME_ENV', Path.home() / '.cache' / 'raphael' / 'vame-env'))


def lock_sha256():
    """Checksum of VAME's lockfile: which exact VAME environment ran."""
    return hashlib.sha256((ENV_PROJECT / 'uv.lock').read_bytes()).hexdigest()


def _uv():
    uv = shutil.which('uv')
    if not uv:
        raise VameError('uv is not on PATH; VAME runs in its own uv environment '
                        '(install uv: https://docs.astral.sh/uv/)')
    return uv


def _env():
    env = dict(os.environ)
    env['UV_PROJECT_ENVIRONMENT'] = str(env_path())
    env.pop('VIRTUAL_ENV', None)              # never the caller's environment
    env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    env.setdefault('OMP_NUM_THREADS', '1')
    return env


def ensure_env(quiet=True):
    """Create or update VAME's environment from its lockfile (first use
    downloads VAME and torch: a few minutes)."""
    cmd = [_uv(), 'sync', '--frozen', '--project', str(ENV_PROJECT)]
    if quiet:
        cmd.append('-q')
    subprocess.run(cmd, check=True, env=_env())
    return env_path()


def settings_with_defaults(settings=None):
    s = json.loads(json.dumps(DEFAULTS))
    for key, value in (settings or {}).items():
        if isinstance(value, dict) and isinstance(s.get(key), dict):
            s[key].update(value)
        else:
            s[key] = value
    return s


def check(settings):
    """Refuse settings that would run for hours and produce nothing."""
    c = settings.get('config', {})
    epochs, kl_start, anneal = (c.get('max_epochs'), c.get('kl_start'), c.get('annealtime'))
    if epochs is not None:
        kl_start = 2 if kl_start is None else kl_start           # VAME's defaults
        anneal = 4 if anneal is None else anneal
        if epochs <= kl_start + anneal:
            raise VameError(f'max_epochs ({epochs}) must exceed kl_start + annealtime '
                            f'({kl_start} + {anneal}): VAME saves a model only once its KL '
                            'weight has reached 1, so a shorter run saves nothing')
    if not settings.get('fps') and not settings.get('videos'):
        raise VameError('VAME needs the pose frame rate: set fps (or give videos)')
    steps = settings.get('steps')
    if steps and set(steps) - set(STEPS):
        raise VameError(f'unknown steps {sorted(set(steps) - set(STEPS))}; known: {STEPS}')


def run(poses, work_dir, settings=None, steps=None, log=None):
    """Run VAME on pose files (DLC h5/csv, or SLEAP/LightningPose/NWB with
    `source_software`) in `work_dir`. Returns the runner's summary, plus
    the VAME lockfile checksum. `log`: a file to write VAME's output to
    (default: printed)."""
    s = settings_with_defaults(settings)
    if steps:
        s['steps'] = list(steps)
    check(s)
    ensure_env()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        'project_name': s['project_name'],
        'working_directory': str(work_dir),
        'poses': [str(Path(p).resolve()) for p in poses],
        'source_software': s['source_software'],
        'fps': s.get('fps'),
        'videos': s.get('videos'),
        'device': s.get('device', 'auto'),
        'config': s.get('config') or {},
        'preprocessing': s.get('preprocessing') or {},
        'trainset': s.get('trainset') or {},
        'community': s.get('community'),
        'steps': s.get('steps') or list(STEPS),
        'labels_dir': str(work_dir / 'motifs'),
        'out': str(work_dir / 'vame_summary.json'),
    }
    spec_path = work_dir / 'vame_spec.json'
    spec_path.write_text(json.dumps(spec, indent=2))
    # -P: don't put the runner's folder on sys.path, where this module
    # (vame.py) would shadow the real VAME package.
    cmd = [_uv(), 'run', '--frozen', '--project', str(ENV_PROJECT), 'python', '-P', str(RUNNER),
           str(spec_path)]
    if log:
        with open(log, 'w') as output:
            done = subprocess.run(cmd, check=False, env=_env(), stdout=output,
                                  stderr=subprocess.STDOUT)
    else:
        done = subprocess.run(cmd, check=False, env=_env())
    if done.returncode != 0:
        where = f' (see {log})' if log else ''
        raise VameError(f'VAME failed with exit code {done.returncode}{where}')
    summary = json.loads(Path(spec['out']).read_text())
    summary['vame_env_lock_sha256'] = lock_sha256()
    summary['settings'] = s
    Path(spec['out']).write_text(json.dumps(summary, indent=2, default=str))
    return summary


def motifs_path(session, root, segmentation):
    """Where a session's motifs live in the processed tree."""
    return session.folder(root) / f'{session.stem}_src-vame-{segmentation}_motifs.csv'


def write_motifs(motif_csv, session, root, segmentation):
    """Copy one exported motif CSV into the processed tree; returns the path."""
    target = motifs_path(session, root, segmentation)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(motif_csv, target)
    return target
