"""One-photon (miniscope) cell extraction with CaImAn's CNMF-E, run in
CaImAn's own environment.

CaImAn is not on PyPI and brings a GUI stack and Keras/torch with it, so,
as with VAME, it runs in a small environment of its own, locked in
caiman_env/uv.lock (CaImAn 1.13.2) and created on first use in
~/.cache/raphael/caiman-env ($RAPHAEL_CAIMAN_ENV to move it). Raphael writes
a spec, runs caiman_runner.py there with `uv run --frozen`, and reads back
plain arrays. Building CaImAn needs a C compiler; uv must be on PATH.

    from raphael.imaging import cnmfe, inscopix
    rec = inscopix.open_isxd('..._video.isxd')
    rate, times = inscopix.downsample(rec, work / 'movie.tif', spatial=2, temporal=2)
    summary = cnmfe.run(work / 'movie.tif', work / 'caiman', 'name', rate, settings)
    plane = cnmfe.to_plane(cnmfe.load(work / 'caiman' / 'name_cnmfe.npz'), times, rate)

Settings are the `cnmfe` section of configs/imaging_1p.yaml: CaImAn's own
parameter names, grouped as `motion`, `cnmfe` and `quality`. The defaults
are the helping-behaviour analysis's (CogEmLab/helping-behaviour-calcium-
imaging, config/master.yaml), which follow CaImAn's 1p example except for
smaller patches (rf 20) and a smaller bounding box (gSiz = 2 gSig + 1).
"""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from raphael.imaging.nwb import Plane

ENV_PROJECT = Path(__file__).with_name('caiman_env')
RUNNER = Path(__file__).with_name('caiman_runner.py')

DEFAULTS = {
    'spatial_downsample': 2,
    'temporal_downsample': 2,
    'motion': {
        'pw_rigid': False,
        'gSig_filt': [3, 3],
        'max_shifts': [5, 5],
        'strides': [48, 48],
        'overlaps': [24, 24],
        'max_deviation_rigid': 3,
        'border_nan': 'copy',
    },
    'cnmfe': {
        'decay_time': 0.3,
        'p': 1,
        'gSig': [3, 3],
        'gSiz': [7, 7],
        'merge_thr': 0.7,
        'rf': 20,
        'stride': 10,
        'tsub': 2,
        'ssub': 1,
        'min_corr': 0.8,
        'min_pnr': 7,
        'ssub_B': 2,
        'ring_size_factor': 1.4,
        'update_background_components': True,
        'del_duplicates': True,
    },
    'quality': {
        'use_cnn': False,
        'min_SNR': 3.0,
        'SNR_lowest': 2.0,
        'rval_thr': 0.85,
        'rval_lowest': 0.3,
    },
}


class CnmfeError(RuntimeError):
    pass


def settings_with_defaults(settings=None):
    """DEFAULTS with `settings` merged over them, one level deep per group."""
    s = json.loads(json.dumps(DEFAULTS))
    for key, value in (settings or {}).items():
        if isinstance(value, dict) and isinstance(s.get(key), dict):
            s[key].update(value)
        else:
            s[key] = value
    return s


def env_path():
    return Path(os.environ.get('RAPHAEL_CAIMAN_ENV',
                               Path.home() / '.cache' / 'raphael' / 'caiman-env'))


def lock_sha256():
    """Checksum of CaImAn's lockfile: which exact CaImAn environment ran."""
    return hashlib.sha256((ENV_PROJECT / 'uv.lock').read_bytes()).hexdigest()


def _uv():
    uv = shutil.which('uv')
    if not uv:
        raise CnmfeError('uv is not on PATH; CaImAn runs in its own uv environment '
                         '(install uv: https://docs.astral.sh/uv/)')
    return uv


def _env(temp=None):
    env = dict(os.environ)
    env['UV_PROJECT_ENVIRONMENT'] = str(env_path())
    env.pop('VIRTUAL_ENV', None)              # never the caller's environment
    if temp:
        # CaImAn writes its motion-corrected memmaps to CAIMAN_TEMP, by
        # default ~/caiman_data/temp: on a cluster, the home quota.
        env['CAIMAN_TEMP'] = str(temp)
    # CaImAn parallelises over processes; one thread each avoids oversubscription.
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env.setdefault(var, '1')
    return env


def ensure_env(quiet=True):
    """Create or update CaImAn's environment from its lockfile (first use
    downloads and builds CaImAn: a few minutes)."""
    cmd = [_uv(), 'sync', '--frozen', '--project', str(ENV_PROJECT)]
    if quiet:
        cmd.append('-q')
    subprocess.run(cmd, check=True, env=_env())
    return env_path()


def temporal_factor(rate, settings):
    """The integer number of raw frames per output frame. `temporal_downsample`
    is a factor (int) or, as {'target_hz': 30}, the rate to reach, which must
    divide the recorded rate (to 1%): 60 -> 30 Hz is 2, 20 Hz stays 1, 40 Hz
    can't reach 30."""
    value = settings.get('temporal_downsample', 1)
    if isinstance(value, dict):
        target = float(value['target_hz'])
        if rate <= 1.01 * target:
            return 1
        factor = round(rate / target)
        if abs(rate / factor - target) > 0.01 * target:
            raise CnmfeError(f'{rate:.2f} Hz is not a whole multiple of {target} Hz; '
                             'set temporal_downsample to a factor instead')
        return factor
    factor = int(value)
    if factor < 1:
        raise CnmfeError(f'temporal_downsample must be >= 1, got {value}')
    return factor


#: CaImAn holds about this many float32 copies of the movie at its peak
#: (memmap pages, CNMF-E patches across workers, evaluation), so a session
#: needs roughly this multiple of the float32 movie size in RAM.
MEMORY_FACTOR = 3.0


def physical_memory():
    """Bytes of RAM on this machine (a cluster job's limit can be lower)."""
    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')


def memory_needed(movie):
    """Rough peak RAM (bytes) CaImAn needs for a uint16 TIFF movie: its
    float32 size times MEMORY_FACTOR."""
    return int(Path(movie).stat().st_size * 2 * MEMORY_FACTOR)


def check_memory(movie, available=None):
    """Refuse a movie this machine can't hold: CaImAn past the RAM swaps
    until the machine stops responding (an 8 GB laptop on a 4-minute
    session froze and restarted)."""
    needed, available = memory_needed(movie), available or physical_memory()
    if needed > available:
        raise CnmfeError(
            f'{Path(movie).name} needs about {needed / 1e9:.0f} GB of RAM in CaImAn and this '
            f'machine has {available / 1e9:.0f} GB: run it on the cluster (raphael-jobs, '
            f'--mem-gb {int(np.ceil(needed / 1e9)) + 4}), or a shorter stretch with max_frames')
    return needed


def run(movie, out_dir, name, rate, settings=None, n_processes=None, keep_movies=False,
        log=None, memory_check=True):
    """Motion correction, CNMF-E and evaluation on `movie` (a TIFF at `rate`
    Hz), in CaImAn's environment. Writes the CaImAn model, `{name}_cnmfe.npz`
    and `{name}_summary.json` to `out_dir`; returns the summary, with the
    settings and the lockfile checksum added. Refuses a movie too big for
    this machine's RAM (`check_memory`) unless `memory_check` is False."""
    s = settings_with_defaults(settings)
    if memory_check:
        check_memory(movie)
    ensure_env()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        'movie': str(Path(movie).resolve()), 'out_dir': str(out_dir), 'name': name,
        'fr': float(rate), 'n_processes': n_processes or os.cpu_count(),
        'motion': s['motion'], 'cnmfe': s['cnmfe'], 'quality': s['quality'],
        'keep_movies': bool(keep_movies),
    }
    spec_path = out_dir / f'{name}_spec.json'
    spec_path.write_text(json.dumps(spec, indent=2))
    temp = out_dir / 'tmp'
    temp.mkdir(exist_ok=True)
    env = _env(temp)
    # -P: don't put the runner's folder on sys.path, where Raphael's modules
    # would shadow CaImAn's.
    cmd = [_uv(), 'run', '--frozen', '--project', str(ENV_PROJECT), 'python', '-P',
           str(RUNNER), str(spec_path)]
    if log:
        with open(log, 'w') as output:
            done = subprocess.run(cmd, check=False, env=env, stdout=output,
                                  stderr=subprocess.STDOUT)
    else:
        done = subprocess.run(cmd, check=False, env=env)
    if not keep_movies:
        shutil.rmtree(temp, ignore_errors=True)
    if done.returncode != 0:
        where = f' (see {log})' if log else ''
        raise CnmfeError(f'CaImAn failed with exit code {done.returncode}{where}')
    path = out_dir / f'{name}_summary.json'
    summary = json.loads(path.read_text())
    summary['caiman_env_lock_sha256'] = lock_sha256()
    summary['settings'] = s
    path.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def load(path):
    """The runner's `_cnmfe.npz` as a dict; `A` as a scipy CSC matrix
    (pixels x components, pixels in CaImAn's column-major order)."""
    from scipy import sparse

    with np.load(path, allow_pickle=False) as f:
        out = {k: f[k] for k in f.files}
    out['A'] = sparse.csc_matrix((out.pop('A_data'), out.pop('A_indices'), out.pop('A_indptr')),
                                 shape=tuple(out.pop('A_shape')))
    return out


def masks(A, dims):
    """Each component's footprint as an (n_pixels, 3) array of x, y, weight.
    CaImAn flattens pixels column-major: index = y + x * height."""
    height = int(dims[0])
    out = []
    for k in range(A.shape[1]):
        col = A[:, k]
        idx, weight = col.indices, col.data
        out.append(np.column_stack([idx // height, idx % height, weight]).astype(np.float32))
    return out


def to_plane(result, timestamps, rate, description='CaImAn CNMF-E output.'):
    """A `Plane` from the runner's arrays. Traces are frames x components:
    F = C + YrA (the component's raw activity), denoised = C,
    deconvolved = S, detrended = CaImAn's detrended trace. No dF/F: the
    CNMF-E background is not a baseline to divide by."""
    C, YrA = result['C'].T, result['YrA'].T
    n = min(len(C), len(timestamps))
    rois = pd.DataFrame({
        'accepted': result['accepted'].astype(bool),
        'snr': result['snr'],
        'r_value': result['r_value'],
        'cnmfe_index': np.arange(C.shape[1]),
    })
    images = {'mean': result['mean'], 'correlation': result['correlation'],
              'peak_to_noise': result['pnr']}
    return Plane(
        timestamps=np.asarray(timestamps)[:n],
        rate=float(rate),
        masks=masks(result['A'], result['dims']),
        image_shape=tuple(int(d) for d in result['dims']),
        F=(C + YrA)[:n],
        denoised=C[:n],
        deconvolved=result['S'].T[:n],
        detrended=result['detrended'].T[:n],
        shifts=result['shifts'][:n],
        images=images,
        rois=rois,
        description=description,
    )
