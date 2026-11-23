"""Runs inside VAME's own environment (vame_env/), started by
raphael.behavioural_tracking.vame; imports nothing from Raphael.

    python vame_runner.py spec.json

The spec (JSON) says which pose files to use and which steps to run:
init, preprocess, trainset, train, evaluate, segment, community, export.
Steps already done are skipped where VAME allows it (the project is reused
if it exists). `export` writes one CSV per session and segmentation:
frame, time (s, on the video's clock), motif and, if community analysis
ran, community. VAME drops time_window // 2 frames at each end of a
session; the frame numbers account for that. A summary JSON goes to
spec['out'].
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

STEPS = ('init', 'preprocess', 'trainset', 'train', 'evaluate', 'segment', 'community', 'export')


def _hide_mps():
    """Make torch report no Apple GPU (VAME picks its device at import)."""
    import torch

    original = torch.backends.mps.is_available

    def no_mps():
        return False

    # torch's own code expects the attribute that functools.wraps leaves.
    no_mps.__wrapped__ = getattr(original, '__wrapped__', original)
    torch.backends.mps.is_available = no_mps


def _macos_major():
    if platform.system() != 'Darwin':
        return None
    try:
        return int(platform.mac_ver()[0].split('.')[0])
    except (ValueError, IndexError):
        return None


def choose_device(requested):
    """Set up torch so VAME uses `requested` ('auto', 'cpu', 'cuda', 'mps').

    'auto' avoids the Apple GPU before macOS 15: torch's GPU code for the
    recurrent layers VAME uses needs a macOS 15 function and aborts the
    process otherwise."""
    if requested == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        _hide_mps()
        return 'cpu'
    major = _macos_major()
    if requested == 'auto' and major is not None and major < 15:
        _hide_mps()
        return 'cpu (Apple GPU needs macOS 15 for VAME)'
    return requested


def _labels_path(project, session, model, algorithm, n):
    return (Path(project) / 'results' / session / model / f'{algorithm}-{n}'
            / f'{n}_{algorithm}_label_{session}.npy')


def _community_path(project, session, model, algorithm, n):
    return (Path(project) / 'results' / session / model / f'{algorithm}-{n}' / 'community'
            / f'cohort_community_label_{session}.npy')


def export(config, out_dir):
    """Per session and segmentation: a CSV of frame, time, motif, community."""
    import numpy as np
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project, model = config['project_path'], config['model_name']
    n, fps = int(config['n_clusters']), float(config.get('fps') or 0) or None
    offset = int(config['time_window']) // 2
    written = []
    for session in config['session_names']:
        for algorithm in config['segmentation_algorithms']:
            path = _labels_path(project, session, model, algorithm, n)
            if not path.exists():
                continue
            motif = np.load(path)
            frame = np.arange(len(motif)) + offset
            table = pd.DataFrame({'frame': frame, 'motif': motif})
            if fps:
                table.insert(1, 'time', np.round(frame / fps, 6))
            community = _community_path(project, session, model, algorithm, n)
            if community.exists():
                table['community'] = np.load(community)
            target = out_dir / f'{session}_{algorithm}-{n}_motifs.csv'
            table.to_csv(target, index=False)
            written.append({'session': session, 'segmentation': f'{algorithm}-{n}',
                            'path': str(target), 'frames': len(table),
                            'motif_counts': {int(k): int(v) for k, v in
                                             zip(*np.unique(motif, return_counts=True))}})
    return written


def main(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    device = choose_device(spec.get('device', 'auto'))
    import torch
    import vame

    steps = spec.get('steps') or list(STEPS)
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise SystemExit(f'unknown VAME steps {sorted(unknown)}; known: {STEPS}')
    work = Path(spec['working_directory'])
    config_file = work / spec['project_name'] / 'config.yaml'
    timings = {}

    def timed(name, call):
        start = time.perf_counter()
        call()
        timings[name] = round(time.perf_counter() - start, 1)

    if config_file.exists():
        config = vame.read_config(str(config_file))
    elif 'init' in steps:
        start = time.perf_counter()
        _, config = vame.init_new_project(
            project_name=spec['project_name'], poses_estimations=spec['poses'],
            source_software=spec.get('source_software', 'DeepLabCut'),
            working_directory=str(work), videos=spec.get('videos'), fps=spec.get('fps'),
            video_type=spec.get('video_type', '.mp4'), config_kwargs=spec.get('config') or {})
        timings['init'] = round(time.perf_counter() - start, 1)
    else:
        raise SystemExit(f'no VAME project at {config_file}; run the init step first')
    config['fps'] = config.get('fps') or spec.get('fps')

    pre = spec.get('preprocessing') or {}
    if 'preprocess' in steps:
        timed('preprocess', lambda: vame.preprocessing(config, **pre))
    if 'trainset' in steps:
        timed('trainset', lambda: vame.create_trainset(config, **(spec.get('trainset') or {})))
    if 'train' in steps:
        timed('train', lambda: vame.train_model(config))
    if 'evaluate' in steps:
        timed('evaluate', lambda: vame.evaluate_model(config))
    if 'segment' in steps:
        try:
            timed('segment', lambda: vame.segment_session(config))
        except ValueError as e:
            if 'positive-definite' in str(e):
                raise SystemExit(
                    'VAME segmentation failed fitting its HMM (covariance not positive-'
                    'definite): the latent space is degenerate, usually because the model '
                    'trained too little. Train for longer, or use segmentation_algorithms: '
                    '[kmeans].') from e
            raise
    if 'community' in steps and spec.get('community') is not False:
        timed('community', lambda: vame.community(config, **(spec.get('community') or {})))
    files = export(config, spec['labels_dir']) if 'export' in steps else []

    summary = {
        'vame_version': vame.get_version() if hasattr(vame, 'get_version') else None,
        'torch_version': torch.__version__,
        'device': device,
        'project_path': config['project_path'],
        'model_name': config['model_name'],
        'sessions': config['session_names'],
        'n_clusters': config['n_clusters'],
        'time_window': config['time_window'],
        'segmentation_algorithms': config['segmentation_algorithms'],
        'steps': steps,
        'seconds': timings,
        'motif_files': files,
    }
    Path(spec['out']).write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ('device', 'sessions', 'seconds')}, default=str))


if __name__ == '__main__':
    main(sys.argv[1])
