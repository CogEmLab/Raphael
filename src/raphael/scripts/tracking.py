"""raphael-tracking: behavioural tracking from video, with DeepLabCut and VAME.

Settings: your project's copy of raphael/configs/tracking.yaml (--config).
DeepLabCut steps need the tracking extra (uv sync --extra tracking); VAME
runs in its own environment, installed on first use. See
docs/behavioural_tracking.md.

A project from labels you already have, several models on one split:

    raphael-tracking build                  # DLC project from labels + day-held-out split
    raphael-tracking shuffles               # one shuffle per model (where you train)
    raphael-tracking benchmark              # register an already-trained model (optional)
    raphael-tracking train bu_resnet50      # resumes from the newest snapshot
    raphael-tracking evaluate
    raphael-tracking package                # zip the project for Colab

A project from scratch:

    raphael-tracking create VIDEO...        # DLC project over these videos
    raphael-tracking extract VIDEO...       # frames to label (k-means), then `label`
    raphael-tracking label                  # DLC's labelling GUI
    raphael-tracking dataset SPLIT.csv      # training set with a day-held-out split

Tracking videos, and what comes after:

    raphael-tracking capture MODEL --out captures/ [--folder] [--match ...]
    raphael-tracking sweep captures/        # re-track at every threshold
    raphael-tracking analyze VIDEO...       # plain DLC tracking (dlc projects)
    raphael-tracking pose POSE.h5 --fps 30 --out ROOT --mouse 1 --day 1 --session-type OF1
    raphael-tracking vame POSE... --work vame/

Two interacting animals (identity correction):

    raphael-tracking skeleton
    raphael-tracking capture-labels MODEL --out identity/
    raphael-tracking identity-eval identity/*_frames.npz
    raphael-tracking identity captures/MODEL/
    raphael-tracking identity-tune folds/ --jobs 8
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from raphael.behavioural_tracking import zoo

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'tracking.yaml'


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=Path('configs/tracking.yaml'),
                   help='tracking settings (default: configs/tracking.yaml, else Raphael\'s)')
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('build')
    s.add_argument('--overwrite', action='store_true')
    s = sub.add_parser('package')
    s.add_argument('--out', type=Path, default=None)
    s = sub.add_parser('shuffles')
    s.add_argument('models', nargs='*')
    sub.add_parser('benchmark')
    s = sub.add_parser('train')
    s.add_argument('model')
    s.add_argument('--epochs', type=int, default=None)
    s.add_argument('--device', default=None)
    sub.add_parser('evaluate')
    s = sub.add_parser('create')
    s.add_argument('videos', nargs='+', type=Path)
    s = sub.add_parser('extract')
    s.add_argument('videos', nargs='+', type=Path)
    s.add_argument('--frames', type=int, default=None)
    s = sub.add_parser('dataset')
    s.add_argument('split', type=Path, help='CSV with video_id and split (train/test)')
    s.add_argument('--shuffle', type=int, default=1)
    s = sub.add_parser('analyze')
    s.add_argument('videos', nargs='+', type=Path)
    s.add_argument('--shuffle', type=int, default=1)
    s.add_argument('--out', type=Path, default=None)
    s = sub.add_parser('capture')
    s.add_argument('model')
    s.add_argument('--match', nargs='*', default=[])
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--max-frames', type=int, default=None)
    s.add_argument('--start-frame', type=int, default=0)
    s.add_argument('--scratch', type=Path, default=None)
    s.add_argument('--folder', action='store_true',
                   help='read videos from videos.folder instead of videos.zip')
    s = sub.add_parser('sweep')
    s.add_argument('captures', type=Path)
    s.add_argument('--save', nargs='*', default=[],
                   help='pcutoff,bbox pairs to write tracked h5 for, e.g. 0.4,0.5')
    s = sub.add_parser('pose', help='DLC output -> arena cm -> the processed tree')
    s.add_argument('pose', type=Path)
    s.add_argument('--fps', type=float, required=True)
    s.add_argument('--out', type=Path, required=True, help='processed root')
    s.add_argument('--mouse', required=True)
    s.add_argument('--day', required=True)
    s.add_argument('--session-type', required=True)
    s.add_argument('--pixels', action='store_true', help='keep pixels (no arena fit)')
    s = sub.add_parser('vame', help='VAME motifs from pose files (own environment)')
    s.add_argument('poses', nargs='+', type=Path)
    s.add_argument('--work', type=Path, required=True)
    s.add_argument('--steps', nargs='*', default=None)
    sub.add_parser('label')
    sub.add_parser('check')
    sub.add_parser('skeleton')
    s = sub.add_parser('capture-labels')
    s.add_argument('model')
    s.add_argument('--split', default='test')
    s.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('identity-eval')
    s.add_argument('captures', type=Path, nargs='+')
    s = sub.add_parser('identity-tune')
    s.add_argument('folds', type=Path, help='folder of fold*/ (capture + skeleton.json)')
    s.add_argument('--jobs', type=int, default=1)
    s = sub.add_parser('identity')
    s.add_argument('captures', type=Path)
    s.add_argument('--benchmark', type=Path, default=None,
                   help="a zip of another model's tracked _el.csv, to compare against")
    return p


def _settings(path):
    path = path if path.exists() else DEFAULT_CONFIG
    if path == DEFAULT_CONFIG:
        print(f'note: using Raphael\'s template settings ({path}); copy it to '
              'configs/tracking.yaml and edit it', file=sys.stderr)
    return zoo.load_settings(path)


def main(argv=None):
    args = parser().parse_args(argv)
    settings = _settings(args.config)
    config = zoo.config_path(settings)

    if args.command == 'build':
        path, summary = zoo.build_project(settings, overwrite=args.overwrite)
        print(f'project: {path}')
        print(summary.groupby(['rig', 'split'])[['frames', 'two_animal_frames']].sum())
    elif args.command == 'package':
        out = args.out or Path(settings['project']['root']) / 'colab_upload.zip'
        print(f'wrote {zoo.package_for_colab(settings, out, [args.config])}')
    elif args.command == 'shuffles':
        print(zoo.create_shuffles(settings, args.models or None).to_string(index=False))
    elif args.command == 'benchmark':
        print(f'benchmark registered at {zoo.register_benchmark(settings)}')
    elif args.command == 'train':
        zoo.split_and_fraction(settings)
        zoo.train(settings, args.model, epochs=args.epochs, device=args.device)
    elif args.command == 'evaluate':
        from raphael.behavioural_tracking.dlc import _dlc

        dlc = _dlc()
        zoo.split_and_fraction(settings)
        shuffles = [m['shuffle'] for m in settings['models'].values()]
        if settings.get('benchmark'):
            shuffles.insert(0, zoo.benchmark_shuffle(settings))
        for shuffle in shuffles:
            if list((zoo.model_folder(config, shuffle) / 'train').glob('snapshot*.pt')):
                dlc.evaluate_network(str(config), shuffles=[shuffle])
    elif args.command in ('create', 'extract', 'dataset', 'analyze'):
        from raphael.behavioural_tracking import dlc as D

        project_config = Path(settings['project']['root']).expanduser()
        found = sorted(project_config.glob('*/config.yaml'))
        if args.command == 'create':
            print(f'project: {D.create_project(settings, args.videos)}')
            return 0
        if not found:
            raise SystemExit(f'no DLC project under {project_config}; run `create` first')
        cfg = found[-1]
        if args.command == 'extract':
            D.extract_frames(cfg, args.videos, args.frames or settings['dlc']['numframes2pick'])
        elif args.command == 'dataset':
            split = pd.read_csv(args.split)
            print(D.create_training_dataset(cfg, dict(zip(split['video_id'], split['split'])),
                                            settings['dlc']['net_type'], args.shuffle))
        else:
            D.analyze(cfg, args.videos, args.shuffle, args.out)
    elif args.command == 'capture':
        if args.folder:
            videos = zoo.folder_videos(settings['videos']['folder'])
            if args.match:
                videos = videos[videos['path'].apply(lambda m: any(s in m for s in args.match))]
            folder, snap, det = zoo.snapshots_for(settings, args.model)
            zoo.capture_folder(list(videos['path']), args.out, folder, snap, det,
                               settings['capture'], args.model, max_frames=args.max_frames,
                               start_frame=args.start_frame)
        else:
            import tempfile

            videos = zoo.zip_videos(settings['videos']['zip'])
            if args.match:
                videos = videos[videos['member'].apply(lambda m: any(s in m for s in args.match))]
            folder, snap, det = zoo.snapshots_for(settings, args.model)
            zoo.capture_from_zip(settings['videos']['zip'], list(videos['member']), args.out,
                                 folder, snap, det, settings['capture'], args.model,
                                 args.scratch or Path(tempfile.gettempdir()) / 'raphael-capture',
                                 max_frames=args.max_frames)
    elif args.command == 'sweep':
        from raphael.behavioural_tracking import sweep as S

        save = [tuple(float(v) for v in pair.split(',')) for pair in args.save]
        tables = []
        for npz in sorted(Path(args.captures).rglob('*.npz')):
            if npz.name.endswith('.identity.npz'):
                continue
            table = S.sweep_capture(npz, settings, save_dir=npz.with_suffix(''), save=save)
            table.to_csv(npz.with_suffix('.sweep.csv'), index=False)
            tables.append(table)
            print(f'swept {npz}')
        if tables:
            out = Path(args.captures) / 'sweep_results.csv'
            pd.concat(tables).to_csv(out, index=False)
            print(f'-> {out}')
    elif args.command == 'pose':
        from raphael.behavioural_tracking import io
        from raphael.ephys.layout import Session

        arena = settings.get('arena') or {}
        pose = io.load_dlc(args.pose, args.fps,
                           None if args.pixels else arena.get('size_cm'),
                           arena.get('pcutoff', 0.9),
                           corners=tuple(arena.get('corners') or io.CORNERS),
                           door=arena.get('reference'))
        session = Session(str(args.mouse), str(args.day), args.session_type)
        print(f'-> {io.write_pose(pose, session, args.out)}  ({pose.attrs["arena"]})')
    elif args.command == 'vame':
        from raphael.behavioural_tracking import vame

        summary = vame.run(args.poses, args.work, settings.get('vame'), steps=args.steps,
                           log=Path(args.work) / 'vame.log')
        for f in summary['motif_files']:
            print(f"{f['session']} {f['segmentation']}: {f['frames']} frames -> {f['path']}")
    elif args.command == 'skeleton':
        model = zoo.fit_skeleton(settings)
        print(f'{model.n_skeletons} complete training skeletons -> {zoo.skeleton_path(settings)}')
    elif args.command == 'capture-labels':
        print(f'wrote {zoo.capture_labels(settings, args.model, args.out, args.split)}')
    elif args.command == 'identity-eval':
        for npz in args.captures:
            summary, parts, _ = zoo.evaluate_identity(settings, npz)
            summary.to_csv(npz.with_suffix('.identity.csv'), index=False)
            parts.to_csv(npz.with_suffix('.parts.csv'), index=False)
            print(npz.name)
            print(summary.to_string(index=False))
    elif args.command == 'identity-tune':
        from concurrent.futures import ProcessPoolExecutor

        configs = zoo.identity_grid(settings['identity']['correct'], settings['identity']['tune'])
        folds = sorted(p for p in args.folds.glob('fold*') if p.is_dir())
        tasks = [(label, cfg, f) for label, cfg in configs for f in folds]
        with ProcessPoolExecutor(args.jobs) as pool:
            results = pool.map(_tune_one, [(settings, cfg, f) for _, cfg, f in tasks])
            rows = [{'config': label, 'fold': int(f.name.removeprefix('fold')), **r}
                    for (label, _, f), r in zip(tasks, results, strict=True)]
        table = pd.DataFrame(rows)
        table.to_csv(args.folds / 'identity_tune.csv', index=False)
        lofo = zoo.leave_one_fold_out(table)
        lofo.to_csv(args.folds / 'identity_tune_lofo.csv', index=False)
        print(lofo.to_string(index=False))
    elif args.command == 'identity':
        from raphael.behavioural_tracking import identity

        tables = []
        for npz in sorted(Path(args.captures).rglob('*.npz')):
            if npz.name.endswith('.identity.npz'):
                continue
            bench = (zoo.benchmark_poses(args.benchmark, npz.stem, settings['individuals'],
                                         settings['bodyparts']) if args.benchmark else None)
            table = identity.correct_capture(npz, zoo.skeleton_path(settings),
                                             settings['identity']['correct'], bench)
            tables.append(table)
            print(table.to_string(index=False))
        if tables:
            out = Path(args.captures) / 'identity_results.csv'
            pd.concat(tables).to_csv(out, index=False)
            print(f'-> {out}')
    elif args.command in ('label', 'check'):
        from raphael.behavioural_tracking.dlc import _dlc

        dlc = _dlc()
        (dlc.label_frames if args.command == 'label' else dlc.check_labels)(str(config))
    return 0


def _tune_one(task):
    settings, config, fold = task
    npz = next(fold.glob('*_test_frames.npz'))
    return zoo.score_identity(settings, npz, fold / 'skeleton.json', config)


if __name__ == '__main__':
    sys.exit(main())
