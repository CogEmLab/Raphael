"""Build one DLC project from existing labels and train several models on it.

Every model in the settings' `models` becomes a shuffle of a single project:
same labelled frames, same train/test split, different architecture or
initialisation. That is what makes the models comparable, e.g. when looking
for the one that confuses two interacting mice least.

An already-trained model (a `benchmark`, e.g. a colleague's) can be
registered as one more shuffle on the same split, and new top-down models
can reuse its training recipe (`recipe: benchmark`) and its trained detector
(`detector: benchmark`), so they don't each spend hours retraining one.

`build_project` runs locally (it only copies labels and writes config.yaml).
`create_shuffles` should run where training runs (Colab), because DLC writes
absolute paths to downloaded SuperAnimal weights into each pytorch_config.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from raphael.behavioural_tracking.dlc import (
    _dlc,
    labelled_frames,
    split_indices,
)
from raphael.behavioural_tracking.videos import parse_name

DEFAULT_BENCHMARK_SHUFFLE = 100

SPLIT = 'split.csv'
PROJECT_DATE = 'Oct22'  # fixed, so model folder names never depend on build day


def load_settings(path):
    with open(path) as handle:
        settings = yaml.safe_load(handle)
    for key in ('root', 'labels'):
        if settings['project'].get(key):
            settings['project'][key] = str(Path(settings['project'][key]).expanduser())
    set_name_patterns(settings)
    return settings


def project_dir(settings):
    project = settings['project']
    return Path(project['root']) / f'{project["task"]}-{project["scorer"]}'


def config_path(settings):
    return project_dir(settings) / 'config.yaml'


# ------------------------------------------------------------- the project


def label_folders(labels_root, scorer):
    """Folders holding CollectedData; DLC's *_labeled check images are skipped."""
    return sorted(
        p for p in Path(labels_root).iterdir()
        if p.is_dir() and (p / f'CollectedData_{scorer}.h5').exists()
    )


_PATTERNS = {}


def _name_patterns():
    return _PATTERNS.get('day', r'(20\d{6})'), _PATTERNS.get('rig')


def set_name_patterns(settings):
    """Read how video names give the recording day and rig (settings
    `videos.day_pattern` and `videos.rig_pattern`; see videos.parse_name)."""
    v = settings.get('videos', {}) or {}
    _PATTERNS.update({'day': v.get('day_pattern', r'(20\d{6})'), 'rig': v.get('rig_pattern')})


def summarise_labels(folders, scorer, individuals):
    """Frames, and frames with every individual labelled, per video."""
    rows = []
    for folder in folders:
        data = pd.read_hdf(folder / f'CollectedData_{scorer}.h5')
        present = [
            data.xs(ind, axis=1, level='individuals').notna().any(axis=1)
            for ind in individuals
        ]
        info = parse_name(folder.name, *_name_patterns())
        rows.append(
            {
                'video_id': folder.name,
                'date': info['date'],
                'rig': info['rig'],
                'frames': len(data),
                'two_animal_frames': int(np.logical_and.reduce(present).sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_split(summary, test_fraction, seed, min_two_animal, min_per_rig=0, tries=5000):
    """Hold out whole days per rig, closest to the target test fraction.

    Searches random day assignments (reproducibly, from `seed`) and keeps the
    one whose test share of frames is nearest `test_fraction` while holding
    at least `min_two_animal` two-animal frames overall, and `min_per_rig`
    from every rig that has any, since those are the frames where identity
    confusion shows up and the rigs differ in appearance.
    """
    rng = np.random.default_rng(seed)
    days = summary.groupby(['rig', 'date'])[['frames', 'two_animal_frames']].sum()
    total = days['frames'].sum()
    best, best_cost = None, np.inf
    for _ in range(tries):
        held = []
        for _, group in days.groupby(level='rig'):
            if len(group) < 2:  # a one-day rig cannot be on both sides
                continue
            # Days in random order, taken until this rig's share is reached;
            # always at least one day held out and one kept.
            order = group.iloc[rng.permutation(len(group))]
            share = order['frames'].cumsum() / order['frames'].sum()
            n_test = int(np.clip((share <= test_fraction).sum(), 1, len(group) - 1))
            held += list(order.index[:n_test])
        test = days.loc[held]
        if test['two_animal_frames'].sum() < min_two_animal:
            continue
        rig_two = test['two_animal_frames'].groupby(level='rig').sum()
        rigs_with_two = days[days['two_animal_frames'] > 0].index.get_level_values('rig').unique()
        if any(rig_two.get(rig, 0) < min_per_rig for rig in rigs_with_two):
            continue
        cost = abs(test['frames'].sum() / total - test_fraction)
        if cost < best_cost:
            best, best_cost = set(held), cost
    if best is None:
        raise ValueError('no day split satisfies the two-animal minimum')
    keys = list(zip(summary['rig'], summary['date']))
    return pd.Series(
        ['test' if key in best else 'train' for key in keys], index=summary.index
    )


def random_split(summary, test_fraction, seed):
    """Per-video random split; only for comparison with a random-frame split."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(summary))
    cumulative = summary['frames'].to_numpy()[order].cumsum() / summary['frames'].sum()
    test = set(order[cumulative <= test_fraction])
    return pd.Series(
        ['test' if i in test else 'train' for i in range(len(summary))],
        index=summary.index,
    )


def build_project(settings, overwrite=False):
    """Write config.yaml, copy labelled-data, record the split."""
    from deeplabcut.utils import auxiliaryfunctions

    _dlc()
    project = settings['project']
    out = project_dir(settings)
    if out.exists() and not overwrite:
        raise FileExistsError(f'{out} exists; pass overwrite to rebuild')
    folders = label_folders(project['labels'], project['scorer'])
    if not folders:
        raise ValueError(f'no CollectedData_{project["scorer"]}.h5 under {project["labels"]}')

    labelled = out / 'labeled-data'
    labelled.mkdir(parents=True, exist_ok=True)
    for sub in ('videos', 'training-datasets', 'dlc-models-pytorch'):
        (out / sub).mkdir(exist_ok=True)

    import cv2

    video_sets = {}
    for folder in folders:
        target = labelled / folder.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            folder, target,
            ignore=shutil.ignore_patterns('._*', '.DS_Store', 'machinelabels*'),
        )
        first = next(target.glob('*.png'))
        height, width = cv2.imread(str(first)).shape[:2]
        # The video need not exist: training only reads labeled-data, and DLC
        # keys label folders by the video's stem.
        video_sets[str(out / 'videos' / f'{folder.name}.mp4')] = {
            'crop': f'0, {width}, 0, {height}'
        }

    cfg, _ = auxiliaryfunctions.create_config_template(multianimal=True)
    cfg.update(
        {
            'Task': project['task'],
            'scorer': project['scorer'],
            'date': PROJECT_DATE,
            'project_path': str(out),
            'video_sets': video_sets,
            'individuals': list(settings['individuals']),
            'multianimalbodyparts': list(settings['bodyparts']),
            'uniquebodyparts': [],
            'bodyparts': 'MULTI!',
            'multianimalproject': True,
            'identity': False,
            'skeleton': [list(e) for e in settings['skeleton']],
            'TrainingFraction': [0.85],
            'iteration': 0,
            'engine': 'pytorch',
            'default_track_method': settings['sweep']['track_method'],
            'numframes2pick': 20,
            'pcutoff': 0.6,
        }
    )
    auxiliaryfunctions.write_config(str(out / 'config.yaml'), cfg)

    summary = summarise_labels(folders, project['scorer'], settings['individuals'])
    spec = settings['split']
    if spec['method'] == 'day':
        summary['split'] = choose_split(
            summary, spec['test_fraction'], spec['seed'],
            spec['min_two_animal_test_frames'],
            spec.get('min_two_animal_test_frames_per_rig', 0),
        )
    else:
        summary['split'] = random_split(summary, spec['test_fraction'], spec['seed'])
    summary.to_csv(out / SPLIT, index=False)
    return out / 'config.yaml', summary


# ---------------------------------------------------------------- shuffles


def model_folder(config, shuffle):
    from deeplabcut.core.engine import Engine
    from deeplabcut.utils import auxiliaryfunctions

    cfg = auxiliaryfunctions.read_config(str(config))
    return Path(cfg['project_path']) / auxiliaryfunctions.get_model_folder(
        cfg['TrainingFraction'][0], shuffle, cfg, engine=Engine.PYTORCH
    )


def split_and_fraction(settings):
    """Train/test frame indices from split.csv, and set TrainingFraction.

    Every DLC call (train, evaluate, analyze) finds a shuffle through
    TrainingFraction[0], so it must match the split's actual fraction. Call
    this whenever config.yaml may have been replaced, e.g. at the start of
    every Colab session.
    """
    _dlc()
    from deeplabcut.utils import auxiliaryfunctions

    config = config_path(settings)
    split = pd.read_csv(project_dir(settings) / SPLIT)
    frames = labelled_frames(config)
    train, test = split_indices(frames, dict(zip(split['video_id'], split['split'])))
    cfg = auxiliaryfunctions.read_config(str(config))
    cfg['TrainingFraction'] = [round(len(train) / (len(train) + len(test)), 2)]
    auxiliaryfunctions.write_config(str(config), cfg)
    return train, test


SPEC_FILE = 'zoo_spec.yaml'
RECIPE_VERSION = 2  # bump when shuffle construction changes, to force rebuilds


def spec_record(spec):
    """What a shuffle was built from, saved next to it."""
    return {**spec, 'recipe_version': RECIPE_VERSION}


def benchmark_shuffle(settings):
    return int((settings.get('benchmark') or {}).get('shuffle', DEFAULT_BENCHMARK_SHUFFLE))


def benchmark_source(settings):
    """The benchmark model's folder, or the project's registered copy.

    Once registered (`register_benchmark`), the project's copy carries the
    same config and detector, so building shuffles no longer depends on the
    original (e.g. on an external drive) being mounted.
    """
    bench = settings.get('benchmark') or {}
    original = Path(bench.get('model_dir', ''))
    if (original / 'train' / 'pytorch_config.yaml').exists():
        return original
    copy = model_folder(config_path(settings), benchmark_shuffle(settings))
    if (copy / 'train' / bench.get('detector_snapshot', '')).exists():
        return copy
    raise FileNotFoundError(f'benchmark model not found at {original} and not yet registered '
                            f'as shuffle {benchmark_shuffle(settings)}')


def apply_recipe(pose, settings):
    """Copy the benchmark model's augmentation, optimiser schedule and
    detector settings into a new shuffle's pytorch_config.

    Only the parts that define its training recipe are copied; the model,
    metadata and weight initialisation stay as DLC built them for this
    shuffle. With this, two top-down models differ from the benchmark, and
    from each other, only in their starting weights.
    """
    from deeplabcut.utils import auxiliaryfunctions

    source = auxiliaryfunctions.read_plainconfig(
        str(benchmark_source(settings) / 'train' / 'pytorch_config.yaml'))
    pose['data']['train'] = source['data']['train']
    pose['data']['inference'] = source['data']['inference']
    pose['runner']['optimizer'] = source['runner']['optimizer']
    pose['runner']['scheduler'] = source['runner']['scheduler']
    if source.get('detector') and pose.get('detector'):
        for key in ('model', 'data', 'runner'):
            pose['detector'][key] = source['detector'][key]


def shuffles_to_build(settings):
    """Models whose shuffle is missing or was built from different settings.

    A shuffle that already has training snapshots is never listed, even if
    stale: rebuilding would discard training. It is reported instead.
    """
    config = config_path(settings)
    todo = []
    for name, spec in settings['models'].items():
        train = model_folder(config, spec['shuffle']) / 'train'
        saved = train / SPEC_FILE
        current = saved.exists() and yaml.safe_load(saved.read_text()) == spec_record(spec)
        if (train / 'pytorch_config.yaml').exists() and current:
            continue
        if list(train.glob('snapshot-*.pt')) and any(
            not p.name.startswith('snapshot-detector') for p in train.glob('snapshot-*.pt')
        ):
            print(f'{name}: settings changed but it already has training snapshots; '
                  'left as is (delete its folder to rebuild)')
            continue
        todo.append(name)
    return todo


def create_shuffles(settings, names=None):
    """One shuffle per model, all on the recorded split. Returns a table."""
    dlc = _dlc()
    from deeplabcut.core.engine import Engine
    from deeplabcut.utils import auxiliaryfunctions

    config = config_path(settings)
    project = project_dir(settings)
    train, test = split_and_fraction(settings)

    models = settings['models']
    names = names or list(models)
    rows = []
    for name in names:
        spec = models[name]
        cfg = auxiliaryfunctions.read_config(str(config))
        # identity is project-level in DLC but only read when a shuffle is
        # created, so it is set per model here.
        cfg['identity'] = bool(spec.get('identity', False))
        cfg['default_net_type'] = spec['net_type']
        auxiliaryfunctions.write_config(str(config), cfg)

        kwargs = {
            'Shuffles': [spec['shuffle']],
            'trainIndices': [train],
            'testIndices': [test],
            'net_type': spec['net_type'],
            'engine': Engine.PYTORCH,
            'userfeedback': False,
        }
        if spec.get('superanimal'):
            from deeplabcut.modelzoo import build_weight_init

            kwargs['weight_init'] = build_weight_init(
                cfg=str(config),
                super_animal=spec['superanimal'],
                model_name=spec['net_type'].replace('top_down_', ''),
                detector_name='fasterrcnn_resnet50_fpn_v2',
                with_decoder=False,
            )
        if spec.get('conditions'):
            kwargs['ctd_conditions'] = (models[spec['conditions']]['shuffle'], -1)
        if spec['net_type'].startswith('top_down_'):
            # DLC silently defaults top-down detectors to SSDLite; a benchmark
            # detector reused by `detector: benchmark` is usually a Faster R-CNN.
            kwargs['detector_type'] = spec.get('detector_type', 'fasterrcnn_resnet50_fpn_v2')
        dlc.create_training_dataset(str(config), **kwargs)

        folder = model_folder(config, spec['shuffle'])
        pytorch_cfg = folder / 'train' / 'pytorch_config.yaml'
        pose = auxiliaryfunctions.read_plainconfig(str(pytorch_cfg))
        if spec.get('recipe') == 'benchmark':
            apply_recipe(pose, settings)
        pose['train_settings']['epochs'] = int(spec.get('epochs', 200))
        pose['train_settings']['batch_size'] = int(spec.get('batch_size', 8))
        if spec.get('detector') == 'benchmark' and pose.get('detector'):
            bench = settings['benchmark']
            src = benchmark_source(settings) / 'train' / bench['detector_snapshot']
            shutil.copy2(src, folder / 'train' / src.name)
            pose['detector']['train_settings']['epochs'] = 0
        auxiliaryfunctions.write_plainconfig(str(pytorch_cfg), pose)
        with open(folder / 'train' / SPEC_FILE, 'w') as handle:
            yaml.safe_dump(spec_record(spec), handle)

        rows.append(
            {
                'model': name,
                'shuffle': spec['shuffle'],
                'net_type': spec['net_type'],
                'method': pose.get('method'),
                'identity': pose['metadata'].get('with_identity'),
                'heads': ','.join(pose['model'].get('heads', {})),
                'folder': str(folder.relative_to(project)),
            }
        )

    cfg = auxiliaryfunctions.read_config(str(config))
    cfg['identity'] = False
    auxiliaryfunctions.write_config(str(config), cfg)
    table = pd.DataFrame(rows)
    table.to_csv(project / 'shuffles.csv', index=False)
    return table


def latest_snapshot(folder):
    """(path, epoch) of the newest regular snapshot, or (None, 0)."""
    import re

    best = (None, 0)
    for path in (Path(folder) / 'train').glob('snapshot-[0-9]*.pt'):
        epoch = int(re.search(r'snapshot-(\d+)\.pt$', path.name).group(1))
        if epoch > best[1]:
            best = (path, epoch)
    return best


def ensure_pretrained_weights(settings, name):
    """Re-download SuperAnimal starting weights if their file has vanished.

    DLC downloads them into its own install folder when a shuffle is created
    and writes that absolute path into pytorch_config. On Colab the shuffle
    persists on Drive but the install folder is wiped with every runtime, so
    a later session finds the path dead. Download again and repoint.
    """
    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
        get_super_animal_snapshot_path,
    )
    from deeplabcut.utils import auxiliaryfunctions

    spec = settings['models'][name]
    pytorch_cfg = model_folder(config_path(settings), spec['shuffle']) / 'train' / 'pytorch_config.yaml'
    pose = auxiliaryfunctions.read_plainconfig(str(pytorch_cfg))
    init = pose['train_settings'].get('weight_init') or {}
    dataset = init.get('dataset')
    if not dataset:
        return  # ImageNet-initialised: nothing to fetch
    changed = False
    for key, model_name in [
        ('snapshot_path', spec['net_type'].replace('top_down_', '')),
        ('detector_snapshot_path', 'fasterrcnn_resnet50_fpn_v2'),
    ]:
        path = init.get(key)
        if path and not Path(path).exists():
            init[key] = str(get_super_animal_snapshot_path(
                dataset=dataset, model_name=model_name, download=True
            ))
            changed = True
    if changed:
        pose['train_settings']['weight_init'] = init
        auxiliaryfunctions.write_plainconfig(str(pytorch_cfg), pose)
        print(f'{name}: re-downloaded {dataset} starting weights')


def train(settings, name, epochs=None, save_epochs=10, device=None):
    """Train one model, resuming from its newest snapshot if there is one.

    DLC continues the epoch counter from a snapshot and then trains `epochs`
    MORE, so only the remaining epochs are requested; a run interrupted at
    epoch 120 of 200 resumes for 80, not 200.
    """
    dlc = _dlc()
    spec = settings['models'][name]
    config = config_path(settings)
    folder = model_folder(config, spec['shuffle'])
    target = int(epochs or spec.get('epochs', 200))
    resume, done = latest_snapshot(folder)
    remaining = target - done
    if remaining <= 0:
        print(f'{name}: already trained to epoch {done}')
        return folder
    print(f'{name}: training epochs {done + 1}-{target}'
          + (f' (resuming {resume.name})' if resume else ''))
    # Needed even when resuming: DLC builds the model from its starting
    # weights first and only then loads the snapshot over them.
    ensure_pretrained_weights(settings, name)
    kwargs = {
        'shuffle': spec['shuffle'],
        'epochs': remaining,
        'save_epochs': save_epochs,
        'batch_size': int(spec.get('batch_size', 8)),
        'snapshot_path': str(resume) if resume else None,
        'device': device,
    }
    if spec.get('detector') == 'benchmark':
        kwargs['detector_epochs'] = 0
    elif spec.get('detector') == 'train':
        kwargs['detector_epochs'] = int(spec.get('detector_epochs', 100))
    dlc.train_network(str(config), **kwargs)
    return folder


# ------------------------------------------------------- Colab and videos


def package_for_colab(settings, out_zip, extra=()):
    """The project (labels, config, split) as one zip, for training on Colab.

    Model folders and training-datasets are left out on purpose: shuffles
    are created on Colab, so DLC records Colab paths to SuperAnimal weights.
    Raphael itself is installed there with pip, so no code is bundled;
    `extra` files (e.g. the settings YAML) are added under `extra/`.
    """
    import zipfile

    project = project_dir(settings)
    skip_dirs = {'dlc-models-pytorch', 'training-datasets', 'evaluation-results-pytorch'}
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(project.rglob('*')):
            rel = path.relative_to(project)
            if path.is_dir() or rel.parts[0] in skip_dirs or path.name.startswith('._'):
                continue
            archive.write(path, Path('project') / project.name / rel)
        for path in extra:
            archive.write(path, Path('extra') / Path(path).name)
    return Path(out_zip)


def zip_videos(zip_path):
    """Every video in the raw-video zip, with cohort and filename stem."""
    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        infos = [
            i for i in archive.infolist()
            if i.filename.lower().endswith(('.mp4', '.avi'))
            and not Path(i.filename).name.startswith('._')
        ]
    return pd.DataFrame(
        {
            'member': [i.filename for i in infos],
            'video_id': [Path(i.filename).stem for i in infos],
            'cohort': [Path(i.filename).parts[1] if len(Path(i.filename).parts) > 2 else '' for i in infos],
            'gb': [round(i.file_size / 1e9, 2) for i in infos],
        }
    )


def capture_from_zip(zip_path, members, out_dir, model_dir, snapshot, detector_snapshot,
                     capture, name, scratch, max_frames=None):
    """Extract one video at a time, capture it, delete it. Resumable.

    Free disk only ever needs to hold the largest single video (~1.8GB),
    never the whole 129GB archive.
    """
    import time

    from raphael.behavioural_tracking import multipeak as MP
    from raphael.behavioural_tracking.videos import extract_member

    out_dir, scratch = Path(out_dir), Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    runners = None
    done = []
    for member in members:
        out = out_dir / name / f'{Path(member).stem}.npz'
        if out.exists():
            done.append(out)
            continue
        if runners is None:
            runners = MP.build_runners(model_dir, snapshot, detector_snapshot, capture)
        start = time.time()
        video = extract_member(zip_path, member, scratch)
        try:
            MP.capture_video(
                video, out, model_dir, snapshot, detector_snapshot,
                capture, runners=runners, name=name, max_frames=max_frames,
            )
        finally:
            video.unlink(missing_ok=True)
        print(f'{name}: {Path(member).name} in {(time.time() - start) / 60:.1f} min')
        done.append(out)
    return done


def snapshots_for(settings, name):
    """(model_dir, pose snapshot, detector snapshot) for a model name."""
    from raphael.behavioural_tracking.multipeak import load_model_config

    config = config_path(settings)
    if name == (settings.get('benchmark') or {}).get('name'):
        folder = model_folder(config, benchmark_shuffle(settings))
        bench = settings['benchmark']
        return folder, bench['snapshot'], bench['detector_snapshot']
    folder = model_folder(config, settings['models'][name]['shuffle'])
    train = folder / 'train'
    best = sorted(train.glob('snapshot-best-*.pt')) or sorted(
        train.glob('snapshot-[0-9]*.pt'), key=lambda p: int(p.stem.split('-')[-1])
    )
    if not best:
        raise FileNotFoundError(f'{name} has no snapshots yet; train it first')
    detector = None
    if load_model_config(folder)['method'].lower() == 'td':
        dets = sorted(train.glob('snapshot-detector-*.pt'))
        detector = dets[-1].name if dets else None
    return folder, best[-1].name, detector


def register_benchmark(settings, model_dir=None):
    """Install an already-trained model (settings `benchmark`) as a shuffle
    of this project.

    The shuffle gets this project's split, so `evaluate_network` scores it on
    the same test frames as the new models. Caveat: if the benchmark was
    trained on a random split of these same labelled frames, most of the
    test frames were probably in its training set and its test score here
    is optimistic. Its tracked output on unlabelled session videos is the
    fair comparison.
    """
    dlc = _dlc()
    from deeplabcut.core.engine import Engine
    from deeplabcut.utils import auxiliaryfunctions

    bench = settings['benchmark']
    src = Path(model_dir or bench['model_dir'])
    config = config_path(settings)
    project = project_dir(settings)
    train, test = split_and_fraction(settings)
    dlc.create_training_dataset(
        str(config),
        Shuffles=[benchmark_shuffle(settings)],
        trainIndices=[train],
        testIndices=[test],
        net_type='top_down_hrnet_w32',
        engine=Engine.PYTORCH,
        userfeedback=False,
    )
    folder = model_folder(config, benchmark_shuffle(settings))
    for name in (bench['snapshot'], bench['detector_snapshot']):
        shutil.copy2(src / 'train' / name, folder / 'train' / name)
    pose = auxiliaryfunctions.read_plainconfig(str(src / 'train' / 'pytorch_config.yaml'))
    pose['metadata']['project_path'] = str(project)
    pose['metadata']['pose_config_path'] = str(folder / 'train' / 'pytorch_config.yaml')
    # Weight init only matters for training, and points at the other machine.
    pose['train_settings']['weight_init'] = None
    pose['logger'] = None
    auxiliaryfunctions.write_plainconfig(str(folder / 'train' / 'pytorch_config.yaml'), pose)
    return folder


# ------------------------------------------------- identity correction (identity.py)


def labelled_points(settings):
    """Every labelled frame of the project: points (frames, individuals,
    parts, 2) and a table with video, image path and split."""
    from raphael.behavioural_tracking import skeleton

    project = project_dir(settings)
    scorer = settings['project']['scorer']
    csv = next((project / 'training-datasets').rglob(f'CollectedData_{scorer}.csv'))
    points, frames, _, bodyparts = skeleton.read_labels(csv)
    split = pd.read_csv(project / SPLIT).set_index('video_id')['split']
    frames['split'] = frames['video'].map(split)
    frames['path'] = [str(project / 'labeled-data' / v / i)
                      for v, i in zip(frames['video'], frames['image'], strict=True)]
    return points, frames, bodyparts


def skeleton_path(settings):
    return project_dir(settings) / 'skeleton.json'


def fit_skeleton(settings):
    """Skeleton model from the training split only; saved in the project."""
    from raphael.behavioural_tracking import skeleton

    points, frames, bodyparts = labelled_points(settings)
    train = (frames['split'] == 'train').to_numpy()
    model = skeleton.SkeletonModel.fit(points[train], frames['video'][train], bodyparts,
                                       **settings.get('identity', {}).get('skeleton', {}))
    model.save(skeleton_path(settings))
    return model


def capture_labels(settings, name, out_dir, split='test'):
    """Multi-peak capture of the labelled frames of one split."""
    from raphael.behavioural_tracking import multipeak as MP

    _, frames, _ = labelled_points(settings)
    frames = frames[frames['split'] == split]
    folder, snap, det = snapshots_for(settings, name)
    out = Path(out_dir) / f'{name}_{split}_frames.npz'
    MP.capture_images(list(frames['path']), out, folder, snap, det,
                      settings['capture'], name=name)
    return out


def evaluate_identity(settings, capture_npz, model=None, overrides=None):
    """DLC's own top-2 detections vs the corrected poses, on labelled frames.
    Frames are corrected independently; each video's scale comes from the
    model's own confident predictions on that video's frames, never from
    labels. Returns a summary table and the per-part outcomes."""
    from raphael.behavioural_tracking import identity
    from raphael.behavioural_tracking.multipeak import load_capture
    from raphael.behavioural_tracking.skeleton import SkeletonModel, scale_of

    model = model or SkeletonModel.load(skeleton_path(settings))
    ident = identity.Settings.from_dict({**settings.get('identity', {}).get('correct', {}),
                                         **(overrides or {})})
    raw = load_capture(capture_npz)
    points, frames, _ = labelled_points(settings)
    # matched on video/image, so captures made elsewhere (the MSC) still match
    def tail(path):
        return '/'.join(Path(path).parts[-2:])

    index = {tail(p): i for i, p in enumerate(frames['path'])}
    rows = [index[tail(p)] for p in raw['meta']['images']]
    truth, videos = points[rows], frames['video'].to_numpy()[rows]
    poses = np.full((len(rows), 2, truth.shape[2], 3), np.nan)
    dlc = poses.copy()
    label_scales = np.empty(len(rows))
    for video in np.unique(videos):
        sel = np.flatnonzero(videos == video)
        part = {k: v[sel] for k, v in raw.items() if k != 'meta'}
        out = identity.correct(part, model, ident, track=False)
        poses[sel], dlc[sel] = out['poses'], out['dlc']
        # the tolerance for scoring uses the labels' own scale
        label_scales[sel] = scale_of(points[(frames['video'] == video).to_numpy()])
    tables, summary = {}, []
    for method, pred in (('dlc', dlc), ('corrected', poses)):
        table, s = identity.evaluate_frames(pred, truth, label_scales)
        tables[method] = table.assign(method=method, video=videos[table['frame']],
                                      path=np.asarray(raw['meta']['images'])[table['frame']])
        for subset, values in s.items():
            summary.append({'method': method, 'frames': subset, **values})
    return pd.DataFrame(summary), pd.concat(tables.values()), {'dlc': dlc, 'corrected': poses}


def folder_videos(folder):
    """Every video under a folder."""
    paths = sorted(p for p in Path(folder).rglob('*')
                   if p.suffix.lower() in ('.mp4', '.avi') and not p.name.startswith('._'))
    return pd.DataFrame({'path': [str(p) for p in paths],
                         'video_id': [p.stem for p in paths],
                         'gb': [round(p.stat().st_size / 1e9, 2) for p in paths]})


def capture_folder(paths, out_dir, model_dir, snapshot, detector_snapshot, capture, name,
                   max_frames=None, start_frame=0):
    """Capture videos read in place. Resumable: finished captures are skipped."""
    import time

    from raphael.behavioural_tracking import multipeak as MP

    runners, done = None, []
    for video in paths:
        out = Path(out_dir) / name / f'{Path(video).stem}.npz'
        if not out.exists():
            runners = runners or MP.build_runners(model_dir, snapshot, detector_snapshot, capture)
            start = time.time()
            MP.capture_video(video, out, model_dir, snapshot, detector_snapshot, capture,
                             runners=runners, name=name, max_frames=max_frames,
                             start_frame=start_frame)
            print(f'{name}: {Path(video).name} in {(time.time() - start) / 60:.1f} min', flush=True)
        done.append(out)
    return done


def benchmark_poses(export_zip, video_id, individuals, bodyparts):
    """Another model's tracked output for one video (its `_el.csv` inside a
    zip of exports) as (frames, mice, parts, 3), or None if it has none."""
    import zipfile

    from raphael.behavioural_tracking.identity import poses_from_table

    with zipfile.ZipFile(export_zip) as archive:
        names = [n for n in archive.namelist()
                 if n.endswith('_el.csv') and Path(n).name.startswith(f'{video_id}DLC')]
        if not names:
            return None
        with archive.open(names[0]) as handle:
            df = pd.read_csv(handle, header=[0, 1, 2, 3], index_col=0)
    return poses_from_table(df, individuals, bodyparts)


def close_contact(points, frames, threshold=0.5):
    """Per labelled frame: both mice labelled and their nearest parts closer
    than `threshold` x the video's scale."""
    import warnings

    from raphael.behavioural_tracking.skeleton import scale_of

    scale = frames['video'].map({v: scale_of(points[(frames['video'] == v).to_numpy()])
                                 for v in frames['video'].unique()}).to_numpy()
    both = np.isfinite(points).all(-1).any(-1).all(-1)
    d = np.linalg.norm(points[:, 0, :, None] - points[:, 1, None, :], axis=-1)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        nearest = np.nanmin(d.reshape(len(d), -1), axis=1) / scale
    return both & (nearest < threshold)


def fold_splits(settings, k, threshold=0.5):
    """k day-level splits, every labelled day held out exactly once. Days go,
    most close-contact frames first, to the fold with the fewest so far, so
    every fold can be scored on close contact. Returns a table: video_id,
    date, rig, frames, two_animal_frames, close_frames, fold."""
    points, frames, _ = labelled_points(settings)
    frames['close'] = close_contact(points, frames, threshold)
    videos = pd.read_csv(project_dir(settings) / SPLIT).drop(columns='split')
    videos['close_frames'] = videos['video_id'].map(
        frames.groupby('video')['close'].sum()).fillna(0).astype(int)
    days = (videos.groupby(['rig', 'date'])[['frames', 'two_animal_frames', 'close_frames']]
            .sum().sort_values(['close_frames', 'two_animal_frames', 'frames'], ascending=False))
    load = np.zeros((k, 3))
    fold_of = {}
    for day, row in days.iterrows():
        if row['close_frames'] > 0:
            f = int(np.lexsort((load[:, 2], load[:, 0]))[0])   # fewest close, then frames
        else:
            f = int(np.argmin(load[:, 2]))
        fold_of[day] = f
        load[f] += [row['close_frames'], row['two_animal_frames'], row['frames']]
    videos['fold'] = [fold_of[(r, d)] for r, d in zip(videos['rig'], videos['date'], strict=True)]
    return videos


def use_fold(settings, folds, fold):
    """Write split.csv for one fold (its days are the test set)."""
    split = folds.drop(columns=['fold', 'close_frames']).assign(
        split=np.where(folds['fold'] == fold, 'test', 'train'))
    split.to_csv(project_dir(settings) / SPLIT, index=False)
    return split


def identity_grid(base, grid):
    """Every combination of `grid` (name -> list of values; a value may be a
    dict of several settings, e.g. one gating choice) on top of `base`."""
    import itertools

    names = list(grid)
    configs = []
    for values in itertools.product(*(grid[n] for n in names)):
        config, label = dict(base), []
        for name, value in zip(names, values, strict=True):
            if isinstance(value, dict):
                config.update(value)
            else:
                config[name] = value
            label.append(f'{name}={value}')
        configs.append((';'.join(label), config))
    return configs


def score_identity(settings, capture_npz, skeleton_json, overrides, close_threshold=0.5):
    """One fold, one setting: DLC vs corrected on all, two-mouse and
    close-contact frames, with net parts gained or lost."""
    from raphael.behavioural_tracking import identity
    from raphael.behavioural_tracking.skeleton import SkeletonModel

    model = SkeletonModel.load(skeleton_json)
    _, parts, _ = evaluate_identity(settings, capture_npz, model, overrides)
    points, frames, _ = labelled_points(settings)
    close = dict(zip(['/'.join(Path(p).parts[-2:]) for p in frames['path']],
                     close_contact(points, frames, close_threshold), strict=True))
    parts['close'] = parts['path'].map(lambda p: close['/'.join(Path(p).parts[-2:])])
    out = {}
    for subset, sub in (('all', parts), ('two_mice', parts[parts['two_mice'] == 1]),
                        ('close', parts[parts['close']])):
        per = {m: identity.summarise(g)['all'] for m, g in sub.groupby('method')}
        n = per['dlc']['parts']
        for m in ('dlc', 'corrected'):
            for key in ('correct_fraction', 'mixed_fraction', 'wrong_fraction', 'missing_fraction'):
                out[f'{subset}_{m}_{key.removesuffix("_fraction")}'] = per[m][key]
            out[f'{subset}_{m}_mixed_mice'] = per[m]['mixed_mice']
        out[f'{subset}_parts'] = n
        out[f'{subset}_net_correct'] = round(
            (per['corrected']['correct_fraction'] - per['dlc']['correct_fraction']) * n)
        out[f'{subset}_mixed_removed'] = round(
            (per['dlc']['mixed_fraction'] - per['corrected']['mixed_fraction']) * n)
    return out


def leave_one_fold_out(table, objective='all_net_correct'):
    """For each fold, the setting with the best summed `objective` on the
    other folds, and how it does on the held-out fold."""
    rows = []
    for fold in sorted(table['fold'].unique()):
        others = table[table['fold'] != fold].groupby('config')[objective].sum()
        best = others.idxmax()
        held = table[(table['fold'] == fold) & (table['config'] == best)].iloc[0]
        rows.append({'held_out': fold, 'chosen': best,
                     'objective_on_others': int(others.max()), **held.drop(['fold', 'config'])})
    return pd.DataFrame(rows)
