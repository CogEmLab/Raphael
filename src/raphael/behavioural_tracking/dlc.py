"""Thin, opinionated wrappers around the DeepLabCut 3 project workflow, for a
new project labelled from scratch (`zoo` builds one from existing labels).

Everything DLC-specific lives here and imports `deeplabcut` lazily, so the
rest of the package never needs it. The choices (bodyparts, individuals,
landmarks, net) come from the project's tracking settings (`dlc` section) and are written into the
DLC config.yaml, so that file is generated rather than hand-edited.

The one non-default step is the train/test split. DLC's own split is random
over labelled frames, which puts near-identical frames from one video on both
sides and measures memorisation. Here test frames come only from videos
recorded on held-out days (see `videos.assign_split`), so test error
estimates how the model will do on the unlabelled session videos.
"""

import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import yaml

EXCLUDE_FROM_PACKAGE = ('.mp4', '.avi', '.pickle')


def load_settings(path):
    with open(path) as handle:
        settings = yaml.safe_load(handle)
    settings['project']['root'] = str(
        Path(settings['project']['root']).expanduser()
    )
    return settings


def _dlc():
    # conda's libomp and the one bundled with PyTorch both load on macOS, and
    # training then dies in pthread_mutex_init (segfault) unless both are set.
    # One OpenMP thread only affects CPU ops; the network runs on MPS/CUDA.
    # Must be set before import.
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    try:
        import deeplabcut
    except ImportError as error:
        raise RuntimeError(
            'DeepLabCut is not installed: install Raphael with the tracking extra '
            '(uv sync --extra tracking, or "raphael[tracking]" in a project).'
        ) from error
    return deeplabcut


def create_project(settings, videos):
    """Create the DLC project over `videos` and apply the YAML settings."""
    dlc = _dlc()
    project = settings['project']
    spec = settings['dlc']
    config = dlc.create_new_project(
        project.get('name') or project['task'],
        project.get('experimenter') or project.get('scorer', 'me'),
        [str(v) for v in videos],
        working_directory=project['root'],
        copy_videos=False,
        multianimal=spec['multianimal'],
        individuals=list(spec['individuals']) if spec['multianimal'] else None,
    )
    if config == 'nothingcreated':
        raise RuntimeError('DeepLabCut did not create the project')
    apply_settings(config, settings)
    return Path(config)


def apply_settings(config, settings):
    """Write bodyparts, individuals, landmarks and skeleton into config.yaml."""
    from deeplabcut.utils import auxiliaryfunctions

    spec = settings['dlc']
    cfg = auxiliaryfunctions.read_config(str(config))
    if spec['multianimal']:
        cfg['individuals'] = list(spec['individuals'])
        cfg['multianimalbodyparts'] = list(spec['bodyparts'])
        cfg['uniquebodyparts'] = list(spec['uniquebodyparts'])
        cfg['bodyparts'] = 'MULTI!'
        cfg['identity'] = bool(spec['identity'])
    else:
        cfg['bodyparts'] = list(spec['bodyparts']) + list(
            spec['uniquebodyparts']
        )
    cfg['skeleton'] = [list(edge) for edge in spec['skeleton']]
    cfg['numframes2pick'] = int(spec['numframes2pick'])
    cfg['pcutoff'] = float(spec['pcutoff'])
    cfg['default_net_type'] = spec['net_type']
    auxiliaryfunctions.write_config(str(config), cfg)


def add_videos(config, videos):
    dlc = _dlc()
    dlc.add_new_videos(str(config), [str(v) for v in videos], copy_videos=False)


def extract_frames(config, videos, numframes):
    """k-means frame selection on `videos` only, `numframes` per video."""
    dlc = _dlc()
    from deeplabcut.utils import auxiliaryfunctions

    cfg = auxiliaryfunctions.read_config(str(config))
    # DLC stores its own symlinked copies (project/videos/*.mp4), so match the
    # requested videos to config entries by stem rather than by path.
    wanted = {Path(v).stem for v in videos}
    selected = [v for v in cfg['video_sets'] if Path(v).stem in wanted]
    missing = wanted - {Path(v).stem for v in selected}
    if missing:
        raise ValueError(f'videos not in the DLC project: {sorted(missing)}')
    original = cfg['numframes2pick']
    cfg['numframes2pick'] = int(numframes)
    auxiliaryfunctions.write_config(str(config), cfg)
    try:
        dlc.extract_frames(
            str(config),
            mode='automatic',
            algo='kmeans',
            crop=False,
            userfeedback=False,
            videos_list=selected,
        )
    finally:
        cfg = auxiliaryfunctions.read_config(str(config))
        cfg['numframes2pick'] = original
        auxiliaryfunctions.write_config(str(config), cfg)


def labelled_frames(config):
    """Every labelled frame in DLC's merge order, as (video_id, image) rows.

    `create_training_dataset` indexes train/test by position in the table
    DLC's own `merge_annotateddatasets` returns (concatenated, then
    sort_index'd), so that function is called here rather than imitated.
    """
    import tempfile

    from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
        merge_annotateddatasets,
    )
    from deeplabcut.utils import auxiliaryfunctions

    cfg = auxiliaryfunctions.read_config(str(config))
    with tempfile.TemporaryDirectory() as scratch:
        data = merge_annotateddatasets(cfg, Path(scratch))
    if data is None:
        raise ValueError('no labelled frames found in labeled-data/')
    # Rows are ('labeled-data', video_id, image) tuples.
    return pd.DataFrame(
        [{'video_id': index[1], 'image': index[2]} for index in data.index]
    )


def split_indices(frames, split):
    """Train/test positions from a video_id -> 'train'/'test' mapping."""
    side = frames['video_id'].map(split)
    if side.isna().any():
        missing = sorted(frames.loc[side.isna(), 'video_id'].unique())
        raise ValueError(f'videos with labels but no split: {missing}')
    train = [int(i) for i in frames.index[side == 'train']]
    test = [int(i) for i in frames.index[side == 'test']]
    if not train or not test:
        raise ValueError(
            f'need labelled frames on both sides (train={len(train)}, '
            f'test={len(test)})'
        )
    return train, test


def create_training_dataset(config, split, net_type, shuffle=1):
    """Build the training set with the day-held-out split."""
    dlc = _dlc()
    from deeplabcut.utils import auxiliaryfunctions

    frames = labelled_frames(config)
    train, test = split_indices(frames, split)
    # train/evaluate/analyze find a shuffle through
    # TrainingFraction[trainingsetindex=0], so the custom split's fraction
    # (rounded as DLC rounds it) must be the first entry, or nothing
    # downstream can locate the model.
    cfg = auxiliaryfunctions.read_config(str(config))
    cfg['TrainingFraction'] = [round(len(train) / (len(train) + len(test)), 2)]
    auxiliaryfunctions.write_config(str(config), cfg)
    kwargs = {
        'Shuffles': [shuffle],
        'trainIndices': [train],
        'testIndices': [test],
        'net_type': net_type,
        'userfeedback': False,
    }
    try:
        from deeplabcut.core.engine import Engine

        kwargs['engine'] = Engine.PYTORCH
    except ImportError:
        pass
    dlc.create_training_dataset(str(config), **kwargs)
    return {'train_frames': len(train), 'test_frames': len(test)}


def package_for_colab(config, out_zip):
    """Zip the project without videos, for upload to Google Drive.

    Training only needs config.yaml, labeled-data, training-datasets and the
    model folders DLC created; videos stay local. The Colab notebook rewrites
    project_path after unzipping.
    """
    project = Path(config).parent
    out_zip = Path(out_zip)
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(project.rglob('*')):
            if path.is_dir() or path.name.startswith('._'):
                continue
            relative = path.relative_to(project)
            if relative.parts[0] == 'videos':
                continue
            if path.suffix in EXCLUDE_FROM_PACKAGE and 'labeled-data' not in (
                relative.parts
            ):
                continue
            archive.write(path, Path(project.name) / relative)
    return out_zip


def import_from_colab(config, trained_zip):
    """Unpack a trained model zip (from the notebook) into the project."""
    project = Path(config).parent
    written = []
    with zipfile.ZipFile(trained_zip) as archive:
        for member in archive.namelist():
            parts = Path(member).parts
            # The notebook zips `{project}/dlc-models-pytorch/...`; drop the
            # project folder name, which differs between machines.
            if len(parts) < 2 or member.endswith('/'):
                continue
            target = project.joinpath(*parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            written.append(target)
    # pytorch_config.yaml and friends carry the Colab project path.
    marker = project / 'colab_project_path.txt'
    if marker.exists():
        colab_root = marker.read_text().strip()
        for path in written:
            if path.suffix == '.yaml':
                text = path.read_text()
                if colab_root in text:
                    path.write_text(text.replace(colab_root, str(project)))
        marker.unlink()


def analyze(config, videos, shuffle=1, destfolder=None):
    """Pose + tracking on new videos, then filtering. Returns nothing; DLC
    writes `{video}DLC_..._el.h5` (and `_filtered.h5`) beside each video or
    into `destfolder`."""
    dlc = _dlc()
    videos = [str(v) for v in videos]
    kwargs = {'shuffle': shuffle, 'videotype': '.mp4'}
    if destfolder:
        kwargs['destfolder'] = str(destfolder)
    dlc.analyze_videos(str(config), videos, auto_track=True, **kwargs)
    dlc.filterpredictions(str(config), videos, **kwargs)
