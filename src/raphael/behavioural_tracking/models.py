"""Trained DeepLabCut models on the data server, and which videos use them.

A model lives in `data/processed/_dlcmodels/{name}/`: the model folder as
DeepLabCut wrote it (train/pytorch_config.yaml and its snapshots) plus a
`model.yaml` saying how to run it:

    snapshot: snapshot-200.pt       # in train/ (required)
    detector_snapshot:              # top-down models only
    skeleton:                       # e.g. skeleton.json in this folder: two-animal identity fixes
    bodyparts: [nose, ears, tailbase]
    trained_by: hclark
    trained_on: 2026-11-10
    description: open field, top camera, one mouse

A model folder is never changed once used: a retrained model gets a new
name (openfield-top-v3), so every output can be traced to what made it.

A task uses a model when its entry in metadata/sessions.yaml names one:

    OF:
      name: open field
      dlc_model: openfield-top-v2          # every video of the session
    HBT:
      dlc_model: {top: hbt-top-v1}         # by camera label (_video-top)

A task without `dlc_model` has no DeepLabCut job. An unlabelled `_video`
(the session's only camera) takes the string, or a mapping's only entry.

On the cluster the models are copied once to `paths.models` of
configs/cluster.yaml (`~/models/`); the job finds its model there by name
and checks it against the checksum in the plan.
"""

import hashlib
from pathlib import Path

import yaml

from raphael.cluster import provenance

#: Under data/processed/.
MODELS_DIR = '_dlcmodels'
MODEL_FILE = 'model.yaml'


class ModelError(ValueError):
    pass


def models_root(data_root):
    """data/processed/_dlcmodels/ of a data/ folder."""
    return Path(data_root) / 'processed' / MODELS_DIR


def read(model_dir):
    """A model folder's model.yaml, checked: it has a snapshot that exists."""
    model_dir = Path(model_dir)
    path = model_dir / MODEL_FILE
    if not path.is_file():
        raise ModelError(f'{model_dir.name}: no {MODEL_FILE}')
    try:
        spec = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ModelError(f'{model_dir.name}/{MODEL_FILE} can’t be read: {e}') from e
    if not isinstance(spec, dict) or not spec.get('snapshot'):
        raise ModelError(f'{model_dir.name}/{MODEL_FILE} names no snapshot')
    if not (model_dir / 'train' / 'pytorch_config.yaml').is_file():
        raise ModelError(f'{model_dir.name}: no train/pytorch_config.yaml (not a DeepLabCut '
                         'model folder)')
    for key in ('snapshot', 'detector_snapshot'):
        if spec.get(key) and not (model_dir / 'train' / str(spec[key])).is_file():
            raise ModelError(f'{model_dir.name}: {key} train/{spec[key]} not found')
    if spec.get('skeleton') and not (model_dir / str(spec['skeleton'])).is_file():
        raise ModelError(f'{model_dir.name}: skeleton {spec["skeleton"]} not found')
    return spec


def digest(model_dir):
    """One sha256 for a model folder: every file's path and checksum."""
    model_dir = Path(model_dir)
    h = hashlib.sha256()
    for f in sorted(p for p in model_dir.rglob('*') if p.is_file()
                    and not p.name.startswith('.')):
        h.update(f'{f.relative_to(model_dir).as_posix()}\0{provenance.sha256(f)}\n'.encode())
    return h.hexdigest()


def for_video(task_entry, camera, only_camera=False):
    """The model name a video of this task uses, or None. `camera` is the
    video's label ('' for an unlabelled _video); `only_camera` when it is
    the session's one video."""
    spec = (task_entry or {}).get('dlc_model') if isinstance(task_entry, dict) else None
    if not spec:
        return None
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        if camera in spec:
            return str(spec[camera]) if spec[camera] else None
        if not camera and len(spec) == 1:
            value = next(iter(spec.values()))
            return str(value) if value else None
    return None


def find(root, name, expected_sha256=None):
    """The model folder `name` under `root` (the models folder), checked; with
    `expected_sha256`, it must be the same model the plan was made with."""
    if not name or '/' in str(name) or str(name).startswith('.'):
        raise ModelError(f'not a model name: {name!r}')
    model_dir = Path(root) / str(name)
    if not model_dir.is_dir():
        raise ModelError(f'model {name} not found in {root}: copy it there first '
                         f'(rsync -a data/processed/{MODELS_DIR}/ CLUSTER:models/)')
    read(model_dir)
    if expected_sha256 and digest(model_dir) != expected_sha256:
        raise ModelError(f'model {name} in {root} differs from the one the plan was made '
                         'with: copy it again, or remake the plan')
    return model_dir
