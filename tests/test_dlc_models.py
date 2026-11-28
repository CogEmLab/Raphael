"""Trained DeepLabCut models in data/processed/_dlcmodels/: their checks,
which videos use them, and the job that finds one on the cluster."""

import pytest
import yaml

from raphael.behavioural_tracking import models
from raphael.cluster import jobs


def model_folder(root, name='of-top-v1', skeleton=True):
    """A DeepLabCut model folder as Raphael expects it (small stand-in files)."""
    folder = root / name
    (folder / 'train').mkdir(parents=True)
    (folder / 'train' / 'pytorch_config.yaml').write_text('method: bu\n')
    (folder / 'train' / 'snapshot-200.pt').write_bytes(b'weights')
    spec = {'snapshot': 'snapshot-200.pt', 'bodyparts': ['nose', 'tailbase'],
            'trained_by': 'jsmith'}
    if skeleton:
        (folder / 'skeleton.json').write_text('{}')
        spec['skeleton'] = 'skeleton.json'
    (folder / 'model.yaml').write_text(yaml.safe_dump(spec))
    return folder


def test_a_model_folder_is_checked(tmp_path):
    folder = model_folder(tmp_path)
    assert models.read(folder)['snapshot'] == 'snapshot-200.pt'
    (folder / 'train' / 'snapshot-200.pt').unlink()
    with pytest.raises(models.ModelError, match='snapshot train/snapshot-200.pt not found'):
        models.read(folder)
    (tmp_path / 'not-a-model').mkdir()
    with pytest.raises(models.ModelError, match='no model.yaml'):
        models.read(tmp_path / 'not-a-model')


def test_the_digest_follows_every_file(tmp_path):
    folder = model_folder(tmp_path)
    before = models.digest(folder)
    assert models.digest(folder) == before
    (folder / 'train' / 'snapshot-200.pt').write_bytes(b'retrained')
    assert models.digest(folder) != before


def test_which_videos_use_a_model():
    assert models.for_video({'dlc_model': 'of-top-v1'}, 'side') == 'of-top-v1'
    by_camera = {'dlc_model': {'top': 'hbt-top-v1'}}
    assert models.for_video(by_camera, 'top') == 'hbt-top-v1'
    assert models.for_video(by_camera, 'side') is None
    assert models.for_video(by_camera, '', only_camera=True) == 'hbt-top-v1'
    assert models.for_video({'name': 'habituation'}, '') is None
    assert models.for_video(None, '') is None


def test_the_job_finds_its_model_and_refuses_a_changed_copy(tmp_path, monkeypatch):
    folder = model_folder(tmp_path)
    params = {'model': 'of-top-v1', 'model_sha256': models.digest(folder),
              'snapshot': 'snapshot-200.pt'}
    monkeypatch.delenv('RAPHAEL_MODELS', raising=False)
    with pytest.raises(jobs.JobError, match='no models folder'):
        jobs.registered_model(params)
    monkeypatch.setenv('RAPHAEL_MODELS', str(tmp_path))
    assert jobs.registered_model(params) == folder
    (folder / 'train' / 'snapshot-200.pt').write_bytes(b'retrained')
    with pytest.raises(jobs.JobError, match='differs from the one the plan was made with'):
        jobs.registered_model(params)
    with pytest.raises(jobs.JobError, match='not found'):
        jobs.registered_model({**params, 'model': 'other'})


def test_the_catalogue_names_the_model(tmp_path):
    from raphael import dataserver

    record = {'params': {'model': 'of-top-v1'},
              'environment': {'packages': {'deeplabcut': '3.0.2'}}}
    assert dataserver._software('dlc_capture', record, tmp_path) == 'deeplabcut 3.0.2 (of-top-v1)'
