"""Toy data: the in-memory session has the tuning it claims, the processed
tree loads through Raphael's own loaders, and the raw tree goes through
Raphael's readers and sync."""

import numpy as np
import pandas as pd
import pytest

from raphael import population, toy
from raphael.ephys.layout import Session

SESSION = Session('7', '3', 'OF1')


@pytest.fixture(scope='module')
def data():
    return toy.session(duration=300, seed=1)


def test_session_tuning_is_true(data):
    tuning = data['tuning']
    assert set(tuning['kind']) <= set(toy.KINDS)
    beh = data['behaviour']
    for _, row in tuning[(tuning['modality'] == 'spikes') & (tuning['kind'] == 'place')].iterrows():
        spikes = data['spikes'][int(row['id'])].t
        x = np.interp(spikes, beh['P_x'].t, beh['P_x'].values)
        y = np.interp(spikes, beh['P_y'].t, beh['P_y'].values)
        assert np.hypot(np.median(x) - row['x'], np.median(y) - row['y']) < 20
    hd = tuning[tuning['kind'] == 'head_direction'].iloc[0]
    angles = np.interp(data['spikes'][int(hd['id'])].t, beh['H'].t, beh['H'].values)
    mean = np.angle(np.mean(np.exp(1j * angles)))
    assert abs(np.angle(np.exp(1j * (mean - hd['preferred'])))) < 0.4


def test_processed_tree_loads_with_raphael(data, tmp_path):
    from raphael.photometry.nwb import load as load_photometry

    paths = toy.processed(tmp_path, SESSION, data)
    spikes = population.load(str(paths['clusters']).removesuffix('_clusters.npz'))
    assert spikes.kind == 'spikes' and len(spikes.ids) == len(data['spikes'])
    assert set(spikes.table['kind']) == set(data['tuning'].query('modality == "spikes"')['kind'])
    dff = population.load(str(paths['rois']).removesuffix('_rois.csv'))
    assert dff.kind == 'dff' and len(dff.ids) == data['dff'].shape[1]
    fp = load_photometry(SESSION.folder(tmp_path) / SESSION.stem)
    assert fp.signals == ['green'] and len(fp.dff('green')) == len(data['photometry'])
    beh = pd.read_csv(paths['behaviour'])
    assert list(beh.columns) == ['time', 'P_x', 'P_y', 'H', 'S']
    assert pd.read_csv(paths['truth']).shape[0] == len(data['tuning'])


def test_raw_tree_syncs(data, tmp_path):
    from raphael.sync.run import sync_session

    paths = toy.raw(tmp_path / 'raw', SESSION, data, recording_duration_s=60)
    folder = SESSION.folder(tmp_path / 'raw')
    names = sorted(p.name for p in folder.iterdir())
    assert all(n.startswith(SESSION.stem) for n in names)
    manifest = pd.read_csv(paths['manifest'], sep='\t')
    assert {'path', 'original_name', 'bytes', 'sha256'} <= set(manifest.columns)
    result, written = sync_session(paths['behaviour'], paths['ephys'], tmp_path / 'synced',
                                   SESSION)
    assert result.passed
    assert abs(result.intercept - paths['clock']['intercept']) < 0.005
    assert 'behaviour' in written
    from raphael.photometry import doric
    assert doric.read(paths['photometry'])


def test_raw_ground_truth_neuropixels(tmp_path):
    from raphael.ephys import sorting as srt

    paths = toy.raw(tmp_path, SESSION, toy.session(duration=20), ephys='spikes',
                    photometry=False)
    recording = srt.read(paths['ephys'])
    assert recording.get_num_channels() == 64 and recording.get_total_duration() == 20
    assert (SESSION.folder(tmp_path) / f'{SESSION.stem}_ephys-truth').exists()
