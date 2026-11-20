"""The example downstream project (examples/example-project) runs against
Raphael: its analysis, its figure script and its own job stage. This keeps
the README's "Calling Raphael from another repo" honest."""

import importlib
import sys
from importlib.metadata import EntryPoint
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap
import pytest

from raphael.cluster import jobs
from raphael.ephys import nwb as ephys_nwb
from raphael.ephys.layout import Session

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'example-project'
SESSION = Session('25', '18', 'OF1')


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE / 'src'))
    monkeypatch.syspath_prepend(str(EXAMPLE / 'scripts'))
    yield
    for name in [m for m in sys.modules if m.startswith('example_project')
                 or m == 'figure_tuning']:
        del sys.modules[name]


@pytest.fixture
def processed(tmp_path):
    """A processed session as Raphael writes it (clusters.npz), plus the
    project's own behaviour file: unit 0 is a place cell, unit 1 is not,
    unit 2 failed curation."""
    rng = np.random.default_rng(0)
    rate, duration = 50.0, 600.0
    t = np.arange(0, duration, 1 / rate)
    # A random walk in the 100 x 100 cm box, reflected at the walls.
    xy = np.cumsum(rng.normal(0, 1.5, (len(t), 2)), axis=0) + 50
    xy = 100 - np.abs(100 - np.abs(xy) % 200)
    speed = np.r_[0, np.hypot(*np.diff(xy, axis=0).T) * rate]
    heading = np.r_[0, np.arctan2(*np.diff(xy, axis=0).T[::-1])]
    beh = nap.TsdFrame(t=t, d=np.c_[xy, speed, heading], columns=['P_x', 'P_y', 'S', 'H'])
    root = tmp_path / 'processed'
    SESSION.folder(root).mkdir(parents=True)
    beh.save(str(SESSION.folder(root) / f'{SESSION.stem}_behaviour.npz'))

    in_field = np.hypot(xy[:, 0] - 30, xy[:, 1] - 70) < 15
    place = t[in_field & (rng.random(len(t)) < 0.4)]
    flat = np.sort(rng.uniform(0, duration, 3000))
    spikes = {0: place, 1: flat, 2: flat}
    units = pd.DataFrame({'unit': [0, 1, 2], 'accepted': [True, True, False]})
    ephys_nwb.export_clusters(units, spikes, duration,
                              SESSION.clusters(root, 'kilosort4'))
    return root


def test_analysis_uses_raphael_core(example, processed):
    analysis = importlib.import_module('example_project.analysis')
    table = analysis.score_session(processed, SESSION, n_shuffles=20)
    assert list(table['unit']) == [0, 1]                  # unit 2 not accepted
    place, flat = table.set_index('unit').loc[[0, 1], 'spatial_information']
    assert place > 5 * flat
    assert bool(table.set_index('unit').loc[0, 'si_sig'])


def test_figure_script(example, processed, tmp_path):
    figure = importlib.import_module('figure_tuning')
    path = figure.main([str(processed), 'M25', 'D18', 'OF1', '--out', str(tmp_path / 'fig'),
                        '--n-shuffles', '10'])
    assert path.exists() and path.with_suffix('.csv').exists()


def test_project_stage_runs_through_raphael_jobs(example, processed, tmp_path, monkeypatch):
    ep = EntryPoint('example', 'example_project.stages', jobs.ENTRY_POINT_GROUP)
    monkeypatch.setattr(jobs, 'entry_points',
                        lambda group: [ep] if group == jobs.ENTRY_POINT_GROUP else [])
    try:
        assert jobs.load_project_stages() == ['example']
        rel = SESSION.folder(processed).relative_to(processed.parent)
        with pytest.raises(jobs.JobError, match='mouse'):
            jobs.make(tmp_path / 'jobs' / 'bad', 'tuning_scores', [str(rel)], processed.parent,
                      resources={'gpu': False})
        job = jobs.make(tmp_path / 'jobs' / 'scores', 'tuning_scores', [str(rel)],
                        processed.parent, resources={'gpu': False},
                        params={'mouse': 25, 'day': 18, 'session_type': 'OF1',
                                'n_shuffles': 10})
        jobs.run(job, processed.parent, allow_env_mismatch=True)
        assert jobs.read_status(job)['state'] == 'DONE'
        out = pd.read_csv(job / 'outputs' / f'{SESSION.stem}_tuning_scores.csv')
        assert list(out['unit']) == [0, 1]
    finally:
        jobs.STAGES.pop('tuning_scores', None)
        jobs.CHECKS.pop('tuning_scores', None)
