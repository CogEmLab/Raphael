"""raphael.plots on processed data: a synthetic session with units of known
tuning, written as Raphael's processed files (spikes as _clusters.npz,
imaging dF/F as _traces.npz + _rois.csv), loaded back with
raphael.population, then plotted. Each plot's returned numbers are
checked against the truth."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from raphael import plots, population, style
from raphael.ephys import nwb as ephys_nwb
from raphael.ephys.layout import Session
from raphael.plots import example

SESSION = Session('1', '1', 'OF1')


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    """The example session as processed files, loaded as a project would."""
    root = tmp_path_factory.mktemp('processed')
    d = example.example_data(duration=900.0)
    spikes = d['spikes']
    units = pd.DataFrame({'unit': list(spikes.keys()), 'accepted': True})
    ephys_nwb.export_clusters(units, {u: spikes[u].t for u in list(spikes.keys())}, 900.0,
                              SESSION.clusters(root, 'kilosort4'))
    prefix = SESSION.folder(root) / f'{SESSION.stem}_src-suite2p'
    d['dff'].save(f'{prefix}_traces.npz')
    pd.DataFrame({'roi': d['dff'].columns, 'accepted': True}).to_csv(f'{prefix}_rois.csv',
                                                                      index=False)
    d['units'] = population.load(SESSION.folder(root) / f'{SESSION.stem}_srt-kilosort4')
    d['rois'] = population.load(prefix)
    return d


@pytest.fixture(autouse=True)
def _styled():
    with style.context():
        yield
    plt.close('all')


def _centre(m, bounds):
    """(x, y) in cm of a map's peak."""
    y, x = np.unravel_index(np.nanargmax(m), m.shape)
    x0, x1, y0, y1 = bounds
    return x0 + (x + 0.5) * (x1 - x0) / m.shape[1], y0 + (y + 0.5) * (y1 - y0) / m.shape[0]


def test_rate_maps_find_the_fields(session):
    beh, arena = session['behaviour'], session['arena']
    _, (a, b, c) = plt.subplots(1, 3)
    for ax, unit, field in ((a, 0, example.FIELDS_CM[0]), (b, 1, example.FIELDS_CM[1])):
        m = plots.rate_map(ax, session['units'].data[unit], beh, arena)
        assert np.hypot(*np.subtract(_centre(m, arena), field)) < 8
    m = plots.rate_map(c, session['rois'].data[:, 0], beh, arena, colorbar=True)   # imaging
    assert np.hypot(*np.subtract(_centre(m, arena), example.FIELDS_CM[0])) < 8
    # A non-square arena keeps its shape.
    m = plots.rate_map(a, session['units'].data[0], beh, (0, 100, 0, 50), bins=40)
    assert m.shape == (20, 40)


def test_head_direction_and_theta_phase(session):
    fig = plt.figure()
    polar = fig.add_subplot(121, projection='polar')
    _, preferred, mvl = plots.head_direction(polar, session['units'].data[2],
                                             session['behaviour'])
    assert abs(np.angle(np.exp(1j * (preferred - example.HD_PREFERRED)))) < np.deg2rad(15)
    assert mvl > 0.3
    _, _, flat = plots.head_direction(polar, session['units'].data[5], session['behaviour'])
    assert flat < 0.1
    with pytest.raises(ValueError, match='polar'):
        plots.head_direction(fig.add_subplot(122), session['units'].data[2],
                             session['behaviour'])
    _, phase, mvl = plots.phase_histogram(fig.add_subplot(122), session['units'].data[3],
                                          session['theta_phase'])
    assert abs(np.angle(np.exp(1j * (phase - example.THETA_PREFERRED)))) < np.deg2rad(20)


def test_tuning_curve_blanks_unvisited_bins(session):
    _, ax = plt.subplots()
    curve = plots.tuning_curve(ax, session['units'].data[6], session['behaviour']['S'],
                               bins=12, minmax=(0, 60), xlabel='Speed (cm/s)')
    visited = curve.dropna()
    assert visited.index.max() < 50 and curve[curve.index > 50].isna().all()   # never visited
    assert np.corrcoef(visited.index, visited.values)[0, 1] > 0.9              # speed cell


def test_peri_event_spikes_and_signals(session):
    _, axes = plt.subplots(2, 3)
    times, mean, _ = plots.peth(axes[0, 0], axes[1, 0], session['units'].data[4],
                                session['events'], window=(-0.5, 1.0), binsize=0.025)
    assert 0.08 < times[np.argmax(mean)] < 0.25
    times, mean, sem = plots.peth(axes[0, 1], axes[1, 1], session['photometry'],
                                  session['events'], window=(-1, 4))
    assert 0 < times[np.argmax(mean)] < 1 and np.all(sem >= 0)
    times, mean, _ = plots.peth(None, axes[1, 2], session['rois'].data[:, 3], session['events'],
                                window=(-1, 3))
    assert 0 < times[np.argmax(mean)] < 1.5


def test_spike_statistics(session):
    _, (a, b, c) = plt.subplots(1, 3)
    cc = plots.autocorrelogram(a, session['units'].data[3], binsize=0.005, window=0.3)
    lag = cc[(cc.index > 0.08) & (cc.index < 0.2)].idxmax()
    assert abs(lag - 1 / example.THETA_HZ) < 0.02                  # theta-rhythmic
    counts, _, violations = plots.isi_histogram(b, session['units'].data[5])
    assert counts.sum() > 0 and 0 <= violations < 0.05
    rng = np.random.default_rng(0)
    fraction = plots.score_histogram(c, rng.normal(1, 1, 200), null=rng.normal(0, 1, 2000),
                                     threshold=1.64)
    assert 0.2 < fraction < 0.33              # P(N(1,1) > 1.64) = 0.26


def test_population_plots_and_saving(session, tmp_path):
    fig, (a, b) = plt.subplots(1, 2)
    ids = plots.raster(a, session['units'], 100, 110)
    assert ids == list(session['units'].data.keys())
    offsets = plots.traces(b, session['rois'], 100, 200, scale=0.5, scale_label='0.5 dF/F')
    assert len(offsets) == 4 and offsets[0] > offsets[-1]
    with pytest.raises(ValueError, match='one unit'):
        plots.rate_map(a, session['units'].data, session['behaviour'], session['arena'])
    written = style.save(fig, tmp_path / 'plots')
    assert all(p.stat().st_size > 1000 for p in written)
