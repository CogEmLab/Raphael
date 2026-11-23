"""Machine-learning encoders and decoders on simulated neurons with known
tuning: each model family recovers what it should, scores mean what they
say, and the example notebook runs."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from raphael import ml
from raphael.ml.example import population

ROOT = Path(__file__).resolve().parents[1]
SKLEARN_MODELS = ['mean', 'linear', 'poisson_glm', 'random_forest', 'gradient_boosting', 'mlp']


@pytest.fixture(scope='module')
def data():
    d = population(n_place=20, n_hd=8, n_speed=2, n_untuned=4, duration=600)
    t, counts = ml.bin_counts(d['spikes'], 0.25, ep=d['behaviour']['moving'])
    beh = ml.behaviour_at(d['behaviour'], ['P_x', 'P_y', 'H', 'S'], t, circular=['H'])
    return d, counts, beh


def kinds(d, kind):
    return d['tuning'].query('type == @kind')['unit'].tolist()


def test_pseudo_r2_reference_points():
    y = np.array([0, 1, 3, 0, 2, 5])
    assert ml.poisson_pseudo_r2(y, np.full(6, y.mean()), y.mean()) == pytest.approx(0)
    assert ml.poisson_pseudo_r2(y, y + 1e-12, y.mean()) == pytest.approx(1, abs=1e-6)
    assert ml.poisson_pseudo_r2(y, np.full(6, 10.0), y.mean()) < 0


def test_blocked_folds_are_contiguous_and_respect_the_gap():
    for train, test in ml.folds(100, 4, gap=3):
        assert np.all(np.diff(test) == 1)
        assert np.min(np.abs(train[:, None] - test[None, :])) > 3
    covered = np.concatenate([test for _, test in ml.folds(100, 4)])
    assert np.array_equal(np.sort(covered), np.arange(100))


def test_history_never_sees_the_present():
    x = np.zeros(30)
    x[10] = 1
    h = ml.history(x, n_bases=3, max_lag_bins=6, name='x')
    assert np.all(h.iloc[:11].to_numpy() == 0)          # nothing at or before the impulse
    assert h.iloc[11:17].to_numpy().sum() == pytest.approx(3)
    assert np.allclose(ml.raised_cosines(4, 10).sum(axis=1), 1)


@pytest.mark.parametrize('model', SKLEARN_MODELS)
def test_every_model_encodes_a_place_cell(data, model):
    d, counts, beh = data
    unit = kinds(d, 'place')[0]
    X = beh[['P_x', 'P_y', 'S']].join(ml.sin_cos(beh['H'], 'H'))
    r = ml.cross_validate(model, X, counts[unit], task='encode', n_folds=3)
    assert len(r.predicted) == len(counts) and np.all(r.predicted >= 0)
    if model == 'mean':
        assert r.score == pytest.approx(0, abs=1e-9)
    elif model in ('random_forest', 'gradient_boosting'):
        assert r.score > 0.1                     # trees fit a field from raw x, y


def test_basis_features_let_a_glm_fit_place_and_direction(data):
    d, counts, beh = data
    X = (ml.spatial_basis(beh['P_x'], beh['P_y'], d['arena'])
         .join(ml.circular_basis(beh['H'], 8, name='H')))
    scores = ml.encode_population('poisson_glm', X, counts, n_folds=3).set_index('neuron')
    place, hd, untuned = (scores.loc[kinds(d, k), 'pseudo_r2']
                          for k in ('place', 'head_direction', 'untuned'))
    assert place.median() > 0.2 and hd.median() > 0.2
    assert untuned.max() < 0.05              # nothing to find: no better than the mean


def test_decoding_position_direction_and_category(data):
    _, counts, beh = data
    floor = ml.cross_validate('mean', counts, beh['P_x'], task='continuous')
    ridge = ml.cross_validate('ridge', counts, beh['P_x'], task='continuous')
    assert ridge.score > 0.2 > floor.score
    hd = ml.cross_validate('ridge', counts, beh['H'], task='circular')
    assert hd.score < 30                                  # degrees; chance is 90
    assert np.all((hd.predicted >= 0) & (hd.predicted < 2 * np.pi))
    half = np.where(beh['P_x'] > 50, 'east', 'west')
    acc = ml.cross_validate('logistic', counts, half, task='categorical')
    assert acc.score > 0.75 and set(acc.predicted) <= {'east', 'west'}


def test_compare_orders_best_first(data):
    _, counts, beh = data
    table, results = ml.compare(['mean', 'ridge', 'random_forest'], counts, beh['P_x'],
                                task='continuous', n_folds=3)
    assert list(table.columns) == ['model', 'r2', 'sem']
    assert table['model'].iloc[-1] == 'mean' and set(results) == {'mean', 'ridge', 'random_forest'}


def test_shuffle_test_separates_tuned_from_untuned(data):
    d, counts, beh = data
    X = ml.spatial_basis(beh['P_x'], beh['P_y'], d['arena'])
    _, _, p_place = ml.shuffle_test('poisson_glm', X, counts[kinds(d, 'place')[0]],
                                    n_shuffles=9, n_folds=3)
    _, _, p_untuned = ml.shuffle_test('poisson_glm', X, counts[kinds(d, 'untuned')[0]],
                                      n_shuffles=9, n_folds=3)
    assert p_place == pytest.approx(0.1) and p_untuned > 0.1


def test_any_estimator_and_helpful_errors(data):
    from sklearn.linear_model import Lasso
    _, counts, beh = data
    r = ml.cross_validate(Lasso(alpha=0.1), counts, beh['H'], task='circular', n_folds=3)
    assert r.model == 'Lasso' and r.score < 90
    with pytest.raises(ValueError, match='known'):
        ml.make_model('svm')
    with pytest.raises(ValueError, match='does not do'):
        ml.make_model('poisson_glm', 'categorical')


def test_xgboost_encodes_and_decodes(data):
    pytest.importorskip('xgboost')
    d, counts, beh = data
    unit = kinds(d, 'place')[0]
    enc = ml.cross_validate('xgboost', beh[['P_x', 'P_y']], counts[unit], n_folds=3)
    assert enc.score > 0.2
    half = np.where(beh['P_x'] > 50, 'east', 'west')
    assert ml.cross_validate('xgboost', counts, half, task='categorical', n_folds=3).score > 0.7


def test_example_notebook_runs(tmp_path, monkeypatch):
    """Every code cell of examples/ml_encoding_decoding.ipynb, in order."""
    pytest.importorskip('xgboost')
    import matplotlib
    matplotlib.use('Agg')
    nb = json.loads((ROOT / 'examples' / 'ml_encoding_decoding.ipynb').read_text())
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Our own notebook's code, run as a test (no Jupyter kernel needed).
        exec(compile(code, 'ml_encoding_decoding.ipynb', 'exec'), {'__name__': '__main__'})  # noqa: S102
