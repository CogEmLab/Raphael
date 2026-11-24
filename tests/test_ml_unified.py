"""Unified CEBRA (raphael.ml.unified) on toy recordings with different
neurons: the label matching, the pooled input, the Jacobian (against finite
differences), the attribution map's invariance to rotations of the latent
space, saving, and the controls. Training is kept to a few iterations, so it
runs on a CPU; the opt-in test checks that a real fit learns."""

import os

import numpy as np
import pandas as pd
import pytest

from raphael import ml, toy
from raphael.ml import unified

cebra = pytest.importorskip('cebra')
torch = pytest.importorskip('torch')


def recordings(n=3, duration=240, binsize=0.2):
    neural, labels, truth = [], [], []
    for m in range(n):
        s = toy.session(duration=duration, seed=10 + m, tuning_seed=100 + m, n_place=6 + 2 * m,
                        n_hd=2, n_speed=0, n_theta=0, n_event=0, n_untuned=2)
        t, counts = ml.bin_counts(s['spikes'], binsize=binsize, ep=s['behaviour']['moving'])
        neural.append(counts)
        labels.append(ml.behaviour_at(s['behaviour'], ['P_x', 'P_y'], t))
        truth.append(s['tuning'].query("modality == 'spikes'").set_index('id'))
    return neural, labels, truth


@pytest.fixture(scope='module')
def fitted():
    neural, labels, truth = recordings()
    train, test = unified.split(neural, labels)
    model = unified.Unified(max_iterations=30, batch_size=128, output_dimension=4,
                            hidden_units=16, device='cpu').fit(*train)
    return model, train, test, truth


def test_split_is_blocked_per_recording():
    neural, labels, _ = recordings(n=2, duration=60)
    (n_tr, _, _), (n_te, l_te, extra_te) = unified.split(
        neural, labels, labels, test_fraction=0.25)
    for i in range(2):
        assert len(n_tr[i]) + len(n_te[i]) == len(neural[i])
        assert n_tr[i].index[-1] < n_te[i].index[0]          # test block comes after
        assert len(n_te[i]) == pytest.approx(0.25 * len(neural[i]), abs=1)
        pd.testing.assert_frame_equal(extra_te[i], l_te[i])
    with pytest.raises(ValueError, match='rows'):
        unified.split(neural, [labels[0].iloc[:5], labels[1]])


def test_matching_and_pooled_input(fitted):
    model, train, _, _ = fitted
    neural, labels = train
    assert model.n_neurons_ == [x.shape[1] for x in neural]
    idx = model.match(labels, target=1)
    assert np.array_equal(idx[1], np.arange(len(labels[1])))
    for i in (0, 2):                                      # each match is the nearest label
        z = [(np.asarray(y) - model.label_mean_) / model.label_sd_ for y in labels]
        d = ((z[i][None] - z[1][:50, None]) ** 2).sum(-1)
        assert np.array_equal(idx[i][:50], d.argmin(1))
    X = model.pseudo_population(neural, labels, target=1)
    assert X.shape == (len(labels[1]), sum(model.n_neurons_))
    np.testing.assert_array_equal(X[:, model.n_neurons_[0]:sum(model.n_neurons_[:2])],
                                  np.asarray(neural[1], np.float32))
    with pytest.raises(ValueError, match='neurons per recording'):
        model.transform([neural[0].iloc[:, :2], neural[1], neural[2]], labels, target=0)


def test_transform_shapes_and_normalised(fitted):
    model, _, test, _ = fitted
    zs = model.transform_all(*test)
    assert [z.shape for z in zs] == [(len(x), 4) for x in test[0]]
    np.testing.assert_allclose(np.linalg.norm(zs[0], axis=1), 1, atol=1e-5)   # cosine model
    assert np.isfinite(model.goodness_of_fit)
    assert len(model.loss_) == 30


def test_jacobian_matches_finite_differences(fitted):
    model, train, _, _ = fitted
    neural, labels = train
    short = [x.iloc[:40] for x in neural], [y.iloc[:40] for y in labels]
    J = model.jacobian(*short, target=0, over_time=True)            # T, D, N
    X = model.pseudo_population(*short, target=0)
    t, n, eps = 20, 3, 1e-2
    base = model.transform(*short, target=0)[t]
    windows = model._windows(X)
    K = windows.shape[-1]
    numeric = np.zeros(base.shape)
    for k in range(K):                                  # nudge neuron n at one window bin
        w = windows[t:t + 1].clone()
        w[0, n, k] += eps
        with torch.no_grad():
            numeric += (model.encoder_(w)[0].numpy() - base) / eps
    np.testing.assert_allclose(J[t, :, n], numeric / K, atol=2e-3)
    averaged = model.jacobian(*short, target=0)
    np.testing.assert_allclose(averaged, J.mean(0), atol=1e-6)


def test_attribution_map_is_rotation_invariant(fitted):
    model, train, _, _ = fitted
    M = unified.attribution(model, train, train[1])
    assert list(M.columns) == ['P_x', 'P_y']
    assert M.index.names == ['recording', 'neuron'] and len(M) == sum(model.n_neurons_)
    rotation, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(4, 4)))
    rotated = unified.Unified(**model.settings())
    for name in ('label_mean_', 'label_sd_', 'n_neurons_', 'neuron_names_', 'device_'):
        setattr(rotated, name, getattr(model, name))
    layer = torch.nn.Linear(4, 4, bias=False)
    layer.weight.data = torch.tensor(rotation, dtype=torch.float32)

    class Rotated(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def get_offset(self):
            return self.inner.get_offset()

        def forward(self, x):
            return layer(self.inner(x))

    rotated.encoder_ = Rotated(model.encoder_).eval()
    M_rotated = unified.attribution(rotated, train, train[1])
    np.testing.assert_allclose(M_rotated.to_numpy(), M.to_numpy(), rtol=1e-3, atol=1e-6)


def test_attribution_over_time(fitted):
    model, train, _, _ = fitted
    Mt = unified.attribution(model, train, train[1], over_time=True, target=2)
    assert Mt.shape == (len(train[0][2]), sum(model.n_neurons_), 2)
    with pytest.raises(ValueError, match='target'):
        unified.attribution(model, train, train[1], over_time=True)


def test_save_and_load(fitted, tmp_path):
    model, _, test, _ = fitted
    path = model.save(tmp_path / 'unified.pt')
    loaded = unified.Unified.load(path, device='cpu')
    np.testing.assert_allclose(loaded.transform(*test, target=1), model.transform(*test, target=1),
                               atol=1e-6)
    assert loaded.neuron_index().equals(model.neuron_index())


def test_one_recording_is_plain_cebra():
    neural, labels, _ = recordings(n=1, duration=60)
    model = unified.Unified(max_iterations=5, batch_size=64, device='cpu').fit(neural, labels)
    assert model.transform(neural, labels, target=0).shape == (len(neural[0]), 8)


def test_controls():
    neural, _, _ = recordings(n=2, duration=60)
    shuffled = unified.shuffle_neurons(neural)
    for x, s in zip(neural, shuffled):
        assert list(s.columns) == list(x.columns)
        np.testing.assert_array_equal(np.sort(s.to_numpy(), 0), np.sort(x.to_numpy(np.float32), 0))
        assert not np.array_equal(s.to_numpy(), x.to_numpy())
    z = np.random.default_rng(0).normal(size=(100, 3))
    assert unified.consistency(z, z @ np.diag([2, -1, 0.5]) + 1) == pytest.approx(1)


@pytest.mark.skipif(not os.environ.get('RAPHAEL_TEST_CEBRA'),
                    reason='trains for real (minutes on a GPU); set RAPHAEL_TEST_CEBRA=1')
def test_pooling_recovers_position_and_place_cells():
    neural, labels, truth = recordings(n=4, duration=600, binsize=0.1)
    train, test = unified.split(neural, labels)
    model = unified.Unified(max_iterations=2000).fit(*train)
    scores = unified.decode(model, train, test, train[1], test[1])
    assert (scores['r2'] > 0.7).all()
    control = unified.decode(model, train, (unified.shuffle_neurons(test[0]), test[1]),
                             train[1], test[1])
    assert (control['r2'] < 0.2).all()
    M = unified.attribution(model, train, train[1])
    fields = pd.concat({i: t for i, t in enumerate(truth)}).reindex(M.index)
    place = (fields['kind'] == 'place').to_numpy()
    assert np.hypot(M.P_x, M.P_y)[place].mean() > 3 * np.hypot(M.P_x, M.P_y)[~place].mean()
    assert np.corrcoef(M.P_x[place], fields.x[place])[0, 1] > 0.7
