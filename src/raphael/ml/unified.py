"""Unified CEBRA: one contrastive encoder over many recordings, and each
neuron's contribution to what it encodes, on one scale across recordings.

    from raphael.ml import unified

    train, test = unified.split(neural, labels)           # per recording, blocked in time
    model = unified.Unified(max_iterations=5000).fit(*train)
    z = model.transform(*test, target=0)                   # recording 0's timeline
    unified.decode(model, train, test, targets_train, targets_test)
    M = unified.attribution(model, train, features_train)  # neurons x features

`neural` is a list with one (time x neurons) array or DataFrame per
recording, with any number of neurons each. `labels` is a list of the
matching (time x labels) auxiliary variables, defined the same way in every
recording (position in cm, trial time, a stimulus feature, ...).

How it works (Benquet, Mirzaei, Schneider & Mathis 2026, bioRxiv
10.64898/2026.11.18.767506): at each training step one recording is the
reference. Every other recording contributes its activity at the timepoint
whose label is nearest, and the matched activities are concatenated into one
pseudo-population of all the neurons. One encoder embeds it with CEBRA's
contrastive (InfoNCE) objective, so neurons from different recordings share
a latent space. Inference does the same, along one target recording's
timeline: the result embeds the pooled population, not that recording on
its own. The labels choose which timepoints are combined, but they are
never an input to the encoder: shuffling the activity in time makes the
embedding uninformative (`shuffle_neurons`).

Because every neuron enters the same encoder, the encoder's Jacobian
chained with a linear readout of a feature gives each neuron a signed
attribution to that feature, M = J^T W. With normalised embeddings, M is
invariant to the rotation that relates independently trained runs.

Training uses CEBRA's own unified sampler and solver (cebra >= 0.6.1, the
`cebra` extra). Matching at inference, the pooled input and the Jacobian are
computed here, so they work the same on CUDA, Apple GPUs and the CPU.
"""

import dataclasses
import random

import numpy as np
import pandas as pd

DEFAULTS = {
    'model': 'offset10-model',   # CEBRA architecture: a 10-bin receptive field
    'output_dimension': 8,
    'hidden_units': 32,
    'batch_size': 512,
    'max_iterations': 5000,
    'time_offset': 10,           # bins: the label change that defines a positive pair
    'temperature': 1.0,
    'learnable_temperature': True,
    'learning_rate': 3e-4,
    'standardise_labels': True,  # labels are matched by Euclidean distance
    'device': 'auto',            # auto | cuda | mps | cpu
}


def _cebra():
    try:
        import cebra
        import cebra.data
        import cebra.models
        import cebra.solver
    except ImportError as e:
        raise ImportError(
            'Unified CEBRA needs the cebra extra: uv sync --extra cebra (in Raphael) or '
            '"raphael[cebra]" (in a project).') from e
    return cebra


def choose_device(requested='auto'):
    """'auto': CUDA, else the Apple GPU, else the CPU. The CPU is ~50x
    slower than an Apple GPU for CEBRA's models: fine for tests, slow for
    real fits."""
    import torch

    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _array(x, dtype=np.float32):
    a = np.asarray(x.to_numpy() if hasattr(x, 'to_numpy') else x, dtype)
    return a[:, None] if a.ndim == 1 else a


def _names(x):
    if isinstance(x, pd.DataFrame):
        return list(x.columns)
    return list(range(_array(x).shape[1]))


def split(neural, labels, *others, test_fraction=0.2):
    """Each recording split into a training block (the start) and a test
    block (the last `test_fraction`), so test bins are not interleaved with
    the training bins they are correlated with. Extra lists (e.g. decoding
    targets) are split the same way. Returns (train, test), each a tuple of
    lists in the order given."""
    lists = (neural, labels) + others
    train, test = [[] for _ in lists], [[] for _ in lists]
    for i in range(len(neural)):
        n = len(neural[i])
        cut = round(n * (1 - test_fraction))
        for k, values in enumerate(lists):
            v = values[i]
            if len(v) != n:
                raise ValueError(f'recording {i}: {len(v)} rows, but {n} rows of neural data')
            train[k].append(v.iloc[:cut] if hasattr(v, 'iloc') else v[:cut])
            test[k].append(v.iloc[cut:] if hasattr(v, 'iloc') else v[cut:])
    return tuple(train), tuple(test)


@dataclasses.dataclass
class Unified:
    """A Unified CEBRA encoder. Settings as in DEFAULTS."""

    model: str = DEFAULTS['model']
    output_dimension: int = DEFAULTS['output_dimension']
    hidden_units: int = DEFAULTS['hidden_units']
    batch_size: int = DEFAULTS['batch_size']
    max_iterations: int = DEFAULTS['max_iterations']
    time_offset: int = DEFAULTS['time_offset']
    temperature: float = DEFAULTS['temperature']
    learnable_temperature: bool = DEFAULTS['learnable_temperature']
    learning_rate: float = DEFAULTS['learning_rate']
    standardise_labels: bool = DEFAULTS['standardise_labels']
    device: str = DEFAULTS['device']
    seed: int = 0

    def settings(self):
        return dataclasses.asdict(self)

    # -- labels and matching --------------------------------------------------

    def _labels(self, labels):
        ys = [_array(y) for y in labels]
        return [(y - self.label_mean_) / self.label_sd_ for y in ys]

    def _check(self, neural, labels):
        if len(neural) != len(labels):
            raise ValueError(f'{len(neural)} neural arrays but {len(labels)} label arrays')
        if hasattr(self, 'n_neurons_'):
            n = [_array(x).shape[1] for x in neural]
            if n != self.n_neurons_:
                raise ValueError(f'neurons per recording {n}, but the model was fitted on '
                                 f'{self.n_neurons_}')

    def match(self, labels, target):
        """For every timepoint of recording `target`, the index of the
        nearest-label timepoint in each recording (squared Euclidean
        distance on the standardised labels). Returns a list of index arrays,
        one per recording; the target's is 0..T-1."""
        from scipy.spatial import cKDTree

        ys = self._labels(labels)
        query = ys[target]
        out = []
        for i, y in enumerate(ys):
            if i == target:
                out.append(np.arange(len(query)))
            else:
                out.append(cKDTree(y).query(query, k=1)[1])
        return out

    def pseudo_population(self, neural, labels, target):
        """The encoder's input along recording `target`'s timeline: each
        recording's activity at its matched timepoints, concatenated
        (time x all neurons, in recording order)."""
        self._check(neural, labels)
        idx = self.match(labels, target)
        return np.concatenate([_array(x)[j] for x, j in zip(neural, idx)], axis=1)

    # -- training -------------------------------------------------------------

    def fit(self, neural, labels):
        """Train on a list of recordings (time x neurons each) and their
        labels (time x labels each)."""
        import torch

        cebra = _cebra()
        self._check(neural, labels)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        ys = [_array(y) for y in labels]
        widths = {y.shape[1] for y in ys}
        if len(widths) != 1:
            raise ValueError(f'every recording needs the same labels; got widths {widths}')
        every = np.concatenate(ys)
        self.label_mean_ = every.mean(0) if self.standardise_labels else np.zeros(every.shape[1])
        sd = every.std(0) if self.standardise_labels else np.ones(every.shape[1])
        self.label_sd_ = np.where(sd > 0, sd, 1.0)
        self.n_neurons_ = [_array(x).shape[1] for x in neural]
        self.neuron_names_ = [_names(x) for x in neural]
        self.device_ = choose_device(self.device)

        datasets = [cebra.data.TensorDataset(torch.from_numpy(_array(x)),
                                             continuous=torch.from_numpy(y.astype(np.float32)),
                                             device=self.device_)
                    for x, y in zip(neural, self._labels(labels))]
        dataset = cebra.data.UnifiedDataset(d for d in datasets)  # generator: also one recording
        encoder = cebra.models.init(self.model, num_neurons=int(dataset.input_dimension),
                                    num_units=self.hidden_units,
                                    num_output=self.output_dimension).to(self.device_)
        dataset.configure_for(encoder)
        loader = cebra.data.UnifiedLoader(dataset, batch_size=self.batch_size,
                                          num_steps=self.max_iterations,
                                          time_offset=self.time_offset)
        criterions = cebra.models.criterions
        criterion = (criterions.LearnableCosineInfoNCE if self.learnable_temperature
                     else criterions.FixedCosineInfoNCE)(temperature=self.temperature)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(criterion.parameters()),
                                     lr=self.learning_rate)
        self.solver_ = cebra.solver.init('unified-session', model=encoder, criterion=criterion,
                                         optimizer=optimizer, tqdm_on=False)
        # CEBRA's multi-session sampler searches labels on a device fixed at
        # import (CUDA, else the CPU). On an Apple GPU, searching there too
        # halves the time per step; results are unchanged.
        import cebra.distributions.multisession as sampler_module

        before = sampler_module._device
        sampler_module._device = self.device_
        try:
            self.solver_.fit(loader)
        finally:
            sampler_module._device = before
        self.encoder_ = encoder.eval()
        self.loss_ = np.asarray(self.solver_.log['total'], float)
        return self

    @property
    def goodness_of_fit(self):
        """Bits above chance, (log B - InfoNCE) / log 2, from the mean loss
        of the last 10 iterations. Compare only models on the same data and
        batch size."""
        return float((np.log(self.batch_size) - self.loss_[-10:].mean()) / np.log(2))

    # -- inference ------------------------------------------------------------

    def _windows(self, X):
        """(T, N) -> (T, N, K) windows of the encoder's receptive field,
        edges padded by repeating the first and last bins."""
        import torch

        offset = self.encoder_.get_offset()
        left, right = offset.left, offset.right
        X = torch.as_tensor(X, dtype=torch.float32)
        padded = torch.cat([X[:1].repeat(left, 1), X, X[-1:].repeat(right - 1, 1)])
        return padded.unfold(0, left + right, 1)            # (T, N, K)

    def transform(self, neural, labels, target, batch_size=1024):
        """The embedding (time x output_dimension) along recording
        `target`'s timeline."""
        import torch

        X = self._windows(self.pseudo_population(neural, labels, target))
        out = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                out.append(self.encoder_(X[start:start + batch_size].to(self.device_)).cpu())
        return torch.cat(out).numpy()

    def transform_all(self, neural, labels):
        """One embedding per recording (each along its own timeline)."""
        return [self.transform(neural, labels, i) for i in range(len(neural))]

    def jacobian(self, neural, labels, target, batch_size=256, over_time=False):
        """d embedding / d neuron along recording `target`'s timeline,
        averaged over the encoder's receptive field: (output_dimension x
        neurons), averaged over time, or (time x output_dimension x neurons)
        with `over_time`."""
        import torch
        from torch.func import jacrev, vmap

        X = self._windows(self.pseudo_population(neural, labels, target))
        encoder = self.encoder_

        def one(window):                                     # (N, K) -> (D,)
            return encoder(window[None])[0]

        jac = vmap(jacrev(one))
        total, frames = None, []
        for start in range(0, len(X), batch_size):
            J = jac(X[start:start + batch_size].to(self.device_)).mean(-1).detach().cpu()
            if over_time:
                frames.append(J)
            s = J.sum(0)
            total = s if total is None else total + s
        if over_time:
            return torch.cat(frames).numpy()
        return (total / len(X)).numpy()

    def neuron_index(self):
        """(recording, neuron) for every input neuron, in encoder order."""
        return pd.MultiIndex.from_tuples(
            [(i, n) for i, names in enumerate(self.neuron_names_) for n in names],
            names=['recording', 'neuron'])

    # -- saving ---------------------------------------------------------------

    def save(self, path):
        import torch

        torch.save({'settings': self.settings(), 'state': self.encoder_.state_dict(),
                    'label_mean': self.label_mean_, 'label_sd': self.label_sd_,
                    'n_neurons': self.n_neurons_, 'neuron_names': self.neuron_names_,
                    'loss': self.loss_}, path)
        return path

    @classmethod
    def load(cls, path, device='auto'):
        import torch

        cebra = _cebra()
        saved = torch.load(path, map_location='cpu', weights_only=False)
        self = cls(**{**saved['settings'], 'device': device})
        self.device_ = choose_device(device)
        self.label_mean_, self.label_sd_ = saved['label_mean'], saved['label_sd']
        self.n_neurons_, self.neuron_names_ = saved['n_neurons'], saved['neuron_names']
        self.loss_ = saved['loss']
        self.encoder_ = cebra.models.init(self.model, num_neurons=sum(self.n_neurons_),
                                          num_units=self.hidden_units,
                                          num_output=self.output_dimension)
        self.encoder_.load_state_dict(saved['state'])
        self.encoder_ = self.encoder_.to(self.device_).eval()
        return self


# -- analyses -----------------------------------------------------------------

def _knn(k):
    from sklearn.neighbors import KNeighborsRegressor

    return KNeighborsRegressor(n_neighbors=k)


def decode(model, train, test, targets_train, targets_test, task='continuous', k=36):
    """Decode each recording's targets from its embedding: a k-nearest-
    neighbour decoder fitted on the training embedding, scored on the test
    embedding (as in the paper). `train` and `test` are (neural, labels)
    pairs of lists, as `split` returns; targets are lists of (time x
    targets) arrays, which may differ from the labels. task 'continuous'
    scores R2 (mean over target columns) and median absolute error;
    'circular' (one angle, rad) the median angular error in degrees.

    Unified CEBRA uses the labels at inference to choose which timepoints
    of the other recordings are pooled, so this measures how much of the
    targets the pooled population carries, not how well one recording
    alone predicts them."""
    from raphael.ml.evaluate import angular_error, r2

    rows = []
    for i in range(len(train[0])):
        z_train = model.transform(*train, target=i)
        z_test = model.transform(*test, target=i)
        y_train, y_test = _array(targets_train[i], float), _array(targets_test[i], float)
        if task == 'circular':
            fit = _knn(k).fit(z_train, np.c_[np.cos(y_train[:, 0]), np.sin(y_train[:, 0])])
            p = fit.predict(z_test)
            angle = np.arctan2(p[:, 1], p[:, 0])
            rows.append({'recording': i,
                         'median_error_deg': np.degrees(angular_error(y_test[:, 0], angle))})
        else:
            p = _knn(k).fit(z_train, y_train).predict(z_test)
            rows.append({'recording': i,
                         'r2': float(np.mean([r2(y_test[:, j], p[:, j])
                                              for j in range(y_test.shape[1])])),
                         'median_abs_error': float(np.median(np.abs(p - y_test)))})
    return pd.DataFrame(rows).set_index('recording')


def readout(model, train, features, alpha=9.0):
    """A ridge readout W (output_dimension x features) from the training
    embeddings (all recordings, pooled) to standardised features, as in the
    paper (alpha = 9). Returns W and the feature names."""
    from sklearn.linear_model import Ridge

    z = np.concatenate(model.transform_all(*train))
    ys = [_array(f, float) for f in features]
    y = np.concatenate(ys)
    y = (y - y.mean(0)) / np.where(y.std(0) > 0, y.std(0), 1)
    names = list(features[0].columns) if isinstance(features[0], pd.DataFrame) \
        else list(range(y.shape[1]))
    return Ridge(alpha=alpha).fit(z, y).coef_.T, names


def attribution(model, train, features, alpha=9.0, over_time=False, target=None):
    """The neuron-to-feature attribution map M = J^T W: how far a change in
    each neuron's activity moves the readout of each feature (standardised
    units per unit of activity). J is the encoder Jacobian averaged over its
    receptive field and over time, along every recording's timeline in turn
    (or only `target`'s); W is `readout`. Returns a DataFrame indexed by
    (recording, neuron), one column per feature.

    With `over_time`, returns instead (time x neurons x features) along
    `target`'s timeline, to see where (e.g. at which positions) each neuron
    pushes the readout."""
    W, names = readout(model, train, features, alpha)
    if over_time:
        if target is None:
            raise ValueError('over_time needs a target recording')
        J = model.jacobian(*train, target=target, over_time=True)      # T, D, N
        return np.einsum('tdn,df->tnf', J, W)
    targets = range(len(train[0])) if target is None else [target]
    lengths = np.array([len(train[0][i]) for i in targets], float)
    J = sum(model.jacobian(*train, target=i) * n for i, n in zip(targets, lengths)) / lengths.sum()
    return pd.DataFrame(J.T @ W, index=model.neuron_index(), columns=names)


def shuffle_neurons(neural, seed=0):
    """Each neuron's activity permuted in time, independently (labels left
    alone): the control that the labels alone carry no information."""
    rng = np.random.default_rng(seed)
    out = []
    for x in neural:
        a = _array(x).copy()
        for j in range(a.shape[1]):
            a[:, j] = a[rng.permutation(len(a)), j]
        out.append(pd.DataFrame(a, index=x.index, columns=x.columns)
                   if isinstance(x, pd.DataFrame) else a)
    return out


def consistency(source, target):
    """R2 of a linear map (with intercept) from one embedding to another of
    the same timepoints: near 1 when they agree up to a linear transform.
    Use it across training runs (seeds) to check a fit is reproducible."""
    from sklearn.linear_model import LinearRegression

    return float(LinearRegression().fit(source, target).score(source, target))
