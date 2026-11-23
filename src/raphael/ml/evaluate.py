"""Cross-validation, scores, model comparison and a shuffle control.

Folds are contiguous blocks of time by default. Neighbouring time bins are
alike (behaviour and activity are smooth), so randomly chosen test bins sit
next to training bins and scores come out too high. Random folds are
available (`folds='random'`) for data without temporal structure.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from raphael.ml.models import make_model

EPS = np.spacing(1)


# ------------------------------------------------------------------ scores

def poisson_pseudo_r2(y, y_hat, y_null):
    """Poisson pseudo-R2 (Cameron & Windmeijer; as in Benjamin et al. 2018):
    the fraction of the possible log-likelihood gain over the null model
    (predicting `y_null`, the training mean) that `y_hat` achieves. 1 is
    perfect, 0 no better than the mean, below 0 worse."""
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float).reshape(np.shape(y))
    ll = np.sum(y * np.log(EPS + y_hat) - y_hat)
    ll_null = np.sum(y * np.log(EPS + y_null) - y_null)
    ll_sat = np.sum(y * np.log(EPS + y) - y)
    return float(1 - (ll_sat - ll) / (ll_sat - ll_null))


def r2(y, y_hat):
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    return float(1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))


def angular_error(y, y_hat):
    """Median absolute angular difference (rad)."""
    d = np.angle(np.exp(1j * (np.asarray(y_hat) - np.asarray(y))))
    return float(np.median(np.abs(d)))


def accuracy(y, y_hat):
    return float(np.mean(np.asarray(y) == np.asarray(y_hat)))


#: The score reported for each task, and whether higher is better.
SCORES = {
    'encode': ('pseudo_r2', True),
    'continuous': ('r2', True),
    'circular': ('median_error_deg', False),
    'categorical': ('accuracy', True),
}


def _score(task, y_test, y_hat, y_train):
    if task == 'encode':
        return poisson_pseudo_r2(y_test, y_hat, np.mean(y_train))
    if task == 'continuous':
        return r2(y_test, y_hat)
    if task == 'circular':
        return float(np.degrees(angular_error(y_test, y_hat)))
    return accuracy(y_test, y_hat)


# ------------------------------------------------------------------- folds

def folds(n, n_folds=5, kind='blocked', gap=0, random_state=0):
    """(train, test) index arrays. 'blocked': contiguous test blocks, with
    `gap` bins dropped from training on each side of the test block (use
    at least the history length when using history features). 'random':
    shuffled K-fold."""
    idx = np.arange(n)
    if kind == 'random':
        rng = np.random.default_rng(random_state)
        parts = np.array_split(rng.permutation(n), n_folds)
        return [(np.setdiff1d(idx, p), np.sort(p)) for p in parts]
    if kind != 'blocked':
        raise ValueError("folds must be 'blocked' or 'random'")
    out = []
    for test in np.array_split(idx, n_folds):
        keep = (idx < test[0] - gap) | (idx > test[-1] + gap)
        out.append((idx[keep], test))
    return out


# ---------------------------------------------------------- cross-validate

@dataclass
class Result:
    """A cross-validated fit: predictions for every bin (each made by a
    model that never saw that bin), and the score of each fold."""
    model: str
    task: str
    score_name: str
    fold_scores: list
    predicted: np.ndarray
    observed: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def score(self):
        return float(np.mean(self.fold_scores))

    @property
    def sem(self):
        return float(np.std(self.fold_scores, ddof=1) / np.sqrt(len(self.fold_scores)))

    def __repr__(self):
        return (f'Result({self.model}, {self.task}: {self.score_name} = {self.score:.3f} '
                f'+/- {self.sem:.3f} over {len(self.fold_scores)} folds)')


def _name(model):
    return model if isinstance(model, str) else type(model).__name__


def cross_validate(model, X, y, task='encode', n_folds=5, kind='blocked', gap=0,
                   random_state=0, **params):
    """Fit `model` (a name from MODELS, or an estimator) on all but one fold
    and predict the held-out fold, for every fold. Returns a Result."""
    X = np.asarray(X, float)
    X = X[:, None] if X.ndim == 1 else X
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f'X has {len(X)} rows but y has {len(y)}')
    predicted = np.empty(len(y), dtype=object if task == 'categorical' else float)
    scores = []
    for train, test in folds(len(y), n_folds, kind, gap, random_state):
        m = make_model(model, task, random_state=random_state, **params)
        m.fit(X[train], y[train])
        predicted[test] = m.predict(X[test])
        scores.append(_score(task, y[test], predicted[test], y[train]))
    if task != 'categorical':
        predicted = predicted.astype(float)
    return Result(_name(model), task, SCORES[task][0], scores, predicted, y)


def compare(models, X, y, task='encode', **kwargs):
    """Cross-validate several models on the same folds. Returns (table,
    {model: Result}); the table has one row per model, best first."""
    results = {_name(m): cross_validate(m, X, y, task, **kwargs) for m in models}
    name, higher = SCORES[task]
    table = pd.DataFrame([{'model': k, name: r.score, 'sem': r.sem} for k, r in results.items()])
    table = table.sort_values(name, ascending=not higher, ignore_index=True)
    return table, results


def encode_population(model, X, Y, **kwargs):
    """Cross-validated encoding of every neuron (column of Y) from the same
    features. Returns a DataFrame: neuron, pseudo-R2, sem."""
    Y = pd.DataFrame(Y)
    rows = []
    for col in Y.columns:
        r = cross_validate(model, X, Y[col].to_numpy(), 'encode', **kwargs)
        rows.append({'neuron': col, 'pseudo_r2': r.score, 'sem': r.sem})
    return pd.DataFrame(rows)


def shuffle_test(model, X, y, task='encode', n_shuffles=20, min_shift=None, random_state=0,
                 **kwargs):
    """Is the score better than chance? Re-scores with y circularly shifted
    against X by a random offset (at least `min_shift` bins, default a
    tenth of the data) `n_shuffles` times. Shifting keeps each signal's own
    autocorrelation, so the null isn't flattered by smoothness. Returns
    (real Result, null scores, p-value)."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    real = cross_validate(model, X, y, task, random_state=random_state, **kwargs)
    min_shift = min_shift or len(y) // 10
    null = []
    for _ in range(n_shuffles):
        shift = int(rng.integers(min_shift, len(y) - min_shift))
        null.append(cross_validate(model, X, np.roll(y, shift), task,
                                   random_state=random_state, **kwargs).score)
    null = np.array(null)
    higher = SCORES[task][1]
    beats = np.sum(null >= real.score) if higher else np.sum(null <= real.score)
    return real, null, float((beats + 1) / (n_shuffles + 1))
