"""The models, by name, from the simplest to the most flexible.

Every model is a scikit-learn style estimator (`fit`, `predict`), so any
other estimator with those two methods can be passed in its place.

Encoding (behaviour -> one neuron's spike counts) uses a Poisson objective
where the model has one, since counts are not Gaussian. Decoding predicts a
continuous variable (position, speed), a circular one (head direction,
phase: fitted as sin and cos) or a category (context, trial type).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TASKS = ('encode', 'continuous', 'circular', 'categorical')

#: name: (what it is, for which tasks). Ordered from simplest to most flexible.
MODELS = {
    'mean': (('predicts the training mean (or most common class): the floor every model '
             'must beat'), TASKS),
    'linear': ('ordinary least squares', ('encode', 'continuous', 'circular')),
    'ridge': ('linear with L2 shrinkage, strength chosen by internal CV',
              ('encode', 'continuous', 'circular')),
    'poisson_glm': ('Poisson GLM (log link, L2): the standard encoding model',
                    ('encode',)),
    'logistic': ('multinomial logistic regression (L2)', ('categorical',)),
    'random_forest': ('random forest', TASKS),
    'gradient_boosting': (("scikit-learn's histogram gradient boosting (Poisson loss for "
                          'encoding)'), TASKS),
    'xgboost': (('XGBoost gradient-boosted trees (count:poisson for encoding); needs the '
                'ml extra'), TASKS),
    'mlp': ('feedforward neural network, two hidden layers', TASKS),
}


class _Mean(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.mean_ = np.mean(y, axis=0)
        return self

    def predict(self, X):
        return np.tile(self.mean_, (len(X), 1)).squeeze()


class _MostCommon(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        self.classes_, self.value_ = values, values[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


class _Positive(BaseEstimator, RegressorMixin):
    """Wraps a regressor so its predicted counts are never negative (a
    linear model of counts can predict below zero; Poisson scores need > 0)."""

    def __init__(self, estimator, floor=1e-6):
        self.estimator, self.floor = estimator, floor

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        return np.maximum(self.estimator_.predict(X), self.floor)


class _Labelled(BaseEstimator, ClassifierMixin):
    """Lets a classifier that needs labels 0..k-1 (XGBoost) take any labels."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_, codes = np.unique(np.asarray(y), return_inverse=True)
        self.estimator_ = clone(self.estimator).fit(X, codes)
        return self

    def predict(self, X):
        return self.classes_[np.asarray(self.estimator_.predict(X), int)]


class Circular(BaseEstimator, RegressorMixin):
    """Decodes an angle (rad) by fitting sin and cos with `estimator` (one
    copy each, so trees and boosting work too), then taking arctan2."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y = np.asarray(y, float)
        self.sin_ = clone(self.estimator).fit(X, np.sin(y))
        self.cos_ = clone(self.estimator).fit(X, np.cos(y))
        return self

    def predict(self, X):
        return np.arctan2(self.sin_.predict(X), self.cos_.predict(X)) % (2 * np.pi)


def _xgboost():
    try:
        import xgboost
    except ImportError as e:
        raise ImportError("the 'xgboost' model needs XGBoost: install Raphael with the ml "
                          'extra (uv sync --extra ml, or add "raphael[ml]" to the project)') from e
    return xgboost


def make_model(name, task='encode', random_state=0, **params):
    """A fresh, unfitted model `name` (see MODELS) for `task` (see TASKS).
    `params` override its defaults. A non-string `name` is taken to be an
    estimator already (anything with fit and predict) and is used as is,
    wrapped for circular targets."""
    if task not in TASKS:
        raise ValueError(f'task must be one of {TASKS}, got {task!r}')
    if not isinstance(name, str):
        model = clone(name)
        return Circular(model) if task == 'circular' else model
    if name not in MODELS:
        raise ValueError(f'unknown model {name!r}; known: {", ".join(MODELS)}')
    if task not in MODELS[name][1]:
        raise ValueError(f'{name!r} does not do {task!r}; models that do: '
                         + ', '.join(n for n, (_, t) in MODELS.items() if task in t))
    classify = task == 'categorical'
    poisson = task == 'encode'
    rs = {'random_state': random_state}

    if name == 'mean':
        model = _MostCommon() if classify else _Mean()
    elif name == 'linear':
        from sklearn.linear_model import LinearRegression
        model = make_pipeline(StandardScaler(), LinearRegression(**params))
    elif name == 'ridge':
        from sklearn.linear_model import RidgeCV
        model = make_pipeline(StandardScaler(),
                              RidgeCV(**{'alphas': np.logspace(-3, 3, 13), **params}))
    elif name == 'poisson_glm':
        from sklearn.linear_model import PoissonRegressor
        model = make_pipeline(StandardScaler(),
                              PoissonRegressor(**{'alpha': 1e-3, 'max_iter': 1000, **params}))
    elif name == 'logistic':
        from sklearn.linear_model import LogisticRegression
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(**{'C': 1.0, 'max_iter': 2000, **params}))
    elif name == 'random_forest':
        from sklearn.ensemble import (
            RandomForestClassifier,
            RandomForestRegressor,
        )
        defaults = {'n_estimators': 200, 'min_samples_leaf': 20, 'max_features': 0.5,
                    'n_jobs': -1, **rs}
        if poisson:
            defaults['criterion'] = 'poisson'
        model = (RandomForestClassifier if classify else RandomForestRegressor)(
            **{**defaults, **params})
    elif name == 'gradient_boosting':
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )
        defaults = {'max_iter': 300, 'learning_rate': 0.05, 'max_leaf_nodes': 15,
                    'min_samples_leaf': 40, **rs}
        if poisson:
            defaults['loss'] = 'poisson'
        model = (HistGradientBoostingClassifier if classify else HistGradientBoostingRegressor)(
            **{**defaults, **params})
    elif name == 'xgboost':
        xgb = _xgboost()
        defaults = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 4,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5,
                    'n_jobs': -1, **rs}
        if poisson:
            defaults['objective'] = 'count:poisson'
        model = (_Labelled(xgb.XGBClassifier(**{**defaults, **params})) if classify
                 else xgb.XGBRegressor(**{**defaults, **params}))
    elif name == 'mlp':
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        defaults = {'hidden_layer_sizes': (64, 16), 'alpha': 1e-3, 'learning_rate_init': 1e-3,
                    'max_iter': 300, 'early_stopping': True, **rs}
        model = make_pipeline(StandardScaler(), (MLPClassifier if classify else MLPRegressor)(
            **{**defaults, **params}))

    if poisson and name in ('linear', 'ridge', 'mlp'):
        model = _Positive(model)       # counts can't be negative
    if task == 'circular':
        model = Circular(model)
    return model
