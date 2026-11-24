"""Machine-learning encoding and decoding, from linear regression to XGBoost.

    from raphael import ml

    # Encoding: how well does behaviour predict a neuron's spikes?
    t, counts = ml.bin_counts(spikes, binsize=0.1, ep=moving)
    beh = ml.behaviour_at(behaviour, ['P_x', 'P_y', 'H', 'S'], t, circular=['H'])
    X = beh[['P_x', 'P_y', 'S']].join(ml.sin_cos(beh['H'], 'H'))
    table, results = ml.compare(['mean', 'poisson_glm', 'random_forest', 'xgboost'],
                                X, counts[0], task='encode')

    # Decoding: how well does the population predict position?
    r = ml.cross_validate('ridge', counts, beh['P_x'], task='continuous')

Tasks: 'encode' (spike counts, Poisson objective, scored by pseudo-R2),
'continuous' (R2), 'circular' (angles, fitted as sin and cos, scored by
median error in degrees) and 'categorical' (accuracy). Models are listed in
MODELS, simplest first; any scikit-learn style estimator also works.

Built on the approach of Benjamin et al. (2018), Modern machine learning as
a benchmark for fitting neural responses, Front. Comput. Neurosci., as used
in spatial-manifolds (mlencoding.py): the same models where they install
alongside Raphael (scikit-learn's Poisson GLM, random forest and neural
network replace pyglmnet and Keras), its Poisson pseudo-R2 and its
raised-cosine history features. Folds are contiguous blocks of time by
default, and `shuffle_test` gives a time-shift null. See docs/ml.md.

`raphael.ml.unified` pools recordings with different neurons into one
encoder (Unified CEBRA, the cebra extra), with each neuron's attribution to
the features it encodes.
"""

from raphael.ml.evaluate import (
    SCORES,
    Result,
    accuracy,
    angular_error,
    compare,
    cross_validate,
    encode_population,
    folds,
    poisson_pseudo_r2,
    r2,
    shuffle_test,
)
from raphael.ml.features import (
    behaviour_at,
    bin_counts,
    circular_basis,
    history,
    raised_cosines,
    sin_cos,
    spatial_basis,
)
from raphael.ml.models import MODELS, TASKS, Circular, make_model

__all__ = [
    'MODELS', 'SCORES', 'TASKS', 'Circular', 'Result', 'accuracy', 'angular_error',
    'behaviour_at', 'bin_counts', 'circular_basis', 'compare', 'cross_validate',
    'encode_population', 'folds', 'history', 'make_model', 'poisson_pseudo_r2', 'r2',
    'raised_cosines', 'shuffle_test', 'sin_cos', 'spatial_basis',
]
