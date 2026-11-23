# Machine learning: encoding and decoding

`raphael.ml` fits models in two directions, from a constant mean up to
XGBoost, and scores them all the same honest way.

- **Encoding**: how well does behaviour predict a neuron's spikes? One
  model per neuron, with a Poisson objective, scored by **pseudo-R²**.
- **Decoding**: how well does the population predict behaviour? Scored by
  **R²** (continuous variables), **median error in degrees** (angles) or
  **accuracy** (categories).

**Example notebook:**
[examples/ml_encoding_decoding.ipynb](../examples/ml_encoding_decoding.ipynb).
It works through a simulated population with known tuning, showing every
step with its figures.

## Install

scikit-learn's models come with Raphael. XGBoost is in the `ml` extra:

```bash
uv add "raphael[ml] @ git+https://github.com/CogEmLab/Raphael" --tag vX.Y.Z   # in a project
uv sync --extra ml                                                            # in Raphael
```

## Quick start

```python
from raphael import ml

t, counts = ml.bin_counts(spikes, binsize=0.25, ep=behaviour['moving'])   # bins x neurons
beh = ml.behaviour_at(behaviour, ['P_x', 'P_y', 'H', 'S'], t, circular=['H'])

# Encoding one neuron: compare models on identical folds
X = beh[['P_x', 'P_y', 'S']].join(ml.sin_cos(beh['H'], 'H'))
table, results = ml.compare(['mean', 'poisson_glm', 'random_forest', 'xgboost'],
                            X, counts[unit], task='encode')

# Decoding from the population
ml.cross_validate('ridge', counts, beh['P_x'], task='continuous')    # R²
ml.cross_validate('xgboost', counts, beh['H'], task='circular')      # median error, degrees
ml.cross_validate('logistic', counts, context_labels, task='categorical')

# Is it better than chance?
real, null, p = ml.shuffle_test('ridge', counts, beh['P_x'], task='continuous')
```

Every call returns a `Result`. It holds the mean score and its SEM over
folds, each fold's score, and a held-out prediction for every bin, so the
predictions can be plotted against the data.

## Models

From simplest to most flexible (`ml.MODELS`):

| Name | What | Tasks |
|---|---|---|
| `mean` | the training mean (or most common class): the floor to beat | all |
| `linear` | ordinary least squares | encode, continuous, circular |
| `ridge` | linear with L2 shrinkage, strength chosen by internal CV | encode, continuous, circular |
| `poisson_glm` | Poisson GLM, log link, L2: the standard encoding model | encode |
| `logistic` | multinomial logistic regression | categorical |
| `random_forest` | random forest (Poisson criterion for encoding) | all |
| `gradient_boosting` | scikit-learn histogram gradient boosting (Poisson loss for encoding) | all |
| `xgboost` | XGBoost (count:poisson for encoding); `ml` extra | all |
| `mlp` | feedforward neural network, two hidden layers | all |

Any scikit-learn style estimator (anything with `fit` and `predict`) can
be passed instead of a name: `ml.cross_validate(Lasso(alpha=0.1), ...)`.
Override a named model's settings with keywords:
`ml.cross_validate('xgboost', X, y, max_depth=6, n_estimators=500)`.

The defaults are sensible starting points, not tuned for your data. For a
paper, tune them on data held out from the final comparison. Tuning on
the same folds you report inflates the scores.

**Circular targets** (head direction, phase) are fitted as sin and cos,
and the angle is recovered with arctan2. This works with every model,
including trees. **Encoding with linear models** clips predictions at a
tiny positive rate, because counts can't be negative.

## Features

Raw x and y can't describe a place field to a linear model or GLM, and an
angle can't describe a tuning curve. Tree models find those shapes on
their own; linear models need the features expanded:

- `spatial_basis(x, y, bounds, n=8)`: Gaussian bumps on an n × n grid
  over the arena.
- `circular_basis(angle, n=8)`: von Mises bumps around the circle.
- `sin_cos(angle)`: the plain circular encoding.
- `history(values, n_bases, max_lag_bins)`: each variable's past (never
  the present), through log-spaced raised-cosine filters. Use it for
  covariate history, or a neuron's own spike history, as in Benjamin et
  al. (2018).

In the notebook's place cell, a Poisson GLM on the spatial basis does as
well as XGBoost on raw coordinates (pseudo-R² ≈ 0.50). On raw
coordinates, the GLM manages only 0.20.

## How scores are computed

- **Folds are contiguous blocks of time** (5 by default). Neighbouring
  bins are alike, so random folds put near-copies of each test bin into
  training and inflate scores. Use `kind='random'` only for data without
  temporal structure.
- **With history features, leave a gap** (`gap=` the history length in
  bins), so no training bin's history overlaps the test block.
- **The pseudo-R²** compares the model's Poisson log-likelihood on the
  test block with two references: predicting the training mean (score 0)
  and predicting the data exactly (score 1). It can be negative. Decoding
  R² can also be below 0, even for `mean`, because a test block can come
  from a part of the session unlike the training data.
- **The shuffle test** shifts the target circularly in time against the
  features, by at least a tenth of the session, and re-scores. Unlike a
  random permutation, this keeps each signal's own smoothness, so the
  null isn't too easy to beat. The p-value is
  (shuffles scoring at least as well + 1) / (shuffles + 1).

## Calcium imaging and other signals

`bin_counts` averages a continuous signal (a dF/F `TsdFrame`, or a
`Population` loaded from imaging) per bin. dF/F isn't a count, so encode
it with `task='continuous'` (Gaussian loss, R²), not `'encode'`.

## Where it comes from

This module follows Benjamin et al. (2018), *Modern machine learning as a
benchmark for fitting neural responses*, Frontiers in Computational
Neuroscience, as implemented in spatial-manifolds' `mlencoding.py`. It
keeps the same model families, the Poisson pseudo-R² and the
raised-cosine history features. Some of that code's libraries can't be
installed alongside Raphael (which pins numpy below 2), so scikit-learn
equivalents replace them:

- the GLM uses scikit-learn's `PoissonRegressor` instead of pyglmnet;
- the neural network uses `MLPRegressor` instead of Keras;
- the LSTM isn't included.

Other changes from the original:

- folds are blocked in time by default;
- decoding is added (continuous, circular and categorical targets);
- the time-shift shuffle test is new.

The original's cross-validated stacking (an ensemble trained on the other
models' predictions) isn't ported yet. Ask if you need it.
