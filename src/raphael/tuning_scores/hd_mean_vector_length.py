"""Head-direction mean vector length, with a shuffled null.

Ported from spatial-manifolds. `variable` is exposed so the same computation
covers head direction and the egocentric bearing to the victim door, which is
also a circular variable. `bin_config` is as for spatial_information.
"""

import numpy as np
import pynapple as nap
from scipy.stats import norm


def compute_hd_mean_vector_length(
    session,
    bin_config,
    cluster_spikes,
    cluster_id,
    alpha,
    n_shuffles,
    variable='H',
):
    results = {'cluster_id': cluster_id, 'variable': variable}
    shuffles = [
        nap.shift_timestamps(cluster_spikes, min_shift=20.0)
        for _ in range(n_shuffles)
    ]
    bin_config = bin_config[variable]
    tcs = nap.compute_1d_tuning_curves(
        nap.TsGroup([cluster_spikes] + shuffles),
        session[variable],
        nb_bins=bin_config['num_bins'],
        minmax=bin_config['bounds'],
        ep=session['moving'],
    )
    scores = mean_vector_length(tcs.index.values, tcs.values.T)

    results['hd_mean_vector_length'] = scores[0]
    results['null_distribution'] = scores[1:]
    results['sig'] = results['hd_mean_vector_length'] > norm.ppf(
        1 - alpha, loc=np.mean(scores[1:]), scale=np.std(scores[1:])
    )
    return [results]


def mean_vector_length(angles, weights):
    dx = np.cos(angles)
    dy = np.sin(angles)
    totx = np.sum(dx * weights, axis=1) / np.sum(weights, axis=1)
    toty = np.sum(dy * weights, axis=1) / np.sum(weights, axis=1)
    return np.sqrt(totx**2 + toty**2)
