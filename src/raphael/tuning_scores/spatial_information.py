"""Skaggs spatial information with a shift-shuffled null.

Ported from spatial-manifolds. The VR branch (`'trials' in session`) is gone;
this version takes an explicit `variable` so the same code covers 2D tuning in
the box and 1D tuning along the shuttle axis.

`bin_config` maps each variable to its binning, e.g.
`{('P_x', 'P_y'): {'dim': '2d', 'bounds': (0, 30, 0, 20), 'num_bins': 20,
'smooth_sigma': 1.5}, 'P': {...}}`. Arenas differ between projects, so the
project supplies it (social-manifolds: `data.binning.get_bin_config`).
"""

import numpy as np
import pynapple as nap
from scipy.stats import norm

from raphael.util import gaussian_filter_nan


def compute_spatial_information(
    session,
    bin_config,
    cluster_spikes,
    cluster_id,
    alpha,
    n_shuffles,
    variable='2d',
    ep='moving',
):
    results = {'cluster_id': cluster_id, 'variable': variable}
    shuffles = [
        nap.shift_timestamps(cluster_spikes, min_shift=20.0)
        for _ in range(n_shuffles)
    ]
    group = nap.TsGroup([cluster_spikes] + shuffles)
    epoch = session[ep] if isinstance(ep, str) else ep

    with np.errstate(invalid='ignore', divide='ignore'):
        if variable == '2d':
            config = bin_config[('P_x', 'P_y')]
            features = np.stack([session['P_x'], session['P_y']], axis=1)
            tcs = gaussian_filter_nan(
                np.stack(
                    list(
                        nap.compute_2d_tuning_curves(
                            group,
                            features,
                            nb_bins=config['num_bins'],
                            minmax=config['bounds'],
                            ep=epoch,
                        )[0].values()
                    ),
                    axis=0,
                ),
                sigma=(0, config['smooth_sigma'], config['smooth_sigma']),
            )
            scores = nap.compute_2d_mutual_info(
                tcs, features, ep=epoch, minmax=config['bounds']
            )['SI'].values
        else:
            config = bin_config[variable]
            feature = session[variable]
            tcs = gaussian_filter_nan(
                nap.compute_1d_tuning_curves(
                    group,
                    feature,
                    nb_bins=config['num_bins'],
                    minmax=config['bounds'],
                    ep=epoch,
                ),
                sigma=(config.get('smooth_sigma', 0), 0),
            )
            scores = nap.compute_1d_mutual_info(
                tcs, feature, ep=epoch, minmax=config['bounds']
            )['SI'].values

    results['spatial_information'] = scores[0]
    results['null_distribution'] = scores[1:]
    results['sig'] = results['spatial_information'] > norm.ppf(
        1 - alpha,
        loc=np.nanmean(scores[1:]),
        scale=np.nanstd(scores[1:]),
    )
    return [results]
