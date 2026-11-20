"""Downstream analysis: Raphael's loaders and scores, this project's
behaviour and arena."""

import pandas as pd

from example_project.arena import bin_config
from example_project.behaviour import load_behaviour
from raphael import population
from raphael.ephys.layout import Session
from raphael.tuning_scores import (
    compute_hd_mean_vector_length,
    compute_spatial_information,
    compute_speed_correlation,
)


def score_session(root, session: Session, sorter='kilosort4', n_shuffles=100, alpha=0.05):
    """One row per accepted unit: spatial information, HD tuning and speed
    correlation, each with whether it beats its shuffled null."""
    # Raphael: spikes, whatever sorted them, in the shared processed tree.
    pop = population.load(session.folder(root) / f'{session.stem}_srt-{sorter}').accepted()
    # The project: behaviour and arena.
    beh = load_behaviour(root, session)
    bins = bin_config()

    rows = []
    for unit in pop.ids:
        spikes = pop.data[unit]
        si, = compute_spatial_information(beh, bins, spikes, unit, alpha, n_shuffles)
        hd, = compute_hd_mean_vector_length(beh, bins, spikes, unit, alpha, n_shuffles)
        sc, = compute_speed_correlation(beh, spikes, unit, alpha, n_shuffles)
        rows.append({'unit': unit,
                     'spatial_information': si['spatial_information'], 'si_sig': si['sig'],
                     'hd_mvl': hd['hd_mean_vector_length'], 'hd_sig': hd['sig'],
                     'speed_corr': sc['speed_correlation'], 'speed_sig': sc['sig']})
    return pd.DataFrame(rows)
