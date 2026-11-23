"""A simulated population for trying encoders and decoders: place cells
tiling the arena, head-direction cells around the circle, speed cells and
untuned cells, during the same random walk as raphael.plots.example.

    from raphael.ml.example import population
    d = population()          # d['spikes'] (TsGroup), d['behaviour'], d['tuning']
"""

import numpy as np
import pandas as pd
import pynapple as nap

from raphael.plots.example import ARENA_CM, RATE, _poisson, _walk


def population(n_place=30, n_hd=10, n_speed=4, n_untuned=6, duration=900.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / RATE)
    xy, speed, hd = _walk(rng, t.size)
    rows, rates = [], {}
    unit = 0
    for _ in range(n_place):
        cx, cy = rng.uniform(10, 90, 2)
        width, peak = rng.uniform(8, 14), rng.uniform(6, 15)
        rates[unit] = 0.2 + peak * np.exp(-((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
                                         / (2 * width ** 2))
        rows.append({'unit': unit, 'type': 'place', 'x': cx, 'y': cy})
        unit += 1
    for k in range(n_hd):
        pref = 2 * np.pi * k / n_hd + rng.normal(0, 0.1)
        rates[unit] = 0.5 + rng.uniform(8, 15) * np.exp(2.5 * (np.cos(hd - pref) - 1))
        rows.append({'unit': unit, 'type': 'head_direction', 'preferred': pref % (2 * np.pi)})
        unit += 1
    for _ in range(n_speed):
        rates[unit] = 1.0 + rng.uniform(0.3, 0.6) * speed
        rows.append({'unit': unit, 'type': 'speed'})
        unit += 1
    for _ in range(n_untuned):
        rates[unit] = np.full(t.size, rng.uniform(1, 6))
        rows.append({'unit': unit, 'type': 'untuned'})
        unit += 1
    spikes = nap.TsGroup({u: nap.Ts(t=_poisson(rng, t, r)) for u, r in rates.items()},
                         time_support=nap.IntervalSet(0, duration))
    moving = nap.Tsd(t=t, d=speed).threshold(2.0).time_support
    return {
        'spikes': spikes,
        'behaviour': {'P_x': nap.Tsd(t=t, d=xy[:, 0]), 'P_y': nap.Tsd(t=t, d=xy[:, 1]),
                      'H': nap.Tsd(t=t, d=hd), 'S': nap.Tsd(t=t, d=speed), 'moving': moving},
        'tuning': pd.DataFrame(rows),
        'arena': ARENA_CM,
    }
