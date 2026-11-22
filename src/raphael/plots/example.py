"""A synthetic processed session, for trying the plots and testing them.

    from raphael.plots import example_data
    d = example_data()

Units with known tuning, so a plot can be checked against the truth:

    unit  0, 1   place cells (fields at FIELDS_CM)
    unit  2      head-direction cell (preferred direction HD_PREFERRED)
    unit  3      theta-locked cell (preferred phase THETA_PREFERRED), theta-rhythmic
    unit  4      responds to the stimulus events (latency ~0.15 s)
    unit  5      untuned
    unit  6      speed cell (rate rises with running speed)

and, on the same clock: behaviour (P_x, P_y in cm, H in rad, S in cm/s,
`moving`), theta phase, stimulus events, imaging dF/F (ROIs 0-2 place,
ROI 3 event-responsive) and a photometry dF/F trace that rises after
each event.
"""

import numpy as np
import pynapple as nap

ARENA_CM = (0.0, 100.0, 0.0, 100.0)          # x min, x max, y min, y max
FIELDS_CM = [(30.0, 70.0), (70.0, 30.0)]
HD_PREFERRED = np.pi / 3
THETA_PREFERRED = np.pi                       # spikes near the theta trough
THETA_HZ = 8.0
RATE = 100.0                                  # behaviour and traces (Hz)


def _walk(rng, n):
    """A smooth random walk that covers the arena, reflected at the walls."""
    size = ARENA_CM[1] - ARENA_CM[0]
    heading = np.cumsum(rng.normal(0, 0.15, n))
    speed = np.clip(15 + np.convolve(rng.normal(0, 70, n), np.ones(50) / 50, mode='same'), 0, 40)
    step = np.c_[np.cos(heading), np.sin(heading)] * (speed / RATE)[:, None]
    xy = np.cumsum(step, axis=0) + size / 2
    xy = size - np.abs(size - np.abs(xy) % (2 * size))
    d = np.diff(xy, axis=0, prepend=xy[:1])
    return xy, np.hypot(*d.T) * RATE, np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)


def _poisson(rng, t, rate_hz):
    """Spike times from a rate sampled on `t`, jittered within each sample."""
    spikes = t[rng.random(t.size) < rate_hz / RATE]
    return np.sort(spikes + rng.uniform(0, 1 / RATE, spikes.size))


def example_data(duration=1200.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / RATE)
    xy, speed, hd = _walk(rng, t.size)
    theta = (2 * np.pi * THETA_HZ * t + np.cumsum(rng.normal(0, 0.02, t.size))) % (2 * np.pi)
    events = np.arange(5.0, duration - 5, 12.0) + rng.uniform(-1, 1, int((duration - 10) / 12) + 1)
    events = events[events < duration - 5]
    since = np.full(t.size, np.inf)                 # time since the last event
    for e in events:
        after = (t >= e) & (t < e + 3)
        since[after] = np.minimum(since[after], t[after] - e)
    response = np.where(np.isfinite(since), np.exp(-((since - 0.15) / 0.08) ** 2), 0)

    def field(x0, y0, width=10.0):
        return np.exp(-((xy[:, 0] - x0) ** 2 + (xy[:, 1] - y0) ** 2) / (2 * width ** 2))

    rates = {
        0: 0.3 + 12 * field(*FIELDS_CM[0]),
        1: 0.3 + 10 * field(*FIELDS_CM[1]),
        2: 0.5 + 15 * np.exp(2.5 * (np.cos(hd - HD_PREFERRED) - 1)),
        3: 6 * (1 + 0.9 * np.cos(theta - THETA_PREFERRED)),
        4: 1.0 + 40 * response,
        5: np.full(t.size, 4.0),
        6: 1.0 + 0.6 * speed,
    }
    moving = nap.Tsd(t=t, d=speed).threshold(2.0).time_support
    spikes = nap.TsGroup({u: nap.Ts(t=_poisson(rng, t, r)) for u, r in rates.items()},
                         time_support=nap.IntervalSet(0, duration))

    # Imaging dF/F (a calcium-like kernel on each ROI's drive) at 30 Hz.
    t_img = np.arange(0, duration, 1 / 30.0)
    idx = np.searchsorted(t, t_img)
    kernel = np.exp(-np.arange(0, 3, 1 / 30) / 0.7)
    drives = [field(*FIELDS_CM[0])[idx], field(*FIELDS_CM[1])[idx],
              field(50, 50, 15)[idx], response[idx]]
    dff = np.column_stack([np.convolve(rng.random(t_img.size) < 0.4 * d, kernel)[:t_img.size]
                           * 0.5 + rng.normal(0, 0.05, t_img.size) for d in drives])

    photometry = (2.0 * np.convolve(response, np.exp(-np.arange(0, 4, 1 / RATE) / 1.0))
                  [:t.size] / RATE * 10 + np.convolve(rng.normal(0, 0.3, t.size),
                                                       np.ones(20) / 20, mode='same'))
    return {
        'spikes': spikes,
        'behaviour': {'P_x': nap.Tsd(t=t, d=xy[:, 0]), 'P_y': nap.Tsd(t=t, d=xy[:, 1]),
                      'H': nap.Tsd(t=t, d=hd), 'S': nap.Tsd(t=t, d=speed), 'moving': moving},
        'theta_phase': nap.Tsd(t=t, d=theta),
        'events': nap.Ts(t=events),
        'dff': nap.TsdFrame(t=t_img, d=dff, columns=[0, 1, 2, 3]),
        'photometry': nap.Tsd(t=t, d=photometry),
        'arena': ARENA_CM,
    }
