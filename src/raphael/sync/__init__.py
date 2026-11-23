"""Synchronising behaviour with a recording from shared TTL pulses, with
sanity checks that stop a bad alignment from reaching the processed data.

Preliminary: one TTL channel in a behaviour CSV (sampled with each row)
and one TTL line in a Neuropixels recording (Open Ephys events).

    from raphael import sync
    beh = sync.read_behaviour('beh.csv')                 # DataFrame
    rec = sync.read_openephys('raw/.../Record Node 101')
    result = sync.align(sync.edges(beh['time'], beh['ttl']), rec.pulses,
                        behaviour_span=(beh['time'].iloc[0], beh['time'].iloc[-1]),
                        recording_span=(0, rec.duration))
    result.passed, result.checks                         # pass/fail, and why
    beh['time_recording'] = result.to_recording(beh['time'])

How it works:

1. **Pulses** are the rising edges of each TTL trace, with their widths.
2. **Matching** pairs pulses across the two clocks by their intervals:
   the lag at which the most inter-pulse intervals agree. Random intervals
   (e.g. 0.5-1.5 s) make this unambiguous; regular pulses work too if both
   devices see the same first pulse.
3. **Fit**: recording time = slope * behaviour time + intercept, by least
   squares on the matched pulses, refined twice by re-pairing every pulse
   to its nearest partner under the fit.
4. **Checks** (see CHECKS) decide whether the alignment can be used.
"""

from dataclasses import dataclass, field

import numpy as np

from raphael.sync.io import Recording, read_behaviour, read_openephys

__all__ = [
    'CHECKS',
    'DEFAULTS',
    'Pulses',
    'Recording',
    'SyncError',
    'SyncResult',
    'align',
    'edges',
    'read_behaviour',
    'read_openephys',
]

#: Settings for align(); override per rig.
DEFAULTS = {
    'min_pulses': 10,              # fewer matched pulses than this cannot be trusted
    'max_drift_ppm': 50.0,         # clock drift allowed between the devices
    'residual_ms': None,           # max residual; None: 1 ms + one behaviour sample interval
    'expected_interval_s': None,   # regular pulses: their interval (checked to +/- tolerance)
    'interval_tolerance': 0.05,
    'expected_width_s': None,      # pulse width, checked to +/- width_tolerance
    'width_tolerance': 0.5,
}

#: What each check means (also written into the report).
CHECKS = {
    'pulses': 'each device has at least min_pulses pulses',
    'matched': 'at least min_pulses pulses pair up across the devices',
    'unambiguous': 'only one way to pair the pulses: pairing every pulse with its partner '
                   'shifted by 1-3 pulses fits far worse (regular pulses with one device '
                   'starting late can be off by whole pulses)',
    'counts_agree': 'no unpaired pulse while both devices were recording '
                    '(a dropped, extra or bounced pulse)',
    'intervals': 'inter-pulse intervals within tolerance of expected_interval_s',
    'widths': 'pulse widths as expected (or, if not set, none under half or over '
              'twice the median: noise read as pulses)',
    'drift': 'clock drift under max_drift_ppm (beyond 2 standard errors)',
    'residuals': 'every matched pulse within residual_ms of the fit',
}


class SyncError(RuntimeError):
    pass


@dataclass
class Pulses:
    """Rising-edge times (s, on one device's clock) and pulse widths (s)."""
    onsets: np.ndarray
    widths: np.ndarray
    sample_interval: float = 0.0       # the trace's sampling interval (s); 0 for event times

    def __len__(self):
        return len(self.onsets)


def edges(times, values, threshold=None):
    """Pulses of a sampled TTL trace, crossing `threshold` (default: halfway
    between the trace's min and max). Each edge is dated halfway between
    the last sample before it and the first after: the true edge lies
    somewhere between them, so dating it at the first high sample would put
    every pulse half a sample late on average."""
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    if threshold is None:
        threshold = (np.nanmin(values) + np.nanmax(values)) / 2
    high = (values > threshold).astype(np.int8)
    step = np.diff(high)
    mid = (times[:-1] + times[1:]) / 2
    rises, falls = mid[step == 1], mid[step == -1]
    nxt = np.searchsorted(falls, rises, side='right')
    widths = np.full(len(rises), np.nan)
    ok = nxt < len(falls)
    widths[ok] = falls[nxt[ok]] - rises[ok]
    interval = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    return Pulses(rises, widths, interval)


@dataclass
class SyncResult:
    slope: float
    intercept: float
    behaviour: Pulses
    recording: Pulses
    pairs: np.ndarray                   # (n, 2) indices into behaviour, recording pulses
    residuals: np.ndarray               # s, recording - fit, per pair
    drift_ppm: float
    drift_se_ppm: float
    checks: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    waived: str | None = None
    unpaired: dict = field(default_factory=dict)   # indices unpaired while both recorded

    @property
    def passed(self):
        return all(c['passed'] is not False for c in self.checks.values())

    @property
    def usable(self):
        """Passed, or failed with a written waiver."""
        return self.passed or bool(self.waived)

    def to_recording(self, t):
        """Behaviour-clock times -> recording-clock times (s)."""
        return self.slope * np.asarray(t, float) + self.intercept

    def to_behaviour(self, t):
        return (np.asarray(t, float) - self.intercept) / self.slope

    def summary(self):
        return {
            'passed': self.passed,
            'waived': self.waived,
            'recording_time': f'{self.slope!r} * behaviour_time + {self.intercept!r}',
            'slope': self.slope,
            'intercept_s': self.intercept,
            'drift_ppm': self.drift_ppm,
            'drift_se_ppm': self.drift_se_ppm,
            'pulses': {'behaviour': len(self.behaviour), 'recording': len(self.recording),
                       'matched': len(self.pairs)},
            'residual_ms': {'max': float(np.max(np.abs(self.residuals)) * 1e3),
                            'rms': float(np.sqrt(np.mean(self.residuals ** 2)) * 1e3)},
            'checks': self.checks,
            'settings': self.settings,
        }


def _match_by_intervals(a, b, tol, min_pulses):
    """Indices (ia, ib) of pulses whose surrounding intervals agree, at the
    lag where most intervals agree."""
    da, db = np.diff(a), np.diff(b)
    counts = {}
    for k in range(-(len(db) - 1), len(da)):
        i0, i1 = max(0, -k), min(len(db), len(da) - k)
        if i1 <= i0:
            continue
        counts[k] = int(np.count_nonzero(np.abs(da[i0 + k:i1 + k] - db[i0:i1]) < tol))
    best_k = max(counts, key=lambda k: (counts[k], -abs(k)))
    best_n = counts[best_k]
    runner_up = max((n for k, n in counts.items() if k != best_k), default=0)
    if best_n < max(min_pulses - 1, 1):
        raise SyncError(f'could not match the pulse trains: at best {best_n} intervals agree '
                        f'(need {min_pulses - 1}). Are these two files from the same '
                        'session, and is the TTL channel right?')
    # Start from the longest unbroken run of agreeing intervals: intervals
    # that agree by chance elsewhere would pair pulses across a fault.
    i = np.arange(max(0, -best_k), min(len(db), len(da) - best_k))
    agree = np.concatenate([[0], (np.abs(da[i + best_k] - db[i]) < tol).astype(int), [0]])
    starts, ends = np.flatnonzero(np.diff(agree) == 1), np.flatnonzero(np.diff(agree) == -1)
    longest = np.argmax(ends - starts)
    if ends[longest] - starts[longest] < max(min_pulses - 1, 1):
        raise SyncError(f'could not match the pulse trains: at most '
                        f'{ends[longest] - starts[longest]} consecutive intervals agree '
                        f'(need {min_pulses - 1}). Are these two files from the same '
                        'session, and is the TTL channel right?')
    run = i[starts[longest]:ends[longest]]
    ib = np.append(run, run[-1] + 1)
    return ib + best_k, ib, {'lag': best_k, 'agreeing': best_n, 'next_best': runner_up}


def _pair_nearest(a, b, slope, intercept, window):
    """One-to-one pairs of each a (mapped) with its nearest b within window."""
    pred = slope * a + intercept
    j = np.clip(np.searchsorted(b, pred), 1, len(b) - 1)
    j = np.where(np.abs(b[j - 1] - pred) < np.abs(b[j] - pred), j - 1, j)
    dist = np.abs(b[j] - pred)
    ok = dist < window
    ia, ib, dist = np.flatnonzero(ok), j[ok], dist[ok]
    best = {}
    for x, y, d in zip(ia, ib, dist, strict=True):     # one a per b: keep the closest
        if y not in best or d < best[y][1]:
            best[y] = (x, d)
    ib = np.array(sorted(best), dtype=int)
    ia = np.array([best[y][0] for y in ib], dtype=int)
    return ia, ib


def _fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    res = y - (slope * x + intercept)
    n = len(x)
    se = (np.sqrt(np.sum(res ** 2) / (n - 2) / np.sum((x - x.mean()) ** 2))
          if n > 2 else np.inf)
    return slope, intercept, res, se


def _check(passed, value, limit, detail):
    return {'passed': None if passed is None else bool(passed), 'value': value, 'limit': limit, 'detail': detail}


def align(behaviour, recording, behaviour_span=None, recording_span=None, **settings):
    """Match `behaviour` and `recording` Pulses, fit the clock mapping, and
    run the checks. Spans (start, end) on each device's own clock say when
    each was recording, so pulses one device couldn't have seen are not
    counted as missing. Raises SyncError only if no alignment is possible;
    otherwise check `result.passed`."""
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unknown sync settings {sorted(unknown)}; known: {sorted(DEFAULTS)}')
    s = {**DEFAULTS, **settings}
    a, b = np.asarray(behaviour.onsets, float), np.asarray(recording.onsets, float)
    if min(len(a), len(b)) < s['min_pulses']:
        raise SyncError(f'too few pulses to align: {len(a)} in behaviour, {len(b)} in the '
                        f'recording (need {s["min_pulses"]}). Is the TTL channel right?')
    sample = max(behaviour.sample_interval, recording.sample_interval)
    tol = (s['residual_ms'] / 1e3 if s['residual_ms'] is not None else 1e-3 + sample)
    s['residual_ms'] = tol * 1e3

    ia, ib, lags = _match_by_intervals(a, b, 2 * tol, s['min_pulses'])
    slope, intercept, _, _ = _fit(a[ia], b[ib])
    window = 0.25 * float(np.median(np.diff(b)))
    for _ in range(2):
        ia, ib = _pair_nearest(a, b, slope, intercept, window)
        slope, intercept, res, se = _fit(a[ia], b[ib])
        window = max(5 * tol, 3 * float(np.max(np.abs(res))))
        window = min(window, 0.25 * float(np.median(np.diff(b))))

    drift, drift_se = (slope - 1) * 1e6, se * 1e6
    checks = {}
    checks['pulses'] = _check(True, {'behaviour': len(a), 'recording': len(b)},
                              s['min_pulses'], 'enough pulses on both devices')
    checks['matched'] = _check(len(ia) >= s['min_pulses'], len(ia), s['min_pulses'],
                               f'{len(ia)} pulses paired')
    # Could the pulses be paired off by whole pulses just as well? Shift every
    # pairing by m pulses, refit, and count pairs still within the limit.
    shifted = {}
    for m in (-3, -2, -1, 1, 2, 3):
        ok = (ib + m >= 0) & (ib + m < len(b))
        if ok.sum() < 3:
            continue
        x, y = a[ia[ok]], b[ib[ok] + m]
        _, _, r, _ = _fit(x, y)
        shifted[m] = round(float(np.mean(np.abs(r) <= tol)), 3)
    worst = max(shifted.values(), default=0.0)
    whole = len(a) == len(b) and lags['lag'] == 0 and len(ia) == len(a)
    checks['unambiguous'] = _check(
        whole or worst < 0.5, {'fraction_fitting_if_shifted': shifted},
        'under half the pairs fit when shifted',
        'both devices saw every pulse' if whole else
        f'shifted by whole pulses, at most {worst:.0%} of pairs fit'
        + ('' if worst < 0.5 else '; the pulses are too regular to tell which is which: '
           'send them at random intervals, or start the recording before the first pulse'))

    # Pulses one device should have seen while the other was recording.
    b_span = recording_span or (b[0], b[-1])
    a_span = behaviour_span or (a[0], a[-1])
    lo = max(b_span[0], slope * a_span[0] + intercept) + window
    hi = min(b_span[1], slope * a_span[1] + intercept) - window
    a_rec = slope * a + intercept
    lone_ia = [k for k in range(len(a)) if k not in set(ia) and lo < a_rec[k] < hi]
    lone_ib = [k for k in range(len(b)) if k not in set(ib) and lo < b[k] < hi]
    lone_a, lone_b = [float(a[k]) for k in lone_ia], [float(b[k]) for k in lone_ib]
    outside = (len(a) - len(ia) - len(lone_a)) + (len(b) - len(ib) - len(lone_b))
    detail = (f'{len(lone_a)} behaviour and {len(lone_b)} recording pulses unpaired while both '
              f'were recording; {outside} outside the overlap (explained: one device was '
              'not recording)')
    checks['counts_agree'] = _check(not lone_a and not lone_b,
                                    {'behaviour_unpaired_s': lone_a[:20],
                                     'recording_unpaired_s': lone_b[:20]}, 0, detail)

    if s['expected_interval_s']:
        e, t = s['expected_interval_s'], s['interval_tolerance']
        bad = {name: int(np.count_nonzero(np.abs(np.diff(x) - e) > t * e))
               for name, x in (('behaviour', a), ('recording', b))}
        checks['intervals'] = _check(not any(bad.values()), bad, f'{e} s +/- {t:.0%}',
                                     'intervals outside tolerance per device')
    else:
        checks['intervals'] = _check(None, None, None, 'not checked: expected_interval_s not set')

    widths = {}
    for name, p in (('behaviour', behaviour), ('recording', recording)):
        w = np.asarray(p.widths, float)
        w = w[np.isfinite(w)]
        if not len(w):
            continue
        if s['expected_width_s']:
            e, t = s['expected_width_s'], s['width_tolerance']
            slack = p.sample_interval
            widths[name] = int(np.count_nonzero(np.abs(w - e) > t * e + slack))
        else:
            med = np.median(w)
            widths[name] = int(np.count_nonzero((w < 0.5 * med - p.sample_interval)
                                                | (w > 2 * med + p.sample_interval)))
    checks['widths'] = _check(not any(widths.values()), widths,
                              s['expected_width_s'] or 'within 0.5-2x the median',
                              'pulses with unexpected widths per device')
    checks['drift'] = _check(abs(drift) - 2 * drift_se <= s['max_drift_ppm'],
                             round(drift, 2), s['max_drift_ppm'],
                             f'{drift:.1f} +/- {drift_se:.1f} ppm')
    worst = float(np.max(np.abs(res)))
    checks['residuals'] = _check(worst <= tol, round(worst * 1e3, 3), round(tol * 1e3, 3),
                                 f'max {worst * 1e3:.2f} ms (limit {tol * 1e3:.2f} ms)')
    for name, c in checks.items():
        c['means'] = CHECKS[name]
    return SyncResult(slope, intercept, behaviour, recording, np.column_stack([ia, ib]), res,
                      drift, drift_se, checks, s,
                      unpaired={'behaviour': np.array(lone_ia, int),
                                'recording': np.array(lone_ib, int)})
