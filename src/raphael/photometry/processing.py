"""Raw photometry channels -> dF/F and z-score, per signal channel.

Follows Simpson et al. 2024, Neuron 112:718 ("Lights, fiber, action! A
primer on in vivo fiber photometry") and the Akam lab's reference
implementation (github.com/ThomasAkam/photometry_preprocessing):

1. Trim the start: when the LEDs come on, every channel falls steeply for a
   few seconds (fast bleaching, LED warm-up); no exponential with one
   slow time constant fits it.
2. Low-pass filter (zero-phase Butterworth): sensor kinetics are slow, and
   this removes noise above them.
3. Photobleaching: fit a double exponential to each channel and subtract
   it. The fit is also each channel's baseline fluorescence, F0.
4. Motion and other shared artefacts: regress the detrended reference
   (isosbestic) channel onto the detrended signal and subtract the fit.
   A signal with no reference skips this step.
5. dF/F = corrected signal / F0 (in %), and z-score = corrected signal /
   its standard deviation.

Every fitted parameter is returned, so a session's processing can be
checked and reported.
"""

import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt
from scipy.stats import linregress

DEFAULTS = {
    'trim_start_s': 5.0,
    'lowpass_hz': 10.0,
    'filter_order': 2,
    'bleaching': 'double_exponential',     # or 'linear', or 'none'
}


def trim(times, start_s):
    """Index of the samples to keep: after `start_s` seconds from the first."""
    return times >= times[0] + start_s


def lowpass(values, rate, cutoff_hz, order=2):
    if not cutoff_hz:
        return np.asarray(values, float)
    if cutoff_hz >= rate / 2:
        raise ValueError(f'lowpass_hz ({cutoff_hz}) must be below half the sampling rate '
                         f'({rate / 2:.1f} Hz)')
    sos = butter(order, cutoff_hz, btype='low', fs=rate, output='sos')
    return sosfiltfilt(sos, values)


def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    """const + amp_slow e^(-t/tau_slow) + amp_fast e^(-t/tau_fast), with
    tau_fast = tau_slow * tau_multiplier (< 1) so the two cannot swap."""
    tau_fast = tau_slow * tau_multiplier
    return const + amp_slow * np.exp(-t / tau_slow) + amp_fast * np.exp(-t / tau_fast)


def fit_bleaching(t, values, method='double_exponential'):
    """(F0 curve, {method, params}). Falls back to a linear fit, with a
    warning, if the double exponential does not converge."""
    t = np.asarray(t, float) - t[0]
    values = np.asarray(values, float)
    if method == 'none':
        return np.full_like(values, np.mean(values)), {'method': 'none'}
    if method == 'double_exponential':
        top = np.max(values)
        p0 = [np.min(values), 0.1 * top, 0.1 * top, max(t[-1] / 2, 1.0), 0.1]
        bounds = ([0, 0, 0, 1.0, 0.0], [top, top, top, 36000.0, 1.0])
        try:
            params, _ = curve_fit(double_exponential, t, values, p0=p0, bounds=bounds,
                                  maxfev=2000)
            names = ['const', 'amp_fast', 'amp_slow', 'tau_slow_s', 'tau_multiplier']
            return double_exponential(t, *params), {
                'method': method, **{n: float(v) for n, v in zip(names, params, strict=True)}}
        except (RuntimeError, ValueError) as e:
            warnings.warn(f'double-exponential bleaching fit failed ({e}); using a linear fit',
                          stacklevel=2)
            method = 'linear'
    if method == 'linear':
        slope, intercept = np.polyfit(t, values, 1)
        return intercept + slope * t, {'method': 'linear', 'slope': float(slope),
                                       'intercept': float(intercept)}
    raise ValueError(f'unknown bleaching method {method!r}')


def motion_correct(signal, reference):
    """Subtract the detrended reference's linear fit to the detrended
    signal. Returns (corrected, {slope, intercept, r})."""
    fit = linregress(reference, signal)
    estimate = fit.intercept + fit.slope * reference
    return signal - estimate, {'slope': float(fit.slope), 'intercept': float(fit.intercept),
                               'r': float(fit.rvalue)}


def process(times, channels, roles, settings=None):
    """`channels`: {name: raw values}, all on `times`. `roles`: {name:
    {'role': 'signal'|'reference', 'reference': name or None}}.

    Returns (times kept, {signal name: {'dff', 'zscore'}}, {name: fit
    parameters}). Reference channels are processed (filtered, detrended)
    but have no output of their own."""
    s = {**DEFAULTS, **(settings or {})}
    times = np.asarray(times, float)
    rate = 1 / np.median(np.diff(times))
    keep = trim(times, s['trim_start_s'])
    t = times[keep]

    detrended, baseline, fits = {}, {}, {}
    for name, values in channels.items():
        if name not in roles:
            continue
        filtered = lowpass(np.asarray(values, float)[keep], rate, s['lowpass_hz'],
                           s['filter_order'])
        baseline[name], fits[name] = fit_bleaching(t, filtered, s['bleaching'])
        fits[name]['role'] = roles[name]['role']
        detrended[name] = filtered - baseline[name]

    out = {}
    for name, spec in roles.items():
        if spec['role'] != 'signal' or name not in detrended:
            continue
        corrected = detrended[name]
        reference = spec.get('reference')
        if reference:
            if reference not in detrended:
                raise ValueError(f'{name}: reference channel {reference!r} not in the recording '
                                 f'(has {sorted(channels)})')
            corrected, motion = motion_correct(corrected, detrended[reference])
            fits[name].update(reference=reference, **{f'motion_{k}': v
                                                      for k, v in motion.items()})
        else:
            fits[name]['reference'] = None
        out[name] = {'dff': 100 * corrected / baseline[name],
                     'zscore': corrected / np.std(corrected)}
    return t, out, fits
