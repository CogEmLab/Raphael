"""Toy data: one simulated session with known tuning, in memory, as a raw
tree, or as a processed tree, so pipelines, analyses and teaching material
run without lab data.

    from raphael import toy
    from raphael.ephys.layout import Session

    s = toy.session()                          # in memory: spikes, behaviour, dF/F, ...
    s['tuning']                                # what every unit and ROI really encodes

    toy.processed('/tmp/toy/processed', Session('1', '1', 'OF1'))   # processed tree
    toy.raw('/tmp/toy/raw', Session('1', '1', 'OF1'))               # raw tree

What the session holds (all on the recording's clock, seconds):

    spikes        TsGroup: place, head-direction, speed, theta-locked,
                  event-responsive and untuned units (n_* arguments)
    behaviour     P_x, P_y (cm), H (rad), S (cm/s) as Tsd, and `moving`
    theta_phase   Tsd (rad), 8 Hz
    events        Ts: stimulus times
    dff           TsdFrame: imaging dF/F of place-tuned and event-responsive ROIs
    photometry    Tsd: dF/F (%) rising after each event
    tuning        DataFrame: one row per unit / ROI, its kind and parameters
    ttl           TTL pulse onsets on the rig's clock (random 0.5-1.5 s intervals)

`processed` writes the files Raphael's loaders read: `_srt-toy_clusters.npz`
(raphael.population.load), `_src-toy_traces.npz` + `_rois.csv`,
`_photometry.npz` + `.csv` (raphael.photometry.nwb.load), `_beh.csv` (as
raphael.sync writes it) and `_toy-truth.csv` (the tuning table).

`raw` writes the BIDS-style raw tree of docs/data_server.md: a behaviour CSV
on the rig's clock with a TTL column, a light Neuropixels 2.0 recording in
Open Ephys format whose TTL line carries the same pulses on a drifting
clock (so raphael-sync can align them), and a Doric photometry file, plus
the manifest. With `ephys='spikes'`, the recording is instead a full
ground-truth Neuropixels recording (spikeinterface format) for testing spike
sorting; its units are not the behavioural ones.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap

from raphael.plots.example import ARENA_CM, RATE, THETA_HZ, _poisson, _walk

KINDS = ('place', 'head_direction', 'speed', 'theta', 'event', 'untuned')


def session(duration=600.0, seed=0, n_place=20, n_hd=8, n_speed=3, n_theta=3, n_event=3,
            n_untuned=5, n_rois=12):
    """One simulated session (see the module docstring for what it holds)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / RATE)
    xy, speed, hd = _walk(rng, t.size)
    theta = (2 * np.pi * THETA_HZ * t + np.cumsum(rng.normal(0, 0.02, t.size))) % (2 * np.pi)
    events = np.sort(rng.uniform(5, duration - 5, max(int(duration / 12), 1)))
    since = np.full(t.size, np.inf)
    for e in events:
        after = (t >= e) & (t < e + 3)
        since[after] = np.minimum(since[after], t[after] - e)
    response = np.where(np.isfinite(since), np.exp(-((since - 0.15) / 0.08) ** 2), 0)

    def field(cx, cy, width):
        return np.exp(-((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2) / (2 * width ** 2))

    rates, rows = {}, []

    def add(kind, rate, **params):
        unit = len(rates)
        rates[unit] = rate
        rows.append({'id': unit, 'modality': 'spikes', 'kind': kind, **params})

    for _ in range(n_place):
        cx, cy = rng.uniform(10, 90, 2)
        width, peak = rng.uniform(8, 14), rng.uniform(6, 15)
        add('place', 0.2 + peak * field(cx, cy, width), x=cx, y=cy, width=width, peak_hz=peak)
    for k in range(n_hd):
        pref = (2 * np.pi * k / max(n_hd, 1) + rng.normal(0, 0.1)) % (2 * np.pi)
        peak = rng.uniform(8, 15)
        add('head_direction', 0.5 + peak * np.exp(2.5 * (np.cos(hd - pref) - 1)),
            preferred=pref, peak_hz=peak)
    for _ in range(n_speed):
        slope = rng.uniform(0.3, 0.6)
        add('speed', 1.0 + slope * speed, slope_hz_per_cm_s=slope)
    for _ in range(n_theta):
        pref = rng.uniform(0, 2 * np.pi)
        add('theta', 6 * (1 + 0.9 * np.cos(theta - pref)), preferred=pref)
    for _ in range(n_event):
        add('event', 1.0 + rng.uniform(25, 45) * response, latency_s=0.15)
    for _ in range(n_untuned):
        add('untuned', np.full(t.size, rng.uniform(1, 6)))
    spikes = nap.TsGroup({u: nap.Ts(t=_poisson(rng, t, r)) for u, r in rates.items()},
                         time_support=nap.IntervalSet(0, duration))

    # Imaging at 30 Hz: place-tuned ROIs, and one in four event-responsive.
    t_img = np.arange(0, duration, 1 / 30.0)
    idx = np.minimum(np.searchsorted(t, t_img), t.size - 1)
    kernel = np.exp(-np.arange(0, 3, 1 / 30) / 0.7)
    traces = []
    for k in range(n_rois):
        roi = len(rates) + k
        if k % 4 == 3:
            drive = response[idx]
            rows.append({'id': roi, 'modality': 'dff', 'kind': 'event', 'latency_s': 0.15})
        else:
            cx, cy = rng.uniform(15, 85, 2)
            drive = field(cx, cy, 12)[idx]
            rows.append({'id': roi, 'modality': 'dff', 'kind': 'place', 'x': cx, 'y': cy,
                         'width': 12.0})
        traces.append(np.convolve(rng.random(t_img.size) < 0.4 * drive, kernel)[:t_img.size]
                      * 0.5 + rng.normal(0, 0.05, t_img.size))
    dff = nap.TsdFrame(t=t_img, d=np.column_stack(traces) if traces else np.zeros((t_img.size, 0)),
                       columns=[len(rates) + k for k in range(n_rois)])

    photometry = (20 * np.convolve(response, np.exp(-np.arange(0, 4, 1 / RATE)))[:t.size] / RATE
                  + np.convolve(rng.normal(0, 0.3, t.size), np.ones(20) / 20, mode='same'))
    ttl = [0.5]
    while ttl[-1] < duration - 1.5:
        ttl.append(ttl[-1] + rng.uniform(0.5, 1.5))

    return {
        'spikes': spikes,
        'behaviour': {'P_x': nap.Tsd(t=t, d=xy[:, 0]), 'P_y': nap.Tsd(t=t, d=xy[:, 1]),
                      'H': nap.Tsd(t=t, d=hd), 'S': nap.Tsd(t=t, d=speed),
                      'moving': nap.Tsd(t=t, d=speed).threshold(2.0).time_support},
        'theta_phase': nap.Tsd(t=t, d=theta),
        'events': nap.Ts(t=events),
        'dff': dff,
        'photometry': nap.Tsd(t=t, d=photometry),
        'tuning': pd.DataFrame(rows),
        'ttl': np.array(ttl),
        'arena': ARENA_CM,
        'duration': duration,
        'seed': seed,
    }


def _session_id(sess):
    if sess is None:
        from raphael.ephys.layout import Session
        return Session('1', '1', 'OF1')
    return sess


def processed(root, sess=None, data=None, **kwargs):
    """Write `data` (default: session(**kwargs)) as session `sess` (a
    raphael Session; default M1/D1/OF1) of the processed tree under `root`.
    Returns {name: path}."""
    from raphael.ephys import nwb as ephys_nwb

    sess = _session_id(sess)
    data = data or session(**kwargs)
    folder = sess.folder(root)
    folder.mkdir(parents=True, exist_ok=True)
    tuning = data['tuning']
    paths = {}

    units = tuning[tuning['modality'] == 'spikes'].rename(columns={'id': 'unit'})
    units = units.drop(columns='modality').reset_index(drop=True)
    units['accepted'] = True
    spikes = {int(u): data['spikes'][int(u)].t for u in units['unit']}
    paths['clusters'] = sess.clusters(root, 'toy')
    ephys_nwb.export_clusters(units, spikes, data['duration'], paths['clusters'])

    rois = tuning[tuning['modality'] == 'dff'].rename(columns={'id': 'roi'})
    if len(rois):
        rois = rois.drop(columns='modality').reset_index(drop=True)
        rois['accepted'] = True
        rois['plane'] = 0
        paths['traces'], paths['rois'] = sess.traces(root, 'toy'), sess.rois(root, 'toy')
        data['dff'].save(str(paths['traces']))
        rois.to_csv(paths['rois'], index=False)

    stem = folder / sess.stem
    fp = data['photometry']
    z = (fp.values - fp.values.mean()) / fp.values.std()
    nap.TsdFrame(t=fp.t, d=np.c_[fp.values, z], columns=['green_dff', 'green_zscore']).save(
        f'{stem}_photometry.npz')
    pd.DataFrame([{'channel': 'green', 'role': 'signal', 'reference': None,
                   'indicator': 'toy', 'description': 'simulated: rises after each event'}]
                 ).to_csv(f'{stem}_photometry.csv', index=False)
    paths['photometry'] = Path(f'{stem}_photometry.npz')

    beh = data['behaviour']
    pd.DataFrame({'time': beh['P_x'].t, **{k: beh[k].values for k in ('P_x', 'P_y', 'H', 'S')}}
                 ).to_csv(f'{stem}_beh.csv', index=False)
    paths['behaviour'] = Path(f'{stem}_beh.csv')
    pd.DataFrame({'time': data['events'].t}).to_csv(f'{stem}_events.csv', index=False)
    paths['events'] = Path(f'{stem}_events.csv')
    tuning.to_csv(f'{stem}_toy-truth.csv', index=False)
    paths['truth'] = Path(f'{stem}_toy-truth.csv')
    return paths


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def raw(root, sess=None, data=None, ephys='light', recording_start_s=3.2,
        recording_duration_s=120.0, drift_ppm=20.0, behaviour_rate_hz=60.0, n_channels=4,
        photometry=True, **kwargs):
    """Write `data` (default: session(**kwargs)) as session `sess` of the
    raw tree under `root`, named as docs/data_server.md describes. Returns
    {name: path}, and 'clock' (recording time = slope * rig time + intercept).

    The light recording covers `recording_duration_s` of the session (from
    `recording_start_s`; None for all of it): Open Ephys stores a sample
    number and a timestamp for every 30 kHz sample, about 0.7 MB per second
    whatever the channel count, and two minutes is plenty to test sync."""
    from raphael.sync.simulate import write_openephys

    sess = _session_id(sess)
    data = data or session(**kwargs)
    rng = np.random.default_rng(data['seed'] + 1)
    folder = sess.folder(root)
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / sess.stem
    duration = data['duration']
    onsets, width = data['ttl'], 0.05
    paths = {}

    # Behaviour CSV on the rig's clock (= the session's behaviour time),
    # one row per frame (~60 Hz, jittered), with the TTL sampled per row.
    n = int(duration * behaviour_rate_hz * 1.1)
    times = np.cumsum(np.clip(1 / behaviour_rate_hz + rng.normal(0, 0.002, n),
                              0.5 / behaviour_rate_hz, None))
    times = times[times < duration]
    beh = data['behaviour']
    k = np.searchsorted(onsets, times, side='right') - 1
    ttl = ((k >= 0) & (times - onsets[np.clip(k, 0, None)] < width)).astype(int)
    frame = {'time': times.round(6)}
    for name in ('P_x', 'P_y', 'S'):
        frame[name] = np.interp(times, beh[name].t, beh[name].values).round(3)
    frame['H'] = (np.arctan2(np.interp(times, beh['H'].t, np.sin(beh['H'].values)),
                             np.interp(times, beh['H'].t, np.cos(beh['H'].values)))
                  % (2 * np.pi)).round(4)
    frame['ttl'] = ttl
    paths['behaviour'] = Path(f'{stem}_beh-rig.csv')
    pd.DataFrame(frame).to_csv(paths['behaviour'], index=False)

    # Neuropixels.
    ephys_dir = Path(f'{stem}_ephys')
    if ephys == 'light':
        rec_duration = duration - recording_start_s - 2.0
        if recording_duration_s:
            rec_duration = min(rec_duration, recording_duration_s)
        write_openephys(ephys_dir, onsets, width, recording_start_s, rec_duration, drift_ppm,
                        rng, n_channels=n_channels)
        paths['ephys'] = ephys_dir / 'Record Node 101'
    elif ephys == 'spikes':
        import shutil
        import tempfile

        from raphael.ephys import benchmark
        with tempfile.TemporaryDirectory() as tmp:
            benchmark.simulate(tmp, durations=(min(duration, 60.0),), num_units=20,
                               per_shank=16, seed=data['seed'])
            shutil.move(str(Path(tmp) / 'session0'), ephys_dir)
            shutil.move(str(Path(tmp) / 'truth' / 'session0'), f'{stem}_ephys-truth')
        paths['ephys'] = ephys_dir
    elif ephys is not None:
        raise ValueError("ephys must be 'light', 'spikes' or None")

    if photometry:
        from raphael.photometry import benchmark as fp_benchmark
        paths['photometry'] = Path(f'{stem}_photometry.doric')
        fp_benchmark.simulate(paths['photometry'], duration_s=min(duration, 300.0),
                              seed=data['seed'])

    rows = []
    for f in sorted(p for p in folder.rglob('*') if p.is_file()):
        rel = f.relative_to(folder)
        rows.append({'path': rel.as_posix(), 'original_name': f.name, 'bytes': f.stat().st_size,
                     'sha256': _sha256(f)})
    paths['manifest'] = Path(f'{stem}_manifest.tsv')
    pd.DataFrame(rows).to_csv(paths['manifest'], sep='\t', index=False)
    scale = 1 + drift_ppm * 1e-6
    paths['clock'] = {'slope': scale, 'intercept': -recording_start_s * scale}
    (Path(f'{stem}_toy-truth.json')).write_text(json.dumps(
        {'clock': paths['clock'], 'ttl_onsets_rig_s': onsets.tolist(), 'seed': data['seed'],
         'ephys': ephys}, indent=2))
    return paths
