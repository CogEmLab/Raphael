"""Simulated Doric recordings with a known signal, for tests and demos.

`simulate` writes a .doric file in the layout `doric.read` expects: an
isosbestic and a green channel on one detector, sharing bleaching and
motion artefacts, with sensor transients only in green; and a red channel on
a second detector. `truth` holds the green transients as dF/F (%).
"""

from pathlib import Path

import h5py
import numpy as np

RATE = 60.24


def simulate(path, duration_s=300.0, seed=0, start='Thu Aug 28 14:08:22 2025'):
    """Write `path` (.doric); return (times, true green dF/F in %)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.1, duration_s, 1 / RATE)
    n = len(t)
    # Transients: random events convolved with a sensor-like kernel (rise
    # ~0.2 s, decay ~2 s), ~3% peak dF/F.
    events = (rng.random(n) < 0.05 / RATE) * rng.uniform(2, 4, n)
    k_t = np.arange(0, 10, 1 / RATE)
    kernel = np.exp(-k_t / 2.0) - np.exp(-k_t / 0.2)
    true_dff = np.convolve(events, kernel / kernel.max())[:n]
    # Shared artefacts: slow motion-like wander and occasional sharp drops.
    motion = np.convolve(rng.normal(0, 1, n), np.ones(60) / 60, mode='same') * 0.15
    motion[rng.random(n) < 0.002] -= 0.05
    motion = np.convolve(motion, np.ones(10) / 10, mode='same')

    def bleach(f0, fast, slow):
        return f0 * (1 + fast * np.exp(-t / 3) + slow * np.exp(-t / 150))

    iso_f0, green_f0 = bleach(0.27, 0.08, 0.03), bleach(1.05, 0.08, 0.03)
    iso = iso_f0 * (1 + motion) + rng.normal(0, 0.0015, n)
    green = green_f0 * (1 + motion) * (1 + true_dff / 100) + rng.normal(0, 0.003, n)
    red = bleach(0.85, 0.05, 0.02) + rng.normal(0, 0.01, n)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.attrs.update({'Created': start, 'SoftwareName': 'Doric Neuroscience Studio (simulated)',
                        'SoftwareVersion': '6.6.3.0'})
        config = f.create_group('Configurations/FPConsole')
        config.attrs.update({'DeviceName': 'Simulated Console', 'SerialNumber': 'SIM'})
        config.create_group('GlobalSettings').attrs['SamplingRate'] = 12166
        config.create_group('SavingSettings').attrs.update(
            {'DecimationEnabled': 1, 'DecimationFactor': 200})
        mod = config.create_group('DIO01/Modulations/Modulation1')
        mod.attrs.update({'Frequency': 59.874, 'Period': 16.7017, 'StartingDelay': 0})
        acq = f.create_group('DataAcquisition')
        acq.attrs['MasterStartTime'] = start.split()[3] + '.000'
        signals = acq.create_group('FPConsole/Signals/Series0001')
        for i, (name, ain, carrier, level, values) in enumerate(
                [('iso', 'AIN01', 208.616, 30.0, iso), ('green', 'AIN01', 572.205, 60.0, green),
                 ('red', 'AIN02', 333.786, 40.0, red)], start=1):
            group = signals.create_group(f'LockInAOUT0{i}')
            group['Time'] = t
            group[ain] = values
            group[ain].attrs['Username'] = name
            lock = config.require_group(f'{ain}/Settings/LockInAOUT{i}')
            lock.attrs.update({'ReferenceFrequency': carrier, 'OutputLevel': level})
    return t, true_dff
