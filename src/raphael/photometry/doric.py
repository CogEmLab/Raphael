"""Read Doric Neuroscience Studio recordings (.doric, HDF5).

Written against FP Console files from Neuroscience Studio 6 with lock-in
demodulation, where each excitation LED is modulated at its own carrier
frequency and each photodetector input (AIN) is demodulated against each
carrier it is paired with:

    DataAcquisition/FPConsole/Signals/Series0001/
        LockInAOUT01/AIN01, Time      one demodulated channel per LED x detector
        AnalogIn/AIN01, Time          raw photodetector voltage (decimated)
        AnalogOut/AOUT01, Time        LED drive
        DigitalIO/DIO01, Time         digital lines (decimated)
    Configurations/FPConsole/...      carrier frequencies, LED levels,
                                      digital-output modulations, saving settings

A channel is named by the `Username` the experimenter gave it in the
console (e.g. iso, green, red); those names are what configs/photometry.yaml
refers to. Doric's per-channel CSV exports hold the same samples.

Digital lines in the file are decimated like everything else (here 12166 Hz
/ 200 = 60.24 Hz), so a fast TTL such as a ~60 Hz camera trigger aliases
and cannot be read back from the samples. When the console itself generates
the TTL (a digital output with a modulation), its pulse times are fixed by
the configuration and `ttl_times` reconstructs them on the same clock as
the signals.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np


@dataclass
class Channel:
    name: str                  # the console Username, e.g. 'green'
    source: str                # e.g. 'LockInAOUT02/AIN01'
    times: np.ndarray          # s, device clock (0 = acquisition start)
    values: np.ndarray         # demodulated photodetector signal (V)
    carrier_hz: float | None = None
    led: str | None = None     # analog output driving the LED, e.g. 'AOUT02'
    led_level: float | None = None   # console output level
    detector: str | None = None      # photodetector input, e.g. 'AIN01'

    @property
    def rate(self):
        return float(1 / np.median(np.diff(self.times)))


@dataclass
class Recording:
    path: Path
    channels: dict             # name -> Channel
    start_time: datetime | None     # naive local time of the first sample
    device: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=dict)
    digital_outputs: dict = field(default_factory=dict)   # line -> modulation settings
    software: str = ''

    def ttl_times(self, line='DIO01', stop=None):
        """Pulse onsets (s, device clock) of a TTL the console generated on
        `line`, reconstructed from its modulation settings. None when the
        line was not a console-generated output."""
        mod = self.digital_outputs.get(line)
        if not mod:
            return None
        period = float(mod['Period']) / 1000
        delay = float(mod.get('StartingDelay', 0)) / 1000
        if stop is None:
            stop = max(float(c.times[-1]) for c in self.channels.values())
        return delay + np.arange(int(np.floor((stop - delay) / period)) + 1) * period


def _attrs(obj):
    return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in obj.attrs.items()}


def _start_time(f, console):
    """Recording start from the file's creation date and the acquisition
    start clock time (HH:MM:SS.mmm); naive local time."""
    created = _attrs(f).get('Created')
    master = _attrs(f['DataAcquisition']).get('MasterStartTime')
    if not created:
        return None
    day = datetime.strptime(created, '%a %b %d %H:%M:%S %Y')
    if not master:
        return day
    clock = datetime.strptime(master, '%H:%M:%S.%f').time()
    start = datetime.combine(day.date(), clock)
    if start - day > timedelta(hours=12):      # started just before midnight
        start -= timedelta(days=1)
    first = _attrs(f[f'DataAcquisition/{console}']).get('DifferenceMasterStartToFirstData')
    return start + timedelta(milliseconds=float(first)) if first is not None else start


def read(path):
    """A Doric recording: every lock-in channel, by name, and the metadata
    needed to describe it."""
    path = Path(path)
    with h5py.File(path, 'r') as f:
        if 'DataAcquisition' not in f:
            raise ValueError(f'{path}: no DataAcquisition group; only Doric Neuroscience '
                             'Studio 6 files are supported')
        consoles = list(f['DataAcquisition'].keys())
        if not consoles:
            raise ValueError(f'{path}: no console data')
        console = consoles[0]
        series = sorted(f[f'DataAcquisition/{console}/Signals'].keys())
        if len(series) != 1:
            raise ValueError(f'{path}: {len(series)} series; read one recording per file')
        signals = f[f'DataAcquisition/{console}/Signals/{series[0]}']
        config = f.get(f'Configurations/{console}')

        def settings(name):
            if config is None:
                return {}
            if not name:
                return _attrs(config)
            return _attrs(config[name]) if name in config else {}

        channels = {}
        for group in sorted(k for k in signals if k.startswith('LockIn')):
            led = 'AOUT' + group.removeprefix('LockInAOUT')
            lock = {}
            for ain in sorted(k for k in signals[group] if k != 'Time'):
                number = int(group.removeprefix('LockInAOUT'))     # LockInAOUT02 -> 2
                lock = settings(f'{ain}/Settings/LockInAOUT{number}')
                attrs = _attrs(signals[group][ain])
                name = attrs.get('Username') or f'{ain}x{led}'
                if name in channels:
                    name = f'{name}_{ain}'
                channels[name] = Channel(
                    name=name, source=f'{group}/{ain}',
                    times=signals[group]['Time'][:], values=signals[group][ain][:],
                    carrier_hz=lock.get('ReferenceFrequency'), led=led,
                    led_level=lock.get('OutputLevel'), detector=ain)
        if not channels:
            raise ValueError(f'{path}: no lock-in channels (LockIn*/AIN*); only lock-in '
                             'recordings are supported so far')

        outputs = {}
        if config is not None:
            for line in (k for k in config if k.startswith('DIO')):
                mod = settings(f'{line}/Modulations/Modulation1')
                if mod:
                    outputs[line] = mod
        rec = Recording(
            path=path, channels=channels, start_time=_start_time(f, console),
            device=settings(''), sampling={**settings('GlobalSettings'),
                                           **settings('SavingSettings')},
            digital_outputs=outputs,
            software=f'{_attrs(f).get("SoftwareName", "")} {_attrs(f).get("SoftwareVersion", "")}'
                     .strip())
    return rec


def describe(rec):
    """A few lines about a recording, for `raphael-photometry inspect`."""
    lines = [f'{rec.path.name}  ({rec.software})',
             (f'  start {rec.start_time}  device {rec.device.get("DeviceName", "?")} '
              f'{rec.device.get("SerialNumber", "")}')]
    for c in rec.channels.values():
        lines.append(f'  {c.name:<8} {c.source:<20} {c.rate:6.2f} Hz  {len(c.values):>7} samples '
                     f'({c.times[-1] - c.times[0]:.1f} s)  carrier {c.carrier_hz} Hz  '
                     f'LED {c.led} level {c.led_level}  mean {np.mean(c.values):.3f} V')
    for line, mod in rec.digital_outputs.items():
        lines.append(f'  {line}: console-generated TTL at {mod.get("Frequency"):.3f} Hz '
                     f'(period {mod.get("Period"):.3f} ms), reconstructable with ttl_times')
    decimation = rec.sampling.get('DecimationFactor') if rec.sampling.get('DecimationEnabled') \
        else 1
    lines.append(f'  sampling {rec.sampling.get("SamplingRate")} Hz, decimated x{decimation}')
    return '\n'.join(lines)
