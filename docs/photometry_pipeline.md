# Fibre photometry pipeline

The default pipeline for these recordings; how to run it on your data is in
[processing.md](processing.md).

Doric recordings (`.doric`) to dF/F per fibre channel, in the same processed
tree as ephys and imaging.

```bash
uv run raphael-photometry inspect recording.doric           # channels, rates, carriers, TTLs
uv run raphael-photometry process recording.doric --out /data/processed \
    --mouse 9418 --day 7 --session-type HBT4T2 --config configs/photometry.yaml
```

Or as a job, so the inputs are checksummed and the environment is recorded:

```bash
uv run raphael-jobs make jobs/fp-M9418-D7 --stage photometry --cpu --data-root /data \
    --input raw/baseline.doric --input raw/session_T1.doric \
    --param mouse=9418 --param day=7 --param "session_type=[baseline, T1]" \
    --config configs/photometry.yaml
uv run raphael-jobs run jobs/fp-M9418-D7 --data-root /data
```

## Input

Doric Neuroscience Studio 6, FP Console, lock-in mode. Each LED is modulated at
its own carrier frequency, and each photodetector input is demodulated against
the carriers paired with it. Channels are named by the **Username** given in the
console (e.g. `iso`, `green`, `red`), and `configs/photometry.yaml` refers to
them by those names. Doric's CSV exports contain the same samples, so read
the `.doric` file, which also carries the metadata.

| Example recording | Detector | Carrier | Configured role |
|---|---|---|---|
| `iso` | AIN01 | 208.6 Hz | reference for `green` |
| `green` (e.g. GRAB-NE) | AIN01 | 572.2 Hz | signal |
| `red` | AIN02 | 333.8 Hz | signal, no reference |

Set each project's channels, indicators and locations in its own
`configs/photometry.yaml`.

## Processing

From Simpson et al. 2024, *Neuron* ("Lights, fiber, action! A primer on in vivo
fiber photometry"), and the Akam lab's reference implementation:

1. **Trim** the first `trim_start_s` (5 s): every channel falls steeply while
   the LEDs warm up.
2. **Low-pass** at `lowpass_hz` (10 Hz), zero-phase Butterworth.
3. **Photobleaching**: fit a double exponential to each channel and subtract
   it. The fit is the channel's baseline fluorescence, F0.
4. **Motion**: regress the detrended reference onto the detrended signal
   (ordinary least squares) and subtract the fit. A signal without a reference
   skips this step.
5. **dF/F** = corrected / F0 × 100, and **z-score** = corrected / SD.

The QC figure (`qc/*_photometry.png`) shows each channel with its bleaching
fit, and each signal's dF/F with its reference and fit r. Every fitted
parameter is in `_photometry.csv` and in the NWB file.

## Output

```
{root}/sub-M{mouse}/day-{dd}/ses-{type}/
    sub-..._photometry.nwb   raw channels (acquisition, V); {ch}_dff (%) and {ch}_zscore
                             (processing/photometry), fit parameters in comments
    sub-..._photometry.npz   TsdFrame: {ch}_dff, {ch}_zscore
    sub-..._photometry.csv   one row per channel: role, reference, indicator, LED,
                             carrier, fit parameters
    qc/
```

```python
from raphael.photometry.nwb import load
fp = load('/data/processed/sub-M9418/day-07/ses-HBT4T2/sub-M9418_day-07_ses-HBT4T2')
fp.dff('green')            # pynapple Tsd, %
fp.channels                # the table
```

## Synchronising with video or behaviour

Digital lines are saved decimated like the signals (12166 Hz / 200 =
60.24 Hz). A ~60 Hz camera trigger therefore **aliases** in the saved file:
it looks like a 0.36 Hz square wave, and the frame times cannot be read from
the samples. If the console generated the trigger (a DIO output with a
modulation, as in the example files), its pulse times are fixed by the
configuration, and `doric.read(path).ttl_times('DIO01')` reconstructs them on
the signal clock. Check that the count matches the number of video frames.
For a trigger from another device, record it with decimation off, or on
the other device.

`raphael-photometry simulate OUT.doric` writes a simulated recording with a
known signal, for trying things out.
