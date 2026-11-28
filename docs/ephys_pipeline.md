# In vivo Neuropixels pipeline

The default pipeline for these recordings; how to run it on your data is in
[processing.md](processing.md).

Raw Neuropixels 2.0 recordings go in. Out comes one processed, publishable
NWB file per session, plus the files mecpy and our analysis code load. The
layout is the processed tree ([data_server.md](data_server.md)), the same
one the [imaging pipeline](imaging_pipeline.md) fills, so analysis code
treats the two modalities alike. clark2025's files, named before November
2026 (`M25/D18/OF1/sub-M25_ses-D18_typ-OF1_…`), are read the same way.

The processing reproduces the paper's Methods ("Offline spike sorting")
and the lab's sorting code: protocol `kilosort4A` in
[MattNolanLab/nolanlab-ephys](https://github.com/MattNolanLab/nolanlab-ephys),
and Harry's scripts in
[chrishalcrow/nolanlab-ephys](https://github.com/chrishalcrow/nolanlab-ephys/tree/main/scripts/harry).

- Settings and publication metadata: [configs/ephys.yaml](../src/raphael/configs/ephys.yaml)
- Job stage: `kilosort4` (runs on the CPU by default; see [cluster.md](cluster.md))
- Code: `src/raphael/ephys/`
- Loading either modality: `raphael.population`

**Processing a recording step by step, with the data at each stage:
[ephys_walkthrough.md](ephys_walkthrough.md).**

## What it does

| # | step | as in the Methods / lab code |
|---|---|---|
| 1 | read each recording of the day (SpikeGLX, Open Ephys or a spikeinterface folder) and join them in recording order | `concatenate_recordings` |
| 2 | split by shank; on each shank **remove bad and noisy channels** (IBL method, seed 1205), **then correct the phase shift** | `detect_and_remove_bad_channels`, `phase_shift` |
| 3 | **Kilosort 4.0.16** on each shank, **default parameters, motion correction off** | `run_sorter('kilosort4', do_correction=False, use_binary_file=False)` |
| 4 | analyser recording: the channels the sorter saw, phase shift, common median reference, 300–6000 Hz band-pass, per shank | `preprocessing_for_analyzer` |
| 5 | cut the sorting and recording back into sessions; spike times start at 0 in each | the lab's `split_analyzers_into_sessions` |
| 6 | one sorting analyser per session (`peak_sign="both"`, `radius_um=70`), with the lab's extensions; unit locations by **monopolar triangulation**; waveforms deleted afterwards | `generic_postprocessing` |
| 7 | **bombcell** labels each unit good, mua, noise or non_soma; **good and mua are kept** | `bombcell_label_units` |
| 8 | **ISI rule**: a unit whose ISI violation ratio exceeds 10 in any session of the day is excluded | Methods |
| 9 | write `_clusters.npz` (spikes, curation, metrics, waveforms) and the QC figure | |
| — | separately, the `lfp` stage: LFP and theta phase into `_lfp.nwb` | mecpy `generate_nwb` |

### One environment

Everything, Kilosort included, runs in the repo's one locked environment
(`uv sync --frozen --extra gpu`). Kilosort 4.0.16 sets its versions:
- **numpy < 2**, because Kilosort 4.0.16 requires it;
- **spikeinterface 0.104.9**, the newest release that still runs Kilosort
  4.0.16 (0.105 refuses versions before 4.1.1);
- **and what follows from those:** pandas 2.3 and pyarrow < 26.

The whole test suite, imaging and pose included, passes in it.

### What is not decided by the Methods

- **The spikeinterface version used for sorting.** The Methods name
  Kilosort 4.0.16 only; the lab's repositories have since moved to 4.1.x.
  0.104.9 is the newest release that runs 4.0.16.
- **Sorting shank by shank.** The Methods say "the preprocessed recording
  was passed to Kilosort"; the lab code splits by shank first
  (`split_by('group')`), and this pipeline follows the code.
- **Which sessions the ISI rule looks at.** The Methods use "its VR or its
  open-field session"; here it is every session sorted together, or the
  session types in `curation.isi_sessions`.

## Output layout

The processed dataset is three files per session, so you can take as much
as you need: behaviour, spikes, and (optionally, the heavy one) the LFP.

```
{root}/sub-M{mouse}/day-{dd}/ses-{type}/
    sub-M{mouse}_day-{dd}_ses-{type}_beh.nwb              all behaviour (as clark2025; planned)
    sub-..._ses-{type}_srt-kilosort4_clusters.npz           spike times, curation, waveforms, metrics
    sub-..._ses-{type}_srt-kilosort4_bombcell_curation.json bombcell labels (spikeinterface curation file)
    sub-..._ses-{type}_lfp.nwb                              LFP and theta phase (lfp stage)
    qc/units.png
```

| file | written by | size, 30-min session |
|---|---|---|
| `_beh.nwb` | behaviour/sync step (planned) | tens of MB |
| `_clusters.npz` | `kilosort4` stage (CPU) | tens of MB |
| `_bombcell_curation.json` | `kilosort4` stage | tens of kB |
| `_lfp.nwb` | `lfp` stage (CPU), optional | ~0.2–0.7 GB at the defaults; set by `lfp` settings |

No raw voltage is in the processed dataset; it stays with the acquisition
files. `srt-kilosort4` names the sorter. Every unit is in every session's
`_clusters.npz`; select with `accepted`.

### What each file is for

- **`_beh.nwb`**: behaviour, as in clark2025. VR sessions hold position
  `P`, speed `S`, licks, pupil and DLC points, trial tables and
  moving/stationary epochs. It is not written yet: it needs the TTL sync
  between ephys and behaviour.
- **`_clusters.npz`**: everything about the units, **in the same form as
  clark2025's**. It is a pynapple TsGroup with one spike train per unit,
  keyed by `unit`, and spike times start at 0 at the recording's first
  sample. Its metadata is one row per unit with clark2025's column names:
  - curation: `accepted`, `bombcell_label`, `isi_excluded`, and Kilosort's
    `KSLabel`;
  - position: `x`, `y`, `z`, `group` (shank), `extremum_channel`,
    `coord_probe_x`/`y`;
  - every spikeinterface quality metric, and every template metric, i.e.
    the **waveform measures** (`peak_to_trough_duration`,
    `trough_half_width`, `peak_half_width`, `spread`, `exp_decay`, ...);
  - **the waveforms themselves**: `waveform_mean` and `waveform_sd`
    (samples × 16 channels, µV, extremum channel first),
    `waveform_channels`, `waveform_positions_um`,
    `waveform_sampling_frequency` and `waveform_nbefore` (the sample the
    spike is aligned to).

  It holds everything in one file, so labels, waveforms and spike times
  cannot drift apart. Code that reads clark2025's `_clusters.npz` reads
  these unchanged.
- **`_lfp.nwb`**: local field potential and theta phase, in the lab's
  layout (`processing/ecephys/LFP/lfp`,
  `processing/ecephys/Processed/theta`), on the same clock as the spikes.
  How big and detailed it is, is a setting: see
  [The LFP file](#the-lfp-file).

```python
import numpy as np
import pynapple as nap

units = nap.load_file('.../sub-M25_day-18_ses-VR_srt-kilosort4_clusters.npz')
good = units[units['accepted'].astype(bool)]
good['peak_to_trough_duration'], good['firing_rate']     # waveform measures
mean = np.stack(good['waveform_mean'])                  # units x samples x 16, µV
```

### What is not in the processed tree

- **Sorting analysers (`.zarr`)** are working files. Everything analysis
  needs from them is copied into `_clusters.npz`. By default they are
  deleted after the run. With
  `postprocessing.keep_analyzer: true` they are kept in the job's
  `outputs/kilosort4/analyzers/`, outside the processed tree, e.g. to
  recompute metrics or recurate.
- **`_ecephys.nwb`** (units, waveforms and electrodes as one NWB file,
  e.g. for DANDI) is off by default. Turn it on with
  `postprocessing.write_ecephys_nwb: true`.
- **Kilosort's own files** (for phy) stay in the job's
  `outputs/kilosort4/`.

Compared with clark2025: there the analysers sat in the session folders
(39 of 243 sessions in the downloaded copy, 12 GB of it for M20 alone),
and the waveforms existed only there. mecpy's clark2025 loader opens the
`_analyzer.zarr`; for these sessions, load `_clusters.npz` directly.

## The LFP file

Written by the `lfp` job stage, a CPU job, one `_lfp.nwb` per recording.
The defaults reproduce the lab's files (mecpy):

1. Band-pass 1–400 Hz, then resample to 1 kHz.
2. On each shank, order the channels by depth and average them in blocks
   of 8 neighbouring contacts (~60 µm). This turns 384 channels into 48
   LFP channels, named by their contacts (`CH49_CH50_..._CH56`).
3. Theta phase: band-pass 6–10 Hz, peaks at least 80 ms apart, and phase
   running 0 to 2π from one peak to the next (NaN outside).

**Size and detail are settings** (`lfp` in
[configs/ephys.yaml](../src/raphael/configs/ephys.yaml), or per job with
`--param 'lfp={...}'`):

| setting | default | lighter | richer |
|---|---|---|---|
| `rate_hz` | 1000 | 500 | 2500 (band up to 1 kHz) |
| `band_hz` | [1, 400] | [1, 200] | [1, 1000] |
| `channels.block_size` | 8 (48 channels) | 16 (24) | 1: every contact (384) |
| `channels.every` | 1 | 2: every other LFP channel | 1 |
| `channels.shanks` | all | e.g. [0, 3] | all |
| `dtype` | float32 | float16 | float32 |
| `compression_level` | 4 | 9 (slower to write) | 4 |
| `theta.enabled` | true | false | true |

Size before compression ≈ duration × `rate_hz` × channels × bytes (4 for
float32, 2 for float16), doubled with theta. For 30 minutes at the
defaults that is 0.35 GB per series. 1 kHz, every contact, float32 would
be 2.8 GB; 500 Hz, 24 channels, float16, no theta is 0.04 GB.

Differences from the lab's files (`clark2025_lfp`), which hold the same
arrays:
- **Units:** ours stores µV with NWB conversion 1e-6, so NWB readers get
  volts. The lab's files store the same µV numbers as float64 but label
  them volts with conversion 1.
- **Precision:** ours is float32, not float64.
- **Theta phase** is in radians, and ours says so.

## The same shape as imaging

| | ephys | imaging |
|---|---|---|
| one row per neuron | metadata of `_clusters.npz` (`unit`, `accepted`, ...) | `_rois.csv` (`roi`, `accepted`, ...) |
| activity | `_clusters.npz`: TsGroup of spike times | `_traces.npz`: TsdFrame of dF/F |
| behaviour | `_beh.nwb` | `_beh.nwb` |
| automatic curation | bombcell + ISI rule | suite2p classifier |
| QC | `qc/units.png` | `qc/plane1_rois.png` |

```python
from raphael import population

pop = population.load('.../sub-M25_day-18_ses-OF1_srt-kilosort4').accepted()
# or population.load('.../sub-M17797_day-07_ses-scan4_src-suite2p')
pop.table             # one row per neuron
pop.data              # TsGroup (spikes) or TsdFrame (dF/F)
pop.binned(0.1)       # TsdFrame, time x neuron: rate (Hz) or mean dF/F
```

pynapple's tuning-curve and decoding functions take either a TsGroup or a
TsdFrame, so most analysis code runs unchanged on both modalities.

## The unit metadata in `_clusters.npz`

One row per unit, the same rows in every session of the day:

| column | meaning |
|---|---|
| `unit` | id, shared across the day's sessions; key of the TsGroup |
| `accepted` | kept for analysis: bombcell good or mua, and not `isi_excluded` |
| `bombcell_label` | good, mua, noise or non_soma (in this session) |
| `isi_excluded` | ISI violation ratio > 10 in a session of the day |
| `x`, `y`, `z` | position from monopolar triangulation (µm; `y` along the shank from the tip) |
| `num_spikes`, `firing_rate`, `snr`, `isi_violations_ratio`, `presence_ratio`, `amplitude_median`, `drift_ptp`, ... | spikeinterface quality metrics |
| `peak_to_trough_duration`, `trough_half_width`, `peak_half_width`, `repolarization_slope`, `recovery_slope`, `spread`, `exp_decay`, `velocity_above`, `velocity_below`, ... | spikeinterface template metrics: the waveform measures, from each unit's mean template (multi-channel ones included) |
| `group`, `original_cluster_id` | the shank Kilosort sorted it on, and its cluster id there (what phy shows) |
| `KSLabel`, `KSLabel_repeat`, `Amplitude`, `ContamPct` | Kilosort's own label and properties |
| `extremum_channel`, `coord_probe_x`, `coord_probe_y` | the unit's extremum channel and that channel's position on the probe |
| `waveform_mean`, `waveform_sd` | arrays, samples × 16 channels, µV: the unit's mean and SD waveform, its extremum channel first |
| `waveform_channels`, `waveform_positions_um` | those 16 channels' ids and probe positions (µm) |
| `waveform_sampling_frequency`, `waveform_nbefore` | sampling rate of the waveforms, and the sample the spike is aligned to |

The names are clark2025's. Its `_clusters.npz` also has `coord_CCFs_*`,
`coord_SCs_*` and `brain_region`, from histology (not built here yet), and
`chris_good`, a manual curation. `accepted` and `isi_excluded` are this
pipeline's additions.

## The optional ephys NWB file

With `postprocessing.write_ecephys_nwb: true`, each session also gets
`_srt-kilosort4_ecephys.nwb`, holding the same units as `_clusters.npz` in
NWB form:

| NWB location | contents |
|---|---|
| `general/` | subject, lab, experimenter, surgery/pharmacology, keywords, start time, `notes` (job, inputs, versions, bad channels, Kilosort parameters, curation settings) |
| `devices/Probe` | Neuropixels 2.0, IMEC |
| `extracellular_ephys/shank{i}` | one electrode group per shank |
| `electrodes` | every channel: position on the probe (`rel_x`, `rel_y`, µm), shank, bad-channel label, whether it was sorted |
| `units` | per unit: `spike_times` (s), `obs_intervals`, `electrodes` (the 16 channels nearest its extremum channel, that channel first), `waveform_mean` and `waveform_sd` (samples x those 16 channels, µV), and every units-table column. `resolution` = 1/sampling rate |

The raw voltage is not copied in. It stays with the acquisition files, or
goes in a separate raw NWB for DANDI.

## Settings

All in [configs/ephys.yaml](../src/raphael/configs/ephys.yaml), which is snapshotted
into every job. The defaults are the Methods' values; the comments there
say where each comes from. Per job, `--param` can override
`kilosort4={...}`, `preprocessing={...}`, `postprocessing={...}` (e.g.
`{keep_analyzer: true}`) and `curation={...}`; the `lfp` stage takes
`lfp={...}`. Overrides are recorded in the job summary and the NWB `notes`.

Bad-channel detection compares each channel with its neighbours. Run per
shank, as the protocol does, it rejected 4 of 384 channels of simulated
data. Run on the whole probe at once, it rejected about a third. Real
recordings have not been checked yet.

## Publishing checklist

1. Fill the TODOs in `configs/ephys.yaml` (experiment description,
   surgery, pharmacology, data collection).
2. Pass the brain region (`--param location=CA1`, an Allen CCF acronym)
   until histology assigns regions per channel.
3. Pass per-mouse subject fields (`subject={age: P90D, sex: F, ...}`).
4. Run `raphael-imaging validate {root}`. It runs NWB Inspector on any NWB
   file. With 1–3 done, it should report no issues.

## Status

| piece | status |
|---|---|
| readers: SpikeGLX, Open Ephys, spikeinterface folders | **done**; tested on simulated data so far |
| bad channels, phase shift, Kilosort 4.0.16 per shank (own environment) | **done** (`ephys/sorting.py`, `sort_step.py`, job stage `kilosort4`) |
| analysers, bombcell, ISI rule, `_clusters.npz` with waveforms, QC | **done** (`ephys/sorting.py`, `nwb.py`, `qc.py`) |
| LFP and theta phase, `_lfp.nwb`, size settings | **done** (`ephys/lfp.py`, job stage `lfp`); tested on simulated data |
| simulated ground truth and scoring | **done** (`ephys/benchmark.py`) |
| first run on our own recordings | **to do** |
| histology: brain region per unit (anatomy CSV, as mecpy) | to do |
| behaviour sync (TTL / sync channel) and `_beh.nwb` | to do |
