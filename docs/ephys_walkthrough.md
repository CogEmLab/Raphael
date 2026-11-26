# Processing Neuropixels recordings: walkthrough

What happens to a day of Neuropixels 2.0 recordings, from the acquisition
files to analysis-ready, publishable files, and what the data looks like at
every step. The processing reproduces the paper's Methods ("Offline spike
sorting"); [ephys_pipeline.md](ephys_pipeline.md) maps each step to the
Methods and the lab's code.

The examples are real. They come from a simulated day sorted on an MSC
`1g.5gb` GPU slice (and its LFP on a CPU node) in November 2026, in the
repo's one environment. The day had two 3-minute sessions on
an NP2.0 four-shank probe (384 channels, 96 per shank) with 150 known
units. File names, shapes, sizes and timings are as produced. No recording
of our own has been through it yet.

Each step says whether it **works now** or is **planned**. Related pages:

- [ephys_pipeline.md](ephys_pipeline.md): what each step does, the output
  format, and the LFP size settings
- [imaging_walkthrough.md](imaging_walkthrough.md): the same for two-photon
  data, which ends in the same shape
- [msc_cluster.md](msc_cluster.md): getting access to the MSC and running
  jobs there

## The whole picture

```
  laptop / lab storage                       MSC cluster
  ────────────────────                       ───────────
  0  raw recordings of a day ── rsync ─────▶ ~/data/raw/M../D../{OF1,VR,OF2}/
     + session metadata
           │
  1  raphael-jobs make ─────── rsync ───────▶ ~/jobs/<job>/
     (job bundle)                                 │
                                             2  sbatch submit.slurm
                                                  │
                                             3  raphael-jobs run
                                                checks ▶ join the day
                                                ▶ per shank: bad channels ▶ phase shift
                                                ▶ Kilosort 4.0.16
                                                ▶ analysers per session ▶ bombcell
                                                ▶ ISI rule ▶ _clusters.npz, QC
                                             3b optional, a CPU job: raphael-jobs run
                                                (stage lfp) ▶ _lfp.nwb
                                                  │
  4  results ◀────────────── rsync ─────────── outputs/
           │
  5  check QC  ▶  6  manual curation (optional, planned)  ▶  7  analyse / publish
```

| step | where | status |
|---|---|---|
| 0 raw recordings and metadata | laptop / lab storage | readers for SpikeGLX and Open Ephys **work, but are untested on our files** |
| 1 make a job bundle | laptop | works |
| 2 copy and submit | laptop to MSC | works |
| 3 run: sorting, analysers, curation, writing `_clusters.npz` | MSC GPU node | works (simulated data) |
| 3b LFP and theta phase into `_lfp.nwb` (optional) | MSC CPU node | works (simulated data, locally) |
| 4 copy results back | MSC to laptop | works |
| 5 quality check | laptop | works |
| 6 manual curation in phy or the spikeinterface GUI | laptop | **planned** |
| 7 analysis and publishing | laptop | loading works; histology and behaviour sync **planned** |

## 0. The raw recordings

One folder per recording, as the acquisition software wrote it. The
pipeline recognises the format from the files:

```
raw/M25/D18/
├── OF1/                              Open Ephys (as the lab records)
│   └── Record Node 109/
│       ├── settings.xml
│       └── experiment1/recording1/
│           ├── structure.oebin
│           └── continuous/Neuropix-PXI-100.ProbeA/
│               ├── continuous.dat    int16, samples x 384 channels, ~83 GB/hour
│               ├── sample_numbers.npy
│               └── timestamps.npy
├── VR/ ...
└── OF2/ ...
```

SpikeGLX run folders (`*_g0/*_imec0/*.ap.bin` + `.ap.meta`) and folders
saved by spikeinterface also work. The probe's stream is picked
automatically: the one wideband stream on NP2.0, or the AP stream on NP1.0.
Pass `stream=...` if a folder holds more than one probe.

Each folder must hold **one** continuous recording. A folder with several
Open Ephys recordings (`recording1`, `recording2`) is refused; give each
its own input.

Every day of recordings needs this metadata. All of it ends up in the NWB
files, and the job refuses to be made without the first two:

| metadata | example | required |
|---|---|---|
| each recording's start time | `session_start=[2026-11-03T10:00, 2026-11-03T10:40, ...]` | yes |
| each recording's session type, in input order | `session_type=[OF1, VR, OF2]` | yes, for more than one recording |
| mouse and day | `mouse=25 day=18` | for the output names |
| brain region | `location=CA1` (Allen CCF acronym) | for DANDI, until histology |
| subject | `subject={age: P90D, sex: F, genotype: ...}` | for DANDI |

Times without a UTC offset are read as Vienna time (`nwb.timezone` in
[configs/ephys.yaml](../src/raphael/configs/ephys.yaml)).

## 1. Make a job bundle (laptop)

One job per day. List the recordings in the order they were made: they are
joined in that order and sorted together, so a unit keeps its id across
the day.

```bash
uv run raphael-jobs make ~/jobs/M25-D18 --stage kilosort4 \
    --data-root /Volumes/ephys \
    --input raw/M25/D18/OF1 --input raw/M25/D18/VR --input raw/M25/D18/OF2 \
    --config configs/ephys.yaml --gpu-type 1g.5gb --cpus 8 --mem-gb 64 --hours 8 \
    --param 'session_start=[2026-11-03T10:00, 2026-11-03T10:40, 2026-11-03T11:30]' \
    --param 'session_type=[OF1, VR, OF2]' --param mouse=25 --param day=18 \
    --param 'subject={age: P90D, sex: F}' --param location=CA1
```

`make` checks the metadata before writing anything. It records every file
of every recording folder with its size and checksum, the code commit and
the `uv.lock` hash. Checksumming takes about a minute per 10 GB; `--no-hash`
records sizes only.

**What it creates** (a few kB; the recordings are not copied):

```
~/jobs/M25-D18/
├── job.yaml           what to run (below)
├── config/
│   └── ephys.yaml     snapshot of the processing and NWB settings
├── submit.slurm       SLURM script for the MSC
├── submit.sge
├── status.json        {"state": "PENDING"}
├── logs/
└── outputs/
```

`job.yaml`, abridged, from the simulated day:

```yaml
job_id: ks4016-sim
stage: kilosort4
git: {commit: d5705365..., dirty: false}
uv_lock_sha256: 5d91933b...
inputs:
- path: sim-np2/session0
  bytes: 4147457469
  files:
  - {path: traces_cached_seg0.raw, bytes: 4147200000, sha256: 40838d1b...}
  - {path: probegroup.json, bytes: 194233, sha256: 750d0215...}
  - ...
- path: sim-np2/session1
  ...
params: {session_start: [2026-11-10 09:00:00, 2026-11-10 09:30:00],
         session_type: [OF1, OF2], mouse: sim, day: 1, location: CA1, ...}
resources: {gpu: true, gpu_type: 1g.5gb, cpus: 8, mem_gb: 32, hours: 2, ...}
```

## 2. Copy to the MSC and submit

```bash
# once per day of recordings, keeping paths relative to the data root
rsync -a --relative /Volumes/ephys/./raw/M25/D18 msc:data/
# every job: the bundle
rsync -a ~/jobs/M25-D18 msc:jobs/
# on the MSC
ssh msc 'cd ~/jobs/M25-D18 && sbatch submit.slurm'
```

Once per checkout on the MSC, install the environment (Kilosort included):

```bash
cd ~/social-manifolds && uv sync --frozen --extra gpu
```

**Which GPU.** Kilosort sorts one shank at a time and needed **1.24 GB** of
GPU memory on the simulated day, so the smallest slice (`1g.5gb`, 5 GB) is
enough, and it is usually free. Its jobs may run 8 hours. Use `a16` or
`3g.20gb` (3 days) for days that might take longer: Kilosort took 6 minutes
for 6 minutes of recording on a `1g.5gb` slice, so budget roughly real time.

## 3. The run (MSC GPU node)

`raphael-jobs run` works through these steps in order. Any failure stops the
job with `FAILED` and the reason in `status.json`, and the full traceback in
`logs/traceback.txt`:

| # | step | writes | fails if |
|---|---|---|---|
| 1 | record environment | `logs/env.json`: package versions, commit, lockfile hash, SLURM job and node | |
| 2 | lockfile check | | `uv.lock` differs from the one the job was made with |
| 3 | input check | | a file is missing, a different size or checksum, or a folder has extra files |
| 4 | GPU check | `logs/gpu.json` | no usable GPU |
| 5 | join the day, per shank: remove bad channels, phase shift | `work/sorted/channels.csv` | a folder is not a recognised recording |
| 6 | Kilosort 4.0.16 on each shank | `work/sorted/kilosort4/{shank}/`, `spikes.npz`, `units.csv` | |
| 7 | analyser recording; cut into sessions; one analyser per session | `work/analyzers/.../_analyzer.zarr` | |
| 8 | bombcell, ISI rule | `work/analyzers/.../_bombcell_curation.json` | |
| 9 | write `_clusters.npz` (spikes, curation, metrics, waveforms) and QC | `outputs/M../D../{type}/` | |
| 10 | clean up: Kilosort's files to `outputs/kilosort4/`; delete `work/sorted` and the analysers (kept in `outputs/kilosort4/analyzers/` with `keep_analyzer: true`) | | |

**Steps 5–6, the sorting step** (`ephys/sort_step.py`), leave plain files
that the rest of the stage reads:

```
work/sorted/
├── channels.csv        every channel: shank, position, bad-channel label, used
├── spikes.npz          sample_index (n_spikes,), unit (n_spikes,), sampling_frequency
├── units.csv           unit, group (shank), original_cluster_id, KSLabel, Amplitude, ...
├── sort_summary.json   versions, run times, peak GPU memory, every Kilosort parameter
└── kilosort4/
    ├── 0/sorter_output/   Kilosort's own files for shank 0 (phy format)
    ├── 1/ ...
    └── 3/ ...
```

`channels.csv`:

```
channel_id,x_um,y_um,shank,label,used
0,0.0,0.0,0,good,True
1,32.0,0.0,0,good,True
```

Bad-channel labels are `good`, `dead`, `noise` or `out` (outside the
brain). On the simulated day, 4 of 384 channels were labelled dead.

Units are numbered 0..N-1 across shanks (shank 0's first). `original_cluster_id`
is Kilosort's id within its shank, which is what phy shows for
`outputs/kilosort4/{shank}/`. The simulated day gave 20, 32, 45 and 40
units on shanks 0–3: 137 in all, for 150 true units.

**Step 7, analysers.** On each shank, the channels the sorter saw are
phase-shifted, common-median referenced and band-passed (300–6000 Hz).
The joined recording and sorting are cut back into sessions, so spike
times start at 0 at each recording's first sample. Each session gets a
sorting analyser:
- **Sparsity:** channels within 70 µm of each unit's largest one.
- **Extensions:** the lab's set: unit and spike locations (monopolar
  triangulation), random spikes, noise levels, waveforms, templates,
  amplitudes and scalings, ISI histograms, correlograms, template
  similarity, quality and template metrics.
- **Waveforms** are deleted afterwards (GBs per session); everything else
  is kept.

**Step 8, curation.** Each unit gets these labels, which `accepted`
combines:

| label | rule |
|---|---|
| `bombcell_label` | good, mua, noise or non_soma (spikeinterface's default thresholds) |
| `isi_excluded` | ISI violation ratio above 10 in any session of the day: excluded in every session |
| `accepted` | good or mua, and not `isi_excluded` |

## 4. The outputs

```
outputs/                                                    size (2 x 3 min)
├── sub-Msim/day-01/ses-OF1/                                 ← the processed dataset
│   ├── sub-Msim_day-01_ses-OF1_srt-kilosort4_clusters.npz       4.4 MB
│   ├── sub-Msim_day-01_ses-OF1_lfp.nwb   (from the lfp job)       31 MB
│   └── qc/units.png
├── sub-Msim/day-01/ses-OF2/   the same for the second session
├── kilosort4/          working files, not part of the dataset:
│   ├── 0/ ... 3/       Kilosort's own files per shank, for phy
│   ├── analyzers/      only with keep_analyzer: true: per session the
│   │                   _analyzer.zarr (7.0 MB) and _bombcell_curation.json
│   ├── channels.csv, sort_summary.json                         101 MB in all
└── kilosort4_summary.json
```

The `lfp` job (step 3b) writes `_lfp.nwb` into its own job's outputs, in
the same tree, so the two merge into one dataset. The behaviour step
(planned) adds `_beh.nwb`. Of `_clusters.npz`'s 4.4 MB, the waveforms are
about 1.6 MB (137 units × 2 arrays × 90 samples × 16 channels × 4 bytes).

Every unit appears in every session's `_clusters.npz`; select with
`accepted`. What each file is for:
[ephys_pipeline.md](ephys_pipeline.md#what-each-file-is-for).

**`_clusters.npz`** is a pynapple `TsGroup` in clark2025's form: one `Ts`
of spike times per unit, keyed by `unit`, with a time support of
[0, session length]. Its metadata is one row per unit, the same rows in
every session. The scalar columns look like this (a selection):

```
unit,x,y,group,original_cluster_id,KSLabel,bombcell_label,extremum_channel,coord_probe_x,coord_probe_y,isi_violations_ratio,isi_excluded,accepted
0,24.183,46.722,0,0,good,mua,6,32.0,45.0,0.0,False,True
```

Each unit's waveform is also in the metadata, as arrays:

```
waveform_mean           (90, 16) float32  µV, extremum channel first
waveform_sd             (90, 16) float32  µV
waveform_channels       (16,)    channel ids
waveform_positions_um   (16, 2)  x, y on the probe
waveform_sampling_frequency  30000.0
waveform_nbefore             30           spike at sample 30 (1 ms)
```

Kilosort and bombcell often disagree (unit 0: Kilosort `good`, bombcell
`mua`). Kilosort labelled 120 of the 137 units good, and bombcell 87.

**`kilosort4_summary.json`** from the simulated day (abridged):

```json
{"device": "cuda", "versions": {"kilosort": "4.0.16", "spikeinterface": "0.104.9",
  "numpy": "1.26.4", "torch": "2.14.1+cu126"},
 "channels": 384, "channels_used": 380, "bad_channels": {"dead": 4},
 "units_sorted": 137, "units_per_shank": {"0": 20, "1": 32, "2": 45, "3": 40},
 "sessions": [
   {"session_type": "OF1", "duration_s": 180.0, "units": 137,
    "labels": {"good": 87, "mua": 49, "noise": 1},
    "isi_excluded": 0, "accepted": 136},
   {"session_type": "OF2", ...}],
 "kilosort_params": {"do_correction": false, "use_binary_file": false},
 "prepare_s": 18.3, "kilosort_s": 375.6, "postprocess_s": 175.4,
 "peak_gpu_memory_gb": 1.24}
```

## 4b. The LFP (optional)

The LFP is a separate CPU job on the same recording folders, so it can be
run, skipped, or rerun at another size without sorting again:

```bash
uv run raphael-jobs make ~/jobs/M25-D18-lfp --stage lfp --cpu \
    --data-root /Volumes/ephys \
    --input raw/M25/D18/OF1 --input raw/M25/D18/VR --input raw/M25/D18/OF2 \
    --config configs/ephys.yaml --cpus 8 --mem-gb 32 --hours 4 \
    --param 'session_start=[2026-11-03T10:00, 2026-11-03T10:40, 2026-11-03T11:30]' \
    --param 'session_type=[OF1, VR, OF2]' --param mouse=25 --param day=18 \
    --param location=CA1
# lighter: --param 'lfp={rate_hz: 500, band_hz: [1, 200], dtype: float16}'
# richer:  --param 'lfp={channels: {block_size: 1}}'          (every contact)
```

Per recording it writes `sub-M25/day-18/ses-{type}/sub-M25_day-18_ses-{type}_lfp.nwb`
and a row in `lfp_summary.json` (channels, rate, file size, time). With
the defaults the file has the lab's layout:

```
general/extracellular_ephys/electrodes   (48 rows) channel_name CH49_CH50_..._CH56,
                                         rel_x, rel_y (mean of the 8 contacts),
                                         shank, n_contacts
processing/ecephys/
├── LFP/lfp            (samples, 48) float32 µV (conversion 1e-6), 1 kHz from 0 s
└── Processed/theta    (samples, 48) float32 radians, 0 to 2π peak to peak
```

On the simulated day, at the defaults, each 3-minute session became 48
LFP channels and a 31.7 MB file; that is about 0.32 GB for 30 minutes. It
took 2–3.5 minutes per session on 8 CPUs. The settings and their effect on
size are in [ephys_pipeline.md](ephys_pipeline.md#the-lfp-file). The
`lfp` stage has been tested on simulated recordings only; MSC has no
Neuropixels recording of ours yet.

## 5. Copy back and check

```bash
rsync -a msc:jobs/M25-D18 ~/jobs/
cat ~/jobs/M25-D18/status.json                         # DONE / FAILED + error
cat ~/jobs/M25-D18/outputs/kilosort4_summary.json
uv run raphael-imaging validate ~/jobs/M25-D18-lfp/outputs   # NWB Inspector, on _lfp.nwb
```

Open `qc/units.png` for each session. It has three panels:

- **Units on the probe,** coloured by bombcell label, over the channel
  layout, with bad channels crossed. Check that units sit along the shanks
  where you expect cells, and that bad channels are few. More than about
  10% points to a recording problem.
- **Spike depth over time.** Horizontal bands mean a stable probe. Motion
  correction is off (as in the Methods), so drift shows here as sloping or
  stepping bands. Drift is the main thing that would argue for turning it
  on: `kilosort4={do_correction: true}`.
- **Firing rate by depth.**

NWB Inspector checks the NWB files (`_lfp.nwb`, and `_ecephys.nwb` if
written). Once `location` and the subject are given, the only expected
report is for theta phase, which is stored in an ElectricalSeries but is in
radians, not volts.

## 6. Manual curation (planned)

Kilosort's own files in `outputs/kilosort4/{shank}/sorter_output` open in
phy, but phy works per shank and with Kilosort's cluster ids
(`original_cluster_id`). The spikeinterface curation GUI works on the analysers
directly. Neither is wired back into the outputs yet. The Methods use
bombcell and the two rules only, so this is optional.

## 7. Analysis

```python
from raphael import population

base = '~/jobs/M25-D18/outputs/sub-M25/day-18/ses-OF1/sub-M25_day-18_ses-OF1_srt-kilosort4'
pop = population.load(base).accepted()   # TsGroup + its unit table, accepted units only
rates = pop.binned(0.1)                   # TsdFrame, time x unit, Hz

import numpy as np, pynapple as nap, pynwb
units = nap.load_file(f'{base}_clusters.npz')            # clark2025's form
mean = np.stack(units['waveform_mean'])                   # units x 90 x 16, µV
with pynwb.NWBHDF5IO('.../sub-M25_day-18_ses-OF1_lfp.nwb') as io:
    lfp = io.read().processing['ecephys']['LFP']['lfp']   # samples x 48, ×1e-6 for V
```

The first three lines also load an imaging session (`..._src-suite2p`),
with dF/F in place of rates. Code written for clark2025's `_clusters.npz`
reads these files as they are.

**Planned:**
- **Histology:** a brain region per unit (the anatomy CSV mecpy reads),
  for NWB electrode locations and region-level analyses.
- **Synchronisation** of spike times with behaviour, and `_beh.nwb`.

## Checking the sorter on simulated data

[`ephys/benchmark.py`](../src/raphael/ephys/benchmark.py) makes
recordings with known spikes and scores a job's output against them:

```bash
# on the MSC, as a CPU job: two 3-minute sessions, 150 units, 384 channels (8 GB)
uv run python -m raphael.ephys.benchmark simulate ~/data/sim-np2
# ... make and run a kilosort4 job on sim-np2/session0 and session1 ...
uv run python -m raphael.ephys.benchmark score ~/data/sim-np2 ~/jobs/<job>/outputs
```

With the Methods' settings, Kilosort 4.0.16 found **83 of the 150 true
units well** (accuracy ≥ 0.8). The mean accuracy over all 150 was 0.57,
counting missed units as 0. **89 of the 136 accepted units** matched a true
unit. Accepting mua, as the Methods do, admits some poorly isolated units,
which the Methods accept for population measures.

Simulated noise is independent between channels, unlike real probe data.
Treat these numbers as a check that the pipeline works, not as the
sorter's accuracy on real recordings.

## Not built yet

| piece | needed for |
|---|---|
| a first run on our own recordings | confirming the readers, bad-channel rates and timings on real data |
| histology: brain region per unit | NWB electrode locations and region-level analyses |
| behaviour sync (TTL / sync channel) and `_beh.nwb` | relating spikes to behaviour |
| LFP | theta analyses |
| manual curation round trip | optional review beyond bombcell |
| batch jobs from a session table | many days without one command each |
