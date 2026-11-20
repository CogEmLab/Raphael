# Processing two-photon data: walkthrough

What happens to a recording from the microscope to analysis-ready,
publishable files, and what the data looks like at every step.

The examples are real. They come from processing a 2000-frame two-photon
movie (the DANDI 000402 test bed: 248 x 440 pixels, 6.3 Hz) on an MSC A16
GPU in November 2026. File names, shapes and sizes are as produced.

Each step says whether it **works now** or is **planned**. Related pages:

- [imaging_pipeline.md](imaging_pipeline.md): the output format and what
  goes into each NWB file
- [msc_cluster.md](msc_cluster.md): getting access to the MSC and running
  jobs there
- [cluster.md](cluster.md): how job bundles, GPU checks and the lockfile work

## The whole picture

```
  laptop / lab storage                       MSC cluster
  ────────────────────                       ───────────
  0  raw recording ──────── rsync (once) ──▶ ~/data/...
     + session metadata
           │
  1  raphael-jobs make ─────── rsync ───────▶ ~/jobs/<job>/
     (job bundle)                                 │
                                             2  sbatch submit.slurm
                                                  │
                                             3  raphael-jobs run
                                                checks ▶ blank frames ▶ suite2p
                                                ▶ dF/F ▶ NWB, traces, ROIs, QC
                                                  │
  4  results ◀────────────── rsync ─────────── outputs/
           │
  5  check QC  ▶  6  curate (planned)  ▶  7  analyse / publish
```

| step | where | status |
|---|---|---|
| 0 raw recording and metadata | laptop / lab storage | works for single-plane HDF5; **reader for our microscope planned** |
| 1 make a job bundle | laptop | works |
| 2 copy and submit | laptop to MSC | works |
| 3 run: checks, blank frames, suite2p, dF/F, writing | MSC GPU node | works |
| 4 copy results back | MSC to laptop | works |
| 5 quality check | laptop | works |
| 6 curation (accept/reject cells by hand) | laptop | **planned** |
| 7 analysis and publishing | laptop | loading works; behaviour sync **planned** |

## 0. The raw recording

**Works now:** one imaging plane, one channel, as an HDF5 file with the
movie under the key `data`, shaped frames x height x width:

```
movie.h5                          436 MB
└── data   (2000, 248, 440) int16      frames x height x width
```

**Planned:** a reader for our microscope's own format (e.g. ScanImage TIFF
or Bruker), multi-plane recordings and a second channel. Until then,
recordings need converting to the HDF5 layout above.

Alongside the movie, every recording needs this metadata. All of it ends up
in the NWB file, and the job refuses to start without the first two:

| metadata | example | required |
|---|---|---|
| frame rate (Hz) | `fs=6.3` | yes |
| recording start time | `session_start=2026-11-03T14:05` | yes |
| mouse, day, session type | `mouse=17797 day=7 session_type=scan4` | for the output names |
| pixel size (µm) | `um_per_pixel=2.5` or `[2.5, 2.5]` | strongly advised |
| imaging depth (µm) | `depth_um=200` | advised |
| subject | `subject={age: P84D, sex: M, genotype: ...}` | needed for DANDI |

Times without a UTC offset are read as Vienna time (`nwb.timezone` in
[configs/imaging.yaml](../configs/imaging.yaml)). `age` is an ISO 8601
duration (`P84D` = 84 days); `date_of_birth` can be given instead.

Raw data lives under a **data root**, and jobs refer to files relative to
it. The same relative path must exist on every machine that runs the job:

```
<data root>/                      laptop: e.g. /Volumes/TEMP/InVivoImaging/work
└── raw/M17797/D7/scan4/          MSC:    /msc/home/<MUW-ID>/data
    └── movie.h5
```

## 1. Make a job bundle (laptop)

Commit your code first, so the job is traceable to a commit. Then:

```bash
uv run raphael-jobs make ~/jobs/M17797-D7-scan4 --stage suite2p \
    --data-root /Volumes/TEMP/InVivoImaging/work \
    --input raw/M17797/D7/scan4/movie.h5 \
    --config configs/imaging.yaml --gpu-type a16 --cpus 8 --mem-gb 32 --hours 2 \
    --param fs=6.3 --param session_start=2019-10-30T10:45 \
    --param mouse=17797 --param day=7 --param session_type=scan4 \
    --param um_per_pixel=2.5 --param depth_um=200 \
    --param 'subject={age: P84D, sex: M}' \
    --param 'suite2p={diameter: [5, 5]}'
```

`make` checks the metadata before writing anything. A missing start time or
a malformed age fails here, not after a wait in the cluster queue. It
records each input's size and checksum, the code commit and the lockfile.

**What it creates** (a few kB; the movie is not copied):

```
~/jobs/M17797-D7-scan4/
├── job.yaml           what to run (below)
├── config/
│   └── imaging.yaml   snapshot of the processing and NWB settings
├── submit.slurm       SLURM script for the MSC (GPU type, account, paths)
├── submit.sge         the same for Grid Engine clusters
├── status.json        {"state": "PENDING"}
├── logs/              empty until it runs
└── outputs/           empty until it runs
```

`job.yaml`, abridged:

```yaml
job_id: M17797-D7-scan4
stage: suite2p
git: {commit: b8dc5722..., dirty: false}
uv_lock_sha256: 1b2909d4...
inputs:
- path: raw/M17797/D7/scan4/movie.h5
  bytes: 436484536
  sha256: 707da502...
params: {fs: 6.3, session_start: 2019-10-30 10:45:00, mouse: 17797, ...}
resources: {gpu: true, gpu_type: a16, cpus: 8, mem_gb: 32, hours: 2, ...}
```

## 2. Copy to the MSC and submit

```bash
# once per recording: the movie, keeping its path relative to the data root
rsync -a --relative /Volumes/TEMP/InVivoImaging/work/./raw/M17797/D7/scan4/movie.h5 msc:data/
# every job: the bundle
rsync -a ~/jobs/M17797-D7-scan4 msc:jobs/
# on the MSC
ssh msc 'cd ~/jobs/M17797-D7-scan4 && sbatch submit.slurm'
```

(`msc` is the SSH shortcut described in [msc_cluster.md](msc_cluster.md); use
`<MUW-ID>@s0-l02.hpc.meduniwien.ac.at` without it.) If the code or `uv.lock`
changed, sync the repo and re-run `uv sync --frozen --extra gpu` there first.

**On the MSC:**

```
/msc/home/<MUW-ID>/
├── social-manifolds/        the code, with its .venv from uv.lock
├── data/raw/M17797/D7/scan4/movie.h5
└── jobs/M17797-D7-scan4/    the bundle, waiting in the queue
```

The job waits in the queue (`squeue --me` shows `PD`) until an A16 is free,
then runs on a GPU node.

## 3. The run (MSC GPU node)

`raphael-jobs run` works through these steps in order. Every check comes
before any GPU work, and any failure stops the job with `FAILED` and the
reason in `status.json`:

| # | step | writes | fails if |
|---|---|---|---|
| 1 | record environment | `logs/env.json`: Python, every package version, commit, lockfile hash, SLURM job ID and node | |
| 2 | lockfile check | | `uv.lock` differs from the one the job was made with |
| 3 | input check | | the movie is missing, a different size, or a different checksum |
| 4 | GPU check: a test matrix multiply on the GPU, compared with the CPU | `logs/gpu.json`: GPU model, memory, driver, CUDA build | no usable GPU, or too little free memory |
| 5 | blank-frame scan: frames darker than 30% of the median frame (laser off, gaps) | `work/bad_frames.npy` | |
| 6 | suite2p: motion correction, cell detection, signal extraction, neuropil, deconvolution, classifier | `work/suite2p/plane0/` | |
| 7 | dF/F | | |
| 8 | write outputs | `outputs/` | |
| 9 | clean up: delete `work/` | | |

**Step 5, blank frames.** These are left out of suite2p's detection and
edge-cropping but stay in the timeline. Their dF/F is NaN. Without this, a
blank stretch derails registration: in our test, 1187 blank frames at the
end of a recording cut detection recall from 0.62 to 0.25.

**Step 6, suite2p working files.** These sit in `work/`, preferably on
node-local scratch. During the run, suite2p's registered movie (`data.bin`)
is about the size of the raw movie, so budget that much free space per job.
suite2p's arrays are **ROIs x frames**:

```
work/suite2p/plane0/
├── data.bin            ~436 MB   registered movie (deleted in step 9)
├── F.npy               (3363, 2000) float32   fluorescence per ROI
├── Fneu.npy            (3363, 2000) float32   surrounding neuropil
├── spks.npy            (3363, 2000) float32   deconvolved activity
├── iscell.npy          (3363, 2)    [accepted 0/1, classifier probability]
├── stat.npy            (3363,)      per ROI: pixels, weights, centre, shape stats
├── reg_outputs.npy     motion shifts, mean images, valid crop
├── detect_outputs.npy  max projection, correlation image
└── ops.npy, settings.npy, db.npy    settings used
```

**Step 7, dF/F.** This is computed by us, not suite2p, so every pipeline
baselines the same way. The settings come from `configs/imaging.yaml`:

```
F_corrected = F − 0.7 × Fneu                        neuropil subtraction
F0          = running 8th percentile over 60 s      baseline
dF/F        = (F_corrected − F0) / F0               NaN at blank frames
```

From here on, arrays are **frames x ROIs** (the transpose of suite2p's).

## 4. The outputs

```
outputs/                                                  size (2000 frames)
├── M17797/D7/scan4/                                      ← clark2025 layout
│   ├── sub-M17797_ses-D7_typ-scan4_src-suite2p_ophys.nwb    81 MB
│   ├── sub-M17797_ses-D7_typ-scan4_src-suite2p_traces.npz   27 MB
│   ├── sub-M17797_ses-D7_typ-scan4_src-suite2p_rois.csv    0.5 MB
│   └── qc/plane1_rois.png
├── suite2p/plane0/            suite2p's own files (F, stat, iscell, ...)
│                              for its GUI; 140 MB
└── suite2p_summary.json
```

`src-suite2p` names the pipeline, so outputs of other pipelines (e.g. with
denoising, `src-deepcad-suite2p`) can sit beside it for comparison.

**The NWB file** (abridged; 3363 ROIs, 2000 frames). This is the file to
publish on DANDI:

```
general/
├── subject/                 subject_id M17797, age P84D, sex, genotype, species
├── devices/Microscope
├── optophysiology/ImagingPlane1/
│                            indicator, location, rate, pixel size 2.5 µm, depth
└── experimenter, lab, institution, session start time, notes (job, blank frames)
processing/ophys/
├── ImageSegmentation/PlaneSegmentation1/
│   ├── pixel_mask           (x, y, weight) for every ROI pixel
│   ├── accepted             (3363,) bool       classifier decision
│   ├── cell_probability     (3363,)
│   └── centre_x/y, npix, radius, compact, aspect_ratio, snr, skew, std
├── Fluorescence/RoiResponseSeries1/data    (2000, 3363)  raw F
├── Neuropil/RoiResponseSeries1/data        (2000, 3363)
├── DfOverF/RoiResponseSeries1/data         (2000, 3363)
├── Deconvolved/RoiResponseSeries1/data     (2000, 3363)
├── SummaryImages1/          mean, mean_enhanced, max_projection,
│                            correlation (248, 440)
└── xy_shifts1/data          (2000, 2)   motion correction per frame
```

**`_traces.npz`** is dF/F as a pynapple `TsdFrame`: `t` (2000,) seconds, `d`
(2000, 3363), and `columns` matching the `roi` column of the CSV.

**`_rois.csv`** has one row per ROI:

```
roi,plane,plane_roi,accepted,cell_probability,centre_y,centre_x,npix,radius,compact,aspect_ratio,snr,skew,std,centroid_x_px,centroid_y_px
0,1,0,True,0.9066,229,265,34,2.761,1.011,1.061,0.994,2.035,4719.38,266.21,229.53
```

**`suite2p_summary.json`**, here from a 10,000-frame run of the same plane
whose last 1187 frames were blank:

```json
{"device": "cuda", "frames": 10000, "image_shape": [248, 440], "rois": 3644,
 "accepted": 1806, "blank_frames": 1187, "suite2p_s": 83.4, "write_s": 21.7,
 "peak_gpu_memory_gb": 0.91}
```

## 5. Copy back and check

```bash
rsync -a msc:jobs/M17797-D7-scan4 ~/jobs/
cat ~/jobs/M17797-D7-scan4/status.json                # DONE / FAILED + error
cat ~/jobs/M17797-D7-scan4/outputs/suite2p_summary.json
uv run raphael-imaging validate ~/jobs/M17797-D7-scan4/outputs   # NWB Inspector
```

Open `qc/plane1_rois.png`. It shows every ROI on the mean image (green
accepted, red rejected) and the dF/F of the first 30 accepted cells. Check
that:

- accepted ROIs sit on cells, not on blood vessels or the image edge
- traces show transients, not just noise
- the accepted fraction is plausible (suite2p's default classifier is
  generic and needs checking on our data)

NWB Inspector flags two harmless things on short test movies: "data may be
in the wrong orientation" (there are more ROIs than frames) and "constant
sampling rate". Anything else, such as missing subject fields, should be
fixed before publishing.

## 6. Curation (planned)

Accepting and rejecting ROIs by hand in the suite2p GUI, using
`outputs/suite2p/plane0`, then writing the edited `accepted` column back into
the NWB file and the CSV. Not built yet: the GUI needs extra packages, and
the re-export step does not exist.

## 7. Analysis

```python
import pandas as pd
import pynapple as nap

base = '~/jobs/M17797-D7-scan4/outputs/M17797/D7/scan4/sub-M17797_ses-D7_typ-scan4_src-suite2p'
dff = nap.load_file(f'{base}_traces.npz')            # TsdFrame, frames x ROIs
rois = pd.read_csv(f'{base}_rois.csv')
cells = dff[:, rois.roi[rois.accepted].to_numpy()]   # accepted cells only
```

The NWB file holds everything else: raw F, neuropil, deconvolved activity,
masks and summary images. Read it with `pynwb`.

**Planned:** the matching behaviour file (`_beh.nwb`) for our sessions, and
synchronisation of imaging frames to behaviour.

## Choosing suite2p settings

Pass suite2p settings with `--param 'suite2p={...}'`. They are merged into
suite2p's defaults, so only what you name changes. What we learned on the
test bed (scored against the published segmentation of the same plane):

- **`diameter` does not affect the default detector** (`sparsery`). It uses
  `detection: {sparsery_settings: {spatial_scale: N}}` (0 = auto, 1 = ~6 px,
  2 = ~12 px). `diameter` matters for `cellpose` and for ROI statistics.
- **`detection: {threshold_scaling: 2}`** gives fewer, more reliable ROIs:
  precision rose from 0.50 to 0.58-0.61.
- **`detection: {algorithm: cellpose}`** finds cells by shape in the mean
  image. It was very precise (0.89) but missed most cells, because test-bed
  cells are only 4-5 pixels across. It is likely to do better on
  higher-resolution recordings.
- Settings should be chosen on our own recordings. `imaging/benchmark.py`
  scores any run against a reference segmentation.

## Not built yet

| piece | needed for |
|---|---|
| reader for our microscope's files; multi-plane and two-channel | processing our own recordings |
| behaviour file and imaging-behaviour sync | relating activity to behaviour |
| curation round trip (GUI, then re-export) | trustworthy cell lists |
| batch processing from a session table | many sessions without one command each |
| denoising stage (DeepCAD-RT / SUPPORT) | better signal-to-noise; compare `src-` pipelines |
| CUDA 13 lockfile | using the B200 GPUs |
