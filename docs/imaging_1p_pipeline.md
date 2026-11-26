# One-photon (miniscope) imaging pipeline

The default pipeline for these recordings; how to run it on your data is in
[processing.md](processing.md).

Raw Inscopix recordings go in. Out comes one processed, publishable NWB
file per session, in the same tree and with the same names as the
two-photon pipeline ([imaging_pipeline.md](imaging_pipeline.md)) and the
clark2025 ephys dataset, with pynapple tables alongside.

- Settings and publication metadata: `configs/imaging_1p.yaml`
- Job stage: `cnmfe` (`raphael-jobs make --stage cnmfe`), CPU
- Session table: `raphael-imaging inscopix-sessions`
- Code: `src/raphael/imaging/inscopix.py` (reader), `cnmfe.py` (CaImAn,
  in its own environment), `caiman_runner.py` (what runs there)

The processing is the helping-behaviour analysis's
([CogEmLab/helping-behaviour-calcium-imaging](https://github.com/CogEmLab/helping-behaviour-calcium-imaging),
`batch_cnmfe.py`), with its settings as the defaults and the fixes listed
under [Changes from the helping-behaviour code](#changes-from-the-helping-behaviour-code).

## Output layout

```
{processed}/sub-M{mouse}/day-{dd}/ses-{type}/
    sub-M{mouse}_day-{dd}_ses-{type}_src-cnmfe_ophys.nwb    processed ophys
    sub-M{mouse}_day-{dd}_ses-{type}_src-cnmfe_traces.npz   TsdFrame of C (denoised)
    sub-M{mouse}_day-{dd}_ses-{type}_src-cnmfe_rois.csv     one row per component
    sub-M{mouse}_day-{dd}_ses-{type}_beh.nwb                behaviour, with the DLC pose
    qc/plane1_rois.png                                         mean image, components, traces
    provenance/{job_id}.json
```

`_traces.npz` holds CNMF-E's C (`processing.trace` in imaging_1p.yaml),
which is what the dCA1 analysis reads. The NWB file holds every trace:

| NWB location | contents |
|---|---|
| `.../Fluorescence/RoiResponseSeries1` | C + YrA: the component's activity plus its residual |
| `.../Denoised/RoiResponseSeries1` | C, denoised |
| `.../Detrended/RoiResponseSeries1` | C + YrA, detrended (`detrend_df_f(detrend_only=True)`) |
| `.../Deconvolved/RoiResponseSeries1` | S (OASIS) |
| `.../PlaneSegmentation1` | footprints (sparse), `accepted`, `snr`, `r_value`, `cnmfe_index` |
| `.../SummaryImages1` | mean, correlation, peak-to-noise |
| `.../xy_shifts1` | rigid motion-correction shifts (px) |
| `notes` | the acquisition's own fields (gain, LED power, focus, lens, dropped frames), binning, CaImAn version and lockfile checksum |

There is no dF/F: the CNMF-E background is not a baseline to divide by, so
the detrended trace is labelled as detrended, not dF/F.

`_rois.csv` has every component, rejected ones too (`accepted` False), so
curation can change without re-running CaImAn. Its `roi` column is the
TsdFrame's column.

**Behaviour** (`_beh.nwb`): P_x, P_y, H, S, the `moving`/`stationary`
epochs and the trial events as interval tables, as for every Raphael
session, plus the DeepLabCut pose as ndx-pose `PoseEstimation` (one per
animal: every body part's x, y and likelihood, in arena cm), with the video
linked by path (`nwb.build_beh_nwb(..., pose=table)`).

**Not in the dataset**: the CaImAn model (`.hdf5`), the arrays the NWB file
was built from and CaImAn's log stay in the job's work folder (cluster
scratch), for comparison with earlier runs and re-curation in CaImAn's
tools. The binned and motion-corrected movies (6–33 GB each) are deleted
unless `keep_movies`.

## Days

`day-{dd}` counts the mouse's recording dates, from its first recording:
habituation and image checks included, two sessions on one date share a
day (`day-12/ses-Toy`, `day-12/ses-Pool`). A session keeps its day however a project
filters the table later.

```bash
raphael-imaging --config configs/imaging_1p.yaml inscopix-sessions RAW_ROOT --out sessions.csv
```

lists every raw recording under `RAW_ROOT` (a folder with `session.json`
beside the `.isxd`): animal ID, local start time, day, duration, frame rate,
dropped frames, gain, LED power, lens. Only each file's end is read, so it
takes seconds on the share. A recording that can't be read (no Inscopix
description at its end, as when acquisition never closed the file) gets a
row with an `error` and the date and animal from `session.json`.

## Running a session

Make the job where the data is mounted, run it on the cluster
([cluster.md](cluster.md)):

```bash
raphael-jobs make jobs/cnmfe-M8472-D6 --stage cnmfe --cpu --cpus 16 --mem-gb 64 --hours 4 \
    --data-root /path/to/raw --config configs/imaging_1p.yaml \
    --input May_2024/.../Moises_B6J_8472_M_06HBT03_2024-06-05-17-13-10_video.isxd \
    --param mouse=8472 --param day=6 --param session_type=HBTD3 \
    --param 'subject={sex: M, date_of_birth: 2024-01-23, strain: C57BL/6J}'
```

For a whole project, `raphael-jobs plan`, `files` and `make-batch` make
every session's job and run them in batches that fit the cluster's disk:
[processing.md](processing.md). `raphael-jobs make` without `--stage`
fills in all of the above from the `.isxd`.

The start time comes from the `.isxd`. The work folder is the site's
scratch (`paths.scratch/{job_id}` in cluster.yaml), which is where the
CaImAn model stays, and where CaImAn's temporary memmaps go (Raphael sets
`CAIMAN_TEMP`; CaImAn's default is `~/caiman_data/temp`, the home quota).
A session needs about 3x its binned movie free there while it runs (the
binned movie, the motion-corrected memmap, the C-order memmap): ~10 GB for
a 4-minute recording, ~35 GB for 18 minutes. To run on a laptop, put the
work folder on a big drive: `raphael-jobs run JOB --data-root ... --work
/Volumes/BIG/work/JOB`. `--param 'cnmfe={quality: {min_SNR: 2.5}}'`
overrides any setting for one job (recorded in the NWB notes and the
provenance); `--param max_frames=3600` runs the first minute only.

A TIFF already binned for CaImAn (the helping-behaviour inputs) also works,
with `--param fs=30 --param session_start=...`: that runs CaImAn on exactly
the movie the earlier analysis used, for a like-for-like comparison.

The first run on a machine builds CaImAn's environment
(`~/.cache/raphael/caiman-env`, or `$RAPHAEL_CAIMAN_ENV`): a few minutes,
and it needs a C compiler. CaImAn is not installed alongside Raphael
because it is not on PyPI and brings a GUI stack and Keras/torch; its
versions are locked in `src/raphael/imaging/caiman_env/uv.lock`.

## The steps

1. **Read** the `.isxd` directly (`inscopix.open_isxd`): frames as a
   memmap, start time, frame period, dropped frames, the acquisition's
   settings. Checked against Inscopix's TIFF export of the same recording:
   equal pixel for pixel.
2. **Bin** 2 x 2 in space and to 30 Hz in time (block means;
   `spatial_downsample`, `temporal_downsample`). The rate comes from the
   file (59.988 Hz on nVista3, so 29.994 Hz after binning), never from the
   file name. Dropped frames are replaced by the frame before them first.
3. **Motion correction**: rigid NoRMCorre on a high-pass-filtered copy
   (`gSig_filt` 3 px), shifts up to 5 px.
4. **CNMF-E**: correlation / peak-to-noise seeding (`min_corr` 0.8,
   `min_pnr` 7), ring-model background, OASIS AR(1) deconvolution,
   `gSig` 3 px, `gSiz` 7 px, patches `rf` 20 / `stride` 10.
5. **Component evaluation**: SNR >= 3 (floor 2), spatial correlation >=
   0.85 (floor 0.3), no CNN. Sets `accepted`; nothing is removed.
6. **Write** the NWB file, the tables and the QC figure.

Cross-day registration (CaImAn's `register_multisession`) and the
behaviour clock (below) are next.

## Changes from the helping-behaviour code

| there | here |
|---|---|
| reads Inscopix's TIFF exports; 6 sessions stop at 32,767 frames (1,092 s), 5–48 s short of the video | reads the `.isxd`: every frame |
| CaImAn's `fr` is always 30 (`fps` argument unused), so 20 Hz files were modelled as 30 Hz | `fr` is the binned movie's real rate |
| frame rates and exceptions by file name (`downsample.py`) | from the file's own description |
| Oct_2021 re-curated with other thresholds by a separate script | one set of thresholds; a different set is a recorded override |
| rejected components deleted (`select_components`) | kept, with `accepted` |
| z-scored "dF/F" (`npy_data/`) | detrended trace named as such; z-scoring is analysis |
| uint16 cast truncates | rounded and clipped |
| `maximum shift` recorded as 0 with `border_nan: copy` | the real maximum shift (`cnmfe_summary.json`) |
| outputs by cohort / session type / Inscopix file name | `sub-M{mouse}/day-{dd}/{type}`, one tidy session table |

## Known issues in the raw data

- **Start times.** nVista writes the acquisition computer's local clock as
  "seconds since epoch" with `utcOffset` 0: the `.isxd` start equals the
  file name's time, and is minutes after `session.json`'s true-UTC `date`.
  Raphael reads it as local time in `nwb.timezone` (Europe/Vienna), which
  puts NWB start times right; read as UTC they would be 1–2 h late.
- **The behaviour clock.** The nVista drives the Basler camera from GPIO 1
  (a free-running waveform), and the `.gpio` file next to the movie
  records it, in Inscopix's packet format. Until that is decoded the camera
  and imaging can only be lined up by frame count, as the earlier analysis
  did: for M8472 HBTD3 the camera has 12,728 frames to the imaging's 12,701
  (0.45 s more), so frame-for-frame matching can be off by up to that.
- **An unclosed recording**: May_2024 HAB2, M8476 (41 GB) ends in zeros,
  with no Inscopix description; `inscopix-sessions` lists it with an error.
- **Recovered copies.** Four other unclosed originals have a copy written by
  Inscopix's recovery tool (`_recovered.isxd`, `_fixed.isxd`) beside them;
  detection uses the copy and says so. Three of those copies report
  62.5 Hz, not nVista's 59.99 Hz: check their frame times against the
  behaviour before trusting them.

## The earlier analysis's outputs, ported (temporary)

Until every helping-behaviour recording is reprocessed through the `cnmfe`
stage, `raphael-imaging port-helping-behaviour` ports what the earlier
analysis produced into the same tree, so analysis can start now. **It is
temporary and will be removed** once the reprocessing is done.

```bash
raphael-imaging --config configs/imaging_1p.yaml inscopix-sessions RAW/Mar_2024 --out raw_Mar_2024.csv
# ... one table per cohort with raw recordings, then:
raphael-imaging --config configs/imaging_1p.yaml port-helping-behaviour \
    --analysis .../Analysis/Calcium-Imaging-Analysis \
    --raw-table raw_*.csv --out OUT          # [--session 8472 HBTD3]
```

It writes `OUT/processed/sub-M{mouse}/day-{dd}/ses-{type}/` with the tag
`src-cnmfehb` (never `src-cnmfe`, so the two can't be confused), and
`OUT/metadata/sessions.csv`, `subjects.csv`:

- `_src-cnmfehb_ophys.nwb`, `_traces.npz`, `_rois.csv`: every CaImAn
  component of the earlier model (`cnmfe.from_caiman_hdf5` also reads the
  components it discarded, with `accepted` False; their detrended trace,
  which CaImAn didn't keep, is NaN), plus the correlation and peak-to-noise
  images.
- `_beh.nwb`: the earlier tracking table (`results.csv`): P_x, P_y (cm),
  H (rad), S (cm/s), the front-body point, its egocentric door angle,
  running bouts; the DeepLabCut pose (pixels of the cropped video), with
  the video linked; the session as one `trials` row with the event times
  (victim insertion, door opening, victim head and body crossing, video
  end) as columns.
- `provenance/port-helping-behaviour.json`: every input's path and sha256,
  and the Raphael that ported it.

Carried over unchanged, and fixed only by reprocessing: the traces of the
6 sessions cut at 32,767 frames (`truncated` in sessions.csv); frame times
at the table's 30 Hz (the recordings ran at 29.994 Hz after binning);
behaviour on the video's frames from its first frame, assumed to start
with the imaging; tracking only up to the victim's head crossing in HBTD
sessions; Feb_2025's days from the table's dates (no raw recordings on the
share, so its habituation days are not counted).

## Test bed

May_2024 HBTD3, mouse 8472 (D6): 211.7 s, 12,701 frames at 59.988 Hz,
1280 x 800, no dropped frames; the helping-behaviour run found 38 accepted
components. Its CaImAn model is
`processed-data/May_2024/HBTD3/cnmfe_model/Moises_B6J_8472_M_06HBT03_2024-06-05-17-13-10_video.hdf5`
on the helping-behaviour share.

First run on MSC (2026-11-25, from the `.isxd`, defaults, 16 CPUs): 7 min
45 s, 24.4 GB peak, 69 components (36 accepted) against the earlier run's
93 (38). 27 of the earlier run's 38 accepted cells are accepted here, with
the same footprints (cosine median 0.994) and traces (C correlation median
0.986). Of the other 11, 9 are found with the same footprint (cosine
0.93–0.99) but rejected: all sit at the evaluation thresholds (SNR 2.1–3.0,
r 0.48–0.71). So detection agrees; the threshold calls differ, as they
will when the input movie differs slightly (binned from the `.isxd`,
rounded, at 29.994 Hz, against Inscopix's TIFF export, truncated, at 30).
Running on the earlier TIFF input separates the input from the pipeline.
