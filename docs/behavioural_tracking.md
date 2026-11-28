# Behavioural tracking: DeepLabCut and VAME

`raphael.behavioural_tracking` takes videos to pose (DeepLabCut), then to
the behaviour the rest of Raphael uses, and to behavioural motifs (VAME).
It was developed in social-manifolds for tracking two interacting mice, and
moved here so every project tracks behaviour the same way.

```
videos ──► DeepLabCut model(s) ──► pose (px) ──► arena cm ──► behaviour: P_x, P_y, H, S, moving
                │                                    │           (tuning scores, plots, ml)
     several models, one split               _src-dlc_pose.h5
     multi-peak capture, sweeps                      │
     two-animal identity fixes                       ▼
                                          VAME ──► motifs per frame ──► _src-vame-…_motifs.csv
```

## Install

| What | How |
|---|---|
| DeepLabCut 3.0.2 (PyTorch) | the `tracking` extra: `uv sync --extra tracking` in Raphael, or `"raphael[tracking]"` in a project |
| VAME 0.13 | nothing to do: it runs in its own environment, created on first use (see [VAME](#vame)) |
| ffmpeg (video preparation) | comes with the tracking extra (imageio-ffmpeg), or `brew install ffmpeg` / `winget install ffmpeg` |

Settings live in your project: copy
[`src/raphael/configs/tracking.yaml`](../src/raphael/configs/tracking.yaml)
to `configs/tracking.yaml` and edit it. Every command and job stage reads
it, so the choices stay under version control, not in DeepLabCut's own
config.yaml.

Apple-silicon Macs, Linux and Windows are supported. Intel Macs aren't:
torch has had no Intel-Mac builds since version 2.2.

## Pose with DeepLabCut

### From labels you already have: several models on one split

Every model in the settings' `models` becomes a shuffle of one DeepLabCut
project. They all use the same labelled frames and the same train/test
split, so differences between them come from the models alone.

```bash
raphael-tracking build                 # project from project.labels, plus the split
raphael-tracking shuffles              # one shuffle per model (run where you'll train)
raphael-tracking train bu_resnet50     # resumes from the newest snapshot
raphael-tracking evaluate
```

- **The split holds out whole recording days, per rig.** DeepLabCut's
  default split is random over labelled frames. That puts near-identical
  frames on both sides, so it measures memorisation. With days held out,
  the test error estimates how the model will do on unlabelled sessions.
  The day and rig come from each video's name: set `videos.day_pattern`
  and `videos.rig_pattern`.
- **A benchmark model** is an already-trained model, for example a
  colleague's. Set it in `benchmark`, then `raphael-tracking benchmark`
  registers it as one more shuffle on the same split. New top-down models
  can reuse its training recipe (`recipe: benchmark`) and its trained
  detector (`detector: benchmark`), so they don't each spend hours
  retraining one. If the benchmark was trained on a random split of these
  same frames, its score on this split is optimistic: compare on unlabelled
  sessions instead.
- **Training on Colab:** `raphael-tracking package` zips the project. On
  Colab, install Raphael with pip, unzip, then run `shuffles` and `train`.
- **Training on the cluster:** use the `dlc_train` job stage (below).

### From scratch

```bash
raphael-tracking create videos/*.mp4   # DLC project over these videos (settings: dlc)
raphael-tracking extract videos/*.mp4  # frames to label, chosen by k-means on appearance
raphael-tracking label                 # DeepLabCut's labelling GUI
raphael-tracking dataset split.csv     # training set; split.csv: video_id, split (train/test)
raphael-tracking analyze videos/*.mp4  # track new videos once trained
```

`raphael.behavioural_tracking.videos` makes the split
(`assign_split`, by day and rig), and crops and transcodes raw videos
(`transcode`). It works through a zip one video at a time, so free disk
only needs room for the largest single video.

### Tracking videos: capture once, decide thresholds later

```bash
raphael-tracking capture MODEL --folder --out captures/     # or from videos.zip
raphael-tracking sweep captures/                            # re-track at every threshold
```

- **The capture keeps more than DeepLabCut does.** `multipeak` runs the
  model once at permissive cutoffs. Where DeepLabCut keeps only the
  argmax of each bodypart heatmap, it keeps the top few peaks. Peak 0
  always equals DeepLabCut's own prediction, and a test checks this
  against DeepLabCut's real predictor.
- **The sweep needs no video.** It re-tracks the capture at every
  likelihood and detector threshold, using DeepLabCut's own tracker and
  stitcher.
- **Each threshold pair is scored:** how much pose survives, how often both
  animals are present, how fragmented the tracks are, and how often
  identities look swapped.

In tests on synthetic, perfectly regular animals, DeepLabCut 3.0.2's
`ellipse` tracker started a new tracklet every frame, while `box` tracked
correctly. Check the sweep's `n_tracklets` on your real captures, and set
`sweep.track_method` to whichever links your animals.

### Two interacting animals: identity fixes

When two animals touch, DeepLabCut can give one animal a bodypart that
belongs to the other. The true point is often the second or third heatmap
peak. Two pieces fix this:

- **`skeleton`** learns what a real skeleton looks like from the training
  labels: a Gaussian over log pair distances, calibrated so skeletons with
  different parts missing are comparable.
- **`identity`** reassigns the captured peaks, frame by frame, so each
  animal's skeleton is plausible and its track stays continuous. It
  rejects a skeleton it can't make plausible, rather than guessing.

```bash
raphael-tracking skeleton                          # fit on training-split labels
raphael-tracking capture-labels MODEL --out id/    # capture the held-out labelled frames
raphael-tracking identity-eval id/*_frames.npz     # DLC vs corrected, on labels
raphael-tracking identity captures/MODEL/          # correct session captures
raphael-tracking identity-tune folds/              # choose restraint by leave-one-fold-out
```

The correction is written for two animals. Its settings are in
`identity`. The method and its evaluation are described in
social-manifolds' docs/dlc_identity.md.

## From pose to behaviour and the processed tree

```python
from raphael.behavioural_tracking import io

pose = io.load_dlc('session…DLC_…_el.h5', fps=30, arena_cm=(100, 100))  # cm, landmarks dropped
behaviour = io.to_behaviour(pose)          # P_x, P_y, H (heading), S (speed), moving
io.write_pose(pose, session, processed_root)   # …/sub-M1_day-01_ses-OF1_src-dlc_pose.h5
```

- **Pixels to cm:** each video gets its own mapping, from the four corner
  landmarks DeepLabCut tracks (unique bodyparts, `arena.corners`).
  Optionally, a reference landmark (`arena.reference`, e.g. a door) puts
  its wall at x = 0.
- **The behaviour:** `to_behaviour` gives the `P_x`, `P_y`, `H`, `S` and
  `moving` that `raphael.tuning_scores`, `raphael.plots` and `raphael.ml`
  read.
- **The same from the command line:** `raphael-tracking pose POSE.h5
  --fps 30 --out ROOT --mouse 1 --day 1 --session-type OF1`.

## VAME

VAME learns behavioural motifs from pose: short, recurring movement
patterns, labelled frame by frame. It then groups the motifs into
communities.

```bash
raphael-tracking vame pose/*.h5 --work vame/
```

```python
from raphael.behavioural_tracking import vame
summary = vame.run(pose_files, 'vame/', settings['vame'])
for f in summary['motif_files']:            # frame, time, motif, community per session
    vame.write_motifs(f['path'], session, processed_root, f['segmentation'])
```

**Why VAME runs separately.** Every VAME release pins an exact, older
pynwb than Raphael's NWB checks need, so the two can't share one
environment. VAME instead runs in an environment of its own, locked in
`behavioural_tracking/vame_env/uv.lock` (VAME 0.13). It's created on first
use in `~/.cache/raphael/vame-env` (set `RAPHAEL_VAME_ENV` to move it).
Raphael runs VAME there with `uv run --frozen` and reads back the results.
The summary records the lockfile's checksum, so you know exactly which
VAME ran. Raphael fixes two problems in VAME's own requirements in that
environment:

- **matplotlib:** VAME 0.13 needs 3.9 or newer, but its code breaks on 3.10.
- **netCDF4:** builds newer than 1.7.4 need macOS 15.

**The steps, in order:**
1. **init:** a VAME project from the pose files.
2. **preprocess:** low-confidence points dropped, egocentric alignment,
   outliers removed, smoothing.
3. **trainset**, then **train** (the recurrent variational autoencoder),
   then **evaluate**.
4. **segment:** a motif for every frame, with an HMM and/or k-means.
5. **community:** motifs grouped into communities across sessions.
6. **export:** one CSV per session (frame, time, motif, community).

The project in `--work` is reused, so later steps can be re-run on their
own (`--steps segment community export`).

**Things Raphael handles for you:**
- **Motif times line up with the video.** VAME drops `time_window // 2`
  frames at each end of a session, and the exported frame numbers and
  times account for this.
- **Most points survive.** VAME's default confidence cutoff (0.99)
  discards most DeepLabCut points, so Raphael's default is 0.6.
- **No silently empty runs.** VAME saves a model only once its KL weight
  has ramped up to 1 (after `kl_start + annealtime` epochs). Raphael
  refuses a `max_epochs` too short to reach it.
- **Older Macs work.** Before macOS 15, torch's Apple-GPU code for VAME's
  recurrent layers aborts the process. `device: auto` uses the CPU on
  those Macs. On the cluster, VAME uses CUDA.
- **A failed segmentation says why.** If the HMM fit fails on a degenerate
  latent space (a model that hasn't trained enough), the error says so and
  suggests training longer or using k-means.

## Job stages (cluster)

| Stage | Inputs | What |
|---|---|---|
| `dlc_train` | the DLC project folder, [the benchmark model folder] | trains one model (`--param model=…`). With `--param fold=k`, one fold of a k-fold day-level cross-validation, scored before and after identity correction |
| `dlc_capture` | a model folder, [skeleton.json], [a zip of benchmark `_el.csv`], videos… | multi-peak capture of each video (resumable). With a skeleton model, also identity correction and QC frames. From a plan: the videos only, the model named by `--param model=…` (below) |
| `vame` | pose files, one per session | VAME end to end in its own environment; motif CSVs and the summary |

Give every stage your settings with `--config configs/tracking.yaml`.
Each run writes a provenance record, like every Raphael job.

## Models on the data server

A lab's trained models live in `data/processed/_dlcmodels/{name}/`: the
model folder as DeepLabCut wrote it (`train/pytorch_config.yaml` and its
snapshots) plus a `model.yaml` with the `snapshot` to use and, optionally,
`detector_snapshot`, `skeleton` (a `skeleton.json` in the folder, for
two-animal identity fixes), `bodyparts`, `trained_by`, `trained_on` and
`description`. The folder's README on the server shows one.

**A task names its model** in `metadata/sessions.yaml`:

```yaml
OF:
  name: open field
  dlc_model: openfield-top-v2          # every video of an OF session
HBT:
  dlc_model: {top: hbt-top-v1}         # by camera label (_video-top); other cameras none
```

An unlabelled `_video` (the session's only camera) takes the string, or a
mapping's only entry. A task without `dlc_model` has no DeepLabCut job.

**Ingest plans the jobs.** For each video of a filed session whose task
names a model, its plan gets one GPU `dlc_capture` row, after the
recordings' batches. The row's params name the model, its snapshot and
skeleton, and the **checksum of the model folder**. A task naming a model
that isn't in `_dlcmodels/` (or whose `model.yaml` is incomplete) is a
note in the report, and has no job.

**On the cluster** the models sit in `paths.models` of
`configs/cluster.yaml` (`~/models/`), copied with `rsync -a
data/processed/_dlcmodels/ CLUSTER:models/` (processing.md, step 4.1).
The submit script exports it as `RAPHAEL_MODELS`; the job finds its model
there by name and stops if the folder's checksum differs from the plan's.

**Outputs** go into the session's processed folder, named for the video
and the model, so a re-run with another model sits beside the first:
`sub-M…_day-…_ses-OF1_video-top_src-dlc-openfield-top-v2_capture.npz`,
with, for a model with a skeleton, `…_capture.identity.npz` and a QC
figure in `qc/`. The catalogue's software column names the model
(`deeplabcut 3.0.2 (openfield-top-v2)`), and a session whose task has a
model isn't `processed` until its DeepLabCut jobs have run.

**A model is never changed once used.** A retrained model gets a new name
(`openfield-top-v3`); change the task's `dlc_model` to it. The old folder
stays, so every capture can be traced to the model that made it.

Not yet automatic: choosing tracking thresholds from the capture (the
sweep), pose in cm and the behaviour variables, which follow as above.

## Moving social-manifolds onto this

social-manifolds still has its own copy (`pose/`). To switch it over:

1. Replace `social_manifolds.pose.{arena,dlc_project,multipeak,sweep,
   skeleton,identity,zoo,videos}` imports with
   `raphael.behavioural_tracking.{arena,dlc,multipeak,sweep,skeleton,identity,zoo,videos}`.
2. In its settings, rename `recipe: anna` and `detector: anna` to
   `recipe: benchmark` and `detector: benchmark`, and add
   `benchmark.shuffle: 16` (Anna's original shuffle number).
3. Set `videos.rig_pattern: {bottom: '__bottom__', '23659702':
   '__23659702__'}` to keep its rig names.
4. Pass `arena_cm=ARENA_CM` to `fit_arena` and `dlc_to_arena`.

Its identity analyses, figures and report stay in social-manifolds, because
they belong to that experiment.
