# Processing raw data

How raw recordings become Raphael's processed data. **Start here.** The
pages linked from it give the detail of each pipeline and of the cluster.

Raphael works out what each recording is from its files: you don't name
the modality or the pipeline. It then runs that modality's default
pipeline as one or more job bundles on the cluster, and the outputs make up
the processed tree that analysis reads.

```
laptop   raphael-jobs plan        reads raw/, one job per recording (per day for Neuropixels), in batches
laptop   raphael-jobs files       copy a batch's raw data to the cluster
cluster  raphael-jobs make-batch  make the batch's jobs; sbatch them
laptop   raphael-jobs collect     copy the outputs into processed/{cohort}/
cluster                           delete the batch's raw copy; next batch
```

**Which machine.** Steps marked *(laptop)* run where `data/` is
mounted (the laptop, or wherever Leonardo is mounted).
Steps marked *(cluster)* run in an SSH session on the cluster, which sees
only what has been copied to it: `/Volumes/...` paths don't exist there.

## What Raphael recognises

| on disk | modality | default pipeline (job stages) | details |
|---|---|---|---|
| a folder with Open Ephys `Record Node …/`, or a SpikeGLX run (`*.ap.meta`) | Neuropixels | `kilosort4` (sorting, bombcell curation), then `lfp` | [ephys_pipeline.md](ephys_pipeline.md) |
| a ScanImage TIFF, or a TIFF / HDF5 movie in an `_ophys/` folder | two-photon | `suite2p` | [imaging_pipeline.md](imaging_pipeline.md) |
| an Inscopix `.isxd` movie (with nVista's `session.json` beside it) | one-photon | `cnmfe` (CaImAn CNMF-E) | [imaging_1p_pipeline.md](imaging_1p_pipeline.md) |
| a Doric `.doric` file | fibre photometry | `photometry` | [photometry_pipeline.md](photometry_pipeline.md) |

Behaviour (sync to the recording, [sync.md](sync.md)) and pose
([behavioural_tracking.md](behavioural_tracking.md)) are separate steps,
not picked automatically yet.

From the files Raphael also reads each recording's **start time** (Open
Ephys `sync_messages.txt` or `settings.xml`, SpikeGLX `.meta`, ScanImage,
Inscopix, Doric) and, for movies, the **frame rate**. Not detected: Bruker
two-photon folders (listed, but suite2p can't read them yet), and
ScanImage recordings split over several TIFF files (only the first is
listed). ScanImage's metadata reading has not been checked on our own
recordings yet.

## 1. Before you start (once)

- **The cluster**: an account, the code and its environment installed, your
  login saved: [msc_cluster.md](msc_cluster.md), sections 1–7.
- **Raw data in the raw layout** ([data_server.md](data_server.md#raw)):
  `raw/{cohort}/M{mouse}/D{day}/{type}/sub-M{mouse}_ses-D{day}_typ-{type}_{ephys,ophys,photometry}`.
  The path names the cohort, mouse, day and session type of every output,
  so Raphael needs nothing else to name them. (The cohort level can be
  left out; outputs then go straight into `processed/`.) Raw data in
  another layout works too, with a session table (section 6).
- **The describing files** in the raw tree
  ([data_server.md](data_server.md#the-describing-files)): each session's
  `sub-M…_ses-…_session.yaml` and each mouse's `sub-M…_subject.yaml`.
  Raphael reads them itself: the session's `start` (e.g.
  `2026-11-03T10:00:00`, local time) for recordings whose files don't carry
  it, and the mouse's sex, date of birth, species, strain and genotype for
  the NWB files.

The examples use these names. Set them in the laptop's terminal:

```bash
RAW=/path/to/lab/data/raw                 # the raw tree
OUT=/path/to/lab/data                     # processed/ goes here
PLAN=~/plans/my-project.csv               # the plan: a working file, kept outside data/
CLUSTER=<MUW-ID>@s0-l02.hpc.meduniwien.ac.at
```

## 2. Optional: see what is there *(laptop)*

`plan` (section 3) does this itself; `detect` is a quick look that changes
nothing, e.g. at new data before planning:

```bash
uv run raphael-jobs detect $RAW
```

One line per recording: modality, format, mouse/day/type, size, frame rate
and path, and a `!` line for anything that needs attention (unreadable, no
frame rate, a recovered copy used instead of the original). `--out
recordings.csv` writes the same as a table.

## 3. Plan the jobs *(laptop)*

`plan` finds every recording under `$RAW` (as `detect` does) and plans its
jobs:

```bash
uv run raphael-jobs plan $RAW --out $PLAN
```

The plan has one row per job:

- **which**: `job_id`, `batch`, `stage`, `modality`, `inputs` (paths under
  `$RAW`; a Neuropixels day's recordings together, `;`-separated, because
  Kilosort sorts a day as one, so a unit keeps its id across sessions);
- **names and metadata**: `cohort`, `mouse`, `day`, `session_type`, `session_start`,
  `fs`, `sex`, `date_of_birth`, `strain`, `genotype`, `species`;
- **resources**: `gpu`, `gpu_type`, `cpus`, `mem_gb`, `hours` (from the
  table below; for `cnmfe`, from each movie's size);
- `params` (settings changed for this job, as JSON), `ready` (False when a
  start time or frame rate is missing) and `note`.

The command prints each batch's jobs, raw GB and largest memory request.
**Open the plan and check it.** It is the place to change anything before
jobs are made: fill in a missing `session_start`, raise `mem_gb`, or
override a setting in `params` (e.g. `{"cnmfe": {"quality": {"min_SNR": 2.5}}}`).
Rows that aren't `ready` are skipped when jobs are made.

A batch holds at most 12 recordings (MSC runs 12 CPU jobs at a time; a
Neuropixels day counts as one) and 400 GB of raw data, so the cluster's
shared disk never holds much more at once (`--batch-size`, `--batch-gb`).
`--stage suite2p` plans one stage only; `--configs DIR` uses your project's
copies of the stage configs (`imaging.yaml`, `imaging_1p.yaml`,
`ephys.yaml`, `photometry.yaml`) instead of Raphael's.

| stage | runs on | default request |
|---|---|---|
| `kilosort4` | GPU (`a16`) | 8 CPUs, 64 GB, 8 h |
| `lfp` | CPU | 8 CPUs, 32 GB, 4 h |
| `suite2p` | GPU (`a16`) | 8 CPUs, 32 GB, 2 h |
| `cnmfe` | CPU | 16 CPUs; memory and hours from the movie ([msc_cluster.md](msc_cluster.md#10-process-an-inscopix-recording-with-cnmf-e)) |
| `photometry` | CPU | 2 CPUs, 8 GB, 1 h |

Copy the plan to the cluster *(laptop)*:

```bash
rsync -a $PLAN $CLUSTER:jobs/
```

## 4. Run one batch

Batch 1 is shown; for the others change `--batch 1` and `b01`.

**4.1 Copy its raw data *(laptop)*.**

```bash
uv run raphael-jobs files $PLAN --batch 1 --data-root $RAW > batch01.txt
rsync -a --progress --files-from=batch01.txt $RAW/ $CLUSTER:data/
```

`files` lists every file the batch's jobs read (whole recording folders;
an Inscopix movie with its `.gpio`, `.imu` and `session.json`, without the
large TIFF exports often beside it). They land under `~/data` on the
cluster at the same relative paths, which is where jobs look. An
interrupted copy carries on when the same `rsync` is run again.

**4.2 Make its jobs *(cluster)*.**

```bash
cd ~/Raphael        # your checkout of the code (or your project's)
uv run --frozen --no-sync raphael-jobs make-batch ~/jobs/my-project.csv --batch 1 \
    --data-root ~/data --jobs ~/jobs
```

One bundle per job, `~/jobs/{stage}-b01-M{mouse}-D{day}[-{type}]`, each
with the raw files' checksums, its parameters, its resources and a copy of
its stage config. Jobs already made are left alone, so it is safe to run
again. (On the cluster, `uv run` always gets `--frozen --no-sync`, as in the
job scripts, so it uses the installed environment as it is.)

**4.3 Submit them *(cluster)*.**

```bash
for job in ~/jobs/*-b01-*/; do sbatch "$job/submit.slurm"; done
squeue --me                          # PD = waiting, R = running
```

**4.4 Check them *(cluster)*.** Once they have left `squeue`:

```bash
uv run --frozen --no-sync raphael-jobs status ~/jobs | grep b01      # DONE or FAILED
sacct -S today -u $USER -o JobName%40,State,Elapsed,MaxRSS,ReqMem | grep b01
```

A FAILED job says why in `~/jobs/JOB/status.json` and
`~/jobs/JOB/logs/scheduler-*.out`. To run it again with other settings:
change its row in the plan, copy the plan across, `rm -r ~/jobs/JOB`, and
repeat 4.2 and 4.3 (only the missing job is made).

**4.5 Bring the outputs back *(laptop)*.** Each job's `outputs/M*` is its
part of the processed tree; `collect` copies them into
`processed/{cohort}/`, so they merge into one tree:

```bash
uv run raphael-jobs collect $PLAN --batch 1 --from $CLUSTER:jobs --to $OUT/processed
```

It prints each `rsync` it runs (`--dry-run` prints them only) and names any
job with nothing to copy (not finished, or failed).

Look at a few QC figures (`qc/` in each session folder) before deleting
anything. The job folders on the cluster keep the logs, the provenance and
the stage's working files.

**4.6 Delete the batch's raw copy *(cluster)*.** Only after 4.5:

```bash
cd ~/Raphael
uv run --frozen --no-sync raphael-jobs files ~/jobs/my-project.csv --batch 1 --data-root ~/data \
    | (cd ~/data && xargs -d '\n' rm -v)
```

Then the next batch, from 4.1. Copying batch 2 while batch 1 runs is fine
as long as both fit on the disk (2 × 400 GB at most).

## 5. One recording, by hand *(laptop, or cluster once the data is there)*

For a single session (or one Neuropixels day), skip the plan:

```bash
uv run raphael-jobs make ~/jobs/one-job --data-root $RAW \
    --input M25/D18/OF1/sub-M25_ses-D18_typ-OF1_ephys \
    --input M25/D18/OF2/sub-M25_ses-D18_typ-OF2_ephys
```

`make` reads the inputs and fills in the stage (here `kilosort4`), the
names, the start times (from the recording files or the session files), the
subject, the resources and the stage config. Anything given on the command
line wins: `--stage lfp`, `--param session_start=...`, `--mem-gb 96`.
Then copy the bundle and its raw data to the cluster and submit it
([msc_cluster.md](msc_cluster.md) section 8). Making a job reads every
input for its checksum; for large recordings, copy the data first and make
the job on the cluster.

## 6. Raw data in another layout *(laptop)*

Recordings kept as they were recorded (no `M{mouse}/D{day}/{type}` in the
path) are detected the same way, but have no names. A session table gives
them, by path: columns `input` (path under the raw folder), `mouse`, `day`,
`session_type`, and optionally `cohort`, `session_start` and `fs`.

```bash
uv run raphael-jobs plan $RAW --sessions sessions.csv --subjects subjects.csv --out plan.csv
```

Recordings not in the table are listed as skipped. Everything after the
plan is as above.

**Example: helping behaviour.** The raw miniscope recordings sit in
`helping_behavior/data/raw/dHPC_miniscope/{cohort}/...` as recorded. The
table that names them is the port's `metadata/sessions.csv` (its `cohort`
and `raw_recording` columns give the path, `type` the session type):

```bash
RAW=/Volumes/bigsharesanja/CogEm/Projects/helping_behavior/data/raw/dHPC_miniscope
uv run raphael-jobs plan $RAW \
    --sessions /Volumes/TEMP/helping_behavior/metadata/sessions.csv \
    --subjects /Volumes/TEMP/helping_behavior/metadata/subjects.csv \
    --out /Volumes/TEMP/helping_behavior_raphael/metadata/plan.csv
```

This plans 98 `cnmfe` jobs (6.7 TB of movies) in 19 batches. The processed
tree it builds is `/Volumes/TEMP/helping_behavior_raphael/processed`, one
folder per cohort (`May_2024`, … as the table names them).

## What you get

```
processed/{cohort}/M{mouse}/D{day}/{type}/
    sub-…_srt-kilosort4_clusters.npz            spikes, units, curation      ephys_pipeline.md
    sub-…_srt-kilosort4_bombcell_curation.json  bombcell labels
    sub-…_lfp.nwb                               LFP and theta phase
    sub-…_src-suite2p_ophys.nwb, _traces.npz, _rois.csv                     imaging_pipeline.md
    sub-…_src-cnmfe_ophys.nwb, _traces.npz, _rois.csv                       imaging_1p_pipeline.md
    sub-…_photometry.nwb, .npz, .csv                                         photometry_pipeline.md
    qc/                                         figures to check by eye
    provenance/                                 the job, code version and inputs (sha256)
```

`raphael.population.load` reads the spikes and imaging traces; see
[using_raphael.md](using_raphael.md) for the rest.

## Status

Checked (2026-11-26): detection on the toy dataset (all four modalities in
the raw layout) and on the helping-behaviour miniscope recordings (182 raw
movies); plans, file lists and job bundles in the tests; one `cnmfe` job
run on MSC end to end ([imaging_1p_pipeline.md](imaging_1p_pipeline.md#test-bed)).
**Not yet run:** a whole batch through steps 4.1–4.6. Check the first
batch's outputs before starting the second.
