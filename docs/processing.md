# Processing raw data

How raw recordings become Raphael's processed data. **Start here.** The
pages linked from it give the detail of each pipeline and of the cluster.

Raphael works out what each recording is from its files: you don't name
the modality or the pipeline. It then runs that modality's default
pipeline as one or more job bundles on the cluster, and the outputs make up
the processed tree that analysis reads.

```
server   raphael-data ingest      files new hand-ins into raw/ and writes their plan (metadata/plans/)
laptop   check the plan           one job per recording (per day for Neuropixels), in batches
laptop   raphael-jobs files       copy a batch's raw data to the cluster
cluster  raphael-jobs make-batch  make the batch's jobs; sbatch them
laptop   raphael-jobs collect     copy the outputs into processed/{cohort}/
cluster                           delete the batch's raw copy; next batch
```

**Which machine.** Steps marked *(server)* run on the data server or a
computer with `data/` mounted, as the data manager. Steps marked *(laptop)*
run where `data/` is mounted (the laptop, or wherever Leonardo is mounted).
Steps marked *(cluster)* run in an SSH session on the cluster, which sees
only what has been copied to it: `/Volumes/...` paths don't exist there.

## What Raphael recognises

| on disk | modality | default pipeline (job stages) | details |
|---|---|---|---|
| a folder with Open Ephys `Record Node …/`, or a SpikeGLX run (`*.ap.meta`) | Neuropixels | `kilosort4` (sorting, bombcell curation), then `lfp` | [ephys_pipeline.md](ephys_pipeline.md) |
| a ScanImage TIFF, or a TIFF / HDF5 movie in an `_ophys/` folder | two-photon | `suite2p` | [imaging_pipeline.md](imaging_pipeline.md) |
| an Inscopix `.isxd` movie (with nVista's `session.json` beside it) | one-photon | `cnmfe` (CaImAn CNMF-E) | [imaging_1p_pipeline.md](imaging_1p_pipeline.md) |
| a Doric `.doric` file | fibre photometry | `photometry` | [photometry_pipeline.md](photometry_pipeline.md) |
| a video (`_video`, `_video-top`), when its task names a DeepLabCut model in `metadata/sessions.yaml` | video | `dlc_capture` (GPU), with that model | [behavioural_tracking.md](behavioural_tracking.md#models-on-the-data-server) |

A task without a model (`dlc_model:`) has no DeepLabCut job. Turning pose
into behaviour (cm, heading, speed) and syncing behaviour to the
recording ([sync.md](sync.md)) are separate steps, not planned
automatically yet.

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
- **Raphael's commands where ingest runs** *(server)*:
  `uv tool install --editable /path/to/Raphael`, then `uv tool update-shell`
  and a new terminal if `raphael-data` is "command not found".

## 2. Ingest the new data *(server)*

People hand in sessions by dropping them into their `incoming/{username}/`
folder ([the data README](../templates/data/README.md#adding-data) is their
guide; `incoming/_example/` shows one). Ingest checks a hand-in, files it
into `raw/` and writes its plan.

**What a hand-in needs.** Ingest checks all of this, and files nothing of
a hand-in until all of it holds:

```
incoming/abraun/                                        the folder name is the person
├── 2027-02_dHPC-test_cohort.yaml                       each cohort in the hand-in
├── sub-M20001_subject.yaml                             each mouse in it
├── sub-M20001_procedures.yaml                          each mouse in it ([] before any surgery)
└── sub-M20001_day-01_ses-OF1_2027-02-02_10-15-00/      one folder per session
    ├── Record Node 101/…                               what the rig wrote, not renamed
    ├── top_camera.mp4
    └── sub-M20001_day-01_ses-OF1_session.yaml          optional: notes, weight
```

| what | form | checked |
|---|---|---|
| the person | `incoming/{username}/`, with `{username}` in `metadata/people.yaml` | registered |
| the task | its code in `metadata/sessions.yaml` (`OF` for `ses-OF1`) | registered |
| the project | its code in `metadata/projects.yaml`, given in the cohort file (or a session file) | registered |
| `{cohort}_cohort.yaml` | YAML from `metadata/templates/cohort.yaml`; `cohort:` the same as the file name; a new cohort is named `YYYY-MM_label` (the month it started) | every time |
| `sub-M{mouse}_subject.yaml` | from `metadata/templates/subject.yaml`; required: `mouse` (the same as the file name), `cohort`, `sex` (`M`, `F`, `U` or `O`), `date_of_birth` (`YYYY-MM-DD`), `species` | every time; a mouse's cohort, sex and birth date can't change once in `raw/` |
| `sub-M{mouse}_procedures.yaml` | a YAML list from `metadata/templates/procedures.yaml`, one entry per surgery; just `[]` before the first | every time |
| the session folder | `sub-M{mouse}_day-{dd}_ses-{CODE}{run}_{YYYY-MM-DD}_{HH-MM-SS}`: `day` the mouse's recording dates in order, two digits (a new date after day-03 is `day-04`); `run` 1, 2, … for the same task that day; the date and time the recording started | the time against the recording's own start (within 5 min); the day against the mouse's days in `raw/` and earlier in the hand-in; not already in `raw/` |
| what's in it | what the rig wrote, not renamed: any of the recordings in [What Raphael recognises](#what-raphael-recognises), video and behaviour files. Several recordings of one kind (two fibres, two probes) each carry their procedure `label` in their name, e.g. `fibre_dHPC.doric` | each recording readable; a folder that isn't a recording is a problem |
| `…_session.yaml` | optional, from `metadata/templates/session.yaml`: notes, weight, rig. Blank fields are filled in (start and task from the folder name, experimenter from the person, project from the cohort file) | if present: its task and start agree with the folder name |

Nothing else may be loose in the hand-in (a stray file or an old-style
folder name is a problem). Usernames are the people's server logins, so
whose hand-in it is is never in doubt.

**Run it:**

```bash
cd /path/to/lab/data/incoming/abraun
raphael-data ingest --dry-run            # what it would do; changes nothing
raphael-data ingest
```

**If anything needs clarifying**, nothing of the hand-in is filed: the
list of what to fix is printed. Run ingest again once it's fixed.

**Otherwise it is filed into `raw/`.** The hand-in above becomes:

```
raw/2027-02_dHPC-test/                                  the cohort
├── cohort.yaml                                         was 2027-02_dHPC-test_cohort.yaml
└── sub-M20001/                                         the mouse
    ├── sub-M20001_subject.yaml
    ├── sub-M20001_procedures.yaml
    └── day-01/ses-OF1/                                 the session
        ├── sub-M20001_day-01_ses-OF1_ephys/            was the session folder's Record Node 101/…, insides unchanged
        ├── sub-M20001_day-01_ses-OF1_video.mp4         was top_camera.mp4
        ├── sub-M20001_day-01_ses-OF1_session.yaml      the one handed in, blanks filled in
        └── sub-M20001_day-01_ses-OF1_manifest.tsv      every file: size, sha256, original name
```

The recordings are locked (read-only), the hand-in's folder is emptied,
and `metadata/catalogue.csv` gets a row per session. Each session file has
the fields filled in, and who ingested it and when:

```yaml
weight_g: 23.5                          # from the hand-in
quality: good
notes: cable twisted at 12 min
task: OF                                # filled in: from the folder name
experimenter: [abraun]                  #   whose hand-in it was
project: example-project                #   the cohort file
start: '2027-02-02T10:15:00'            #   the folder name, checked against the recording
ingested: '2026-11-27T13:49:50'
ingested_by: hclark
```

**The plan** for what it filed is written to
`metadata/plans/plan_{date}_{time}_{username}.csv` (the name is printed at
the end of the report, and the catalogue's `plan` column names it for each
session). For the example, one row per job (some columns
left out here; all are described in section 3):

| job_id | batch | stage | modality | inputs | session_type | session_start | ready |
|---|---|---|---|---|---|---|---|
| `kilosort4-b01-M20001-D1` | 1 | kilosort4 | neuropixels | `2027-02_dHPC-test/sub-M20001/day-01/ses-OF1/sub-M20001_day-01_ses-OF1_ephys` | OF1 | 2027-02-02T10:15:00 | True |
| `lfp-b01-M20001-D1` | 1 | lfp | neuropixels | (the same) | OF1 | 2027-02-02T10:15:00 | True |

Inputs are paths under `raw/`. A Neuropixels day is one job: had the
mouse a second session that day (`ses-OF2`), both would be in the same
kilosort4 job, so a unit keeps its id across them. The plan covers only
the sessions that run filed. See [data_server.md](data_server.md) for what
ingest checks and why.

## 3. Check the plan *(laptop)*

The plan from section 2 has one row per job:

- **which**: `job_id`, `batch`, `stage`, `modality`, `inputs` (paths under
  `$RAW`; a Neuropixels day's recordings together, `;`-separated, because
  Kilosort sorts a day as one, so a unit keeps its id across sessions);
- **names and metadata**: `cohort`, `mouse`, `day`, `session_type`, `session_start`,
  `fs`, `sex`, `date_of_birth`, `strain`, `genotype`, `species`;
- **resources**: `gpu`, `gpu_type`, `cpus`, `mem_gb`, `hours` (from the
  table below; for `cnmfe`, from each movie's size);
- `params` (settings changed for this job, as JSON), `ready` (False when a
  start time or frame rate is missing) and `note`.

**Open it and check it.** It is the place to change anything before
jobs are made: fill in a missing `session_start`, raise `mem_gb`, or
override a setting in `params` (e.g. `{"cnmfe": {"quality": {"min_SNR": 2.5}}}`).
Rows that aren't `ready` are skipped when jobs are made.

A batch holds at most 12 recordings (MSC runs 12 CPU jobs at a time; a
Neuropixels day counts as one) and 400 GB of raw data, so the cluster's
shared disk never holds much more at once.

| stage | runs on | default request |
|---|---|---|
| `kilosort4` | CPU | 16 CPUs, 64 GB, 24 h (a first guess: check the first jobs' run time) |
| `lfp` | CPU | 8 CPUs, 32 GB, 4 h |
| `suite2p` | GPU (`a16`) | 8 CPUs, 32 GB, 2 h |
| `cnmfe` | CPU | 16 CPUs; memory and hours from the movie ([msc_cluster.md](msc_cluster.md#10-cnmf-e-on-msc)) |
| `photometry` | CPU | 2 CPUs, 8 GB, 1 h |
| `dlc_capture` | GPU (`a16`) | 8 CPUs, 32 GB, 8 h, one video per job |

**Copy the plan to the cluster *(laptop)*.** The commands from here on use
these names; set them in the laptop's terminal:

```bash
RAW=/path/to/lab/data/raw                 # the raw tree
OUT=/path/to/lab/data                     # processed/ goes here
CLUSTER=<MUW-ID>@s0-l02.hpc.meduniwien.ac.at
PLAN=/path/to/lab/data/metadata/plans/plan_2027-02-05_10-00-00_abraun.csv   # from section 2

rsync -a $PLAN $CLUSTER:jobs/
```

On the cluster, set `PLAN` to the copy:
`PLAN=~/jobs/plan_2027-02-05_10-00-00_abraun.csv`.

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

**DeepLabCut jobs** (`dlc_capture`) also need their model on the cluster.
The models are copied once into `~/models/`, and again after one is added;
the copy costs nothing when nothing changed:

```bash
rsync -a $OUT/processed/_dlcmodels/ $CLUSTER:models/
```

A job checks its model against the checksum in the plan, so a changed or
half-copied model is refused rather than used.

**4.2 Make its jobs *(cluster)*.**

```bash
cd ~/Raphael        # your checkout of the code (or your project's)
uv run --frozen --no-sync raphael-jobs make-batch $PLAN --batch 1 \
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
job with nothing new to copy (already copied, not finished, or failed).
On the data server it then rebuilds `metadata/catalogue.csv`, so the
catalogue shows these sessions as processed.

Look at a few QC figures (`qc/` in each session folder) before deleting
anything. The job folders on the cluster keep the logs, the provenance and
the stage's working files.

**4.6 Delete the batch's raw copy *(cluster)*.** Only after 4.5:

```bash
cd ~/Raphael
uv run --frozen --no-sync raphael-jobs files $PLAN --batch 1 --data-root ~/data \
    | (cd ~/data && xargs -d '\n' rm -v)
```

Then the next batch, from 4.1. Copying batch 2 while batch 1 runs is fine
as long as both fit on the disk (2 × 400 GB at most).

## What you get

```
processed/{cohort}/sub-M{mouse}/day-{dd}/ses-{type}/
    sub-…_srt-kilosort4_clusters.npz            spikes, units, curation      ephys_pipeline.md
    sub-…_srt-kilosort4_bombcell_curation.json  bombcell labels
    sub-…_lfp.nwb                               LFP and theta phase
    sub-…_src-suite2p_ophys.nwb, _traces.npz, _rois.csv                     imaging_pipeline.md
    sub-…_src-cnmfe_ophys.nwb, _traces.npz, _rois.csv                       imaging_1p_pipeline.md
    sub-…_photometry.nwb, .npz, .csv                                         photometry_pipeline.md
    sub-…_video-top_src-dlc-{model}_capture.npz  DeepLabCut capture          behavioural_tracking.md
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
Ingest's plan is checked in the ingest tests, on example hand-ins.
**Not yet run:** a whole batch through steps 4.1–4.6. Check the first
batch's outputs before starting the second.
