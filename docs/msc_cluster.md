# Using the MSC GPU cluster

How to run this repo's GPU jobs on **MSC**, the MedUni Wien Scientific
Cluster, starting from nothing. Every step here has been run end to end
(November 2026: a smoke job, then suite2p on a 2000-frame two-photon
movie, both on an NVIDIA A16). Anything untested is marked as such.

The general job system (bundles, GPU checks, the lockfile) is described in
[cluster.md](cluster.md). This page covers the MSC-specific parts.

## At a glance

| | |
|---|---|
| Login nodes | `s0-l02.hpc.meduniwien.ac.at`, `s0-l03.hpc.meduniwien.ac.at` |
| Login | MUW ID and password, over SSH, **VPN required** |
| Scheduler | SLURM, partitions `gpu` and `cpu` |
| Our account | `neurop` |
| GPU type | chosen with **both** `--qos=TYPE` and `--gres=gpu:TYPE:1` |
| Limits | 6 running GPU jobs, 12 running CPU jobs per user; no GPU-hour budget |
| Priority | fair share: the more you have used recently, the further back you queue |
| Software | no modules, no system conda or uv; Apptainer is installed |
| Wiki | https://msc.hpc.meduniwien.ac.at (MUW login, VPN) |
| Support | hpc@meduniwien.ac.at |

## 1. Get access (once)

1. **MSC user.** Access is granted per department account; ours is
   `neurop`. Submit the intranet form "MedUni Wien Scientific Cluster (MSC)
   - User für Mitarbeiter:innen", or "- User für Externe" if you are an
   external researcher affiliated with the unit.
2. You then get an email saying your MSC user exists, plus an **invitation
   to the NVIDIA private registry** (NGC). Accept it at ngc.nvidia.com; the
   registry holds containers for the cluster. You do not need it for the
   steps below.
3. **VPN.** A separate email says when your VPN access is active. Install
   the client by following https://intranet.meduniwien.ac.at/vpn-zugang (on
   macOS this is Cisco Secure Client / AnyConnect, connection "Med Uni
   Wien").

## 2. Log in

Connect the VPN, then:

```bash
ssh <MUW-ID>@s0-l02.hpc.meduniwien.ac.at
```

On first connection SSH asks you to confirm the server's key. For
`s0-l02` the fingerprint is:

```
ED25519 SHA256:xiry285KoTAkt0fU+kFB/KrNVJUkuyw7bcnrq0/V4AY
```

Type `yes` if it matches. If SSH ever warns that this key has *changed*,
stop and contact hpc@meduniwien.ac.at. Your first login creates your home
directory, `/msc/home/<MUW-ID>`.

Useful commands on the login node:

| command | shows |
|---|---|
| `msc_info` | quick help for the cluster |
| `noru` | each node's CPU, memory and GPU use (used/total) and GPU types |
| `nods` | node states |
| `squeue --me` | your jobs |
| `squeue --me --start` | SLURM's estimate of when pending jobs start |
| `scancel JOBID` | cancel a job |

`msc_job` is an interactive menu that builds a job command; press Ctrl-C to
leave it.

## 3. Choosing a GPU

| QOS / GPU type | GPUs | nodes | time limit |
|---|---|---|---|
| `a100` | A100 | s0-n00, n01, n11 | 3 days |
| `a100_long` | A100 | as above | 5 days |
| `a100-sxm4-80gb` | A100 80 GB | s0-n02 | 3 days |
| `h100` | H100 | s0-n15 | 3 days |
| `b200` | B200 | s0-n03 | 3 days (see below) |
| `a30` | A30 | s0-n12 | 3 days |
| `a16` | A16 (16 GB) | s0-n12 | 3 days |
| `3g.20gb` | slice of an A100, 20 GB | s0-n12 | 3 days |
| `1g.5gb` | slice of an A100, 5 GB | s0-n12 | 8 hours |

**Both the QOS and the GRES must name the type**, e.g.:

```bash
#SBATCH --partition=gpu
#SBATCH --account=neurop
#SBATCH --qos=a16
#SBATCH --gres=gpu:a16:1
```

A plain `--gres=gpu:1` is rejected with `QOSMaxGRESPerJob`. Each QOS allows
zero of every GPU type except its own.

The big cards (A100, H100, B200) are often fully booked, so check `noru`.
`a16` and the A100 slices are usually free, and **they are enough for
suite2p**. On a 2000 x 248 x 440 movie it peaked at 0.9 GB of GPU memory
and took 48 s on an A16. Queue for an A100 or H100 only for jobs that need
their memory or speed, such as training denoising models.

To move a queued job to another GPU type without remaking it:

```bash
scancel JOBID
sbatch --qos=TYPE --gres=gpu:TYPE:1 submit.slurm   # command line overrides the script
```

**B200 is not usable with the current lockfile.** B200s need CUDA 12.8 or
newer; `uv.lock` installs torch built for CUDA 12.6. The node drivers
(580) are new enough for a CUDA 13 build, so moving the lock to CUDA 13 should
fix it. That is untested.

## 4. Storage

- **Home** (`/msc/home/<MUW-ID>`) holds the code, its environment and job
  bundles. It sits on the shared filesystem, which is about **95% full**.
  Do not leave large intermediate files there.
- **Scratch:** new users have no scratch directory. Ask
  hpc@meduniwien.ac.at for one before copying raw data (a recording is
  60-70 GB).

## 5. Rules, limits and priority

Checked November 2026 with `sacctmgr`, `sshare` and `scontrol show config`
on the login node. Limits can change; the commands at the end of this
section show the current values.

### Who pays

MSC is paid for per **department account**, not per user or per job. Units
join by agreeing "financial conditions" with ITSC operations. Users within
an account use the cluster under **fair use**, with no limit on the number
of users. SLURM records no per-job charge and no GPU-hour budget for
`neurop` or its users.

The fee arrangement for our account (flat or usage-based) is not visible
from the cluster. Before running anything unusually large (e.g. weeks of
A100 time), check with the unit's MSC project coordinator
("MSC Projekt Koordinator").

### Limits

| limit | value |
|---|---|
| running GPU jobs per user | **6** (more can be submitted; they wait) |
| running CPU jobs per user | **12** |
| run time per job | set by the QOS: 3 days for most GPU types, 5 days for `a100_long`, 8 hours for `1g.5gb` (section 3) |
| GPU type | one per job, fixed by the QOS |
| GPU or CPU hours | no budget, for the account or the user |

Jobs beyond the running-job limit stay pending until one of yours
finishes. Cancel jobs you no longer need: a forgotten pending job takes a
slot as soon as it starts.

### Priority: fair share

Pending jobs are ordered by SLURM's multifactor priority, and on MSC
**fair share dominates**. Every `neurop` user has an equal share. The more
compute you have used recently, compared with that share, the lower your
priority against colleagues. Usage counts for less over time, with a
**half-life of 14 days**: a heavy week stops costing you priority within a
few weeks.

So nothing stops you from using a lot, but the queue makes you wait longer
for it. A job pending with reason `(Priority)` is normal: other jobs are
ahead of it, or no GPU of its type is free.

### Good practice

- **Use the smallest GPU that does the job.** suite2p needs under 2 GB of
  GPU memory, so an `a16` or `3g.20gb` slice is enough and usually free.
  Keep the A100, H100 and B200 for work that needs them, such as training
  models.
- **Ask for realistic time and memory** (`raphael-jobs make --hours
  --mem-gb --cpus`). Over-asking takes resources others could use and makes
  a job harder to fit.
- **Do not compute on the login nodes.** Installing (`uv sync`), copying
  data and submitting are fine there. For short tests, take a small GPU
  slice interactively instead:
  ```bash
  srun --partition=gpu --account=neurop --qos=1g.5gb --gres=gpu:1g.5gb:1 \
      -t 10 --mem=4G --pty bash
  ```
- **Clean up.** The shared filesystem is about 95% full. The suite2p stage
  deletes its registered movie by default; delete raw movies and finished
  `work/` folders once results are safely back.

### Checking your own usage and priority

```bash
sacctmgr show assoc user=$USER format=account,partition,qos%60,grpjobs  # limits and allowed QOS
sshare -U -l                  # your usage and fair-share factor (1 = best, 0 = worst)
sprio -j JOBID                # why a pending job has the priority it has
squeue --me --start           # estimated start times of your pending jobs
sacctmgr show qos format=name%20,maxwall,maxtres%60   # time and GPU limits per QOS
```

## 6. Install the code and environment (once)

On your laptop, copy the repo to the cluster (skip the local environment
and caches):

```bash
rsync -a --exclude .venv --exclude __pycache__ --exclude .ruff_cache \
    path/to/social-manifolds/ <MUW-ID>@s0-l02.hpc.meduniwien.ac.at:social-manifolds/
```

On the cluster, install uv and the locked environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=$HOME/.cache/uv
cd ~/social-manifolds && uv sync --frozen --extra gpu
mkdir -p ~/data ~/jobs
```

This takes about a minute. The login node has internet access. uv
downloads Python 3.11 and the exact package versions in `uv.lock`, including
`torch 2.14.1+cu126` (about 3 GB). Repeat `uv sync --frozen --extra gpu`
whenever `uv.lock` changes.

## 7. Tell Raphael your login

The shared [cluster config](../src/raphael/configs/cluster.yaml) names no one:
its paths are `/msc/home/{user}/...`, and `{user}` is filled in from your
own setting when the submit scripts are written. Set it once, **on your
laptop**:

```bash
uv run raphael-jobs cluster-user <MUW-ID>     # saved in ~/.config/raphael/cluster.yaml
uv run raphael-jobs cluster-user              # shows what is set
```

With that, the paths become:

```yaml
paths:
  repo: /msc/home/<MUW-ID>/{project}     # the checkout you run from, e.g. social-manifolds
  data: /msc/home/<MUW-ID>/data
  jobs: /msc/home/<MUW-ID>/jobs
env:
  UV_CACHE_DIR: /msc/home/<MUW-ID>/.cache/uv
```

So clone the project into your MSC home under the same folder name as on
your laptop. If something is different for you alone (data on scratch,
another account), add it to `~/.config/raphael/cluster.yaml` under the same
keys as the shared file; it is merged over it:

```yaml
user: <MUW-ID>
paths:
  data: /msc/home/{user}/scratch/data
```

`RAPHAEL_CLUSTER_USER=<MUW-ID>` in the environment overrides the saved
login, e.g. to render a bundle for a colleague. Partition, account and GPU
request are already set for MSC and shared by everyone.

## 8. Run a job

**On your laptop**, make a job bundle. Start with the `smoke` job, which
tests the whole path on a GPU in under a minute:

```bash
uv run raphael-jobs make ~/jobs/smoke-001 --stage smoke --data-root ~/jobs \
    --gpu-type a16 --hours 1 --mem-gb 8 --cpus 2
rsync -a ~/jobs/smoke-001 <MUW-ID>@s0-l02.hpc.meduniwien.ac.at:jobs/
```

`--gpu-type` is the QOS from the table above. It fills in both `--qos` and
`--gres` in the generated `submit.slurm`. For real data, add one
`--input PATH` per file, each relative to `paths.data`, and copy those files
to the same relative paths under the cluster's data folder first.

**On the cluster**, submit the job and watch it:

```bash
cd ~/jobs/smoke-001 && sbatch submit.slurm
squeue --me
```

`PD (Priority)` means the job is waiting behind others. It starts when a
GPU of that type is free.

**When it is gone from `squeue`**, check the result:

```bash
cat ~/jobs/smoke-001/status.json             # DONE or FAILED, with the error
cat ~/jobs/smoke-001/logs/scheduler-*.out    # GPU check report, any errors
cat ~/jobs/smoke-001/outputs/smoke.json      # results
```

A successful smoke job reports `"device": "cuda"` and a matrix multiply
with a relative error around 1e-6. Copy finished jobs back with
`rsync -a <MUW-ID>@s0-l02.hpc.meduniwien.ac.at:jobs/smoke-001 ~/jobs/`.

## 9. Process a movie with suite2p

Copy the movie to the cluster's data folder, keeping its path relative to
your local data root. The `./` marks where the relative part starts:

```bash
rsync -a --relative /local/data/root/./test2000/movie/movie.h5 \
    <MUW-ID>@s0-l02.hpc.meduniwien.ac.at:data/
```

Make the job on your laptop, copy it across and submit it as in step 8:

```bash
uv run raphael-jobs make ~/jobs/s2p-001 --stage suite2p \
    --data-root /local/data/root --input test2000/movie/movie.h5 \
    --config configs/imaging.yaml --gpu-type a16 --cpus 8 --mem-gb 32 --hours 2 \
    --param fs=6.3 --param session_start=2019-10-30T10:45 \
    --param mouse=17797 --param day=7 --param session_type=scan4 \
    --param 'suite2p={diameter: [6, 6]}'
```

`fs` (frame rate) and `session_start` (recording start, local time or with
a UTC offset) are required. Add `um_per_pixel`, `depth_um` and
`subject={age: P84D, sex: M, ...}` for publishable files. `suite2p={...}`
overrides suite2p settings, merged into its defaults. The full processing
walkthrough, with the data at each step, is in
[imaging_walkthrough.md](imaging_walkthrough.md). The job writes to `outputs/`:

- `M{mouse}/D{day}/{type}/..._src-suite2p_ophys.nwb`, `_traces.npz`,
  `_rois.csv` and `qc/plane1_rois.png`: the processed session, clark2025
  layout
- `suite2p/plane0/`: suite2p's own files, which open in its GUI for curation
- `suite2p_summary.json`: ROI counts, run time, peak GPU memory

The job's `work/` folder, which holds the registered movie (about the
size of the raw one), is deleted after the run to save space; pass
`--param keep_binary=true` to keep it.

## 10. Process an Inscopix recording with CNMF-E

The `cnmfe` stage runs on the CPU partition. CaImAn is not in the main
environment: it has its own, locked in Raphael
(`imaging/caiman_env/uv.lock`), which must be built once **on the login
node** (compute nodes may have no internet). Building CaImAn compiles C
code, so check for a compiler first:

```bash
gcc --version          # needed to build CaImAn
cd ~/social-manifolds && uv run python -c \
    "from raphael.imaging import cnmfe; print(cnmfe.ensure_env(quiet=False))"
```

That installs into `~/.cache/raphael/caiman-env` (about 3 GB; set
`RAPHAEL_CAIMAN_ENV` to put it elsewhere) and takes a few minutes. Repeat
it when Raphael's CaImAn lockfile changes. The login node has `gcc` 13.3
(2026-11-25), which is enough. To check that it imports where jobs run:

```bash
srun --partition=cpu --qos=cpu --account=neurop -c 1 --mem=4G -t 0:10:00 \
    ~/.cache/raphael/caiman-env/bin/python -c "import cv2, caiman; print(caiman.__version__)"
```

Copy the raw `.isxd` (the movie only; the `.gpio`, `.imu` and
`session.json` beside it are small and worth keeping too), keeping its
path relative to the raw root:

```bash
rsync -a --relative \
    /Volumes/share/raw/dHPC_miniscope/./May_2024/.../dHPC-SSCW-May2024-06HBT03_20240605-170959 \
    <MUW-ID>@s0-l02.hpc.meduniwien.ac.at:data/
```

Get the mouse's day from the session table
(`raphael-imaging inscopix-sessions RAW_ROOT --out sessions.csv`, on the
laptop: it reads only each file's end), then make the job:

```bash
uv run raphael-jobs make ~/jobs/cnmfe-M8472-D6 --stage cnmfe --cpu \
    --cpus 16 --mem-gb 32 --hours 4 \
    --data-root /Volumes/share/raw/dHPC_miniscope --config configs/imaging_1p.yaml \
    --input May_2024/.../Moises_B6J_8472_M_06HBT03_2024-06-05-17-13-10_video.isxd \
    --param mouse=8472 --param day=6 --param session_type=HBTD3 \
    --param 'subject={sex: M, date_of_birth: 2024-01-23, strain: C57BL/6J}'
```

Copy it across and submit it as in step 8. `--cpu` sends it to the `cpu`
partition. Making the job reads the whole movie once for its checksum (a
26 GB file over the share takes several minutes); `--no-hash` records sizes
only. Once the movie is on the cluster, the job can instead be made there,
with `--data-root ~/data` (and `RAPHAEL_CLUSTER_USER` set), where the
checksum takes seconds; the code to make it from is the cluster's
checkout of the project.

**Memory.** Ask for about 4x the binned movie's float32 size plus a few
GB: the stage estimates it and stops at once, naming the `--mem-gb` to use,
if the job asks for less. The binned movie is frames x 400 x 640 x 2 bytes
(2x2 binning, 30 Hz):

| session | binned movie | `--mem-gb` |
|---|---|---|
| 4 min (7,000 frames) | 3.6 GB | 32 |
| 10 min (18,000 frames) | 9 GB | 80 |
| 18 min (32,000 frames) | 17 GB | 140 |

Measured on MSC (2026-11-25): May_2024 HBTD3 M8472, 6,350 binned frames,
16 CPUs, peaked at 24.4 GB (`sacct -j JOBID -o MaxRSS`) and took 7 min 45 s
(binning 2 min, motion correction 1 min, CNMF-E 1.5 min, evaluation
3 min). Check `MaxRSS` on longer sessions and adjust `MEMORY_FACTOR` in
`raphael.imaging.cnmfe` to follow.

**Disk.** The work folder needs about 3x the binned movie while CaImAn
runs (binned movie, motion-corrected memmap, C-order memmap; CaImAn's
temporary files go there too, not to `~/caiman_data`). With
`paths.scratch` unset that is the job folder in your home, on the 95%-full
shared filesystem: ask for scratch (section 4) before running many
sessions. The movies are deleted at the end; the CaImAn model (`.hdf5`),
the arrays and CaImAn's log stay in `work/caiman/` for comparison and
re-curation.

The job writes `M{mouse}/D{day}/{type}/..._src-cnmfe_ophys.nwb`,
`_traces.npz`, `_rois.csv`, `qc/` and `provenance/` (with
`cnmfe_summary.json`: components, accepted, run times) to `outputs/`. The processing is
described in [imaging_1p_pipeline.md](imaging_1p_pipeline.md).

## 11. Process every session of a project, in batches

For more sessions than one, three `raphael-imaging` commands do what
section 10 does by hand. Each project's raw movies are far bigger than the
free space on the shared filesystem (helping behaviour: 98 sessions, 6.7 TB),
so the sessions go through in **batches**: copy a batch's movies, run its
jobs, bring the outputs back, delete the movies, then the next batch.

**Status (2026-11-26):** the plan, the file lists and the job bundles have
been made and checked for the helping-behaviour sessions; one session has
run on MSC (section 10). A whole batch has not been run yet: check the
first batch's results before starting the second.

The commands below use these names; set them once in each shell:

```bash
# on the laptop
RAW=/Volumes/bigsharesanja/CogEm/Projects/helping_behavior/data/raw/dHPC_miniscope
OUT=/Volumes/TEMP/helping_behavior_raphael        # the processed tree being built
PLAN=$OUT/metadata/cnmfe_plan.csv
```

### Once: the plan (laptop)

The plan is a table with one row per session: its job name and batch, the
movie, the output names (mouse, day, session type), the subject, and the
memory and hours its job asks for, worked out from each movie's own
description (only the end of each file is read, so this takes seconds).

```bash
uv run raphael-imaging cnmfe-plan \
    --sessions /Volumes/TEMP/helping_behavior/metadata/sessions.csv \
    --subjects /Volumes/TEMP/helping_behavior/metadata/subjects.csv \
    --data-root $RAW --out $PLAN
```

It prints each batch's number of sessions, GB of movies and largest
memory request. A batch holds at most 12 sessions (MSC runs 12 CPU jobs at
a time) and 400 GB of movies (`--batch-size`, `--batch-gb`). Open the CSV
and check it: it is the one place to change a session's settings before
its job is made (`mem_gb`, `hours`, or `cnmfe`, a JSON object of
imaging_1p.yaml overrides). Sessions recorded at 40 Hz can't be binned to
30 Hz, so the plan keeps them at their own rate (`note` says so).

`--sessions` takes any table with columns `input` (the movie, relative to
`--data-root`), `mouse`, `day` and `session_type`, and optionally `sex`,
`date_of_birth`, `strain`. The helping-behaviour table above comes from the
port (`cohort` + `raw_recording` stand in for `input`, `type` for
`session_type`).

Copy the plan to the cluster, and update the cluster's checkout of the
code (section 6) if it is older than the plan commands:

```bash
rsync -a $PLAN <MUW-ID>@s0-l02.hpc.meduniwien.ac.at:jobs/
```

### For each batch

Batch 1 is shown; change `--batch 1`, `batch01` and `b01` for the others.

**1. Copy the batch's movies (laptop).** `cnmfe-files` lists each movie and
the small files recorded beside it (`.gpio`, `.imu`, `session.json`), not
the folder, which may hold other large files (Inscopix TIFF exports).

```bash
uv run raphael-imaging cnmfe-files $PLAN --batch 1 --data-root $RAW > batch01.txt
rsync -a --progress --files-from=batch01.txt $RAW/ \
    <MUW-ID>@s0-l02.hpc.meduniwien.ac.at:data/
```

The paths stay relative to `$RAW`, under `~/data` on the cluster, which is
where jobs look (`paths.data`). Time the first batch: the copy (up to
400 GB through the laptop) is likely the slowest part. If it is
interrupted, run the same `rsync` again; it carries on.

**2. Make the batch's jobs (cluster).** Making them here checksums the
movies in seconds (on the laptop it would read each one over the share).
On the cluster, `uv run` always gets `--frozen --no-sync`, as in the job
scripts, so it uses the installed environment as it is.

```bash
cd ~/Raphael
uv run --frozen --no-sync raphael-imaging cnmfe-jobs ~/jobs/cnmfe_plan.csv --batch 1 \
    --data-root ~/data --jobs ~/jobs
```

This makes one bundle per session, `~/jobs/cnmfe-b01-M{mouse}-D{day}-{type}`.
A job already made is left alone, so running it twice is harmless.

**3. Submit them (cluster).**

```bash
for job in ~/jobs/cnmfe-b01-*/; do sbatch "$job/submit.slurm"; done
squeue --me                          # PD = waiting, R = running
```

**4. Check them (cluster).** When they have left `squeue`:

```bash
uv run --frozen --no-sync raphael-jobs status ~/jobs | grep b01          # DONE or FAILED for each
sacct -S today -u $USER -o JobName%40,State,Elapsed,MaxRSS,ReqMem | grep b01
```

For a FAILED job, read `~/jobs/JOB/status.json` and
`~/jobs/JOB/logs/scheduler-*.out`. If it ran out of memory (or the stage
stopped it, naming the `--mem-gb` to use): raise `mem_gb` for that session
in the plan, copy the plan across again, delete the job folder
(`rm -r ~/jobs/JOB`), and repeat steps 2 and 3 for the batch; only the
missing job is made again. Note `MaxRSS` against `ReqMem`: if a long
session used far less than it asked for, lower `MEMORY_FACTOR` (section 10)
before the next plan.

**5. Bring the outputs back (laptop).** Each job's `outputs/` holds its
session's `M{mouse}/D{day}/{type}/` folder, so they merge into one tree:

```bash
rsync -a '<MUW-ID>@s0-l02.hpc.meduniwien.ac.at:jobs/cnmfe-b01-*/outputs/' $OUT/processed/
ls $OUT/processed/M*/D*/*/*_src-cnmfe_ophys.nwb | wc -l      # sessions so far
```

Look at a few `qc/plane1_rois.png` (ROIs on the mean and correlation
images) before deleting anything.

**6. Delete the batch's movies (cluster).** Only once the outputs are on
the laptop:

```bash
cd ~/Raphael
uv run --frozen --no-sync raphael-imaging cnmfe-files ~/jobs/cnmfe_plan.csv --batch 1 --data-root ~/data \
    | (cd ~/data && xargs -d '\n' rm -v)
df -h ~                                                    # space back
```

The binned and motion-corrected movies are deleted by each job; the CaImAn
model, its arrays and log stay in each job's `work/caiman/` (tens of MB).

Then start the next batch at step 1. Steps 1 (copying batch 2) and 3-4
(batch 1 running) can overlap, as long as both batches fit on the disk
together (2 x 400 GB at most).

## 12. Spike-sort Neuropixels recordings with Kilosort 4

Kilosort 4.0.16 (the version in the Methods) is part of the environment
installed in section 6 (`uv sync --frozen --extra gpu`).

Copy each recording folder (SpikeGLX `*_g0` run folder or Open Ephys
record node) to the cluster's data folder, keeping its relative path. A
Neuropixels 2.0 recording is about 80 GB per hour, so ask for scratch
space first (section 4):

```bash
rsync -a --relative /local/data/root/./raw/M25/D18/OF1 \
    <MUW-ID>@s0-l02.hpc.meduniwien.ac.at:data/
```

Make one job per day, with one `--input` per recording, in the order they
were recorded. They are sorted together, so a unit keeps its id across the
day:

```bash
uv run raphael-jobs make ~/jobs/ks4-M25-D18 --stage kilosort4 \
    --data-root /local/data/root \
    --input raw/M25/D18/OF1 --input raw/M25/D18/VR --input raw/M25/D18/OF2 \
    --config configs/ephys.yaml --gpu-type 1g.5gb --cpus 8 --mem-gb 64 --hours 8 \
    --param 'session_start=[2026-11-03T10:00, 2026-11-03T10:40, 2026-11-03T11:30]' \
    --param 'session_type=[OF1, VR, OF2]' --param mouse=25 --param day=18 \
    --param 'subject={age: P90D, sex: F}' --param location=CA1
```

`session_start` and `session_type` take one value per recording and are
required. The processing follows the Methods (see
[ephys_pipeline.md](ephys_pipeline.md)); every setting is in
`configs/ephys.yaml`. The full walkthrough, with the data at each step, is
in [ephys_walkthrough.md](ephys_walkthrough.md).

Kilosort sorts one shank at a time, so a small GPU is enough (see the
walkthrough for measured memory). `1g.5gb` slices allow 8 hours per job;
use `a16` or `3g.20gb` for longer days.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `QOSMaxGRESPerJob` at submission | untyped GPU request; remake the job with `--gpu-type`, or submit with `sbatch --gres=gpu:TYPE:1 submit.slurm` |
| job stays `PD (Priority)` | all GPUs of that type are busy; check `noru`, try another type, or wait |
| `uv not on PATH` in the log | uv is not installed in `~/.local/bin`; redo step 6 |
| `no .venv in ...` | `uv sync --frozen --extra gpu` was not run in the repo on the cluster |
| `uv.lock here differs ...` | the job was made with a different lockfile from the one on the cluster; rsync the repo and re-sync, or remake the job |
| `NO USABLE GPU` / `GpuUnavailable` | see the diagnosis in the log; the table in [cluster.md](cluster.md#checking-the-gpu) lists causes |
| "no kernel image is available" on `b200` | CUDA 12.6 build on a B200; use another GPU type (see section 3) |
| `UserWarning: Sparse invariant checks ...` in a suite2p log | harmless notice from suite2p's internals |
| `needs about N GB of RAM in CaImAn` | the job's `--mem-gb` is below the estimate; remake it with the size the message gives |
| `CaImAn failed` / `uv sync` errors building CaImAn | its environment is not built, or there is no C compiler; see section 10 |
| `ImportError: libGL.so.1` in a CaImAn log | an environment with `opencv-python` instead of the headless build (compute nodes have no libGL); re-run `cnmfe.ensure_env()` from a Raphael checkout whose `caiman_env/uv.lock` has `opencv-python-headless` |
| warning about uncommitted changes | the job cannot be traced to a git commit; fine for tests, commit first for results you keep |
