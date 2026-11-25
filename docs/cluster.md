# Running heavy stages elsewhere

Heavy stages (motion correction, segmentation, denoising, spike sorting,
CNMF-E, pose estimation) run on a cluster or another big machine, not on a
laptop. Three pieces make that reproducible:

1. **One environment, everywhere**: `uv.lock` pins every package, including
   the CUDA build of torch, and `uv sync --frozen` installs exactly that.
2. **GPU checks before any work**: a job proves its GPU computes correctly
   before it starts, and says why when it cannot.
3. **Job bundles**: a small folder that describes one run. Make it here,
   copy it anywhere, run it with one command, and copy it back with its
   results, logs and provenance.

Everything site-specific lives in
[configs/cluster.yaml](../src/raphael/configs/cluster.yaml), which is set up for MSC,
the MedUni Wien Scientific Cluster. **Step-by-step guide for MSC:
[msc_cluster.md](msc_cluster.md).**

## What runs where

| on a laptop | on the cluster |
|---|---|
| making job bundles (`raphael-jobs make`), copying them and the data | every stage below on whole recordings |
| unit tests, linting | anything that loads a full movie, recording or video |
| reading results, figures, analysis on processed tables | runs longer than a few minutes |
| short looks at raw data (`--param max_frames=...`), after checking free RAM and disk | |

Memory is what breaks laptops: a stage past the machine's RAM swaps until
the machine stops responding. CaImAn on a 4-minute miniscope session needs
about 20 GB; on an 8 GB laptop it froze the machine, which restarted. The
`cnmfe` stage now refuses to start when its estimate exceeds the machine's
RAM or the job's `--mem-gb`. Large scratch on a laptop goes on an external
drive (`raphael-jobs run ... --work /Volumes/BIG/work/JOB`), never the
internal disk, which macOS also needs for swap.

## The environment

```bash
uv sync --frozen --extra gpu      # GPU stages: suite2p, Kilosort 4.0.16 + torch
uv sync --frozen --extra dev      # tests and linting
```

- `--frozen` installs from `uv.lock` without re-resolving, so every machine
  gets the same versions. If it complains that the lock is out of date,
  someone edited `pyproject.toml` without running `uv lock`: lock and commit
  on the laptop, never on the cluster.
- Python 3.11 is pinned in `.python-version` and uv downloads it if needed,
  so the cluster's system Python does not matter.
- torch is the same version on every platform (`2.14.1`). On Linux it is the
  CUDA 12.6 build (`+cu126`), which runs on NVIDIA drivers 525 and newer and
  bundles its own CUDA runtime, so `module load cuda` is usually unnecessary.
  On macOS it is the PyPI build, which uses Apple-silicon GPUs (MPS).
- Commit `uv.lock` and `.python-version` with any change to dependencies.
- There is one environment for everything. Kilosort 4.0.16 (the Methods'
  version) needs numpy < 2, which pins numpy 1.26, spikeinterface 0.104.9,
  pandas 2.3 and pyarrow < 26 for the whole repo.

Every job run records the `uv.lock` hash, git commit and the full package
list in `logs/env.json`, and refuses to run if the lock differs from the
one the job was made with.

## Checking the GPU

```bash
uv run raphael-jobs gpu-check                    # any GPU (CUDA, then MPS)
uv run raphael-jobs gpu-check --backend cuda --min-memory-gb 16
```

It runs a matrix multiply on the device and compares it with the CPU, so a
GPU that is listed but unusable fails here, not an hour into a job. On
failure it prints the full report and a diagnosis, e.g.:

| message | usual cause |
|---|---|
| CPU-only build | `uv sync` was run without `--extra gpu`, or on an unsupported platform |
| no nvidia-smi | job is on a CPU node, or did not request a GPU |
| CUDA_VISIBLE_DEVICES hides every GPU | scheduler allocated none |
| driver present but CUDA unusable | driver older than 525 |
| failed to compute | GPU too old for this torch build |
| largest free GPU memory is ... | shared GPU, or the job needs a bigger card |

## Job bundles

```
{jobs}/{job_id}/
    job.yaml        stage, inputs (relative paths, sizes, sha256), params,
                    resources, uv.lock hash and git commit at creation
    config/         snapshot of configs the stage reads (--config)
    submit.slurm    scheduler scripts, rendered from configs/cluster.yaml
    submit.sge
    status.json     PENDING -> RUNNING -> DONE | FAILED (with the error)
    logs/           env.json, gpu.json, provenance.json, scheduler output, traceback.txt
    outputs/        results
```

**Data is not copied into bundles.** Raw movies are tens of GB, so they are
copied to the cluster once, under a data root. Job inputs are paths relative
to that root, and each machine says where its root is at run time. The
recorded size and checksum catch a partial or different copy before any
GPU time is spent.

The data root should mirror the laptop's, e.g.:

```
{data}/
    raw/M{mouse}/D{day}/{type}/...     raw movies, as acquired
    dandi000402/...                    test bed
```

### Workflow

```bash
# 1. laptop: make a bundle (commit first, so it is traceable)
uv run raphael-jobs make ~/jobs/smoke-001 --stage smoke \
    --data-root /Volumes/TEMP/InVivoImaging \
    --input sub-17797_ses-7-scan-4_behavior+image+ophys.nwb --no-hash

# 2. copy the bundle (and, once, the data) to the cluster
rsync -a ~/jobs/smoke-001 cluster:/path/to/jobs/

# 3. cluster: install the environment once per checkout, then submit
uv sync --frozen --extra gpu
cd /path/to/jobs/smoke-001 && sbatch submit.slurm      # or: qsub submit.sge

# 4. check progress, and copy results back
uv run raphael-jobs status /path/to/jobs
rsync -a cluster:/path/to/jobs/smoke-001 ~/jobs/
```

Any machine can also run a bundle directly, without a scheduler:

```bash
uv run raphael-jobs run JOB --data-root DATA [--work /local/scratch/JOB]
```

Flags:
- `make --no-hash` records sizes only; hashing a 66 GB movie takes minutes.
- `make --cpu` is for stages that need no GPU.
- `make --gpu-backend any` lets a job accept a Mac GPU, for local testing.
  The default is `cuda`.
- `run --allow-env-mismatch` is an escape hatch. It is recorded in
  `env.json`, so do not use it for results you keep.

### First steps on a new cluster

1. Fill in `paths`, the scheduler section and `setup` in
   [configs/cluster.yaml](../src/raphael/configs/cluster.yaml). Comments there list
   what each site usually needs.
2. Install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`), clone the
   repo, and run `uv sync --frozen --extra gpu`.
3. Run `uv run raphael-jobs gpu-check` on a GPU node, interactively.
4. Run a `smoke` job through the scheduler. It reads the inputs, runs a
   large matmul on the GPU, and writes `outputs/smoke.json` with read speed
   and GPU timing. That tests the whole path from bundle to results.
5. Re-render existing bundles with `raphael-jobs render JOB` if the site
   config changed.

## Provenance that travels with the results

When a stage finishes, the job writes a record **into every processed
session folder it wrote**, so the record stays with the files wherever they
are copied (to Leonardo, to a laptop, to DANDI's staging area):

```
M1/D2/OF1/
    sub-M1_ses-D2_typ-OF1_photometry.nwb
    ...
    provenance/{job_id}.json
```

Each record (`format: raphael-provenance/1`, plain JSON) holds:

| field | what |
|---|---|
| `raphael` | version, release tag and exact commit (and whether the checkout had uncommitted changes) |
| `project` | the project checkout's commit and `uv.lock` checksum |
| `stage`, `params`, `resources` | what ran, with which settings |
| `configs` | each config file's checksum **and full text** |
| `inputs` | every raw file used, with size and sha256 |
| `outputs` | every file in the folder the job wrote, with size and sha256 |
| `environment` | Python, platform, scheduler job ID, every package version |
| `gpu`, `device`, `host` | where it ran |
| `started`, `finished`, `job_made` | when, and when and where the job was made |

So for any processed file, you can check it hasn't changed since the
pipeline wrote it (its sha256 is in `outputs`), and see exactly which raw
files, settings and code produced it. A folder processed several times
(e.g. re-run with a newer Raphael) keeps one record per run.

Every NWB file Raphael writes also names the Raphael that wrote it in NWB's
`source_script` field (e.g. `raphael 0.3.0 (v0.3.0, commit 27595f7…)`), so
the version is visible from any NWB tool even if the file is separated
from its folder. The job's `logs/provenance.json` has the same record
for the whole job.

Files written by scripts outside a job (e.g. `raphael-photometry` on a
laptop) get the NWB stamp but no record: run a job for anything you keep.

## Stages

| stage | does | key params |
|---|---|---|
| `smoke` | reads inputs, GPU matmul; tests a machine | `size` |
| `kilosort4` | a day's Neuropixels recordings (one input folder each) -> bad channels, phase shift, Kilosort 4.0.16 per shank, analysers per session, bombcell and the ISI rule; writes `_clusters.npz` (spikes, curation, metrics, waveforms) and QC per session in the clark2025 tree, plus Kilosort's files for phy | `session_start` (one per recording, required), `session_type` (one per recording), `subject`, `mouse`, `day`, `location`, `format`, `stream`, `kilosort4`, `preprocessing`, `postprocessing`, `curation` (overrides).
| `lfp` (CPU) | each Neuropixels recording -> LFP and theta phase, `_lfp.nwb` per session; size and detail set by `lfp` in configs/ephys.yaml | as `kilosort4` for the metadata; `lfp` (overrides). Make with `--cpu` Step by step: [ephys_walkthrough.md](ephys_walkthrough.md) |
| `suite2p` | one movie -> registration, ROIs, traces, deconvolution; writes processed NWB, traces, ROI table and QC in the clark2025 tree, plus suite2p's own files for its GUI | `fs` and `session_start` (required), `subject`, `um_per_pixel`, `depth_um`, `mouse`, `day`, `session_type`, `suite2p` (setting overrides), `blank_fraction`, `keep_binary`. Step by step: [imaging_walkthrough.md](imaging_walkthrough.md) |
| `photometry` (CPU) | Doric files -> `_photometry.nwb`, `.npz`, `.csv` and QC per session | `session_type` (one per input), `mouse`, `day`, `channels`, `processing`. See [photometry_pipeline.md](photometry_pipeline.md) |
| `cnmfe` (CPU) | one Inscopix `.isxd` (or a binned TIFF) -> binning, NoRMCorre, CaImAn CNMF-E, component evaluation; writes `_src-cnmfe_ophys.nwb`, `_traces.npz` (C), `_rois.csv` and QC; the CaImAn model stays in `work/`. CaImAn runs in its own environment, built on first use | `mouse`, `day`, `session_type`, `subject`; `session_start` and `fs` for a TIFF; `cnmfe` (overrides of imaging_1p.yaml), `max_frames`, `keep_movies`, `skip_memory_check`. See [imaging_1p_pipeline.md](imaging_1p_pipeline.md) and [msc_cluster.md](msc_cluster.md#10-process-an-inscopix-recording-with-cnmf-e) |
| `sync` (CPU) | behaviour CSV + Open Ephys recording -> behaviour on the recording clock (`_beh.csv`) and sync QC; fails if a sync check fails | `mouse`, `day`, `session_type`; `time_column`, `ttl_column`, `ttl_line`, `stream`; `sync` (limits); `waive`. See [sync.md](sync.md) |

```bash
uv run raphael-jobs make ~/jobs/s2p-001 --stage suite2p --data-root DATA \
    --input raw/movie.h5 --config configs/imaging.yaml --gpu-type a100 \
    --param fs=6.3 --param session_start=2019-10-30T10:45 \
    --param mouse=17797 --param day=7 --param session_type=scan4 \
    --param 'suite2p={diameter: [6, 6]}'
```

`outputs/suite2p_summary.json` records ROI counts, run time and peak GPU
memory.

```bash
uv run raphael-jobs make ~/jobs/ks4-M25-D18 --stage kilosort4 --data-root DATA \
    --input raw/M25/D18/OF1 --input raw/M25/D18/VR --input raw/M25/D18/OF2 \
    --config configs/ephys.yaml --gpu-type 1g.5gb --cpus 8 --mem-gb 64 --hours 8 \
    --param 'session_start=[2026-11-03T10:00, 2026-11-03T10:40, 2026-11-03T11:30]' \
    --param 'session_type=[OF1, VR, OF2]' --param mouse=25 --param day=18
```

An input can be a folder (an ephys recording is one). Every file in it is
recorded with its size and checksum, and the run fails if any differ or if
files were added since.
`outputs/kilosort4_summary.json` records unit counts and labels per session,
bad channels, run times and peak GPU memory.

## Adding a stage

A stage is a function registered in
[cluster/jobs.py](../src/raphael/cluster/jobs.py):

```python
@stage('suite2p')
def suite2p(ctx):
    # ctx.inputs   resolved input paths, already verified
    # ctx.device   'cuda' or 'mps', already checked
    # ctx.work     scratch space (node-local if paths.scratch is set)
    # ctx.outputs  write results here
    # ctx.params   from `make --param KEY=VALUE`
    # ctx.config   snapshot of configs passed with `make --config`
    ...
```

Pass `check=f` to `stage` to validate params when the job is made:
`f(params, inputs)` raises `JobError` on anything missing or malformed.

Imaging and ephys stages should write their results into `ctx.outputs` in
the clark2025 tree, through `imaging.nwb` or `ephys.nwb`, so a finished
bundle's `outputs/` can be merged into the processed dataset as it is.
