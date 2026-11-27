# Using Raphael from a project

The reference for project code that calls Raphael. To set up a new project, start with
[getting_started.md](getting_started.md); [starter_project](https://github.com/HDClark94/starter_project)
is a complete, tested example of everything below.

## How a project uses Raphael

A project repository uses Raphael in four ways, all from its own checkout:

| | How | Example |
|---|---|---|
| **A. Import** core functions in its analysis code | `from raphael import population` | load spikes, score tuning |
| **B. Run** Raphael's processing stages on its raw data | `raphael-data ingest`, then the plan it writes ([processing.md](processing.md)) | spike-sort every day |
| **C. Add** its own stages to the same job runner | `@stage('...')` + an entry point | run its analysis on the cluster |
| **D. Override** Raphael's settings | `configs/ephys.yaml`, `configs/cluster.yaml` | its lab, people, procedures |

The dependency runs one way: **the project imports Raphael, and Raphael
never imports a project**. Raphael and the project share data only through
the processed tree (below). That is what lets several projects use one core.

[starter_project](https://github.com/HDClark94/starter_project) is a complete,
working project laid out this way, with tests; start a new one from it with **Use this template**.

### Project layout

```
my-project/
├── pyproject.toml            depends on raphael; names its own stages
├── uv.lock                   the one environment, Raphael included
├── configs/                  its copies of ephys.yaml / imaging.yaml (and cluster.yaml)
├── src/my_project/
│   ├── arena.py              apparatus: arena size, bin_config()
│   ├── behaviour.py          loads its behaviour into pynapple objects
│   ├── analysis.py           downstream analysis, calling Raphael
│   └── stages.py             its own raphael-jobs stages (optional)
└── scripts/
    └── figure_tuning.py      one script per figure
```

```toml
# my-project/pyproject.toml
[project]
name = "my_project"
requires-python = "==3.11.*"
dependencies = ["raphael @ git+https://github.com/CogEmLab/Raphael@v0.1.0"]

[project.entry-points."raphael.stages"]
my_project = "my_project.stages"
```

### A. Import core functions

What the project's own code writes is behaviour and apparatus. Everything
neural comes from Raphael. From the example's
[`analysis.py`](https://github.com/HDClark94/starter_project/blob/main/src/starter_project/analysis.py):

```python
from raphael import population
from raphael.ephys.layout import Session
from raphael.tuning_scores import (compute_hd_mean_vector_length,
                                   compute_spatial_information,
                                   compute_speed_correlation)

from my_project.arena import bin_config          # the project's arena
from my_project.behaviour import load_behaviour  # the project's behaviour


def score_session(root, session, sorter='kilosort4', n_shuffles=100, alpha=0.05):
    # Raphael: accepted units from the processed tree, whichever sorter made them.
    pop = population.load(session.folder(root) / f'{session.stem}_srt-{sorter}').accepted()
    # The project: a dict of pynapple objects, P_x, P_y, S, H and `moving`.
    beh = load_behaviour(root, session)
    bins = bin_config()

    rows = []
    for unit in pop.ids:
        spikes = pop.data[unit]
        si, = compute_spatial_information(beh, bins, spikes, unit, alpha, n_shuffles)
        hd, = compute_hd_mean_vector_length(beh, bins, spikes, unit, alpha, n_shuffles)
        sc, = compute_speed_correlation(beh, spikes, unit, alpha, n_shuffles)
        rows.append({'unit': unit, 'spatial_information': si['spatial_information'],
                     'si_sig': si['sig'], 'hd_sig': hd['sig'], 'speed_sig': sc['sig']})
    return pd.DataFrame(rows)


score_session('/data/processed', Session(mouse='25', day='18', session_type='OF1'))
```

A figure script then calls the analysis and plots, and is run from the project:

```bash
uv run python scripts/figure_tuning.py /data/processed M25 D18 OF1 --out figures/
```

What to import for what:

| Module | Use it for |
|---|---|
| `raphael.population` | `load(prefix)` spikes (`_clusters.npz`) or dF/F (`_traces.npz` + `_rois.csv`) as one `Population`; `.accepted()`, `.select(ids)`, `.binned(dt)` → time × neuron |
| `raphael.ephys.layout.Session` / `raphael.imaging.layout.Session` | every file name in the processed tree: `.folder(root)`, `.clusters(root, sorter)`, `.lfp(root)`, `.traces(root, pipeline)`, `.rois(...)`, `.beh(root)` |
| `raphael.tuning_scores` | spatial information, HD mean vector length, speed correlation, each with a shuffled null; you pass `bin_config` |
| `raphael.plots` | standard plots from processed data: `rate_map`, `tuning_curve`, `head_direction`, `phase_histogram`, `peth`, `raster`, `traces`, `autocorrelogram`, `isi_histogram`, `score_histogram` ([plots.md](plots.md)) |
| `raphael.style` | the figure style (`use()`), sizes, colours, panel letters, `save()` ([style.md](style.md)) |
| `raphael.util` | `gaussian_filter_nan`, `interpolate_nans`, `wrap` |
| `raphael.ephys.nwb` | `read_waveforms`, and writing `_clusters.npz` / NWB from your own sorting |
| `raphael.ephys.lfp` | `extract`, `theta_phase` on a spikeinterface recording |
| `raphael.ephys.sorting` | reading SpikeGLX / Open Ephys recordings, units tables, curation (what the `kilosort4` stage calls) |
| `raphael.photometry.nwb` | `load(prefix)` a processed photometry session: `.dff(channel)`, `.zscore(channel)`, `.channels` |
| `raphael.photometry.doric` | `read(path)` a Doric file: channels, metadata, `ttl_times` |
| `raphael.imaging.traces` | `dff`, `baseline`, `neuropil_corrected` |
| `raphael.imaging.nwb` | writing behaviour and ophys NWB in the lab's form |

Tuning scores take the binning as `bin_config`, because arenas differ
between projects. Each key is a variable (or `('P_x', 'P_y')`) mapping to
`{'dim', 'bounds', 'num_bins', 'smooth_sigma'}`; see the example's
[`arena.py`](https://github.com/HDClark94/starter_project/blob/main/src/starter_project/arena.py).

### B. Process raw data with Raphael's stages

From the project checkout, so the job records the project's commit and
`uv.lock`, and with the project's settings:

```bash
cd my-project
uv run raphael-jobs make ~/jobs/sort-M25-D18 --stage kilosort4 \
    --data-root /data --input raw/M25_D18_OF1 --input raw/M25_D18_OF2 \
    --param mouse=25 --param day=18 --param "session_type=[OF1, OF2]" \
    --param "session_start=[2026-11-03T14:05, 2026-11-03T14:50]" \
    --config configs/ephys.yaml
uv run raphael-jobs run ~/jobs/sort-M25-D18 --data-root /data    # or on the cluster
```

The outputs land in the processed tree that part A reads. The stages and
the tree are described under [Processing raw data](#processing-raw-data).

### C. Add the project's own stages

A project stage is a function decorated with Raphael's `stage`. It is
checked when the job is made, then run with the same bundles, checksums,
GPU checks and provenance as the core stages. From the example's
[`stages.py`](https://github.com/HDClark94/starter_project/blob/main/src/starter_project/stages.py):

```python
from raphael.cluster.jobs import JobError, stage
from raphael.ephys.layout import Session

from my_project.analysis import score_session


def _check(params, inputs):            # runs at `make`, before any compute
    for key in ('mouse', 'day', 'session_type'):
        if key not in params:
            raise JobError(f'tuning_scores needs --param {key}=...')


@stage('tuning_scores', check=_check)
def tuning_scores_stage(ctx):          # ctx: inputs, params, work, outputs, device
    p = ctx.params
    session = Session(str(p['mouse']), str(p['day']), str(p['session_type']))
    table = score_session(ctx.inputs[0].parents[2], session)
    table.to_csv(ctx.outputs / f'{session.stem}_tuning_scores.csv', index=False)
```

Name the module under `[project.entry-points."raphael.stages"]` (see the
pyproject above), run `uv sync`, and it is a stage:

```bash
uv run raphael-jobs make ~/jobs/scores-M25-D18 --stage tuning_scores --cpu \
    --data-root /data --input processed/M25/D18/OF1 \
    --param mouse=25 --param day=18 --param session_type=OF1
```

social-manifolds registers its DeepLabCut capture and training stages this way.

### D. Override settings

- `configs/ephys.yaml`, `configs/imaging.yaml`: copy them from
  [src/raphael/configs/](../src/raphael/configs/) and fill in the NWB metadata
  (lab, experimenters, procedures, keywords), which Raphael leaves as
  `TODO`. Pass them with `--config`; each job snapshots the file it ran with.
- `configs/cluster.yaml`: only if the project runs somewhere other than MSC.
  `raphael-jobs` picks it up automatically.

### Promoting code into Raphael

When a second project needs something from the first, move it into
Raphael: open a pull request with its tests, release a new version, then
update both projects to it and delete their copies. Don't import one
project from another.

## Processing raw data

Heavy stages run as self-contained job bundles: made on a laptop, run on a
cluster node, a GPU workstation or Colab, with inputs checksummed and the
environment recorded. Run these from the **project** checkout, so the
project's `uv.lock` and commit are what gets recorded.

```bash
uv run raphael-jobs gpu-check
uv run raphael-jobs make JOB --stage kilosort4 --data-root DATA \
    --input raw/M25_D18_OF1 --param session_start=2026-11-03T14:05 \
    --config configs/ephys.yaml
uv run raphael-jobs run JOB --data-root DATA      # or sbatch JOB/submit.slurm
uv run raphael-jobs status JOBS
```

| Stage | What it does | Docs |
|---|---|---|
| `smoke` | proves a machine reads inputs, computes on its GPU and writes back | [cluster.md](cluster.md) |
| `kilosort4` | a day's Neuropixels 2.0 recordings → Kilosort 4.0.16 per shank → bombcell curation → `_clusters.npz`, optional `_ecephys.nwb`, QC | [ephys_pipeline.md](ephys_pipeline.md), [ephys_walkthrough.md](ephys_walkthrough.md) |
| `lfp` | LFP (1 kHz, depth-averaged blocks) and theta phase → `_lfp.nwb` (CPU job) | [ephys_pipeline.md](ephys_pipeline.md) |
| `photometry` | Doric fibre photometry → `_photometry.nwb`, `.npz`, `.csv`, QC (CPU job) | [photometry_pipeline.md](photometry_pipeline.md) |
| `suite2p` | two-photon movie → motion correction, ROIs, dF/F → `_ophys.nwb`, traces, ROI table, QC | [imaging_pipeline.md](imaging_pipeline.md), [imaging_walkthrough.md](imaging_walkthrough.md) |

`raphael-imaging from-dandi` re-exports a public DANDI 000402 (MICrONS) file
into the processed tree, as an end-to-end test of the imaging output, and
`raphael-imaging validate` runs NWB Inspector over a tree.

Every output lands in one tree, the same for every modality and project:

```
{root}/sub-M{mouse}/day-{dd}/ses-{session_type}/
    sub-M{mouse}_day-{dd}_ses-{type}_beh.nwb            behaviour (project-written)
    sub-..._srt-kilosort4_clusters.npz                    spikes, TsGroup + unit table
    sub-..._lfp.nwb                                       LFP and theta phase
    sub-..._src-suite2p_ophys.nwb / _traces.npz / _rois.csv
    qc/
```

### Cluster logins and provenance

Settings, cluster site and project stages are covered in
[Calling Raphael from another repo](#how-a-project-uses-raphael).
Each person sets their cluster login once with
`uv run raphael-jobs cluster-user <MUW-ID>`. Personal overrides go in
`~/.config/raphael/cluster.yaml`. Each run records the project's commit and
`uv.lock` hash, plus Raphael's version and commit, in `logs/env.json`.

## Coding agents (Claude Code)

A coding agent working in a project can't see what Raphael already does,
so it may write its own rate maps, binning or plotting, and the labs' code
drifts apart again. Raphael ships a guide for agents to prevent that.

- **`uv run python -m raphael`** prints the guide (`raphael/AGENTS.md`,
  inside the package): the rules (use Raphael's function, never copy and
  modify it, flag general code as a candidate for Raphael), and a table
  of which Raphael tool to use for each task.
- **`uv run python -m raphael api [module]`** lists every public function
  and class, each with its signature and a one-line description.
- **`uv run python -m raphael path`** prints where the installed source is.

All three come from the installed package, so they describe exactly the
Raphael version the project pins. A test checks that every function the
guide names exists, and that every module is in it.

In a project, point the agent at them from `CLAUDE.md` (starter_project
does this):

```markdown
Before writing any processing, loading, analysis or plotting code, run
`uv run python -m raphael` and follow it: use Raphael's functions rather
than writing your own. `uv run python -m raphael api <module>` lists what
a module offers.
```

When you add something to Raphael, add a row to `src/raphael/AGENTS.md` too.

## Command reference

| Command | What it does |
|---|---|
| `uv sync --frozen [--extra gpu]` | install exactly what `uv.lock` lists |
| `uv add "raphael[gpu] @ git+https://github.com/CogEmLab/Raphael" --tag vX.Y.Z` | pin, or move to, a Raphael release |
| `uv add --editable ../Raphael` | use a local Raphael checkout while developing |
| `uv run raphael-jobs cluster-user [LOGIN]` | save or show your cluster login |
| `uv run raphael-jobs gpu-check` | prove a GPU computes on this machine |
| `uv run raphael-jobs make JOB --stage STAGE --data-root ROOT --input REL ...` | create a job bundle |
| `uv run raphael-jobs render JOB` | rewrite a bundle's submit scripts after a config change |
| `uv run raphael-jobs run JOB --data-root ROOT` | run a bundle here (or `sbatch JOB/submit.slurm`) |
| `uv run raphael-jobs status JOBS` | state of every bundle in a folder |
| `uv run raphael-photometry inspect FILE.doric` | channels and metadata of a Doric recording |
| `uv run raphael-photometry process FILE.doric --out ROOT ...` | process one photometry session |
| `uv run raphael-imaging validate ROOT` | NWB Inspector over a processed tree |
| `uv run pytest` | Raphael's tests (from a Raphael checkout) |

