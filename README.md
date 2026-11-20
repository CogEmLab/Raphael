# Raphael

The shared computational core for neural data analysis in the Mikulovic and
Hangya labs: raw recordings to processed, publishable per-session files,
and the analysis primitives that every project uses on them.

Raphael is a Python package. It is not a project. Each project has its own
repository (e.g. [social-manifolds](../social-manifolds)) that depends on
Raphael, adds its own data conversion, behaviour and figures, and calls
Raphael for everything that would otherwise be copied between projects.

```
            raw Neuropixels / two-photon
                        │
          ┌─────────────▼──────────────┐
          │ Raphael                    │   one implementation, tested,
          │ sort · LFP · ROIs · NWB    │   versioned, shared by everyone
          │ population · tuning scores │
          │ cluster job bundles        │
          └─────────────┬──────────────┘
                        │  processed tree (clark2025 layout)
        ┌───────────────┼────────────────┐
  social-manifolds  next-project   ...         project repos: conversion,
  figures, paper    figures, paper              behaviour, figures, text
```

## What belongs here

Code goes in Raphael when **more than one project would otherwise need it,
and it does not depend on one project's apparatus or task**. Everything else
stays in the project.

| In Raphael | In the project |
|---|---|
| Neuropixels → Kilosort 4 → curated units (`ephys`) | importing a lab's or a collaborator's export |
| LFP and theta phase (`ephys.lfp`) | arena geometry, session tables, trial structure |
| Two-photon → suite2p → ROIs and dF/F (`imaging`) | pose estimation and behaviour classification |
| Writing NWB and the processed tree | task-specific scores and decoders |
| Loading spikes or dF/F the same way (`population`) | figures, statistics for a paper, manuscript |
| Tuning scores with shuffled nulls (`tuning_scores`) | the project's NWB metadata (lab, people, procedures) |
| Cluster job bundles, GPU checks, provenance (`cluster`) | project-only job stages (registered as plug-ins) |

When project code turns out to be general, move it here in its own pull
request, with its tests, and have the project import it from Raphael.

## Getting started

Everything runs through [uv](https://docs.astral.sh/uv/), which installs the
exact environment in a project's `uv.lock`. `uv run CMD` runs a command
inside that environment, so there is no virtualenv to activate.

### 1. Install uv (once per machine)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS, Linux, MSC login node
```

### 2. Set up a project that uses Raphael

From an existing project checkout (e.g. social-manifolds):

```bash
cd ~/Documents/social-manifolds
uv add "raphael[gpu] @ git+https://github.com/HDClark94/Raphael" --tag v0.1.0
uv sync --frozen --extra gpu        # leave out --extra gpu on a laptop that never sorts
```

The project must require **Python 3.11 exactly**, as Raphael does (Kilosort
4.0.16 needs it). `uv init` writes `>=3.11`, which uv then refuses to
resolve, so set this first:

```toml
requires-python = "==3.11.*"
```

`uv add` writes the pin into the project's `pyproject.toml` (as below) and
the exact commit into `uv.lock`; commit both.

```toml
dependencies = ["raphael[gpu]"]

[tool.uv.sources]
raphael = { git = "https://github.com/HDClark94/Raphael", tag = "v0.1.0" }
``` Everyone else on the project then needs only:

```bash
git clone <project>
cd <project>
uv sync --frozen                    # add --extra gpu where Kilosort or suite2p will run
```

Two things the project's `pyproject.toml` must carry:

- **The torch index.** uv reads `[tool.uv]` and `[tool.uv.sources]` only from
  the project being installed, never from its dependencies. A project using
  the `gpu` extra copies the `environments`, `sources` and `index` blocks from
  Raphael's [pyproject.toml](pyproject.toml), so torch comes from the same
  CUDA build on every machine.
- **Its own stages**, if any (see [Project hooks](#project-hooks)).

### 3. Working on Raphael and a project together

Clone Raphael beside the project and point the project at the checkout, so
edits to Raphael take effect at once without a release:

```bash
git clone https://github.com/HDClark94/Raphael ~/Documents/Raphael
cd ~/Documents/social-manifolds
uv add --editable ../Raphael
```

Before committing the project, switch back to a tagged release (step 2), so
the project's results trace to a fixed version of Raphael.

### 4. Per-person cluster set-up (once per laptop)

```bash
uv run raphael-jobs cluster-user <MUW-ID>      # your MSC login, fills {user}
uv run raphael-jobs cluster-user               # shows what is set
```

On MSC itself, clone the project into your home under the same folder name
and run `uv sync --frozen --extra gpu` there once; the full walkthrough is
[docs/msc_cluster.md](docs/msc_cluster.md).

### 5. Check it works

```bash
uv run python -c "import raphael; print(raphael.__version__)"
uv run raphael-jobs gpu-check                  # exit 0 if a GPU computes here
uv run raphael-jobs make ~/jobs/smoke-001 --stage smoke --data-root DATA --input SOME_FILE
uv run raphael-jobs run ~/jobs/smoke-001 --data-root DATA
```

### Command reference

| Command | What it does |
|---|---|
| `uv sync --frozen [--extra gpu]` | install exactly what `uv.lock` lists |
| `uv add "raphael[gpu] @ git+https://github.com/HDClark94/Raphael" --tag vX.Y.Z` | pin, or move to, a Raphael release |
| `uv add --editable ../Raphael` | use a local Raphael checkout while developing |
| `uv run raphael-jobs cluster-user [LOGIN]` | save or show your cluster login |
| `uv run raphael-jobs gpu-check` | prove a GPU computes on this machine |
| `uv run raphael-jobs make JOB --stage STAGE --data-root ROOT --input REL ...` | create a job bundle |
| `uv run raphael-jobs render JOB` | rewrite a bundle's submit scripts after a config change |
| `uv run raphael-jobs run JOB --data-root ROOT` | run a bundle here (or `sbatch JOB/submit.slurm`) |
| `uv run raphael-jobs status JOBS` | state of every bundle in a folder |
| `uv run raphael-imaging validate ROOT` | NWB Inspector over a processed tree |
| `uv run pytest` | Raphael's tests (from a Raphael checkout) |

## Calling Raphael from another repo

A project repository uses Raphael in four ways, all from its own checkout:

| | How | Example |
|---|---|---|
| **A. Import** core functions in its analysis code | `from raphael import population` | load spikes, score tuning |
| **B. Run** Raphael's processing stages on its raw data | `uv run raphael-jobs make --stage kilosort4 ...` | spike-sort a day |
| **C. Add** its own stages to the same job runner | `@stage('...')` + an entry point | run its analysis on the cluster |
| **D. Override** Raphael's settings | `configs/ephys.yaml`, `configs/cluster.yaml` | its lab, people, procedures |

The dependency runs one way: **the project imports Raphael, and Raphael
never imports a project**. Raphael and the project share data only through
the processed tree (below). That is what lets several projects use one core.

[`examples/example-project/`](examples/example-project/) is a complete,
working project laid out this way. Copy it to start a new one.
[`tests/test_example_project.py`](tests/test_example_project.py) runs it
against Raphael on every test run, so it stays correct.

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
dependencies = ["raphael @ git+https://github.com/HDClark94/Raphael@v0.1.0"]

[project.entry-points."raphael.stages"]
my_project = "my_project.stages"
```

### A. Import core functions

What the project's own code writes is behaviour and apparatus. Everything
neural comes from Raphael. From the example's
[`analysis.py`](examples/example-project/src/example_project/analysis.py):

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
| `raphael.util` | `gaussian_filter_nan`, `interpolate_nans`, `wrap` |
| `raphael.ephys.nwb` | `read_waveforms`, and writing `_clusters.npz` / NWB from your own sorting |
| `raphael.ephys.lfp` | `extract`, `theta_phase` on a spikeinterface recording |
| `raphael.ephys.sorting` | reading SpikeGLX / Open Ephys recordings, units tables, curation (what the `kilosort4` stage calls) |
| `raphael.imaging.traces` | `dff`, `baseline`, `neuropil_corrected` |
| `raphael.imaging.nwb` | writing behaviour and ophys NWB in the lab's form |

Tuning scores take the binning as `bin_config`, because arenas differ
between projects. Each key is a variable (or `('P_x', 'P_y')`) mapping to
`{'dim', 'bounds', 'num_bins', 'smooth_sigma'}`; see the example's
[`arena.py`](examples/example-project/src/example_project/arena.py).

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
[`stages.py`](examples/example-project/src/example_project/stages.py):

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
  [src/raphael/configs/](src/raphael/configs/) and fill in the NWB metadata
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
| `smoke` | proves a machine reads inputs, computes on its GPU and writes back | [cluster.md](docs/cluster.md) |
| `kilosort4` | a day's Neuropixels 2.0 recordings → Kilosort 4.0.16 per shank → bombcell curation → `_clusters.npz`, optional `_ecephys.nwb`, QC | [ephys_pipeline.md](docs/ephys_pipeline.md), [ephys_walkthrough.md](docs/ephys_walkthrough.md) |
| `lfp` | LFP (1 kHz, depth-averaged blocks) and theta phase → `_lfp.nwb` (CPU job) | [ephys_pipeline.md](docs/ephys_pipeline.md) |
| `suite2p` | two-photon movie → motion correction, ROIs, dF/F → `_ophys.nwb`, traces, ROI table, QC | [imaging_pipeline.md](docs/imaging_pipeline.md), [imaging_walkthrough.md](docs/imaging_walkthrough.md) |

`raphael-imaging from-dandi` re-exports a public DANDI 000402 (MICrONS) file
into the processed tree, as an end-to-end test of the imaging output, and
`raphael-imaging validate` runs NWB Inspector over a tree.

Every output lands in one tree, the same for every modality and project:

```
{root}/M{mouse}/D{day}/{session_type}/
    sub-M{mouse}_ses-D{day}_typ-{type}_beh.nwb            behaviour (project-written)
    sub-..._srt-kilosort4_clusters.npz                    spikes, TsGroup + unit table
    sub-..._lfp.nwb                                       LFP and theta phase
    sub-..._src-suite2p_ophys.nwb / _traces.npz / _rois.csv
    qc/
```

### Project hooks

Settings, cluster site and project stages are covered in
[Calling Raphael from another repo](#calling-raphael-from-another-repo).
Each person sets their cluster login once with
`uv run raphael-jobs cluster-user <MUW-ID>`. Personal overrides go in
`~/.config/raphael/cluster.yaml`. Each run records the project's commit and
`uv.lock` hash, plus Raphael's version and commit, in `logs/env.json`.

## Developing Raphael

```bash
git clone https://github.com/HDClark94/Raphael && cd Raphael
uv sync --frozen --extra dev            # add --extra gpu to run Kilosort and suite2p tests
uv run pytest
uv run ruff check src tests
```

The tests simulate Neuropixels recordings with known spikes and two-photon
planes with known ROIs, so they run without lab data. The `gpu` extra runs a
real Kilosort 4 sort and suite2p on the CPU.

- **Change carefully.** Several projects depend on what Raphael computes.
  A change that alters outputs (a default threshold, a filter) needs a test
  that shows the new behaviour, a note in the release, and a new version, so
  that projects move to it deliberately.
- **Release** by bumping `version` in `pyproject.toml` (and
  `src/raphael/__init__.py`), committing, then tagging and pushing:
  `git tag v0.2.0 && git push origin v0.2.0`. Projects move with
  `uv add "raphael[gpu] @ git+https://github.com/HDClark94/Raphael" --tag v0.2.0`.
- Python 3.11. numpy < 2 and spikeinterface 0.104.9 are pinned because
  Kilosort 4.0.16, the version in the Methods, needs them.

## Origin

Split from social-manifolds (`cluster`, `ephys`, `imaging`, `population`,
`util`, `tuning_scores`), which in turn ported its conventions and tuning
scores from spatial-manifolds (MEC work, Nolan Lab, Edinburgh). The processed
tree follows the clark2025 dataset so that sessions from all three projects
load with the same code.
