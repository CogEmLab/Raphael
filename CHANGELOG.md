# Changelog

All notable changes to Raphael. Projects pin a released version, so each
entry says what changed and what a project must do to move to it.

Versions: while Raphael is 0.x, the **middle** number goes up for anything
that changes outputs or breaks imports, and the **last** for fixes and
additions that leave existing results unchanged. Each release is a GitHub
Release (and, once Zenodo is connected, gets a DOI); see docs/releases.md.

## [Unreleased]

## [0.4.0] - 2026-11-24
### Added
- `raphael.behavioural_tracking`, moved from social-manifolds (pose/):
  video preparation and day-held-out splits, DeepLabCut projects (from
  scratch, or several models on one split from existing labels, with an
  optional benchmark model), multi-peak capture, threshold sweeps, the
  skeleton model and two-animal identity correction, pixels -> arena cm,
  pose -> Raphael behaviour (P_x, P_y, H, S, moving) and the processed tree
  (`_src-dlc_pose.h5`). VAME 0.13 motifs, run in VAME's own locked
  environment (every VAME pins an older pynwb than Raphael needs);
  `_src-vame-…_motifs.csv`. `raphael-tracking` command, job stages
  `dlc_train`, `dlc_capture`, `vame`, template `configs/tracking.yaml`,
  new `tracking` extra (DeepLabCut 3.0.2). `tables` (PyTables) is now a core
  dependency, for reading DeepLabCut's HDF5 output. See docs/behavioural_tracking.md.
- Provenance with the results: every job writes `provenance/{job_id}.json`
  into each processed session folder it writes (Raphael version, tag and
  commit, project commit and `uv.lock`, settings and config text, inputs'
  and outputs' sha256, package versions, machine, timings), and every NWB
  file names the Raphael that wrote it (`source_script`). Session folders
  now contain a `provenance/` folder. See docs/cluster.md.
- `raphael.sync` (preliminary): aligns a behaviour CSV's TTL column with
  a Neuropixels recording's TTL line (Open Ephys), checks the alignment
  (pulse counts, unpaired pulses, ambiguity, intervals, widths, drift,
  residuals) and writes `_beh.csv` on the recording clock only if it
  passes, plus `qc/_sync.json` and a QC figure. `raphael-sync` command and
  `sync` job stage; simulated behaviour CSV and light Open Ephys NP2
  recording for testing. See docs/sync.md.
- `raphael.ml`: machine-learning encoding (behaviour -> spikes, Poisson
  pseudo-R2) and decoding (population -> continuous, circular or categorical
  behaviour) with models from a mean baseline, linear, ridge and Poisson GLM
  to random forest, gradient boosting, XGBoost and an MLP; blocked-in-time
  cross-validation, model comparison, a time-shift shuffle test, spatial and
  circular basis features and raised-cosine history features. Based on
  spatial-manifolds' mlencoding.py (Benjamin et al. 2018). XGBoost is in the
  new `ml` extra. Example notebook examples/ml_encoding_decoding.ipynb; see
  docs/ml.md.
- `raphael.toy`: a simulated session with known tuning (place,
  head-direction, speed, theta, event-responsive and untuned units; imaging
  ROIs; photometry; TTL pulses), written as a processed tree that Raphael's
  loaders read, or as the BIDS-style raw tree (behaviour CSV, light Open
  Ephys recording sharing the TTL pulses, Doric file, manifest), optionally
  with a ground-truth Neuropixels recording. See docs/toy.md.
- `toy.raw` also writes a raw two-photon movie (`_ophys/`, TIFF) of the
  session's ROIs, and `toy.session(tuning_seed=...)` keeps the same cells
  across days. examples/toy_dataset.ipynb makes a three-day, all-modality
  dataset, raw and processed, with a README. tifffile is now a dependency.
- A guide for coding agents shipped in the package: `python -m raphael`
  prints the rules and which Raphael tool to use for each task
  (`raphael/AGENTS.md`), `python -m raphael api [module]` lists every
  public function of the installed version. Projects point Claude Code at
  it from CLAUDE.md. See docs/using_raphael.md.
### Changed
- Apple-silicon Macs only (with Linux and Windows): torch has had no
  Intel-Mac builds since 2.2, so Intel Macs could not install the gpu extra
  anyway.
- Windows is supported natively (no WSL): uv.lock covers Windows (CUDA 12.6
  torch, as on Linux), CI runs the tests on Windows, and
  docs/getting_started.md gives Windows steps (winget, PowerShell) beside
  the macOS ones. Projects add the Windows line to `environments` and the
  torch sources in their pyproject.
- Raphael moved to the CogEmLab organisation (https://github.com/CogEmLab/Raphael);
  old addresses redirect. New projects start from starter_project with
  **Use this template** (which allows private projects) instead of forking.
- docs/data_server.md: proposed layout and access rights for the labs' data
  on the Leonardo server.

To move from 0.3.0: bump the tag; on Windows, add the `win32` line to
`[tool.uv] environments` and the torch sources (see starter_project);
expect a `provenance/` folder in processed session folders. Intel Macs can
no longer install Raphael.

## [0.3.0] - 2026-11-22
### Added
- `raphael.style`: the labs' figure style (Arial, 6.5-8 pt text, open axes,
  shaded areas without outlines, frameless legends, editable fonts in PDFs),
  journal figure sizes in mm, shared region colours and colormaps, panel
  letters, significance brackets, mean +/- error bands, scale bars and
  reproducible `save()`. Raphael's QC figures use it. See docs/style.md.
- `raphael.plots`: rate maps, tuning curves, head-direction and theta-phase
  plots, peri-event rasters/heatmaps with mean +/- SEM, rasters, stacked
  traces, autocorrelograms, ISI and score histograms, for spikes and
  continuous signals alike; `plots.example_data()`. See docs/plots.md.
- Continuous integration: lint, the test suite with and without the `gpu`
  extra, and every project that uses Raphael (`.github/downstream/*.yml`,
  one config per project) tested against each change. See docs/ci.md.
- Releases: `CHANGELOG.md`, `CITATION.cff`, a release workflow that checks
  versions agree and CI passes before publishing a GitHub Release, and
  `release_tools.py prepare`. See docs/releases.md.
- README banner; the README asks new projects to tell the Raphael owner so
  they join CI.

### Fixed
- Lint errors inherited from social-manifolds. The kilosort4 stage test
  skips when the `gpu` extra is not installed.

## [0.2.0] - 2026-11-20

### Added
- `raphael.photometry`: Doric fibre photometry to dF/F and z-score per
  channel (Simpson et al. 2024 pre-processing), `_photometry.nwb/.npz/.csv`
  and QC; `raphael-photometry inspect|process|simulate` and a `photometry`
  job stage; `configs/photometry.yaml`. Console-generated TTLs are
  reconstructed from the file's settings (`ttl_times`).
- docs/getting_started.md (VS Code, Claude Code, forking starter_project)
  and docs/using_raphael.md.

### Changed
- The in-repo example project is replaced by
  [starter_project](https://github.com/HDClark94/starter_project).

To move from 0.1.0: no changes needed; photometry is new.

## [0.1.0] - 2026-11-20

### Added
- First release, split from social-manifolds: Neuropixels spike sorting
  (Kilosort 4) and LFP (`ephys`), two-photon imaging (`imaging`), NWB
  writing, the population loader, tuning scores and cluster job bundles
  (`raphael-jobs`), with project stages as plug-ins (`raphael.stages`).

[Unreleased]: https://github.com/CogEmLab/Raphael/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/CogEmLab/Raphael/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/CogEmLab/Raphael/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/CogEmLab/Raphael/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/CogEmLab/Raphael/releases/tag/v0.1.0
