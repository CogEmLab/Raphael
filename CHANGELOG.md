# Changelog

All notable changes to Raphael. Projects pin a released version, so each
entry says what changed and what a project must do to move to it.

Versions: while Raphael is 0.x, the **middle** number goes up for anything
that changes outputs or breaks imports, and the **last** for fixes and
additions that leave existing results unchanged. Each release is a GitHub
Release (and, once Zenodo is connected, gets a DOI); see docs/releases.md.

## [Unreleased]

### Added
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

[Unreleased]: https://github.com/HDClark94/Raphael/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/HDClark94/Raphael/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/HDClark94/Raphael/releases/tag/v0.1.0
