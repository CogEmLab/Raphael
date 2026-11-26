# processed/

What the pipeline (Raphael) made from `raw/`, in **the same structure as
`raw/`**: the same cohort, mouse, day and session folders, so a session's
outputs are at the same path as its recordings. **Read-only**: only the
pipeline writes here. To change something, the data is processed again.

```
processed/{cohort}/
└── M{mouse}/
    ├── sub-M{mouse}_…                                       across this mouse's days, e.g. cells or
    │                                                        units matched between sessions (planned)
    ├── subject/                                             from raw subject/: e.g. probe tracks
    │                                                        registered to the Allen atlas (planned)
    └── D{day}/{type}/
        ├── sub-M…_ses-D…_typ-…_srt-kilosort4_clusters.npz   spikes and units
        ├── sub-M…_ses-D…_typ-…_srt-kilosort4_bombcell_curation.json
        ├── sub-M…_ses-D…_typ-…_lfp.nwb                      LFP and theta phase
        ├── sub-M…_ses-D…_typ-…_src-suite2p_ophys.nwb        two-photon ROIs and traces (+ _traces.npz, _rois.csv)
        ├── sub-M…_ses-D…_typ-…_src-cnmfe_ophys.nwb          one-photon ROIs and traces (+ _traces.npz, _rois.csv)
        ├── sub-M…_ses-D…_typ-…_photometry.nwb               photometry dF/F (+ .npz, .csv)
        ├── sub-M…_ses-D…_typ-…_beh.nwb                      behaviour, on the recording's clock
        ├── qc/                                              figures to check by eye
        └── provenance/                                      exactly how these files were made (below)
```

`raw/`'s describing files (`cohort.yaml`, `_subject.yaml`,
`_procedures.yaml`, `_session.yaml`) are **not copied here**: they stay in
`raw/`, written once. The NWB files carry what they need from them, and
`provenance/` records which ones were read. `2027-01_2P-Testing/` mirrors
the example in `raw/`.

`.nwb` files are the record (readable by any NWB tool, ready for DANDI);
`.npz` and `.csv` files are fast copies for analysis. `srt-…` and `src-…`
name what made a file, so re-runs with another method sit side by side.
Loading them: Raphael's `raphael.population.load` and docs/using_raphael.md.

## provenance/

One JSON file per processing run of the session, so anyone can see exactly
how the files were made from the raw data, and make them again:

- **the raw files used**, each with its size and sha256 checksum;
- **every file written**, with its checksum;
- **the code**: Raphael's version and git commit, the project's commit, and
  the checksum of the locked environment (`uv.lock`), plus every package's
  version;
- **the settings**: the full text of the config file used, and every
  parameter given to the run;
- **where and when**: machine, start and end time, CPUs, memory and GPU.
- **the describing files read** (subject, procedures, session), with
  their checksums, so a later correction shows which processed files are
  out of date (planned).

A processing step may add a summary beside it (e.g. `cnmfe_summary.json`:
components found and accepted, run times).
