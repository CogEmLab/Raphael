# raw/

Recordings as acquired, and the files that describe them. Added by the data
manager from `incoming/`; never written here directly. **Recordings are
locked** once added; the describing `.yaml` files stay editable.

## Layout

```
raw/{cohort}/
├── cohort.yaml                                        the cohort: project, owner, start, purpose
└── M{mouse}/
    ├── sub-M{mouse}_subject.yaml                      the mouse: sex, date of birth, strain, genotype
    ├── sub-M{mouse}_procedures.yaml                   its surgeries: injections, implants, coordinates
    ├── subject/                                       not tied to a session: histology/, surgery/, anatomy/
    └── D{day}/{type}/
        ├── sub-M{mouse}_ses-D{day}_typ-{type}_session.yaml      the session: start, experimenter, task, notes
        ├── sub-M{mouse}_ses-D{day}_typ-{type}_ephys/            Neuropixels: Open Ephys or SpikeGLX, inside untouched
        ├── sub-M{mouse}_ses-D{day}_typ-{type}_ophys/            imaging: ScanImage, Bruker, or Inscopix .isxd + session.json
        ├── sub-M{mouse}_ses-D{day}_typ-{type}_photometry.doric
        ├── sub-M{mouse}_ses-D{day}_typ-{type}_video-{camera}.mp4
        ├── sub-M{mouse}_ses-D{day}_typ-{type}_beh-{source}.{ext}    rig logs, task files
        ├── sub-M{mouse}_ses-D{day}_typ-{type}_sync-{device}.{ext}   TTL files not inside another recording
        └── sub-M{mouse}_ses-D{day}_typ-{type}_manifest.tsv          checksum and original name of every file (ingest)
```

| part | means | example |
|---|---|---|
| `{cohort}` | the animals run together: `YYYY-MM_{label}` (month it started, short label) | `2026-03_dHPC-place` |
| `M{mouse}` | the mouse's lab-wide ID | `M10432` |
| `D{day}` | day of experiment for that mouse: D1 = its first recorded day, in any project | `D4` |
| `{type}` | task code (`lab/tasks.yaml`) + run number that day | `OF1`, `OF2`, `VR1` |
| `-{label}` | which one, when a session has several of a kind | `video-top`, `photometry-dHPC`, `ephys-probeA` |

The describing files are written from `lab/templates/`; each template says
what every field is and where it goes in NWB. `2027-01_2P-Testing/` is a
worked example.

## How the files fit together

- A session file names its `task`, `project`, `rig` and `experimenter`;
  those are looked up in `lab/`.
- A recording's label is looked up in the mouse's procedures:
  `…_photometry-dHPC.doric` gets its brain location and indicator from the
  procedure labelled `dHPC`. With one recording of a kind, the label is the
  modality (`ephys`, `ophys`, `photometry`).
- Nothing is written twice: the mouse's sex is in its subject file only,
  the task's description in `lab/tasks.yaml` only.

## Naming rules

- Single files get the `sub-…_ses-…_typ-…` name. **Folders written by
  acquisition software keep their insides as recorded**; only the outer
  folder is renamed.
- Each `_ephys` / `_ophys` folder and each `.doric` file is **one
  continuous recording**. Two recordings of a task on one day are two
  session types: `OF1`, `OF2`.
- A session has only what it recorded. A behaviour-only session has just
  its `_session.yaml`, `_video` and `_beh` files.
