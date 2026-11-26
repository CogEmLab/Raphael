# raw/

Recordings as acquired, and the files that describe them. Added by the data
manager's ingest from `incoming/`, which names and files everything; never
written here directly. **Recordings are
locked** once added; the describing `.yaml` files stay editable.

## Layout

```
raw/{cohort}/
├── cohort.yaml                                        the cohort: project, owner, start, purpose
└── sub-M{mouse}/
    ├── sub-M{mouse}_subject.yaml                      the mouse: sex, date of birth, strain, genotype
    ├── sub-M{mouse}_procedures.yaml                   its surgeries: injections, implants, coordinates
    ├── subject/                                       not tied to a session: histology/, surgery/, anatomy/
    └── day-{dd}/ses-{type}/
        ├── sub-M{mouse}_day-{dd}_ses-{type}_session.yaml      the session: start, experimenter, task, notes
        ├── sub-M{mouse}_day-{dd}_ses-{type}_ephys/            Neuropixels: Open Ephys or SpikeGLX, inside untouched
        ├── sub-M{mouse}_day-{dd}_ses-{type}_ophys/            imaging: ScanImage, Bruker, or Inscopix .isxd + session.json
        ├── sub-M{mouse}_day-{dd}_ses-{type}_photometry.doric
        ├── sub-M{mouse}_day-{dd}_ses-{type}_video-{camera}.mp4
        ├── sub-M{mouse}_day-{dd}_ses-{type}_beh-{source}.{ext}    rig logs, task files
        ├── sub-M{mouse}_day-{dd}_ses-{type}_sync-{device}.{ext}   TTL files not inside another recording
        └── sub-M{mouse}_day-{dd}_ses-{type}_manifest.tsv          checksum and original name of every file (ingest)
```

| part | means | example |
|---|---|---|
| `{cohort}` | the animals run together: `YYYY-MM_{label}` (month it started, short label) | `2026-03_dHPC-place` |
| `sub-M{mouse}` | the mouse's lab-wide ID | `sub-M10432` |
| `day-{dd}` | day of experiment for that mouse, two digits: day-01 = its first recorded day, in any project | `day-04` |
| `ses-{type}` | task code (`metadata/tasks.yaml`) + run number that day | `ses-OF1`, `ses-OF2`, `ses-VR1` |
| `-{label}` | which one, when a session has several of a kind | `video-top`, `photometry-dHPC`, `ephys-probeA` |

The describing files are written from `metadata/templates/`; each template says
what every field is and where it goes in NWB. `2027-01_2P-Testing/` is a
worked example.

## How the files fit together

- A session file names its `task`, `project`, `rig` and `experimenter`;
  those are looked up in `metadata/`.
- A recording's label is looked up in the mouse's procedures:
  `…_photometry-dHPC.doric` gets its brain location and indicator from the
  procedure labelled `dHPC`. With one recording of a kind, the label is the
  modality (`ephys`, `ophys`, `photometry`).
- Nothing is written twice: the mouse's sex is in its subject file only,
  the task's description in `metadata/tasks.yaml` only.

## Naming rules

- Single files get the `sub-…_day-…_ses-…` name. **Folders written by
  acquisition software keep their insides as recorded**; only the outer
  folder is renamed.
- Each `_ephys` / `_ophys` folder and each `.doric` file is **one
  continuous recording**. Two recordings of a task on one day are two
  session types: `OF1`, `OF2`.
- A session has only what it recorded. A behaviour-only session has just
  its `_session.yaml`, `_video` and `_beh` files.
