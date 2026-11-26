# Lab data

Everything the lab records, what is made from it, and the information that
describes it, in one layout for every person and project. Data in this
layout can be processed by [Raphael](https://github.com/CogEmLab/Raphael)
and published to [DANDI](https://dandiarchive.org) as NWB without extra
work.

| folder | what | who writes it |
|---|---|---|
| `incoming/` | new sessions waiting to be added | you, in `incoming/{your username}/` |
| `raw/` | recordings as acquired, with the files that describe each cohort, mouse and session | the data manager's ingest; recordings are then locked |
| `processed/` | spikes, ROIs, traces, NWB files, and how each was made | the pipeline only |
| `lab/` | lists not about one animal: people, projects, task codes, rigs; blank templates | see `lab/README.md` |
| `index/` | tables of every cohort, mouse, procedure and session, built from `raw/` | generated; never edited |

**Projects' own work (analysis results, figures, code) lives outside
`data/`**, in each project's own folder or repository. It reads from
`data/` and never writes to it.

## The layout

```
data/raw/{cohort}/                                     cohort.yaml
data/raw/{cohort}/sub-M{mouse}/                        sub-M{mouse}_subject.yaml, sub-M{mouse}_procedures.yaml
data/raw/{cohort}/sub-M{mouse}/day-{dd}/ses-{type}/    sub-M…_day-…_ses-…_session.yaml, and the recordings
data/processed/{cohort}/sub-M{mouse}/day-{dd}/ses-{type}/   what was made from them, and how
```

Every file is named for its session: `sub-M1_day-01_ses-OF1_…`.
`processed/` has the same structure as `raw/`: a session's outputs sit at
the same path as its recordings, and outputs that span a mouse's days sit
in its mouse folder.

**Information lives at the level it describes**: what is true of the whole
cohort in the cohort folder, of the mouse in the mouse folder, of the
session in the session folder. `raw/2027-01_2P-Testing/` is a worked
example.

## Rules

1. **Recordings are never changed.** Once added, a recording can't be
   edited, renamed or deleted. If something is wrong, ask the data manager.
2. **Describing files stay editable.** The `.yaml` files (cohort, subject,
   procedures, session) can be corrected and added to later, e.g. a
   surgery or a note; the old versions are kept.
3. **Processed data is made only by the pipeline.** To change it, the data
   is processed again. Every processed session has a `provenance/` record
   of exactly how it was made.
4. **Organised by animal, not by person or project.** Who ran what, and for
   which project, is written in the files, so one mouse can serve several
   projects and mistakes are fixed without moving recordings.
5. **One code, one task.** Session types start with a task code from
   `lab/tasks.yaml`, so `OF` means the same thing for everyone.

## Adding data

**Drop your session into `incoming/{your username}/` and tell the data
manager.** You don't rename the recordings or build any folders: the data
manager's ingest works out where everything goes, renames it, checks it
and files it in `raw/`. **`incoming/_example/` shows a hand-in exactly as
it should look.**

### Before your first recording (once)

1. **Add yourself** to `lab/people.yaml`: username, full name ("Last,
   First"), position, email, ORCID.
2. **Check the codes you will use exist** in `lab/`: your task in
   `tasks.yaml` (e.g. `OF`), your project in `projects.yaml`, your rig in
   `rigs/`. If one is missing, ask its owner or the data manager to add it.

### What a hand-in contains

Every hand-in has **four describing files** and **one folder per session**:

```
incoming/jsmith/
├── 2026-03_dHPC-place_cohort.yaml                         the cohort
├── sub-M10432_subject.yaml                                the mouse
├── sub-M10432_procedures.yaml                             its surgeries ([] until the first)
└── sub-M10432_day-04_ses-OF1_2026-03-14_10-05-12/         the session
    ├── sub-M10432_day-04_ses-OF1_session.yaml             its describing file
    ├── 2026-03-14_10-05-12/                               what the rig wrote, not renamed
    ├── cam0_2026-03-14.mp4
    └── rig_log_2026-03-14.csv
```

| file | from | when it changes |
|---|---|---|
| `{cohort}_cohort.yaml` | `lab/templates/cohort.yaml` | rarely: written when the cohort starts |
| `sub-M{mouse}_subject.yaml` | `lab/templates/subject.yaml` | rarely: written when the mouse joins (it names the mouse's cohort) |
| `sub-M{mouse}_procedures.yaml` | `lab/templates/procedures.yaml` | after each surgery: add an entry. Just `[]` before the first |
| `sub-M…_day-…_ses-…_session.yaml` | `lab/templates/session.yaml` | every session: written on the day |

Keep your copies of the cohort, subject and procedures files and hand them
in again each time. When one changes (a surgery, a correction), ingest
updates the server's copy and keeps the old one.

### Naming the session folder

```
sub-M10432_day-04_ses-OF1_2026-03-14_10-05-12
     │        │      │    └── when the session started (date_time)
     │        │      └─────── task code + run number that day: OF1, then OF2 if you run it again
     │        └────────────── day of experiment for this mouse, two digits
     └─────────────────────── the mouse's lab-wide ID
```

- **Day:** the mouse's recording days in date order. If the mouse already
  has days 1–3 and today is a new date for it, today is `day-04`. Every
  recorded date counts, whatever the project or task; two sessions on the
  same date share the day.
- **Run:** the first open field of the day is `ses-OF1`, a second one
  `ses-OF2`.
- **Start:** the time the recording started, as the rig writes it
  (`2026-03-14_10-05-12`). Open Ephys and most acquisition software can be
  set to prefix their folder name with `sub-M10432_day-04_ses-OF1_`, so the
  rig names the folder for you.

The name carries everything that says where the session goes, and ingest
checks it against the recording's own start time, the session file and the
mouse's earlier days. If something doesn't match, it tells you rather than
filing it in the wrong place.

### The session file

Copy `lab/templates/session.yaml` into the session folder, name it like
the folder without the time (`sub-M10432_day-04_ses-OF1_session.yaml`) and
fill in what only you know: at least the notes, and the weight. Fields you
leave blank that can be worked out are filled in for you (start time, task,
experimenter, and project and rig from the mouse's last session).

```yaml
start:                              # blank: taken from the recording
task: OF
experimenter: [jsmith]
project: place-cells
rig: ephys-rig-1
weight_g: 24.1
recordings:
  ephys: probe in CA1, 384 channels, bank 0
  video: top camera, 30 fps
quality: good
notes: cable twisted at 12 min, untwisted at 13 min
```

Write the notes on the day. Months later they are the only record of what
went wrong.

### Examples

**Several sessions in a day:** one folder each, sharing the day.

```
sub-M10432_day-04_ses-OF1_2026-03-14_10-05-12/      open field, morning
sub-M10432_day-04_ses-VR1_2026-03-14_11-30-40/      VR
sub-M10432_day-04_ses-OF2_2026-03-14_14-02-03/      open field again
```

Neuropixels sessions from one day are spike-sorted together, so a unit
keeps its number across them.

**Several of one kind** (two photometry sites, two cameras): ingest tells
them apart by name. A file whose name contains a procedure's `label`
(`dHPC`, `mPFC`) or a rig camera's label (`top`, `side`) is matched to it,
which is where each recording's brain region and indicator come from. If
the names don't say, ingest asks you to rename them.

```
sub-M10432_day-05_ses-OF1_2026-03-15_09-40-00/
├── sub-M10432_day-05_ses-OF1_session.yaml
├── fibre_dHPC_2026-03-15.doric
├── fibre_mPFC_2026-03-15.doric
├── top_2026-03-15.mp4
└── side_2026-03-15.mp4
```

**Miniscope (Inscopix):** the folder nVista wrote, with its `.isxd`,
`.gpio`, `.imu` and `session.json` together. Leave out TIFF exports and
anything made afterwards; they can be made again from the `.isxd`.
**Two-photon (ScanImage):** the folder of TIFFs ScanImage wrote. For one
imaging recording, the procedures' injection and window or lens entries
use `label: ophys`; note the depth and laser or LED power in the session
file.

**Behaviour only:** just the session file and the video and logs.
Recorded habituation days count as days like any other.

**A surgery, without a session:** hand in the updated
`sub-M{mouse}_procedures.yaml` (with the cohort and subject files) on its
own. Histology images go in a folder `sub-M{mouse}_histology/`.

**Something already in `raw/` is wrong** (wrong mouse, day or task):
recordings in `raw/` are locked, so tell the data manager what it should
be. Describing files can be corrected by handing in a new version.

### Before you hand it in

- [ ] Your username, task code, project and rig exist in `lab/`.
- [ ] The cohort, subject and procedures files are there, named for their
      cohort and mouse.
- [ ] Each session folder is named `sub-M…_day-…_ses-…_{date}_{time}` and
      has its session file, named the same without the time.
- [ ] The day number is right (the next new date for this mouse), and runs
      of a task that day are numbered `OF1`, `OF2`.
- [ ] Nothing the rig wrote has been renamed or changed.
- [ ] You still have your own copy.

### Copying to the server

With the server mounted, drag the files into `incoming/{your username}/`,
or from a terminal (Mac or Linux):

```bash
rsync -a --progress --exclude '._*' --exclude '.DS_Store' \
    ./ /path/to/server/data/incoming/jsmith/
```

`rsync` can be run again if it is interrupted; it carries on. Then **tell
the data manager** what you added.

### What the data manager does

Runs ingest (`raphael-data ingest`; `--dry-run` shows what it would do).
For each session it:

1. Checks the session folder's name against the recording's own start
   time, the session file and the mouse's earlier days; that the task,
   project, rig and people exist in `lab/`; and that the cohort, subject
   and procedures files are there.
2. Renames the recordings by what they are (`_ephys`, `_ophys`,
   `_photometry`, `_video`, `_beh`), keeping each original name in the
   session's manifest with its checksum, and fills in blank fields of the
   session file.
3. Moves the session into `raw/{cohort}/sub-M{mouse}/day-{dd}/ses-{type}/`,
   adds or updates the cohort, subject and procedures files (keeping old
   versions), locks the recordings and rebuilds `index/`.
4. Tells you and the data managers what was added, or exactly what is
   wrong and how to fix it.

From then on the session is found by processing (`raphael-jobs plan`), and
its outputs appear at the same path in `processed/`.

## Finding data

Browse `raw/` and `processed/` by cohort, mouse, day and session. To find
every session of a project, a task or a person, filter
`index/sessions.tsv`.

Data named before November 2026 (`M{mouse}/D{day}/{type}/`,
`sub-M…_ses-D…_typ-…`) is still read by Raphael.

Full description: Raphael's
[docs/data_server.md](https://github.com/CogEmLab/Raphael/blob/main/docs/data_server.md).
How data is processed: Raphael's
[docs/processing.md](https://github.com/CogEmLab/Raphael/blob/main/docs/processing.md).
