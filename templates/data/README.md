# Lab data

Everything the lab records, what is made from it, and the information that
describes it, in one layout for every person and project. Data in this
layout can be processed by [Raphael](https://github.com/CogEmLab/Raphael)
and published to [DANDI](https://dandiarchive.org) as NWB without extra
work.

| folder | what | who writes it |
|---|---|---|
| `incoming/` | new sessions waiting to be added | you, in `incoming/{your username}/` |
| `raw/` | recordings as acquired, with the files that describe each cohort, mouse and session | the data manager adds them; recordings are then locked |
| `processed/` | spikes, ROIs, traces, NWB files, and how each was made | the pipeline only |
| `lab/` | lists not about one animal: people, projects, task codes, rigs; blank templates | see `lab/README.md` |
| `index/` | tables of every cohort, mouse, procedure and session, built from `raw/` | generated; never edited |

**Projects' own work (analysis results, figures, code) lives outside
`data/`**, in each project's own folder or repository. It reads from
`data/` and never writes to it.

## The layout

```
data/raw/{cohort}/                                   cohort.yaml
data/raw/{cohort}/M{mouse}/                          sub-M{mouse}_subject.yaml, sub-M{mouse}_procedures.yaml
data/raw/{cohort}/M{mouse}/D{day}/{type}/            sub-M…_ses-D…_typ-…_session.yaml, and the recordings
data/processed/{cohort}/M{mouse}/D{day}/{type}/      what was made from them, and how
```

`processed/` has the same structure as `raw/`: a session's outputs sit at
the same path as its recordings, and outputs that span a mouse's days sit
in its mouse folder.

**Information lives at the level it describes**: what is true of the whole
cohort in the cohort folder, of the mouse in the mouse folder, of the
session in the session folder. Everything about a session is in, or
above, its folder. `raw/2027-01_2P-Testing/` is a worked example.

## Rules

1. **Recordings are never changed.** Once added, a recording can't be
   edited, renamed or deleted. If something is wrong, ask the data manager.
2. **Describing files stay editable.** The `.yaml` files (cohort, subject,
   procedures, session) can be corrected and added to later, e.g. a
   histology result or a note; every change is kept (open: how).
3. **Processed data is made only by the pipeline.** To change it, the data
   is processed again. Every processed session has a `provenance/` record
   of exactly how it was made.
4. **Organised by animal, not by person or project.** Who ran what, and for
   which project, is written in the files, so one mouse can serve several
   projects and mistakes are fixed without moving recordings.
5. **One code, one task.** Session types start with a task code from
   `lab/tasks.yaml`, so `OF` means the same thing for everyone.

## Adding data

How to get a recording from the rig into `raw/`. The examples use cohort
`2026-03_dHPC-place`, mouse `10432` and username `jsmith`.

**The short version:** describe it, name it, copy it into
`incoming/{your username}/`, tell the data manager. You never write in
`raw/` yourself.

### What to write, and when

| when | write | from |
|---|---|---|
| first time | your entry in `lab/people.yaml` | |
| new project, task or rig | an entry in `lab/projects.yaml`, `lab/tasks.yaml`, or `lab/rigs/{rig}.yaml` | `lab/templates/rig.yaml` |
| new cohort | `cohort.yaml` | `lab/templates/cohort.yaml` |
| new mouse | `sub-M{mouse}_subject.yaml` | `lab/templates/subject.yaml` |
| each surgery | an entry in `sub-M{mouse}_procedures.yaml` | `lab/templates/procedures.yaml` |
| each session | `sub-M…_ses-D…_typ-…_session.yaml`, beside the recordings | `lab/templates/session.yaml` |

> **Coming:** a command (`raphael-data add`) that does the naming and fills
> in the session file for you from what the rig wrote. Until then, follow
> the steps below by hand.

### Before your first recording (once)

1. **Add yourself** to `lab/people.yaml`: username, full name ("Last,
   First"), email, ORCID.
2. **Check the codes you will use exist** in `lab/`: your task in
   `tasks.yaml` (e.g. `OF`), your project in `projects.yaml`, your rig in
   `rigs/`. If one is missing, ask its owner (or the data manager) to add
   it. Never invent a new code in a file name.

### The three numbers in every name

```
sub-M10432_ses-D4_typ-OF1_…
     │         │      └── task code + run number that day (OF1, then OF2 if you run it again)
     │         └───────── day of experiment for this mouse
     └─────────────────── the mouse's lab-wide ID
```

**Which day number?** Look in `raw/{cohort}/M{mouse}/`. If the mouse
already has `D1`–`D3` and today is a new date for it, today is `D4`. Every
recorded date counts, whatever the project or task. Two sessions on the
same date share the day number.


### Case 1: a new cohort, a new mouse, its surgery

Before the first recording of a new cohort or mouse, its describing files
go in. Copy each from `lab/templates/` and fill it in.

```
incoming/jsmith/2026-03_dHPC-place/
├── cohort.yaml                          from lab/templates/cohort.yaml (new cohort only)
└── M10432/
    ├── sub-M10432_subject.yaml          from lab/templates/subject.yaml
    └── sub-M10432_procedures.yaml       from lab/templates/procedures.yaml
```

- The cohort name is the month it started and a short label:
  `2026-03_dHPC-place`.
- In `sub-M10432_subject.yaml`, `sex`, `date_of_birth` and `species` are
  required; they go into every NWB file of this mouse.
- In `sub-M10432_procedures.yaml`, add one entry per procedure (injection,
  implant, headplate, …) with coordinates and what went in. Give each
  implant a `label`: `ephys` for the probe if there is one; `dHPC`, `mPFC`
  if there are two fibres. **Recordings use the same label**, which is how
  they get their brain location.

Copy it to the server and tell the data manager (see the end). These files
can be edited later: the data manager adds new procedures and histology
results for you, or you hand them in the same way.

### Case 2: a normal recording day

Neuropixels and a top camera in the open field. The rig wrote:

```
~/Desktop/2026-03-14_10-05-12/Record Node 101/…      Open Ephys
~/Desktop/cam0_2026-03-14.mp4                        camera
~/Desktop/rig_log_2026-03-14.csv                     task log
```

Build the session folder in `incoming/jsmith/`:

```
incoming/jsmith/2026-03_dHPC-place/M10432/D4/OF1/
├── sub-M10432_ses-D4_typ-OF1_session.yaml
├── sub-M10432_ses-D4_typ-OF1_ephys/                 ← the folder 2026-03-14_10-05-12, renamed
│   └── Record Node 101/…                               (nothing inside renamed)
├── sub-M10432_ses-D4_typ-OF1_video-top.mp4          ← cam0_2026-03-14.mp4, renamed
└── sub-M10432_ses-D4_typ-OF1_beh-rig.csv            ← rig_log_2026-03-14.csv, renamed
```

- **Rename the outer folder only** for anything acquisition software wrote
  (Open Ephys, SpikeGLX, ScanImage, Inscopix). Their insides must stay
  exactly as recorded.
- **Single files are renamed** to the session's name plus what they are:
  `_video-{camera}`, `_beh-{source}`, `_sync-{device}`.

Write the session file from `lab/templates/session.yaml`:

```yaml
start: 2026-03-14T10:05:00          # required: local time
task: OF                            # required
experimenter: [jsmith]              # required
project: place-cells                # required
rig: ephys-rig-1
weight_g: 24.1
description:
recordings:
  ephys: probe in CA1, 384 channels, bank 0
  video-top: Basler top camera, 30 fps
  beh-rig: position and TTL log
quality: good
notes: cable twisted at 12 min, untwisted at 13 min
```

Write the notes on the day. Months later they are the only record of what
went wrong.

### Case 3: more than one session on a day

Each session has its own folder; they share the day.

```
incoming/jsmith/2026-03_dHPC-place/M10432/D4/
├── OF1/   sub-M10432_ses-D4_typ-OF1_session.yaml, …_ephys/, …     open field, morning
├── VR1/   sub-M10432_ses-D4_typ-VR1_session.yaml, …_ephys/, …     VR, after
└── OF2/   sub-M10432_ses-D4_typ-OF2_session.yaml, …_ephys/, …     open field again
```

The number after the task code counts runs of that task that day: the
second open field is `OF2`. Each Open Ephys recording goes in its own
session. Neuropixels sessions from one day are spike-sorted together, so a
unit keeps its number across them.

### Case 4: several of one kind

Two photometry sites and two cameras in one session: a `-{label}` says
which.

```
incoming/jsmith/2026-03_dHPC-place/M10432/D5/OF1/
├── sub-M10432_ses-D5_typ-OF1_session.yaml
├── sub-M10432_ses-D5_typ-OF1_photometry-dHPC.doric
├── sub-M10432_ses-D5_typ-OF1_photometry-mPFC.doric
├── sub-M10432_ses-D5_typ-OF1_video-top.mp4
└── sub-M10432_ses-D5_typ-OF1_video-side.mp4
```

The photometry labels (`dHPC`, `mPFC`) must match the fibre implants'
labels in `sub-M10432_procedures.yaml`. That is where each recording's
brain region and indicator come from. Camera labels match the rig file
(`lab/rigs/{rig}.yaml`).

### Case 5: imaging

**Two-photon (ScanImage):** the folder of TIFFs ScanImage wrote, renamed
outside only.

```
incoming/jsmith/2026-03_dHPC-place/M10432/D6/OF1/
├── sub-M10432_ses-D6_typ-OF1_session.yaml
└── sub-M10432_ses-D6_typ-OF1_ophys/
    └── file_00001.tif …
```

**Miniscope (Inscopix):** the session folder nVista wrote, with its
`.isxd`, `.gpio`, `.imu` and `session.json` together, renamed outside only.
Leave out TIFF exports and anything made afterwards; they can be made again
from the `.isxd`.

```
incoming/jsmith/2026-03_dHPC-place/M10432/D6/HBT1/
├── sub-M10432_ses-D6_typ-HBT1_session.yaml
├── sub-M10432_ses-D6_typ-HBT1_ophys/
│   ├── 2026-03-16-11-02-40_video.isxd
│   ├── 2026-03-16-11-02-40_video.gpio
│   ├── 2026-03-16-11-02-40_video.imu
│   └── session.json
└── sub-M10432_ses-D6_typ-HBT1_video-bottom.mp4
```

For one imaging recording per session, the label is `ophys`: the
injection and window or lens entries in the procedures file use
`label: ophys`. In the session file, note the depth, laser power or LED
power.

### Case 6: behaviour only

No neural recording: just the session file, the video and the logs.

```
incoming/jsmith/2026-03_dHPC-place/M10432/D2/HAB1/
├── sub-M10432_ses-D2_typ-HAB1_session.yaml
└── sub-M10432_ses-D2_typ-HAB1_video-top.mp4
```

Habituation days that are recorded count as days, so they take day
numbers like any other session.

### Case 7: adding to a mouse later

- **Another surgery:** add an entry to `sub-M10432_procedures.yaml`.
- **Histology:** images and track reconstructions go in
  `M10432/subject/histology/`; fill in `verified_target` for the matching
  procedure.
- **A note or correction to a session** (it turned out the probe moved, the
  weight was wrong): change its session file.

Hand these in the same way, in `incoming/jsmith/` at the same path:

```
incoming/jsmith/2026-03_dHPC-place/M10432/
├── sub-M10432_procedures.yaml          the whole file, with the new entry
└── subject/histology/M10432_slice07.tif
```

Describing files (`.yaml`) can always be corrected. Recordings can't.

### Case 8: something already in `raw/` is wrong

A recording filed under the wrong mouse, day or task can't be fixed by
you: recordings in `raw/` are locked. Tell the data manager what is wrong
and what it should be. They move it, and note the change in the session
file.


### Before you hand it in

- [ ] Your username, the task code, project and rig exist in `lab/`.
- [ ] The cohort folder has `cohort.yaml`, and the mouse folder has
      `sub-M…_subject.yaml`, either in `raw/` already or in your
      `incoming/`.
- [ ] Every session folder has its `_session.yaml`, with `start`, `task`,
      `experimenter` and `project` filled in.
- [ ] The day number is right (next new date for this mouse), and runs of
      a task that day are numbered `OF1`, `OF2`.
- [ ] Every name starts `sub-M{mouse}_ses-D{day}_typ-{type}_`, matching its
      folders.
- [ ] Acquisition folders are renamed outside only; nothing inside them
      changed.
- [ ] Labels (`photometry-dHPC`, `ephys-probeA`, `video-top`) match the
      procedures file and the rig file.
- [ ] You still have your own copy.

### Copying to the server

With the server mounted, drag the folder into `incoming/{your username}/`,
or from a terminal (Mac or Linux):

```bash
rsync -a --progress --exclude '._*' --exclude '.DS_Store' \
    2026-03_dHPC-place /path/to/server/data/incoming/jsmith/
```

`rsync` can be run again if it is interrupted; it carries on. Then **tell
the data manager** which sessions you added.

### What the data manager does

1. Checks the list above: required fields filled in, codes and people in
   `lab/`, names consistent with their folders.
2. Writes each session's manifest (`…_manifest.tsv`: every file's size,
   sha256 and original name).
3. Moves the session into `raw/{cohort}/M{mouse}/D{day}/{type}/` (and new
   cohort, subject and procedure files into place), and locks the
   recordings. The describing files stay editable.
4. Rebuilds `index/`, and empties the session from `incoming/`.

From then on the session is found by processing (`raphael-jobs plan`), and
its outputs appear at the same path in `processed/`.

## Finding data

Browse `raw/` and `processed/` by cohort, mouse, day and session. To find
every session of a project, a task or a person, filter
`index/sessions.tsv`.

Full description: Raphael's
[docs/data_server.md](https://github.com/CogEmLab/Raphael/blob/main/docs/data_server.md).
How data is processed: Raphael's
[docs/processing.md](https://github.com/CogEmLab/Raphael/blob/main/docs/processing.md).
