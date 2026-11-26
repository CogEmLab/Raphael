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

| when | write | from |
|---|---|---|
| first time | your entry in `lab/people.yaml` | |
| new project, task or rig | an entry in `lab/projects.yaml`, `lab/tasks.yaml`, or `lab/rigs/{rig}.yaml` | `lab/templates/rig.yaml` |
| new cohort | `cohort.yaml` | `lab/templates/cohort.yaml` |
| new mouse | `sub-M{mouse}_subject.yaml` | `lab/templates/subject.yaml` |
| each surgery | an entry in `sub-M{mouse}_procedures.yaml` | `lab/templates/procedures.yaml` |
| each session | `sub-M…_ses-D…_typ-…_session.yaml`, beside the recordings | `lab/templates/session.yaml` |

Then copy the session (or the new mouse or cohort) into `incoming/` and
tell the data manager. **[ingestion.md](ingestion.md) walks through it
step by step, with examples**: a new mouse, a normal recording day, several
sessions in a day, two photometry sites, imaging, behaviour only, and
adding things later.

## Finding data

Browse `raw/` and `processed/` by cohort, mouse, day and session. To find
every session of a project, a task or a person, filter
`index/sessions.tsv`.

Full description: Raphael's
[docs/data_server.md](https://github.com/CogEmLab/Raphael/blob/main/docs/data_server.md).
How data is processed: Raphael's
[docs/processing.md](https://github.com/CogEmLab/Raphael/blob/main/docs/processing.md).
