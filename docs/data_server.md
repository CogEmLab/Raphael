# Data on Leonardo

How the labs' data is laid out on **Leonardo**, the data server, and who may
change what. **This is a proposal**: the layout follows what Raphael already
reads and writes, but the server's paths, accounts and backup are not set up
yet. Points still to decide are marked **(open)** and listed at the end.

## Principles

1. **Raw data is written once and never changed.** After a recording is
   ingested, nobody can edit, rename or delete it without admin rights.
2. **Processed data is written only by the pipeline.** Raphael's stages,
   run on the cluster under a pipeline account, produce it. People read it,
   but never edit it by hand. A change means a new pipeline run, with a
   record of what ran.
3. **NWB wherever a standard exists.** Processed data is NWB, readable by
   any NWB tool and ready for DANDI. Raw data stays in the acquisition
   software's own format (see [Raw data and NWB](#raw-data-and-nwb)).
4. **One naming scheme everywhere.** Raw and processed trees use the same
   `{cohort}/M{mouse}/D{day}/{session type}` folders and the BIDS-style
   `sub-…_ses-…_typ-…` file names that Raphael's `Session` class builds,
   so code finds files without per-project path logic.
5. **Organised by animal; described where it happens.** The tree is
   cohorts, mice, days and sessions, the same for every person and project.
   Each level carries a small file describing it (the cohort, the mouse and
   its procedures, the session), so everything about a session is in or
   above its folder. Who ran it, for which project, on which rig, is written
   there, not encoded in paths, so it can be corrected without moving
   recordings and one mouse can serve several projects.
6. **Enough for NWB, DANDI and BIDS.** Between them, the describing files
   and `lab/` hold every field an NWB file and a DANDI upload need, so
   publishing needs no extra bookkeeping.
7. **Every processed file can be traced back** to the raw files it came
   from (with checksums), the Raphael version and commit, and the settings.

## Layout

One tree per lab, so labs can keep their data apart. Everything recorded,
what is made from it, and what describes it, sits under `data/`:

```
/leonardo/{lab}/data/                     (path open)
├── incoming/{person}/   each person's landing area for new sessions
├── raw/                 recordings as acquired (locked), with a describing
│                        file at each level: cohort, mouse, procedures, session
├── processed/           pipeline-only: the same structure as raw/, NWB first,
│                        each session with the provenance of how it was made
├── lab/                 lists not about one animal: lab, people, projects,
│                        task codes, rigs; blank templates
└── index/               tables of every cohort, mouse, procedure and session,
                         generated from raw/ (never edited)
```

**Projects' own work lives outside `data/`**: analysis results, figures and
code belong to each project (its repository, or a folder of its own next
to `data/`). Projects read from `data/` and never write to it.

A ready-made copy of this tree, with every README, template and a worked
example, is in Raphael's [`templates/data/`](../templates/data/README.md):
copy it to the server to start.

### raw/

Cohort, then mouse, then day, then session, with the BIDS-style names of
`processed/`, so a raw recording and the files made from it share a name
stem (`sub-M{mouse}_ses-D{day}_typ-{type}`) and sit at the same path in
both trees. Each level has the file that describes it:

```
raw/{cohort}/
├── cohort.yaml                                          the cohort
└── M{mouse}/
    ├── sub-M{m}_subject.yaml                            the mouse
    ├── sub-M{m}_procedures.yaml                         its surgeries and procedures
    ├── subject/                                         not tied to a session: histology/, surgery/, anatomy/
    └── D{day}/{type}/
        ├── sub-M{m}_ses-D{d}_typ-{t}_session.yaml       the session
        ├── sub-M{m}_ses-D{d}_typ-{t}_ephys/             Open Ephys (Record Node …/) or SpikeGLX (*_g0/),
        │                                                inside untouched
        ├── sub-M{m}_ses-D{d}_typ-{t}_ophys/             ScanImage TIFF / Bruker, or an Inscopix .isxd
        │                                                with its session.json, inside untouched
        ├── sub-M{m}_ses-D{d}_typ-{t}_photometry.doric
        ├── sub-M{m}_ses-D{d}_typ-{t}_video-{camera}.mp4      one per camera, e.g. video-top
        ├── sub-M{m}_ses-D{d}_typ-{t}_video-{camera}_timestamps.csv
        ├── sub-M{m}_ses-D{d}_typ-{t}_beh-{source}.{ext}      rig logs, task files, e.g. beh-unity.csv
        ├── sub-M{m}_ses-D{d}_typ-{t}_sync-{device}.{ext}     TTL/sync files not inside another recording
        └── sub-M{m}_ses-D{d}_typ-{t}_manifest.tsv            checksum and original name of every file
```

- **`{cohort}`** is the group of animals that went through an experiment
  together, named `YYYY-MM_{label}` (e.g. `2024-05_dHPC-SSCW`) so cohorts
  sort by date. A mouse belongs to one cohort, fixed when it is added. The
  cohort is a folder level only: file names don't carry it, because mouse
  IDs are already unique.
- **`M{mouse}`** is a **lab-wide unique ID** (the animal facility's ID, or
  one the lab hands out) **(open)**, never a per-person number like M1,
  which would collide between people. Each person's own short name for the
  mouse is kept in its subject file (`alias`).
- **`D{day}`** is the day of the experiment for that mouse: `D1` is its
  first recorded day, whichever project that was for, so day numbers never
  clash. The calendar date and time are in the session file.
- **`{type}`** is a **registered task code plus a run number** (`OF1`,
  `OF2`, `VR1`, `HBT3`). Every task code is defined once in
  `lab/tasks.yaml`, so `OF` means the same arena and protocol for everyone;
  ingest refuses codes that aren't registered. Two runs of a task on one
  day are `OF1`, `OF2`. A task's own progression (day 3 of the helping
  test) goes in the run number, not in `D`.
- **One session holds every modality recorded in it**, one file or folder
  each, named by suffix (`_ephys`, `_ophys`, `_photometry`, `_video`,
  `_beh`, `_sync`). When there are several of one kind, `-{label}` says
  which: `video-top`, `photometry-dHPC`, `photometry-mPFC`. Neuropixels
  probes recorded together stay in one `_ephys` folder; probes recorded
  separately are `_ephys-probeA`, `_ephys-probeB`. A session has only what
  it recorded; a behaviour-only session has just its session file, `_video`
  and `_beh` files.
- **`subject/`** holds what belongs to the mouse rather than a session:
  histology, surgery notes, probe-track reconstructions.
- **Single files are renamed** at ingest. **Folders written by acquisition
  software are wrapped, not changed**: Open Ephys and SpikeGLX expect their
  internal names, so ingest renames only the outer folder
  (`…_ephys/Record Node 109/…` stays as recorded). The manifest records
  each file's sha256 and **its original name**.
- Each `_ephys`/`_ophys` folder and each `.doric` file holds **one
  continuous recording**, as Raphael's readers expect.
- This is BIDS's split between `sourcedata/` (data as recorded) and
  derivatives, with our tree's folder levels. DANDI renames to its own
  scheme on upload either way.

#### The describing files

YAML, one per level, written from the blank templates in `lab/templates/`
(each field's comment says where it goes in NWB). **Each fact is written
once, at the level it belongs to**: the mouse's sex in its subject file
only, a task's description in `lab/tasks.yaml` only.

| file | written | holds | becomes |
|---|---|---|---|
| `cohort.yaml` | when the cohort starts | project, owner, start date, animal licence, purpose | NWB `protocol` (licence) |
| `sub-M…_subject.yaml` | when the mouse joins | **ID, sex, date of birth, species**, strain, genotype, alias, owner, source | NWB `Subject`; BIDS `participants.tsv`; the bold fields are required by DANDI |
| `sub-M…_procedures.yaml` | after each surgery | per procedure: date, type, `label`, target, coordinates (mm from bregma), angle, what went in (virus and titre, probe or fibre and serial), indicator, anaesthesia, surgeon, histology-verified target | NWB `surgery`, `virus`; electrode, fibre and imaging-plane locations; imaging indicator |
| `sub-M…_ses-…_session.yaml` | on the day | **start**, **task**, **experimenter**, **project**, rig, weight, what each recording is, pharmacology, stimulus notes, quality, notes | NWB `session_start_time`, `session_description` (with the task's), `experimenter`, `pharmacology`, `stimulus_notes`, `notes`, `Subject.weight`; BIDS `sessions.tsv`. Bold fields are required at ingest |

**`label` joins recordings to anatomy.** A file named
`…_photometry-dHPC.doric` takes its brain location and indicator from the
procedure labelled `dHPC`; `…_ephys-probeA` its probe and target from
`probeA`. With one recording of a kind, the label is the modality (`ephys`,
`ophys`, `photometry`).

**Recordings are locked; describing files stay editable.** Dates of birth
get corrected, histology adds a verified target, a note is added later. So
ingest locks the recordings but leaves the `.yaml` files writable by data
managers (and the mouse's owner **(open)**). Every edit is kept, by git
over the `.yaml` files (a repository over `data/` that ignores everything
else) or by the server's snapshots **(open)**. Each processed session's
provenance records the checksums of the describing files it read, so a
later correction shows which processed files are out of date **(planned)**.

**Why not one folder per person or per project?** A mouse can serve two
projects, projects get renamed and people leave, but raw data is written
once: a path that encodes the project would end up wrong, or need moving.
Writing who and what in the describing files keeps the tree stable, and
`index/sessions.tsv` finds every session of a project, task or person in
one filter.

### processed/

**The same structure as `raw/`**: the same cohort, mouse, day and session
folders, so a session's outputs sit at the same path as its recordings.
Outputs that span a mouse's days (cells or units matched across sessions,
probe tracks registered to the atlas) go in its mouse folder **(planned)**.
`raw/`'s describing files are not copied here: the NWB files carry what
they need from them, and the provenance records which were read.

```
processed/{cohort}/M{mouse}/D{day}/{type}/
├── sub-M{m}_ses-D{d}_typ-{t}_beh.nwb                     behaviour, synced (planned)
├── sub-…_srt-kilosort4_ecephys.nwb                       units, waveforms, electrodes
├── sub-…_srt-kilosort4_clusters.npz                      the same units, pynapple
├── sub-…_srt-kilosort4_bombcell_curation.json            bombcell labels (curation file)
├── sub-…_lfp.nwb                                         LFP and theta phase
├── sub-…_src-suite2p_ophys.nwb                           ROIs, dF/F, deconvolved
├── sub-…_src-suite2p_traces.npz, _rois.csv               the same, pynapple / table
├── sub-…_src-cnmfe_ophys.nwb, _traces.npz, _rois.csv     one-photon (CaImAn CNMF-E)
├── sub-…_photometry.nwb                                  raw channels, dF/F, z-score
├── sub-…_photometry.npz, _photometry.csv                 the same, pynapple / table
├── qc/                                                   QC figures; _sync.json, _sync.png
└── provenance/{job_id}.json                              what each run did (below)
```

- **NWB is the record; `.npz` and `.csv` are fast copies** of the same
  data for pynapple and pandas, written in the same run.
- On Leonardo, units are written as NWB too: set
  `postprocessing.write_ecephys_nwb: true` in the server's
  `configs/ephys.yaml` (it is off by default to save space on laptops).
- `srt-{sorter}` and `src-{pipeline}` name what produced a file, so a
  re-run with another sorter or pipeline sits beside the first instead of
  replacing it.

### lab/

Lists that aren't about any one animal, and the blank templates:

```
lab/
├── lab.yaml           lab, institution, PI, timezone, default licence      NWB lab, institution
├── people.yaml        username: full name ("Last, First"), email, ORCID,   NWB experimenter;
│                      affiliation, DANDI roles, joined, left                DANDI contributors
├── projects.yaml      code: name, owner, members, description, keywords,   NWB experiment_description,
│                      animal licence, funding, publications, dandiset       keywords, protocol, publications
├── tasks.yaml         code: name, description, rig, protocol, stimulus      NWB session_description,
│                      notes, what a session records, owner                  stimulus_notes
├── rigs/{rig}.yaml    devices, cameras, photometry, microscope, TTL wiring  NWB devices; sync
└── templates/         cohort, subject, procedures, session, rig
```

Everywhere else in the tree people are named by username and projects,
tasks and rigs by code, looked up here, so each is described once. People
who leave stay in `people.yaml` (with `left:`), so old sessions still
resolve.

### index/

`cohorts.tsv`, `subjects.tsv`, `procedures.tsv` and `sessions.tsv`: one row
per cohort, mouse, procedure and session, **generated from the describing
files** by ingest and nightly, for searching and filtering (BIDS's
`participants.tsv` and `sessions.tsv` among them). Never edited: correct
the `.yaml` file and the tables follow, so they can't disagree with the
tree. The script that builds them is still to be written **(open)**.

Raphael reads the describing files directly: `raphael-jobs plan` and
`make` take a session's start time from its session file when the
recording's own files don't carry it, and the subject fields from the
subject file ([processing.md](processing.md)).

### Provenance

There is no separate folder of run records: each processed session folder
carries its own `provenance/{job_id}.json`, which Raphael writes with every
run, so the record travels with the files it describes. It records:
- the Raphael version, release tag and exact commit;
- the project's commit and `uv.lock`;
- the settings, and the full text of every config;
- the sha256 of each raw input and of each file written;
- every package version;
- the machine and GPU, and the timings.

Every NWB file also names the Raphael that wrote it (NWB's
`source_script`). So anyone holding a processed file can tell which raw
files, code and settings made it, and check it hasn't changed since. See
[cluster.md](cluster.md#provenance-that-travels-with-the-results). The
job's logs stay with the job on the cluster, not on Leonardo.

## Who can change what

Access rights are enforced by the server, using groups and file
permissions, not left to good intentions:

| Folder | Read | Write | How data gets in |
|---|---|---|---|
| `incoming/{person}/` | acquisition group | that person | experimenters copy new recordings here |
| `raw/` recordings | lab members | **nobody** (admin only) | ingest moves files in and locks them |
| `raw/` describing `.yaml` files | lab members | data managers (owners? **(open)**) | ingest, then corrections; every edit kept |
| `processed/` | lab members | **pipeline account** only | Raphael runs on the cluster |
| `lab/` | lab members | data managers; people add themselves to `people.yaml` | edited through git **(open)** |
| `index/` | lab members | ingest only | generated |

Lab members means members of both labs, unless a lab wants its tree
private **(open)**.

How a person hands data in, with worked examples (a new mouse, a normal
recording day, several sessions a day, several photometry sites, imaging,
behaviour only, later additions): "Adding data" in [`templates/data/README.md`](../templates/data/README.md).

**Ingest** turns a recording in `incoming/{person}/` into locked raw data, in one
step run by a data-manager account (a small script, to be written):

1. Check the session has its session file with the required fields, its
   mouse a subject file and its cohort a cohort file (here or already in
   `raw/`), and that its task, project, rig and people exist in `lab/`.
2. Rename single files, and wrap acquisition folders, to the BIDS-style
   names above. Write the manifest (sha256 and original name of every file).
3. Move the session into `raw/{cohort}/M…/D…/{type}/`, and make its
   recordings owned by the data account and read-only to everyone else.
4. Rebuild `index/`, and remove the session from `incoming/{person}/`.

From then on, nobody without admin rights can modify or delete it. If a
recording really must be corrected (wrong session name, say), an admin does
it and notes it in the session file.

**Permissions protect against accidents, not against everything.** A
mistake by an admin, a failed disk or ransomware gets past them. So the
server also needs:

- **snapshots**, so yesterday's state can be restored, and
- **a backup in another building** of `raw/` and `lab/`. Everything in
  `processed/` can be regenerated from those, but a backup saves weeks of
  cluster time **(open)**.

## Raw to processed

Raphael's core stages do the processing for every data type, with the
cluster (MSC) as the computer. They run as Raphael job bundles
([cluster.md](cluster.md)):

```
Leonardo raw/  ──(transfer, open)──▶  cluster scratch  ──▶  Raphael job
                                                              │ kilosort4, lfp, suite2p,
                                                              │ photometry, project stages
Leonardo processed/  ◀──(transfer, open)──  outputs, with their provenance
```

| Data type | Raw | Raphael stage | Processed NWB |
|---|---|---|---|
| Neuropixels | Open Ephys / SpikeGLX | `kilosort4`, `lfp` | `_ecephys.nwb`, `_lfp.nwb` |
| Two-photon | ScanImage / Bruker | `suite2p` | `_ophys.nwb` |
| Fibre photometry | Doric `.doric` | `photometry` | `_photometry.nwb` |
| Behaviour and video | rig logs, video, TTLs | sync stage (planned) | `_beh.nwb` |

**How data moves between Leonardo and the cluster is still to be decided.**
Options include the cluster mounting Leonardo directly, or copying with
`rsync` or Globus. Whatever is chosen has to keep these rules:

- **Transfers into `processed/` are made by the pipeline
  account**, never by hand.
- **Checksums are checked at both ends.** The job manifest already records
  the sha256 of every input (so jobs from Leonardo shouldn't use `--no-hash`),
  and the outputs are checked after copying back.
- **Only finished runs are copied back.** A run that failed leaves
  `processed/` untouched.
- **Cluster scratch is a working copy.** It can be wiped after a run;
  Leonardo is the only lasting copy.

## Sync checks (TTL pulses)

Streams recorded on different devices (Neuropixels, two-photon, photometry,
cameras, the behaviour rig) are aligned with shared TTL pulses. A wrong
alignment doesn't crash anything; it quietly shifts every result. So
**every processed session gets sync checks before anything is aligned**.
They are written to `qc/…_sync.json` (numbers and pass or fail) and
`qc/…_sync.png` (the figure), and recorded in the provenance:

| Check | Catches | Pass when (default, per rig in config) |
|---|---|---|
| Pulse count on each device, against the protocol's expected count | missing or extra pulses, a disconnected cable | equal |
| Pulse counts agree across devices | a device that started late or stopped early | equal, or the difference explained (e.g. recording started after the first pulse) |
| Inter-pulse intervals | dropped pulses (gaps), doubled pulses (bounces), wrong pulse rate | all within ±5% of the expected interval |
| Pulse widths | noise read as pulses | within the expected range |
| Linear clock fit between each pair of devices | clock drift, an offset | drift under 50 ppm |
| Residuals of that fit | jitter, a pulse matched to the wrong one | max under 1 ms (ephys, photometry), under one frame (video) |
| Camera frames against frame TTLs and the video file | dropped frames | counts equal |
| Recording durations from the aligned clocks | truncated files | agree within one pulse interval |

- **A failed check stops alignment for that session.** `_beh.nwb` and
  anything else that depends on the alignment is not written, and the
  session is listed as failed with the reason, until someone looks at it.
  A check can be waived only by a note in the session file saying
  why, and the waiver is recorded in the provenance.
- **Photometry:** Doric stores digital lines at its decimated rate (about
  60 Hz with the lab's settings: 12,166 Hz / 200), so pulses shorter or
  faster than that alias.
  Pulses the console generated are rebuilt from its settings
  (`Recording.ttl_times`) and checked against what was recorded.
- **Status:** a preliminary version exists ([sync.md](sync.md)): one TTL
  column in a behaviour CSV against one TTL line in an Open Ephys
  Neuropixels recording. It runs these checks, writes the QC report and
  figure, and writes the behaviour on the recording's clock (`_beh.csv`)
  only if they pass. Other devices and `_beh.nwb` come next; these checks
  are their specification.

## Re-processing

When a Raphael release changes outputs (a new middle version number, see
[releases.md](releases.md)), sessions can be re-run. The pipeline account
replaces the files, and the session's `provenance/` gets a new record
naming the new version. The old files remain in snapshots. A paper should
say which Raphael version produced its data. The provenance records answer
that for every file.

## Raw data and NWB

The NWB standard covers raw data too, but converting raw recordings to NWB
on the server would double the largest files (Neuropixels runs at about
83 GB per hour) for no gain in analysis. So:

- **On Leonardo, raw stays in the acquisition format**, which Raphael and
  every other tool can read, with checksums.
- **For publication, raw NWB is written once**, with
  [NeuroConv](https://neuroconv.readthedocs.io), when a dataset goes to
  [DANDI](https://dandiarchive.org). The processed NWB files go as they
  are. Run `raphael-imaging validate` (NWB Inspector) first. DANDI renames
  files to its own `sub-*/sub-*_ses-*` scheme on upload, and the Leonardo
  tree stays the working copy.

## Still to decide

- **Server:** Leonardo's path, filesystem, and whether it supports
  snapshots (e.g. ZFS); who administers it.
- **Accounts:** how groups are managed (MedUni accounts?), who the data
  managers are, and the pipeline account's name.
- **Transfer:** how data moves between Leonardo and the cluster (see above).
- **Backup:** where the off-site copy lives, and whether `processed/` is
  included.
- **Sharing:** whether each lab's tree is readable by the other lab.
- **Describing files:** who may edit them (data managers only, or owners
  too); git or snapshots for their history.
- **Mouse IDs:** the animal facility's IDs, or IDs the lab hands out at
  registration; and how existing datasets (M20–M29, M5592, M8472, …) map
  onto them.
- **Task codes:** who keeps `lab/tasks.yaml`, and the first list of codes.
- **Index:** writing the script that builds `index/` (with ingest).
- **Cohort names:** renaming existing cohorts (`May_2024` →
  `2024-05_…`) when they are ingested.
- **Video:** how long raw video is kept once pose estimation has run.
- **Ingest:** writing the ingest script (renaming, manifest, locking),
  perhaps as a `raphael-data` command.
- **Sync:** each rig's TTL scheme (which device sends pulses, their rate
  and width, expected counts per protocol) and the thresholds above.
