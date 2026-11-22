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
   `M{mouse}/D{day}/{session type}` folders and the BIDS-style
   `sub-…_ses-…_typ-…` file names that Raphael's `Session` class builds,
   so code finds files without per-project path logic.
5. **Every processed file can be traced back** to the raw files it came
   from (with checksums), the Raphael version and commit, and the settings.

## Layout

One tree per lab, so mouse numbers can't collide between labs:

```
/leonardo/{lab}/                          (path open)
├── raw/                 write-once: acquisition files, as recorded
├── processed/           pipeline-only: Raphael's outputs, NWB first
├── metadata/            subjects, sessions, rigs: who, when, what
├── jobs/                record of every pipeline run (no bulk data)
├── projects/            per-project results, writable by the project
└── incoming/            landing area for new recordings before ingest
```

### raw/

```
raw/M{mouse}/D{day}/{type}/
├── ephys/          Open Ephys (Record Node …/) or SpikeGLX (*_g0/), untouched
├── imaging/        ScanImage TIFF / Bruker, untouched
├── photometry/     Doric .doric files
├── video/          camera files (.mp4/.avi) and their timestamps
├── behaviour/      rig logs, task files, TTL/sync files
└── MANIFEST.sha256 checksum of every file, written at ingest
```

- `{day}` is the day of the experiment for that mouse, as in Raphael's
  tree, and `{type}` is the session type (`OF1`, `VR`, `HBT4T2`, …). The
  calendar date and time live in `metadata/sessions.csv`.
- A session has only the modality folders it recorded.
- Each modality folder holds **one continuous recording**, as Raphael's
  readers expect. Two recordings of the same session type on one day become
  two session types (`OF1`, `OF2`).

### processed/

The tree Raphael's stages already write, one folder per session:

```
processed/M{mouse}/D{day}/{type}/
├── sub-M{m}_ses-D{d}_typ-{t}_beh.nwb                     behaviour, synced (planned)
├── sub-…_srt-kilosort4_ecephys.nwb                       units, waveforms, electrodes
├── sub-…_srt-kilosort4_clusters.npz                      the same units, pynapple
├── sub-…_lfp.nwb                                         LFP and theta phase
├── sub-…_src-suite2p_ophys.nwb                           ROIs, dF/F, deconvolved
├── sub-…_src-suite2p_traces.npz, _rois.csv               the same, pynapple / table
├── sub-…_photometry.nwb                                  raw channels, dF/F, z-score
├── sub-…_photometry.npz, _photometry.csv                 the same, pynapple / table
├── qc/                                                   QC figures
└── provenance/                                           one record per run (below)
```

- **NWB is the record; `.npz` and `.csv` are fast copies** of the same
  data for pynapple and pandas, written in the same run.
- On Leonardo, units are written as NWB too: set
  `postprocessing.write_ecephys_nwb: true` in the server's
  `configs/ephys.yaml` (it is off by default to save space on laptops).
- `srt-{sorter}` and `src-{pipeline}` name what produced a file, so a
  re-run with another sorter or pipeline sits beside the first instead of
  replacing it.

### metadata/

```
metadata/
├── subjects.csv      mouse, sex, date of birth, strain/genotype, surgeries, lab
├── sessions.csv      mouse, day, type, date/time, rig, experimenter, modalities, notes
└── rigs/             probe maps, microscope and photometry settings per rig
```

The NWB files take their subject and session fields from these tables, so
they need to be complete before processing. These files are small and
change often. Keeping them under git (with Leonardo holding a checkout)
gives a history of every edit **(open)**.

### jobs/

One folder per pipeline run, copied back from the cluster after the run
finishes. It holds the job's manifest (stage, settings, inputs and their
sha256), logs and provenance (the Raphael version and commit, and
`uv.lock`), but **not** the bulk outputs, which go to `processed/`.
Each session's `provenance/` folder points to the runs that wrote its
files.

### projects/

Results a project derives from `processed/` (tables, statistics, figure
data), one folder per project repository. Project members can write there.
Project code stays on GitHub; only data goes here.

## Who can change what

Access rights are enforced by the server, using groups and file
permissions, not left to good intentions:

| Folder | Read | Write | How data gets in |
|---|---|---|---|
| `incoming/` | acquisition group | acquisition group | experimenters copy new recordings here |
| `raw/` | lab members | **nobody** (admin only) | ingest moves files in and locks them |
| `processed/` | lab members | **pipeline account** only | Raphael runs on the cluster |
| `jobs/` | lab members | pipeline account only | copied back with each run |
| `metadata/` | lab members | data managers | edited through git **(open)** |
| `projects/{name}/` | lab members | project members | project scripts |

Lab members means members of both labs, unless a lab wants its tree
private **(open)**.

**Ingest** turns a recording in `incoming/` into locked raw data, in one
step run by a data-manager account (a small script, to be written):

1. Check the session is in `metadata/sessions.csv` and the folder is named
   correctly.
2. Write `MANIFEST.sha256` for every file.
3. Move the folder into `raw/M…/D…/{type}/`, and make it owned by the data
   account and read-only to everyone else.
4. Remove it from `incoming/`.

From then on, nobody without admin rights can modify or delete it. If a
recording really must be corrected (wrong session name, say), an admin does
it and notes it in the session table.

**Permissions protect against accidents, not against everything.** A
mistake by an admin, a failed disk or ransomware gets past them. So the
server also needs:

- **snapshots**, so yesterday's state can be restored, and
- **a backup in another building** of `raw/` and `metadata/`. Everything in
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
Leonardo processed/ + jobs/  ◀──(transfer, open)──  outputs, logs, provenance
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

- **Transfers into `processed/` and `jobs/` are made by the pipeline
  account**, never by hand.
- **Checksums are checked at both ends.** The job manifest already records
  the sha256 of every input (so jobs from Leonardo shouldn't use `--no-hash`),
  and the outputs are checked after copying back.
- **Only finished runs are copied back.** A run that failed leaves
  `processed/` untouched.
- **Cluster scratch is a working copy.** It can be wiped after a run;
  Leonardo is the only lasting copy.

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
- **Metadata:** whether to keep `metadata/` under git, and the session
  table's exact columns. NWB needs subject age or date of birth, sex,
  species and strain.
- **Video:** how long raw video is kept once pose estimation has run.
- **Ingest:** writing the ingest script, perhaps as a `raphael-data` command.
