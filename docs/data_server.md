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

The same folders and the same BIDS-style names as `processed/`, so a raw
recording and the files made from it share a name stem
(`sub-M{mouse}_ses-D{day}_typ-{type}`) and sit at the same path in both
trees:

```
raw/M{mouse}/D{day}/{type}/
├── sub-M{m}_ses-D{d}_typ-{t}_ephys/                Open Ephys (Record Node …/) or
│                                                   SpikeGLX (*_g0/), inside untouched
├── sub-M{m}_ses-D{d}_typ-{t}_ophys/                ScanImage TIFF / Bruker, or an Inscopix .isxd
│                                                   with its session.json, inside untouched
├── sub-M{m}_ses-D{d}_typ-{t}_photometry.doric
├── sub-M{m}_ses-D{d}_typ-{t}_video-{camera}.mp4    one per camera, e.g. video-top
├── sub-M{m}_ses-D{d}_typ-{t}_video-{camera}_timestamps.csv
├── sub-M{m}_ses-D{d}_typ-{t}_beh-{source}.{ext}    rig logs, task files, e.g. beh-unity.csv
├── sub-M{m}_ses-D{d}_typ-{t}_sync-{device}.{ext}   TTL/sync files not inside another recording
└── sub-M{m}_ses-D{d}_typ-{t}_manifest.tsv          checksum and original name of every file
```

- `{day}` is the day of the experiment for that mouse, as in Raphael's
  tree, and `{type}` is the session type (`OF1`, `VR`, `HBT4T2`, …). The
  calendar date and time live in `metadata/sessions.csv`.
- **Single files are renamed** at ingest. **Folders written by acquisition
  software are wrapped, not changed**: Open Ephys and SpikeGLX expect their
  internal names, so ingest renames only the outer folder
  (`…_ephys/Record Node 109/…` stays as recorded).
- The manifest records each file's sha256 and **its original name**, so
  nothing about the recording is lost by renaming.
- The suffixes (`_ephys`, `_ophys`, `_photometry`, `_beh`, `_video`) match
  the processed files', and `-{label}` marks which camera, rig program or
  device. A session has only the modalities it recorded.
- Each `_ephys`/`_ophys` folder and each `.doric` file holds **one
  continuous recording**, as Raphael's readers expect. Two recordings of
  the same session type on one day become two session types (`OF1`, `OF2`).
- This is BIDS's split between `sourcedata/` (data as recorded) and
  derivatives, with our tree's folder levels. DANDI renames to its own
  scheme on upload either way.

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

### metadata/

```
metadata/
├── subjects.csv      mouse, sex, date of birth, strain/genotype, surgeries, lab
├── sessions.csv      mouse, day, type, start, rig, experimenter, modalities, notes
└── rigs/             probe maps, microscope and photometry settings per rig
```

The NWB files take their subject and session fields from these tables, so
they need to be complete before processing. `start` is the session's start
as ISO 8601 local time (`2026-11-03T10:00:00`); Raphael uses it for
recordings whose files don't carry their own (`raphael-jobs plan` and
`make` read these tables themselves, [processing.md](processing.md)). These files are small and
change often. Keeping them under git (with Leonardo holding a checkout)
gives a history of every edit **(open)**.

### jobs/

One folder per pipeline run, copied back from the cluster after the run
finishes. It holds the job's manifest (stage, settings, inputs and their
sha256), logs and provenance (the Raphael version and commit, and
`uv.lock`), but **not** the bulk outputs, which go to `processed/`.

Each processed session folder carries its own `provenance/{job_id}.json`,
which Raphael writes with every run. It records:
- the Raphael version, release tag and exact commit;
- the project's commit and `uv.lock`;
- the settings, and the full text of every config;
- the sha256 of each raw input and of each file written;
- every package version;
- the machine and GPU, and the timings.

Every NWB file also names the Raphael that wrote it (NWB's
`source_script`). So anyone holding a processed file can tell which raw
files, code and settings made it, and check it hasn't changed since. See
[cluster.md](cluster.md#provenance-that-travels-with-the-results).

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

1. Check the session is in `metadata/sessions.csv`.
2. Rename single files, and wrap acquisition folders, to the BIDS-style
   names above. Write the manifest (sha256 and original name of every file).
3. Move the session into `raw/M…/D…/{type}/`, and make it owned by the data
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
  A check can be waived only by a note in `metadata/sessions.csv` saying
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
- **Metadata:** whether to keep `metadata/` under git, and the session
  table's exact columns. NWB needs subject age or date of birth, sex,
  species and strain.
- **Video:** how long raw video is kept once pose estimation has run.
- **Ingest:** writing the ingest script (renaming, manifest, locking),
  perhaps as a `raphael-data` command.
- **Sync:** each rig's TTL scheme (which device sends pulses, their rate
  and width, expected counts per protocol) and the thresholds above.
