# In vivo two-photon imaging pipeline

Raw two-photon movies go in. Out comes one processed, publishable NWB file
per session, laid out like the clark2025 ephys dataset, with pynapple files
alongside for fast analysis.

- Settings and publication metadata: `configs/imaging.yaml`
- Command line: `raphael-imaging` (analysis env, `sm`)
- Code: `src/raphael/imaging/`
- Surgery protocol: [protocols/surgery/hippocampal_window_ca1.md](protocols/surgery/hippocampal_window_ca1.md)

**Processing a recording step by step, with the data at each stage:
[imaging_walkthrough.md](imaging_walkthrough.md).** The Neuropixels
counterpart, which writes the same shape of output, is
[ephys_pipeline.md](ephys_pipeline.md).

## Output layout

Same tree and naming as clark2025. The spike-sorting outputs
(`_srt-kilosort4_clusters.npz`, `_bombcell_curation.json`, `sync/`) are
replaced by their imaging equivalents:

```
{root}/M{mouse}/D{day}/{type}/
    sub-M{mouse}_ses-D{day}_typ-{type}_beh.nwb               behaviour (as clark2025)
    sub-M{mouse}_ses-D{day}_typ-{type}_src-{pipeline}_ophys.nwb    processed ophys
    sub-M{mouse}_ses-D{day}_typ-{type}_src-{pipeline}_traces.npz   dF/F, pynapple TsdFrame
    sub-M{mouse}_ses-D{day}_typ-{type}_src-{pipeline}_rois.csv     one row per ROI
    qc/plane{i}_rois.png                                            mean image + ROIs + traces
```

`src-{pipeline}` names what produced the traces (`suite2p`, `caiman`,
`microns` for the DANDI re-export), so outputs from different pipelines,
or with and without denoising, sit side by side and can be compared.

```python
import pynapple as nap, pandas as pd
traces = nap.load_file('..._src-suite2p_traces.npz')   # time x ROI, dF/F
rois = pd.read_csv('..._src-suite2p_rois.csv')         # plane, accepted, centroid, ...
cells = traces[:, rois.accepted.to_numpy()]
```

### What goes in each NWB file

**`_beh.nwb`**: one `TimeSeries` per behavioural variable, in clark2025
names and units (`S` speed in cm/s, `P_x`/`P_y` in cm, `H` in rad),
`moving`/`stationary` epochs at 5 cm/s, and every trial or stimulus table
as `TimeIntervals`.

**`_src-{pipeline}_ophys.nwb`**: every plane in one file:

| NWB location | contents |
|---|---|
| `general/` | subject, lab, experimenter, surgery/virus/pharmacology (pointing to the protocol pages), keywords, related publications |
| `devices/Microscope`, `ImagingPlane{i}` | indicator, location (Allen CCF term, e.g. `CA1`), excitation/emission, frame rate, pixel size, depth |
| `processing/ophys/SummaryImages{i}` | mean, correlation, max projection |
| `.../ImageSegmentation/PlaneSegmentation{i}` | sparse pixel masks, plus one column per ROI property (`accepted`, classifier score, ...) |
| `.../Fluorescence/RoiResponseSeries{i}` | raw F |
| `.../Neuropil/RoiResponseSeries{i}` | neuropil F (if the pipeline gives it) |
| `.../DfOverF/RoiResponseSeries{i}` | dF/F |
| `.../Deconvolved/RoiResponseSeries{i}` | inferred activity (if run) |
| `.../xy_shifts{i}` | motion-correction shifts |

The raw movie is not copied into the processed file. It belongs in a
separate raw NWB (or stays as the acquisition files), and the processed
file's `notes` name the source.

Masks are stored sparse (pixel lists), not dense: dense masks for 1,500
ROIs on a 248 x 440 plane are 650 MB per plane.

## Publishing checklist

1. Fill the TODOs in `configs/imaging.yaml` (indicator, microscope, virus,
   pharmacology, experiment description).
2. Per-mouse subject fields (sex, date of birth or age, strain, genotype)
   come from the session table.
3. `raphael-imaging validate {root}` runs NWB Inspector, the checker DANDI
   uses. Fix anything at `CRITICAL` or `BEST_PRACTICE_VIOLATION`.
4. Upload with `dandi upload`. DANDI renames files to its own
   `sub-*/sub-*_ses-*` scheme on upload; our tree stays the working copy.

## Pipeline stages

| stage | status | candidates |
|---|---|---|
| raw reader | **done** for DANDI 000402 (`imaging/dandi000402.py`) | ScanImage TIFF, Bruker, Miniscope readers for our rigs |
| motion correction | **done** via suite2p (`imaging/suite2p.py`, job stage `suite2p`) | NoRMCorre (CaImAn) as an alternative |
| denoising | to do | DeepInterpolation, DeepCAD-RT, SUPPORT (self-supervised: train on our own movies, no clean ground truth needed) |
| segmentation + extraction | **done** via suite2p | CaImAn CNMF |
| neuropil + dF/F | **done** (`imaging/traces.py`) | running 8th percentile over 60 s |
| deconvolution | **done** via suite2p (OASIS) | CASCADE |
| curation | suite2p classifier (`accepted` column); manual review to do | suite2p GUI on `outputs/suite2p/plane0` |
| NWB + tables + QC | **done** (`imaging/nwb.py`, `qc.py`) | |

GPU stages run as job bundles on a cluster, from the shared uv lockfile:
see [cluster.md](cluster.md). Each new stage only has to fill in
`imaging.nwb.Plane`; the writers do the rest. Denoising sits between motion correction and segmentation, so its
effect can be measured by running segmentation with and without it
(`src-suite2p` vs `src-deepinterp-suite2p`) and comparing ROI counts, SNR
and the QC figures.

## Test bed: DANDI 000402 (MICrONS)

Two scans of mouse 17797, session 7, are on `/Volumes/TEMP/InVivoImaging/`
(66 + 70 GB). Each has 8 planes of raw int16 movie (40,000 frames,
248 x 440 px, 2.5 µm/px, 6.3 Hz, GCaMP6, visual cortex), the authors'
segmentation (~1,000–1,700 ROIs per plane, labelled soma or artifact), raw
fluorescence, summary images, and treadmill, pupil and stimulus tables.

```bash
raphael-imaging inspect RAW.nwb
raphael-imaging from-dandi RAW.nwb --out ROOT                 # all planes
raphael-imaging from-dandi RAW.nwb --out ROOT --planes 1 --max-frames 2000   # quick look
raphael-imaging validate ROOT
```

`from-dandi` re-exports the authors' processed data through our writers,
so it checks the template end to end before any processing of our own.
Two things about this source:

- **No dF/F is written.** The authors' F is already background-subtracted
  (baselines near or below zero, minima around −350), so a percentile
  baseline gives undefined ratios for over a quarter of samples. The
  traces table holds F instead.
- NWB Inspector flags the missing experimenter/institution (left blank
  rather than attributed) and the plane location, which is the authors'
  list of four visual areas rather than one Allen term. Both are expected
  for a re-export.

Treadmill velocity is stored with `conversion = 0.01` to m/s, so the raw
values are cm/s; `S` is its absolute value.
