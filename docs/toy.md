# Toy data

`raphael.toy` makes a simulated session in which every neuron's tuning is
known, then writes it as raw data, as processed data, or both. Pipelines,
analyses, tests and teaching material can then run without lab data, and
every result can be checked against the truth.

```python
from raphael import toy
from raphael.ephys.layout import Session

s = toy.session(duration=600, seed=0)     # in memory
s['tuning']                               # what each unit and ROI really encodes

toy.processed('/tmp/toy/processed', Session('1', '1', 'OF1'), s)   # processed tree
toy.raw('/tmp/toy/raw', Session('1', '1', 'OF1'), s)               # raw tree
```

Passing the same `s` to both writes the same session twice: raw and
processed, on the same clock.

**A ready-made dataset.**
[examples/toy_dataset.ipynb](../examples/toy_dataset.ipynb) makes three
days of one mouse, every modality, raw and processed, with a README. It
takes about 30 s and 400 MB. Copies of it live on the lab's drives and
server, so people can start from it without running anything.

## The session

| Key | What |
|---|---|
| `spikes` | TsGroup of place, head-direction, speed, theta-locked, event-responsive and untuned units. Set the counts with `n_place`, `n_hd`, `n_speed`, `n_theta`, `n_event`, `n_untuned` |
| `behaviour` | `P_x`, `P_y` (cm, 1 m arena), `H` (rad), `S` (cm/s) as Tsd, and the `moving` epoch |
| `theta_phase` | 8 Hz theta phase (rad) |
| `events` | stimulus times |
| `dff` | imaging dF/F at 30 Hz: place-tuned ROIs, and every fourth event-responsive (`n_rois`) |
| `photometry` | a dF/F trace (%) that rises after each event |
| `tuning` | one row per unit or ROI: `id`, `modality` (spikes or dff), `kind`, and its parameters (field centre and width, preferred direction or phase, peak rate) |
| `ttl` | TTL pulse times on the behaviour rig's clock, at random 0.5-1.5 s intervals |

`seed` drives behaviour, spikes and noise, and `tuning_seed` (default:
`seed`) drives the cells' tuning. For several days of one mouse, keep
`tuning_seed` fixed and change `seed`. Each day then has the same place
fields and preferred directions, with a new walk and new spikes:
`[toy.session(seed=day, tuning_seed=0) for day in (1, 2, 3)]`.

## Processed tree

`toy.processed` writes one session folder in the layout Raphael's
pipelines produce, so Raphael's loaders read it like real data:

```
{root}/sub-M{mouse}/day-{dd}/ses-{type}/
    sub-..._srt-toy_clusters.npz     spikes + units table (with `kind` etc.)  raphael.population.load
    sub-..._src-toy_traces.npz       dF/F                                      raphael.population.load
    sub-..._src-toy_rois.csv         ROI table (with `kind` etc.)
    sub-..._photometry.npz, .csv     green dF/F and z-score                    raphael.photometry.nwb.load
    sub-..._beh.csv                  time, P_x, P_y, H, S (as raphael.sync writes it)
    sub-..._events.csv               stimulus times
    sub-..._toy-truth.csv            the tuning table
```

`srt-toy` and `src-toy` mark the files as simulated, so they can't be
mistaken for a real sorter's or pipeline's output.

## Raw tree

`toy.raw` writes the BIDS-style raw layout of
[data_server.md](data_server.md):

```
{root}/sub-M{mouse}/day-{dd}/ses-{type}/
    sub-..._beh-rig.csv          behaviour on the rig's clock, ~60 Hz rows, with a ttl column
    sub-..._ephys/Record Node 101/   light Neuropixels 2.0 recording, Open Ephys format
    sub-..._ophys/sub-..._ophys_00001.tif   two-photon movie (30 Hz, 96 x 96 px, first minute)
    sub-..._photometry.doric     Doric photometry file
    sub-..._manifest.tsv         every file's size, sha256 and original name
    sub-..._toy-truth.json       the true clock relation and pulse times
```

- **The recording logs the same TTL pulses** as the behaviour CSV, but on
  its own clock: it starts 3.2 s later and drifts by 20 ppm. Running
  `raphael-sync` on the two files recovers the clock relation, which the
  truth file gives exactly. The recording is "light": a few channels of
  noise in place of 384, and two minutes long by default
  (`recording_duration_s`). Open Ephys stores a sample number and a
  timestamp for every 30 kHz sample, which is about 0.7 MB per second
  whatever the channel count.
- **The two-photon movie** shows each imaging ROI as a blob at its
  position (`px`, `py` in the tuning table), lit by its dF/F trace, over
  neuropil and noise. It covers the first minute by default
  (`imaging_duration_s`), the right size for trying suite2p.
- **`ephys='spikes'`** writes a full ground-truth Neuropixels recording
  instead: 64 channels and 20 units, with the true spike trains in
  `_ephys-truth/`. Use it for testing spike sorting. Its units are
  separate from the behavioural ones.
- **The photometry file** comes from the photometry simulator, with
  bleaching, motion and a known signal. Its signal is separate from the
  session's events.

## Also

- `raphael.plots.example_data()` is the small fixed session behind the
  plotting gallery.
- `raphael.ml.example.population()` is the population used in the ML
  notebook.
- `raphael.sync.simulate`, `raphael.photometry.benchmark` and
  `raphael.ephys.benchmark` are the per-pipeline simulators that `toy`
  builds on.
