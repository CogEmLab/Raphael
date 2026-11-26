# Raphael: guide for coding agents

This project depends on **Raphael**, the labs' shared core for neural data.
Shared code exists so that every project processes and analyses data the
same way. Before writing processing, loading, analysis or plotting code,
**check whether Raphael already does it, and use Raphael's version.**

## Rules

1. **Use Raphael's function rather than writing your own.** This applies
   even if writing it looks quick. A second rate-map function or a second
   way of binning spikes means two projects compute "the same" result
   differently.
2. **Look before you write.** Run `uv run python -m raphael api` (or
   `api <module>`) for every public function in the installed version,
   then read the source of the ones you use. The source is in the project's
   virtual environment, under `site-packages/raphael/`.
3. **Use Raphael's names and layouts.** Processed files follow
   `{root}/sub-M{mouse}/day-{dd}/ses-{type}/sub-M{mouse}_day-{dd}_ses-{type}_...`,
   where `{root}` is `processed/{cohort}` on the data server. Data named
   before 2026-11 (`M{mouse}/D{day}/{type}/sub-M…_ses-D…_typ-…`, e.g.
   clark2025) is found with `Session.find(root, mouse, day, type)`.
   Build paths with `raphael.ephys.layout.Session`, never by hand.
   Behaviour uses `P_x`, `P_y` (cm), `H` (rad), `S` (cm/s) and a `moving`
   epoch, as pynapple objects.
4. **If Raphael almost does what you need,** use it and add only the
   difference in project code. Don't copy Raphael's function and change it.
5. **If Raphael lacks it and other projects would need it too** (anything
   that isn't specific to this project's apparatus or task), write it in
   the project for now. Then tell the user it's a candidate for Raphael,
   so it can be proposed there with tests.
6. **Never edit Raphael inside `site-packages`.** Changes to Raphael go to
   its repository (https://github.com/CogEmLab/Raphael) as a pull request.
7. **Figures use Raphael's style.** Don't set fonts, spine visibility or
   rcParams by hand. Use the style, `style.figsize` for journal sizes and
   `style.save` to save.
8. **Heavy processing runs on the cluster, not on a laptop.** Anything that
   loads a whole recording or runs for more than a few minutes (spike
   sorting, LFP, suite2p, CNMF-E, DeepLabCut training or capture, VAME)
   goes through a job bundle: make it with `raphael-jobs make` on the
   laptop, run it on the cluster. On a laptop, only make bundles, run unit
   tests and take short looks (`max_frames`), after checking free RAM and
   disk. CaImAn on one 4-minute miniscope session needs about 20 GB of RAM
   and froze an 8 GB laptop. How: Raphael's `docs/cluster.md` (bundles and
   stages) and `docs/msc_cluster.md` (MSC, step by step).

## For this task, use this

| Task | Use |
|---|---|
| Load a session's spikes or dF/F (same object for both) | `raphael.population.load(prefix)`, then `.accepted()`, `.binned(...)` |
| Paths of processed files | `raphael.ephys.layout.Session(mouse, day, session_type)`: `.folder(root)`, `.clusters(root, sorter)`, `.lfp(root)` |
| Spatial information, head-direction and speed tuning, with shuffled nulls | `raphael.tuning_scores`: `compute_spatial_information`, `compute_hd_mean_vector_length`, `compute_speed_correlation` (all take a `bin_config`) |
| Rate maps, tuning curves, head-direction polar plots, theta phase, peri-event rasters, traces, autocorrelograms, ISI and score histograms | `raphael.plots`: `rate_map`, `tuning_curve`, `head_direction`, `phase_histogram`, `peth`, `raster`, `traces`, `autocorrelogram`, `isi_histogram`, `score_histogram` |
| Figure style, sizes, panel letters, significance stars, mean +/- error bands, saving | `raphael.style`: `use`, `figsize`, `panel_label`, `significance`, `mean_with_error`, `scale_bar`, `save`, `REGIONS` |
| Encoding (behaviour to spikes) and decoding (population to behaviour), linear models to XGBoost | `raphael.ml`: `bin_counts`, `behaviour_at`, `spatial_basis`, `circular_basis`, `history`, `cross_validate`, `compare`, `encode_population`, `shuffle_test` |
| Pooling recordings with different neurons into one latent space (Unified CEBRA), decoding from it, and each neuron's attribution to a feature | `raphael.ml.unified`: `split`, `Unified` (`Unified.fit`, `Unified.transform`, `Unified.save`, `Unified.load`), `decode`, `attribution`, `shuffle_neurons`, `consistency` (needs the cebra extra) |
| Aligning behaviour with a recording from TTL pulses, with checks | `raphael.sync`: `read_behaviour`, `read_openephys`, `edges`, `align`; the command `raphael-sync` |
| Fibre photometry (Doric) to dF/F | `raphael.photometry`: `run.process_file`, and `nwb.load` to read the results |
| Spike sorting, LFP, suite2p, CNMF-E, photometry on the cluster | job stages via `raphael-jobs make --stage {kilosort4, lfp, suite2p, cnmfe, photometry, sync}` (`raphael.cluster.jobs`) |
| One-photon (Inscopix) movies: reading, binning, recording days, CaImAn CNMF-E | `raphael.imaging.inscopix`: `open_isxd`, `downsample`, `sessions`, `day_numbers`; `raphael.imaging.cnmfe.run`, `raphael.imaging.cnmfe.load`, `raphael.imaging.cnmfe.to_plane` (CaImAn runs in its own environment; job stage cnmfe); the command `raphael-imaging inscopix-sessions` |
| What a raw recording is (modality, start, rate, names) from its files; the describing files above it (session, subject, procedures, cohort) | `raphael.raw.scan`, `raphael.raw.detect`, `raphael.raw.describing`; the command `raphael-jobs detect` |
| Processing raw data: jobs for every recording, in batches | `raphael.cluster.batch.plan`, `raphael.cluster.batch.batch_files`, `raphael.cluster.batch.make_jobs`, `raphael.cluster.batch.collect_commands`, `raphael.cluster.batch.auto_job`; the commands `raphael-jobs plan`, `files`, `make-batch`, `collect`, and `make` without `--stage` (docs/processing.md) |
| Record what code and settings produced a result | job bundles do it (a `provenance/` record beside every output); `raphael.cluster.provenance.environment()` |
| Pose from video (DeepLabCut): projects, training several models on one day-held-out split, tracking, threshold sweeps, two-animal identity fixes | `raphael.behavioural_tracking`: `zoo`, `dlc`, `multipeak`, `sweep`, `identity`, `skeleton`; the command `raphael-tracking`; job stages dlc_train, dlc_capture |
| Pose to arena cm, to Raphael behaviour (P_x, P_y, H, S, moving), into the processed tree | `raphael.behavioural_tracking.io`: `load_dlc`, `to_behaviour`, `write_pose`, `read_pose` |
| Behavioural motifs (VAME) | `raphael.behavioural_tracking.vame`: `run`, `write_motifs` (VAME runs in its own environment; job stage vame) |
| Test data with known tuning, raw or processed | `raphael.toy`: `session`, `processed`, `raw` |
| NWB files and their checks | `raphael.imaging.nwb.new_nwbfile`, `raphael.imaging.nwb.write`, `raphael.imaging.nwb.build_beh_nwb` (with `pose=` for DeepLabCut keypoints); `raphael-imaging validate` |

The docs are in Raphael's repository, under `docs/`: one page per
module, plus `getting_started.md` and `using_raphael.md`.
