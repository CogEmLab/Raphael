"""Simulated Neuropixels 2.0 recordings with known spikes, and scoring a
sorting against them: the ephys counterpart of imaging/benchmark.py.

`simulate` writes a day of recordings (one spikeinterface folder per
session, int16 in µV like acquisition data) and the true spike trains.
After the `kilosort4` stage has sorted them, `score` compares each
session's analyser with the truth: a sorted unit matches a true unit when
they share enough spikes (within 0.4 ms), and accuracy = matched spikes /
(matched + missed + extra).

    uv run python -m raphael.ephys.benchmark simulate ROOT
    uv run python -m raphael.ephys.benchmark score ROOT JOB_OUTPUTS

"""

import argparse
import json
from pathlib import Path

import numpy as np
import spikeinterface.full as si

FS = 30000.0


def np2_probe(per_shank=96):
    """An NP2.0 four-shank probe recording the lowest `per_shank` contacts
    of each shank: 4 x 96 = 384 channels, ~720 µm of each shank."""
    from probeinterface import get_probe

    full = get_probe('imec', 'NP2014')
    shank, y = full.shank_ids.astype(int), full.contact_positions[:, 1]
    keep = np.concatenate([np.flatnonzero(shank == s)[np.argsort(y[shank == s])][:per_shank]
                           for s in range(4)])
    probe = full.get_slice(keep)
    probe.set_device_channel_indices(np.arange(len(keep)))
    return probe


def simulate(root, durations=(180.0, 180.0), num_units=150, seed=0, per_shank=96):
    """`root`/session{i}/ (recordings) and `root`/truth/session{i}/ (spikes)."""
    root = Path(root)
    # Simulated noise is independent between channels, unlike real probe
    # data, where neighbours share signal. Bad-channel detection relies on
    # that sharing; run per shank (as the kilosort4 stage does) it rejects
    # a few simulated channels (4 of 384 here).
    probe = np2_probe(per_shank)
    recording, truth = si.generate_ground_truth_recording(
        durations=[float(sum(durations))], sampling_frequency=FS, num_units=num_units,
        probe=probe, seed=seed,
        generate_sorting_kwargs={'firing_rates': (1.0, 15.0), 'refractory_period_ms': 2.0},
        noise_kwargs={'noise_levels': 7.0, 'strategy': 'on_the_fly'})   # µV
    truth = truth.rename_units(np.arange(num_units))
    start = 0
    for i, seconds in enumerate(durations):
        n = int(seconds * FS)
        recording.frame_slice(start, start + n).astype('int16', round=True).save(
            folder=root / f'session{i}', progress_bar=True)
        truth.frame_slice(start, start + n).save(folder=root / 'truth' / f'session{i}')
        start += n
    (root / 'truth' / 'simulation.json').write_text(json.dumps(
        {'durations_s': list(durations), 'num_units': num_units, 'seed': seed,
         'channels': recording.get_num_channels()}, indent=2))
    return root


def score(truth, sorting, units=None, well_detected=0.8):
    """Per-session summary, and per-true-unit accuracy. With the units
    table, also how many accepted units are real (matched) units."""
    comparison = si.compare_sorter_to_ground_truth(truth, sorting, exhaustive_gt=True)
    performance = comparison.get_performance()
    summary = {
        'true_units': len(truth.unit_ids),
        'sorted_units': len(sorting.unit_ids),
        'well_detected': len(comparison.get_well_detected_units(well_detected)),
        'false_positive': len(comparison.get_false_positive_units()),
        'redundant': len(comparison.get_redundant_units()),
        # over all true units, so units never found count as 0
        'mean_accuracy': round(float(performance['accuracy'].mean()), 3),
        'mean_precision': round(float(performance['precision'].mean()), 3),
        'mean_recall': round(float(performance['recall'].mean()), 3),
    }
    if units is not None:
        matched = {int(u) for u in comparison.hungarian_match_21.index
                   if comparison.hungarian_match_21[u] != -1}
        accepted = units.loc[units['accepted'], 'unit']
        summary['accepted'] = len(accepted)
        summary['accepted_matched'] = int(accepted.isin(matched).sum())
    return summary, performance


def score_job(root, outputs, sorter='kilosort4'):
    """Score every session a job sorted from `simulate`d data, from its
    `_clusters.npz` files."""
    import pynapple as nap

    rows = {}
    summary_file = json.loads((Path(outputs) / f'{sorter}_summary.json').read_text())
    for path in sorted(Path(outputs).rglob(f'*_srt-{sorter}_clusters.npz')):
        session_type = path.parent.name
        source = next(s['input'] for s in summary_file['sessions']
                      if s['session_type'] == session_type)
        truth = si.load(Path(root) / 'truth' / source)
        group = nap.load_file(str(path))
        fs = truth.get_sampling_frequency()
        sorting = si.NumpySorting.from_unit_dict(
            {u: np.round(group[u].index * fs).astype(np.int64) for u in group}, fs)
        units = group.metadata.rename_axis('unit').reset_index()
        rows[session_type], _ = score(truth, sorting, units)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('simulate')
    p.add_argument('root', type=Path)
    p.add_argument('--durations', type=float, nargs='+', default=[180.0, 180.0])
    p.add_argument('--units', type=int, default=150)
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('score')
    p.add_argument('root', type=Path)
    p.add_argument('outputs', type=Path)
    args = parser.parse_args()
    if args.command == 'simulate':
        si.set_global_job_kwargs(n_jobs=-1, chunk_duration='1s')
        simulate(args.root, args.durations, args.units, args.seed)
    else:
        print(json.dumps(score_job(args.root, args.outputs), indent=2))


if __name__ == '__main__':
    main()
