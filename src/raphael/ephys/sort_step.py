"""The sorting step of the kilosort4 job stage: a day's recordings ->
bad channels and phase shift per shank -> Kilosort 4.0.16 per shank.

`run(spec)` takes: inputs (recording folders, in session order), format,
stream, preprocessing (configs/ephys.yaml's section), kilosort
(Kilosort/wrapper parameters), device, n_jobs, out. It writes to `out`:

    spikes.npz, units.csv   every shank's units, numbered 0..N-1
                            (sorting.write_sorting)
    channels.csv            every channel: shank, position, bad-channel label
    kilosort4/{shank}/      Kilosort's own output per shank (for phy)
    sort_summary.json       versions, run time, peak GPU memory, parameters

and returns the summary.
"""

import json
import time
from importlib.metadata import version
from pathlib import Path

import spikeinterface.full as si

from raphael.ephys import sorting as srt


def run(spec):
    out = Path(spec['out'])
    out.mkdir(parents=True, exist_ok=True)
    si.set_global_job_kwargs(n_jobs=spec.get('n_jobs', 1), progress_bar=False)
    recordings = [srt.read(path, spec.get('format'), spec.get('stream'))
                  for path in spec['inputs']]
    start = time.perf_counter()
    shanks, channels = srt.prepare(recordings, spec['preprocessing'])
    channels.to_csv(out / 'channels.csv', index=False)
    prepare_s = time.perf_counter() - start

    if spec['device'] == 'cuda':
        import torch
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    sortings = srt.sort(shanks, out / 'kilosort4', spec['device'], spec.get('kilosort'))
    kilosort_s = time.perf_counter() - start
    # Kilosort >= 4.0.20 logs GPU memory (and resets torch's counter);
    # 4.0.16 does neither, so torch's peak is the measure there.
    peak = srt.kilosort_peak_gpu_gb(out / 'kilosort4')
    if peak is None and spec['device'] == 'cuda':
        peak = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    srt.write_sorting(sortings, out, recordings[0].get_sampling_frequency())

    summary = {
        'versions': {p: version(p) for p in ('kilosort', 'spikeinterface', 'numpy', 'torch')},
        'shanks': {str(k): len(v.unit_ids) for k, v in sortings.items()},
        'session_samples': [r.get_num_samples() for r in recordings],
        'prepare_s': round(prepare_s, 1),
        'kilosort_s': round(kilosort_s, 1),
        'peak_gpu_memory_gb': peak,
        'kilosort_params': srt.sorter_params(out / 'kilosort4'),
    }
    (out / 'sort_summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    return summary
