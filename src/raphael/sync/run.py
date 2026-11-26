"""Sync one session: behaviour CSV + Open Ephys recording -> QC report, and
(if the checks pass) the behaviour on the recording's clock, in the
processed tree:

    {root}/sub-M{mouse}/day-{dd}/ses-{type}/
        sub-..._beh.csv        behaviour rows, `time` on the recording clock
                               (s from its first sample, as spike times);
                               the rig's own time kept as `time_behaviour`
        qc/sub-..._sync.json   every check, the fit, the settings, the sources
        qc/sub-..._sync.png    the QC figure

`_beh.csv` is a preliminary stand-in for `_beh.nwb`. If a check fails it
is not written, unless the failure is waived with a written reason, which
goes into the report.
"""

import json
from pathlib import Path

from raphael.sync import (
    SyncError,
    align,
    edges,
    read_behaviour,
    read_openephys,
)


def sync_session(behaviour_csv, recording, root, session, settings=None, time_column='time',
                 ttl_column='ttl', line=1, stream=None, waive=None):
    """Returns (result or None, {name: path written}). Raises SyncError if
    the files cannot be aligned at all, after writing the report."""
    from raphael import style
    from raphael.sync import qc

    folder = session.folder(root)
    (folder / 'qc').mkdir(parents=True, exist_ok=True)
    report_path = folder / 'qc' / f'{session.stem}_sync.json'
    beh = read_behaviour(behaviour_csv, time_column, ttl_column)
    rec = read_openephys(recording, line=line, stream=stream)
    sources = {'behaviour': str(behaviour_csv), 'time_column': time_column,
               'ttl_column': ttl_column, 'recording': str(rec.folder), 'stream': rec.stream,
               'ttl_line': line, 'sample_rate': rec.sample_rate,
               'first_sample': rec.first_sample, 'recording_duration_s': rec.duration,
               'behaviour_span_s': [float(beh[time_column].iloc[0]),
                                    float(beh[time_column].iloc[-1])]}
    try:
        result = align(edges(beh[time_column], beh[ttl_column]), rec.pulses,
                       behaviour_span=sources['behaviour_span_s'],
                       recording_span=(0.0, rec.duration), **(settings or {}))
    except SyncError as e:
        report_path.write_text(json.dumps({'passed': False, 'error': str(e),
                                           'sources': sources}, indent=2) + '\n')
        raise
    if not result.passed and waive:
        result.waived = str(waive)

    written = {'report': report_path}
    report_path.write_text(json.dumps({**result.summary(), 'sources': sources}, indent=2,
                                      default=float) + '\n')
    fig = qc.figure(result, title=session.stem,
                    behaviour_trace=(beh[time_column], beh[ttl_column]))
    written['figure'] = style.save(fig, folder / 'qc' / f'{session.stem}_sync',
                                   formats=('png',), close=True)[0]

    if result.usable:
        t = result.to_recording(beh[time_column])
        aligned = beh.drop(columns=[time_column]).copy()
        aligned.insert(0, 'time_behaviour', beh[time_column])
        aligned.insert(0, 'time', t.round(6))
        aligned = aligned[(aligned['time'] >= 0) & (aligned['time'] <= rec.duration)]
        path = folder / f'{session.stem}_beh.csv'
        aligned.to_csv(path, index=False)
        written['behaviour'] = path
    return result, written


def describe(result, written):
    lines = [f"sync {'PASSED' if result.passed else 'FAILED'}"
             + (f' (waived: {result.waived})' if result.waived else '')]
    for name, c in result.checks.items():
        mark = {True: 'pass', False: 'FAIL', None: '   -'}[c['passed']]
        lines.append(f'  {mark}  {name:13s} {c["detail"]}')
    lines += [f'  -> {Path(p)}' for p in written.values()]
    if not result.usable:
        lines.append('  behaviour not written: fix the cause, or re-run with --waive "reason"')
    return '\n'.join(lines)
