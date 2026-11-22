"""raphael-sync: align a behaviour CSV with a Neuropixels recording (Open
Ephys) from one shared TTL channel, check the alignment, and write the
behaviour on the recording's clock into the processed tree.

    raphael-sync BEHAVIOUR.csv RECORDING_FOLDER --out /data/processed \\
        --mouse 25 --day 18 --session-type OF1
    raphael-sync --simulate /tmp/sync-demo     # a simulated session to try it on

Exits 1 if a check fails (and writes only the QC report), unless the
failure is waived with --waive "reason".
"""

import argparse
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('behaviour', type=Path, nargs='?', help='behaviour CSV')
    p.add_argument('recording', type=Path, nargs='?', help='Open Ephys record node folder')
    p.add_argument('--out', type=Path, help='processed root')
    p.add_argument('--mouse')
    p.add_argument('--day')
    p.add_argument('--session-type')
    p.add_argument('--time-column', default='time')
    p.add_argument('--ttl-column', default='ttl')
    p.add_argument('--ttl-line', type=int, default=1, help='Open Ephys TTL line (1-based)')
    p.add_argument('--stream', help='Open Ephys stream, if not the one probe stream')
    p.add_argument('--expected-interval', type=float, help='regular pulses: interval (s)')
    p.add_argument('--expected-width', type=float, help='pulse width (s)')
    p.add_argument('--max-drift-ppm', type=float)
    p.add_argument('--residual-ms', type=float)
    p.add_argument('--waive', help='use a failed alignment anyway; the reason is recorded')
    p.add_argument('--simulate', type=Path, metavar='FOLDER',
                   help='write a simulated session (behaviour.csv + recording) and sync it')
    args = p.parse_args(argv)

    from raphael.ephys.layout import Session
    from raphael.sync import SyncError
    from raphael.sync.run import describe, sync_session

    if args.simulate:
        from raphael.sync.simulate import simulate_session
        sim = simulate_session(args.simulate)
        args.behaviour, args.recording = sim['behaviour'], sim['recording']
        args.out = args.out or args.simulate / 'processed'
        args.mouse, args.day, args.session_type = (args.mouse or '1', args.day or '1',
                                                   args.session_type or 'OF1')
        print(f'simulated: {args.behaviour}, {args.recording} '
              f'(true drift {sim["drift_ppm"]} ppm, offset {sim["intercept"]:.4f} s)')
    if not all([args.behaviour, args.recording, args.out, args.mouse, args.day,
                args.session_type]):
        p.error('need BEHAVIOUR RECORDING --out --mouse --day --session-type (or --simulate)')

    settings = {k: v for k, v in {'expected_interval_s': args.expected_interval,
                                  'expected_width_s': args.expected_width,
                                  'max_drift_ppm': args.max_drift_ppm,
                                  'residual_ms': args.residual_ms}.items() if v is not None}
    session = Session(str(args.mouse), str(args.day), args.session_type)
    try:
        result, written = sync_session(args.behaviour, args.recording, args.out, session,
                                       settings, args.time_column, args.ttl_column,
                                       args.ttl_line, args.stream, args.waive)
    except SyncError as e:
        print(f'sync FAILED: {e}', file=sys.stderr)
        return 1
    print(describe(result, written))
    return 0 if result.usable else 1


if __name__ == '__main__':
    sys.exit(main())
