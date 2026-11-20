"""Fibre photometry: inspect Doric files, process them into the processed tree.

Settings, channel roles and publication metadata: configs/photometry.yaml
(pass your project's copy with --config).

Usage:
    raphael-photometry inspect FILE.doric ...
    raphael-photometry process FILE.doric --out ROOT --mouse M --day D --session-type TYPE
                       [--config configs/photometry.yaml]
    raphael-photometry simulate OUT.doric [--duration-s 300]

For provenance (inputs checksummed, environment recorded), run the same
processing as a job: `raphael-jobs make JOB --stage photometry --cpu ...`.
"""

import argparse
import json
from pathlib import Path

from raphael.photometry import doric
from raphael.photometry.layout import DEFAULT_CONFIG, Session, load_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inspect', help='channels, rates and metadata of Doric files')
    p.add_argument('paths', nargs='+', type=Path)
    p = sub.add_parser('process', help='one Doric file -> processed session')
    p.add_argument('path', type=Path)
    p.add_argument('--out', type=Path, required=True, help='processed root')
    p.add_argument('--mouse', required=True)
    p.add_argument('--day', required=True)
    p.add_argument('--session-type', required=True)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p = sub.add_parser('simulate', help='write a simulated Doric file, for trying things out')
    p.add_argument('path', type=Path)
    p.add_argument('--duration-s', type=float, default=300.0)
    args = parser.parse_args(argv)

    if args.command == 'inspect':
        for path in args.paths:
            print(doric.describe(doric.read(path)))
    elif args.command == 'process':
        from raphael.photometry.run import process_file
        session = Session(args.mouse.removeprefix('M'), args.day.removeprefix('D'),
                          args.session_type)
        summary = process_file(args.path, args.out, session, load_settings(args.config))
        print(json.dumps(summary, indent=2, default=str))
    elif args.command == 'simulate':
        from raphael.photometry import benchmark
        benchmark.simulate(args.path, args.duration_s)
        print(f'-> {args.path}')


if __name__ == '__main__':
    main()
