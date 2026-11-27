"""The data server's data/ tree (incoming/, raw/, processed/, metadata/).

Usage:
    raphael-data ingest [DATA_ROOT] [--user NAME ...] [--dry-run]
    raphael-data catalogue [DATA_ROOT]       rebuild metadata/catalogue.csv

`ingest` checks each person's hand-in in incoming/ first. If anything in it
is missing or can't be worked out, nothing of it is filed and the person is
told exactly what to clarify (incoming/{user}/PROBLEMS.md, and email when
metadata/lab.yaml has mail settings); run it again once that's fixed. A
hand-in with nothing to clarify is filed into raw/ whole, with a processing
plan (metadata/plans/plan_{date}_{time}_{user}.csv), and the catalogue is
rebuilt. One person's problems never hold up another's hand-in.
`--dry-run` reports what would happen and changes nothing. Run ingest on or
next to the server: it copies whole recordings. See docs/data_server.md.
"""

import argparse
import sys
from pathlib import Path

from raphael import dataserver


def _root(path):
    root = dataserver.data_root(path)
    if root is None:
        raise SystemExit(f'{path} is not inside a data/ folder (one with raw/ and metadata/)')
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('ingest', help='check hand-ins in incoming/ and file them into raw/')
    p.add_argument('root', type=Path, nargs='?', default=Path('.'),
                   help='the data/ folder (default: here)')
    p.add_argument('--user', dest='users', action='append', default=None,
                   help='only this person’s hand-in (repeatable)')
    p.add_argument('--dry-run', action='store_true', help='report only; change nothing')
    p = sub.add_parser('catalogue', help='rebuild metadata/catalogue.csv')
    p.add_argument('root', type=Path, nargs='?', default=Path('.'),
                   help='the data/ folder (default: here)')
    args = parser.parse_args()

    if args.command == 'catalogue':
        root = _root(args.root)
        rows = dataserver.catalogue(root)
        todo = sum(r['status'] in ('to do', 'partly') for r in rows)
        print(f'{len(rows)} sessions ({todo} to process) -> '
              f'{root / "metadata" / dataserver.CATALOGUE}')
        return
    from raphael import ingest

    reports = ingest.ingest(_root(args.root), users=args.users, dry_run=args.dry_run)
    dry = args.dry_run
    if not reports:
        print('no hand-ins in incoming/')
    for report in reports:
        print(report.text(dry_run=dry))
    if any(r.problems for r in reports):
        sys.exit(1)


if __name__ == '__main__':
    main()
