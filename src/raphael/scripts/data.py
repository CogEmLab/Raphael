"""The data server's data/ tree (incoming/, raw/, processed/, metadata/).

Usage:
    raphael-data catalogue [DATA_ROOT]     rebuild metadata/catalogue.csv

`catalogue` lists every session in raw/ with what it recorded and whether
and when each of its default stages has run (from processed/'s provenance).
Ingest and `raphael-jobs collect` rebuild it themselves. See
docs/data_server.md.
"""

import argparse
from pathlib import Path

from raphael import dataserver


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('catalogue', help='rebuild metadata/catalogue.csv')
    p.add_argument('root', type=Path, nargs='?', default=Path('.'),
                   help='the data/ folder (default: here)')
    args = parser.parse_args()
    if args.command == 'catalogue':
        root = dataserver.data_root(args.root)
        if root is None:
            raise SystemExit(f'{args.root} is not inside a data/ folder (with raw/ and metadata/)')
        rows = dataserver.catalogue(root)
        todo = sum(r['needs_processing'] == 'yes' for r in rows)
        print(f'{len(rows)} sessions ({todo} need processing) -> '
              f'{root / "metadata" / dataserver.CATALOGUE}')


if __name__ == '__main__':
    main()
