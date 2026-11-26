"""In vivo calcium imaging: inspect raw files, write processed NWB, validate.

Settings and publication metadata: configs/imaging.yaml (two-photon),
configs/imaging_1p.yaml (one-photon). Processing itself runs as job stages
(`raphael-jobs make --stage suite2p | cnmfe`).

Usage:
    raphael-imaging inspect RAW.nwb
    raphael-imaging from-dandi RAW.nwb --out ROOT [--planes 1 2] [--max-frames N]
    raphael-imaging validate ROOT_OR_FILES...
    raphael-imaging inscopix-sessions RAW_ROOT --out sessions.csv
    raphael-imaging port-helping-behaviour --analysis DIR --raw-table T.csv... --out OUT

`port-helping-behaviour` is TEMPORARY: it ports the earlier helping-behaviour
analysis's CaImAn models and tracking into the processed tree
(`src-cnmfehb`) until those recordings are reprocessed through Raphael.

`inscopix-sessions` lists every raw Inscopix recording under RAW_ROOT
(animal ID, start time, frames, rate, microscope settings, from each
.isxd's own description) with each mouse's recording day, which names the
`day-{dd}` folders: day 1 is the mouse's first recording date, habituation
included.

`from-dandi` re-exports the processed data inside a DANDI 000402 (MICrONS)
file into our clark2025-style tree. It needs no processing of our own, so
it is the end-to-end test of the output template.
"""

import argparse
from pathlib import Path

import numpy as np

from raphael.imaging import dandi000402 as dandi
from raphael.imaging import nwb, qc
from raphael.imaging.layout import DEFAULT_CONFIG, load_settings


def inspect(path):
    with dandi.open_file(path) as f:
        meta = dandi.metadata(f)
        print(f'{path}\n  {meta["session_description"][:200]}...')
        print(f'  subject {meta["subject"]}, indicator {meta["indicator"]}, '
              f'location {meta["location"]}')
        for name in dandi.plane_names(f):
            i = int(name.removeprefix('TwoPhotonSeries'))
            movie = dandi.raw_movie(f, i)
            t = f[f'acquisition/{name}/timestamps']
            n_rois = f[f'processing/ophys/Fluorescence/RoiResponseSeries{i}/data'].shape[1]
            print(f'  plane {i}: movie {movie.shape} {movie.dtype}, '
                  f'{1 / np.median(np.diff(t[:100])):.2f} Hz, {n_rois} ROIs')


def from_dandi(path, out, settings, planes=None, max_frames=None):
    session = dandi.session_for(path)
    pipeline = 'microns'
    with dandi.open_file(path) as f:
        meta = dandi.metadata(f)
        planes = planes or [int(n.removeprefix('TwoPhotonSeries'))
                            for n in dandi.plane_names(f)]
        processed = []
        for i in planes:
            # No dF/F: the authors' F is background-subtracted (baselines near
            # or below zero, minima around -350), so (F - F0) / F0 is
            # undefined for much of it. traces.npz therefore holds F.
            plane = dandi.read_plane(f, i, max_frames=max_frames)
            processed.append(plane)
            print(f'plane {i}: {len(plane.masks)} ROIs, {len(plane.timestamps)} frames')
        max_time = float(processed[0].timestamps[-1]) if max_frames else None
        beh_series = dandi.behaviour(f, max_time=max_time)
        stimuli = dandi.stimulus_intervals(f, max_time=max_time)

    # The publication fields describe the source study, not our lab.
    source_settings = {**settings, 'nwb': {
        **settings['nwb'],
        'experiment_description': meta['experiment_description'],
        'related_publications': meta['related_publications'],
        'keywords': ['two-photon', 'calcium imaging', 'visual cortex', 'MICrONS'],
        'surgery': None, 'virus': None, 'pharmacology': None, 'data_collection': None,
        'lab': None, 'institution': None, 'experimenter': None,
    }}
    subject = {k: v for k, v in meta['subject'].items() if v}
    notes = {
        'source': 'DANDI 000402 (MICrONS functional data), re-exported',
        'source_file': Path(path).name,
        'planes': planes,
        'max_frames': max_frames,
        'dff': 'not computed: source F is background-subtracted',
    }
    common = {'session_start_time': meta['session_start_time'], 'subject': subject,
              'notes': notes}

    beh_file = nwb.new_nwbfile(source_settings, session,
                               f'Behaviour for {session.stem}. ' + meta['session_description'],
                               **common)
    nwb.build_beh_nwb(beh_file, beh_series, stimuli, settings)
    print(f'-> {nwb.write(beh_file, session.beh(out))}')

    ophys_file = nwb.new_nwbfile(source_settings, session,
                                 f'Processed two-photon data for {session.stem}. '
                                 + meta['session_description'], **common)
    ophys_settings = {**settings, 'microscope': {
        'name': 'Microscope', 'description': meta['microscope'], 'manufacturer': None}}
    nwb.build_ophys_nwb(ophys_file, processed, ophys_settings,
                        indicator=meta['indicator'], location=meta['location'])
    print(f'-> {nwb.write(ophys_file, session.ophys(out, pipeline))}')

    nwb.export_tables(processed, session.traces(out, pipeline), session.rois(out, pipeline))
    print(f'-> {session.traces(out, pipeline)}\n-> {session.rois(out, pipeline)}')
    for i, plane in zip(planes, processed, strict=True):
        print(f'-> {qc.plane_figure(plane, i, session.qc(out))}')


def inscopix_sessions(root, out, settings, raw_only=True):
    from raphael.imaging import inscopix

    table = inscopix.sessions(root, settings['nwb'].get('timezone', 'UTC'), raw_only)
    if table.empty:
        raise SystemExit(f'no raw .isxd recordings under {root}')
    table.insert(2, 'day', inscopix.day_numbers(table))
    table = table.sort_values(['mouse', 'start_local'])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f'{len(table)} recordings of {table.mouse.nunique()} mice -> {out}')
    return table


def validate(paths):
    from nwbinspector import inspect_nwbfile

    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob('*.nwb')) if p.is_dir() else [p]
    worst = 0
    for path in files:
        messages = list(inspect_nwbfile(nwbfile_path=str(path)))
        print(f'{path.name}: {len(messages) or "no"} issues')
        for m in messages:
            print(f'  [{m.importance.name}] {m.check_function_name}: {m.message}')
            worst = max(worst, m.importance.value)
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inspect')
    p.add_argument('path', type=Path)
    p = sub.add_parser('from-dandi')
    p.add_argument('path', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--planes', type=int, nargs='*', default=None)
    p.add_argument('--max-frames', type=int, default=None)
    p = sub.add_parser('validate')
    p.add_argument('paths', nargs='+')
    p = sub.add_parser('port-helping-behaviour',
                       help='TEMPORARY: port the earlier helping-behaviour outputs')
    p.add_argument('--analysis', type=Path, required=True,
                   help='the Calcium-Imaging-Analysis folder')
    p.add_argument('--raw-table', type=Path, nargs='*', default=[],
                   help='inscopix-sessions tables of the raw recordings (for days, start times)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--session', nargs=2, action='append', metavar=('MOUSE', 'SESSION'),
                   help='only this session (repeatable), e.g. --session 8472 HBTD3')
    p = sub.add_parser('inscopix-sessions')
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--all', action='store_true',
                   help='include .isxd files written after acquisition (no session.json)')
    args = parser.parse_args()

    settings = load_settings(args.config)
    if args.command == 'inspect':
        inspect(args.path)
    elif args.command == 'from-dandi':
        from_dandi(args.path, args.out, settings, args.planes, args.max_frames)
    elif args.command == 'validate':
        validate(args.paths)
    elif args.command == 'port-helping-behaviour':
        from raphael.imaging import helping_behaviour

        helping_behaviour.port(args.analysis, args.out, args.raw_table, settings,
                               sessions={tuple(s) for s in args.session} if args.session else None)
    elif args.command == 'inscopix-sessions':
        inscopix_sessions(args.root, args.out, settings, raw_only=not args.all)


if __name__ == '__main__':
    main()
