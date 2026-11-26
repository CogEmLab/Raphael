"""Prepare job bundles here, run them anywhere (cluster, GPU box, Colab).

Usage:
    raphael-jobs detect RAW_ROOT [--out recordings.csv]
    raphael-jobs plan RAW_ROOT --out plan.csv [--sessions S.csv] [--subjects SUBJ.csv]
                     [--batch-size N] [--batch-gb N] [--stage STAGE ...] [--configs DIR]
    raphael-jobs files PLAN --batch N [--data-root ROOT]
    raphael-jobs make-batch PLAN --batch N --data-root ROOT --jobs JOBS [--configs DIR]
                     [--no-hash]
    raphael-jobs collect PLAN --batch N --from JOBS --to PROCESSED [--dry-run]
    raphael-jobs cluster-user [LOGIN]     save (or show) your cluster login
    raphael-jobs gpu-check [--backend cuda|mps|any] [--min-memory-gb N]
    raphael-jobs make JOB_DIR [--stage STAGE] --data-root ROOT [--input REL ...]
                     [--param KEY=VALUE ...] [--cpu | --gpu-backend cuda|mps|any]
                     [--gpu-type TYPE]
                     [--cpus N] [--mem-gb N]
                     [--hours N] [--min-gpu-memory-gb N] [--config FILE ...]
                     [--no-hash]
    raphael-jobs render JOB_DIR [--site configs/cluster.yaml]
    raphael-jobs run JOB_DIR --data-root ROOT [--work DIR] [--no-hash]
                    [--allow-env-mismatch]
    raphael-jobs status JOBS_ROOT

`detect` lists every recording under a raw folder and what it is
(neuropixels, two_photon, one_photon, photometry), read from the files.
`make` without --stage processes its inputs with their modality's default
stage, filling in what the files and the raw layout say (mouse, day,
session type, start time, frame rate) and the resources the stage needs;
anything given on the command line wins. `plan`, `files` and `make-batch`
do the same for every recording of a project, in batches, and `collect`
copies a batch's outputs into processed/{cohort}/.

How to process data: docs/processing.md. Bundle layout and the checks `run`
performs: raphael/cluster/jobs.py. Cluster set-up: docs/cluster.md.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from raphael.cluster import gpu, jobs


def _params(pairs):
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition('=')
        if not sep:
            raise SystemExit(f'--param expects KEY=VALUE, got {pair!r}')
        out[key] = yaml.safe_load(value)  # 8192 -> int, true -> bool
    return out


def main():
    jobs.load_project_stages()    # stages a project registers (raphael.stages)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('cluster-user', help='save or show your cluster login ({user})')
    p.add_argument('login', nargs='?', default=None)

    p = sub.add_parser('gpu-check', help='prove a GPU computes here; exit 1 if not')
    p.add_argument('--backend', choices=['cuda', 'mps', 'any'], default='any')
    p.add_argument('--min-memory-gb', type=float, default=0.0)

    p = sub.add_parser('detect', help='list the recordings under a raw folder')
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, default=None, help='also write them as a CSV')

    p = sub.add_parser('plan', help="plan every recording's jobs, in batches")
    p.add_argument('root', type=Path, help='the raw data root (the data root of the jobs)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sessions', type=Path, default=None,
                   help='for recordings outside the raw layout: mouse, day, session_type '
                        '(and session_start, fs) per input path')
    p.add_argument('--subjects', type=Path, default=None,
                   help='for mice without a subject file: mouse, sex, date_of_birth, strain, '
                        'genotype, species')
    p.add_argument('--batch-size', type=int, default=12,
                   help='most recordings (an ephys day counts as one) per batch')
    p.add_argument('--batch-gb', type=float, default=400,
                   help='most GB of raw data per batch (what the cluster disk holds at once)')
    p.add_argument('--stage', dest='stages', action='append', default=None,
                   help='only these stages (repeatable); default: every default stage')
    p.add_argument('--configs', type=Path, default=None,
                   help="the project's stage configs (imaging_1p.yaml, ...), else Raphael's")

    p = sub.add_parser('files', help='the raw files to copy for one batch of a plan')
    p.add_argument('plan', type=Path)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--data-root', type=Path, default=None,
                   help='list every file of recording folders, and Inscopix companions')

    p = sub.add_parser('make-batch', help="make one batch's job bundles from a plan")
    p.add_argument('plan', type=Path)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--jobs', type=Path, required=True, help='folder the bundles go in')
    p.add_argument('--configs', type=Path, default=None)
    p.add_argument('--no-hash', action='store_true',
                   help='record input sizes only (checksums read every file)')
    p.add_argument('--site', type=Path, default=jobs.SITE_CONFIG)

    p = sub.add_parser('collect', help="copy one batch's outputs into the processed tree")
    p.add_argument('plan', type=Path)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--from', dest='source', required=True,
                   help='the jobs folder: user@host:jobs, or a local folder')
    p.add_argument('--to', dest='processed', type=Path, required=True,
                   help='the processed tree (outputs go into its {cohort}/ folders)')
    p.add_argument('--dry-run', action='store_true', help='print the rsync commands only')

    p = sub.add_parser('make', help='create a job bundle')
    p.add_argument('job_dir', type=Path)
    p.add_argument('--stage', default=None, choices=sorted(jobs.STAGES),
                   help="default: the inputs' modality's first default stage")
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--input', dest='inputs', action='append', default=[],
                   help='path relative to --data-root; repeatable')
    p.add_argument('--param', dest='params', action='append', default=[])
    p.add_argument('--config', dest='configs', action='append', type=Path, default=[])
    p.add_argument('--cpu', action='store_true', help='stage needs no GPU')
    p.add_argument('--gpu-backend', choices=['cuda', 'mps', 'any'], default='cuda',
                   help="'any' also accepts Apple-silicon GPUs, for local testing")
    p.add_argument('--gpu-type', default=None,
                   help='site GPU type, e.g. a100 or h100 on MSC (fills {gpu_type} '
                        'in configs/cluster.yaml)')
    p.add_argument('--cpus', type=int, default=None)
    p.add_argument('--mem-gb', type=int, default=None)
    p.add_argument('--hours', type=int, default=None)
    p.add_argument('--min-gpu-memory-gb', type=float, default=0.0)
    p.add_argument('--no-hash', action='store_true',
                   help='record input sizes only (sha256 of large movies takes minutes)')
    p.add_argument('--site', type=Path, default=jobs.SITE_CONFIG)

    p = sub.add_parser('render', help='(re)write scheduler scripts for a bundle')
    p.add_argument('job_dir', type=Path)
    p.add_argument('--site', type=Path, default=jobs.SITE_CONFIG)

    p = sub.add_parser('run', help='run a bundle on this machine')
    p.add_argument('job_dir', type=Path)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--work', type=Path, default=None)
    p.add_argument('--no-hash', action='store_true', help='check input sizes only')
    p.add_argument('--allow-env-mismatch', action='store_true')

    p = sub.add_parser('status', help='state of every bundle under a folder')
    p.add_argument('root', type=Path)

    args = parser.parse_args()
    if args.command == 'cluster-user':
        if args.login:
            print(f'-> {jobs.set_cluster_user(args.login)}')
        else:
            import os
            user = os.environ.get(jobs.USER_ENV) or jobs.read_user_config().get('user')
            print(user or f'not set: raphael-jobs cluster-user YOUR_LOGIN '
                          f'(saved in {jobs.USER_CONFIG})')
    elif args.command == 'gpu-check':
        try:
            device, info = gpu.check(args.backend, args.min_memory_gb)
        except gpu.GpuUnavailable as e:
            print(json.dumps(gpu.report(), indent=2, default=str))
            print(f'NO USABLE GPU: {e}', file=sys.stderr)
            sys.exit(1)
        print(json.dumps(info, indent=2, default=str))
        print(f'OK: {device}')
    elif args.command == 'detect':
        from raphael import raw

        _exists(args.root, 'the raw folder')
        found = raw.scan(args.root)
        print(raw.describe(found) or f'no recordings found under {args.root}')
        if args.out:
            raw.to_table(found).to_csv(args.out, index=False)
            print(f'{len(found)} recordings -> {args.out}')
    elif args.command == 'plan':
        _plan(args)
    elif args.command == 'files':
        from raphael.cluster import batch

        _exists(args.plan, 'the plan')
        _exists(args.data_root, 'the data root')
        print('\n'.join(batch.batch_files(batch.read_plan(args.plan), args.batch,
                                         args.data_root)))
    elif args.command == 'make-batch':
        from raphael.cluster import batch

        _exists(args.plan, 'the plan')
        _exists(args.data_root, 'the data root')
        try:
            made, existing, not_ready = batch.make_jobs(
                batch.read_plan(args.plan), args.batch, args.data_root, args.jobs,
                args.configs, hash_inputs=not args.no_hash, site=args.site)
        except jobs.JobError as e:
            print(f'NOT MADE: {e}', file=sys.stderr)
            sys.exit(1)
        for job in made:
            print(f'-> {job}')
        for job in existing:
            print(f'   {job} (already made)')
        for job_id, note in not_ready:
            print(f'   {job_id} NOT READY: {note}')
        if not (made or existing or not_ready):
            print(f'batch {args.batch} is not in {args.plan}', file=sys.stderr)
            sys.exit(1)
    elif args.command == 'collect':
        _collect(args)
    elif args.command == 'make':
        _make(args)
    elif args.command == 'render':
        for path in jobs.render(args.job_dir, args.site):
            print(f'-> {path}')
    elif args.command == 'run':
        try:
            jobs.run(args.job_dir, args.data_root, args.work,
                     hash_inputs=not args.no_hash,
                     allow_env_mismatch=args.allow_env_mismatch)
        except (jobs.JobError, gpu.GpuUnavailable) as e:
            print(f'FAILED: {e}', file=sys.stderr)
            sys.exit(1)
        print(f'DONE: {args.job_dir}')
    elif args.command == 'status':
        for row in jobs.summary(args.root):
            print('  '.join(map(str, row)))


def _exists(path, what):
    """Stop with a plain message if `path` isn't on this machine."""
    if path is not None and not Path(path).exists():
        raise SystemExit(f'{what} {path} does not exist on this machine. Raw data and '
                         'metadata are read where they are mounted (usually the laptop); '
                         'the cluster sees only what was copied to it.')


def _collect(args):
    import subprocess

    from raphael.cluster import batch

    _exists(args.plan, 'the plan')
    commands = batch.collect_commands(batch.read_plan(args.plan), args.batch, args.source,
                                      args.processed)
    if not commands:
        raise SystemExit(f'batch {args.batch} is not in {args.plan}')
    missing = []
    for job_id, dest, command in commands:
        print(' '.join(command))
        if args.dry_run:
            continue
        if len(command) < 4:                    # local jobs folder, no outputs/M* to copy
            missing.append(job_id)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        if subprocess.run(command, check=False).returncode != 0:
            missing.append(job_id)
    if missing:
        print(f'no outputs copied for {len(missing)} jobs (not finished, or failed: '
              f'raphael-jobs status on the cluster): {" ".join(missing)}', file=sys.stderr)
        sys.exit(1)


def _plan(args):
    import pandas as pd

    from raphael import raw
    from raphael.cluster import batch

    _exists(args.root, 'the raw folder')
    _exists(args.sessions, 'the session table')
    _exists(args.subjects, 'the subject table')
    found = raw.scan(args.root)
    sessions = pd.read_csv(args.sessions, dtype={'mouse': str}) if args.sessions else None
    subjects = pd.read_csv(args.subjects, dtype={'mouse': str}) if args.subjects else None
    table, skipped = batch.plan(found, args.root, sessions, subjects, args.batch_size,
                                args.batch_gb, args.stages, args.configs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    for path, why in skipped:
        print(f'  skipped {path}: {why}')
    if table.empty:
        raise SystemExit('no jobs planned')
    per_batch = table.groupby('batch').agg(
        jobs=('job_id', 'size'), stages=('stage', lambda s: ' '.join(sorted(set(s)))),
        cohorts=('cohort', lambda s: ' '.join(sorted(set(s) - {''})) or '-'),
        raw_gb=('raw_gb', 'sum'), max_mem_gb=('mem_gb', 'max'))
    print(per_batch.to_string())
    for note, n in table.loc[table['note'] != '', 'note'].value_counts().items():
        print(f'  {n} jobs: {note}')
    ready = table['ready'].sum()
    print(f'{len(table)} jobs ({ready} ready) in {table["batch"].nunique()} batches, '
          f'{len(found) - len(skipped)} recordings -> {args.out}')


def _make(args):
    """make: detect the inputs, fill in what they say, let the command line win."""
    from raphael.cluster import batch

    _exists(args.data_root, 'the data root')
    params = _params(args.params)
    inputs, configs = list(args.inputs), list(args.configs)
    auto = None
    try:
        auto = batch.auto_job(args.inputs, args.data_root, args.stage)
    except (ValueError, OSError) as e:
        print(f'NOT MADE: {e}', file=sys.stderr)
        sys.exit(1)
    stage = args.stage
    resources = dict(jobs.DEFAULT_RESOURCES)
    if auto is not None:
        stage, auto_params, auto_resources, config, row = auto
        params = {**auto_params, **params}
        resources.update(auto_resources)
        inputs = batch._inputs(row)                 # recording order, matching the params
        configs = configs or [config]
        print(f'{stage}: {row["modality"]} ({", ".join(inputs)})'
              + (f'\n  ! {row["note"]}' if row['note'] else ''))
    elif stage is None:
        print('NOT MADE: the inputs are not recordings Raphael recognises; '
              'name the stage with --stage (see raphael-jobs detect)', file=sys.stderr)
        sys.exit(1)
    if args.cpu:
        resources['gpu'] = False
    for key in ('cpus', 'mem_gb', 'hours', 'gpu_type'):
        if getattr(args, key) is not None:
            resources[key] = getattr(args, key)
    resources.update(gpu_backend=args.gpu_backend, min_gpu_memory_gb=args.min_gpu_memory_gb)
    if not resources['gpu']:
        resources['gpu_type'] = None
    try:
        job = jobs.make(args.job_dir, stage, inputs, args.data_root, params, resources,
                        configs, hash_inputs=not args.no_hash, site=args.site)
    except jobs.JobError as e:
        print(f'NOT MADE: {e}', file=sys.stderr)
        sys.exit(1)
    print(f'-> {job}')


if __name__ == '__main__':
    main()
