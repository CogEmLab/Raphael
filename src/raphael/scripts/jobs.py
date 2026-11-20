"""Prepare job bundles here, run them anywhere (cluster, GPU box, Colab).

Usage:
    raphael-jobs cluster-user [LOGIN]     save (or show) your cluster login
    raphael-jobs gpu-check [--backend cuda|mps|any] [--min-memory-gb N]
    raphael-jobs make JOB_DIR --stage STAGE --data-root ROOT [--input REL ...]
                     [--param KEY=VALUE ...] [--cpu | --gpu-backend cuda|mps|any]
                     [--gpu-type TYPE]
                     [--cpus N] [--mem-gb N]
                     [--hours N] [--min-gpu-memory-gb N] [--config FILE ...]
                     [--no-hash]
    raphael-jobs render JOB_DIR [--site configs/cluster.yaml]
    raphael-jobs run JOB_DIR --data-root ROOT [--work DIR] [--no-hash]
                    [--allow-env-mismatch]
    raphael-jobs status JOBS_ROOT

Bundle layout and the checks `run` performs: raphael/cluster/jobs.py.
Cluster set-up: docs/cluster.md.
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

    p = sub.add_parser('make', help='create a job bundle')
    p.add_argument('job_dir', type=Path)
    p.add_argument('--stage', required=True, choices=sorted(jobs.STAGES))
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
    p.add_argument('--cpus', type=int, default=jobs.DEFAULT_RESOURCES['cpus'])
    p.add_argument('--mem-gb', type=int, default=jobs.DEFAULT_RESOURCES['mem_gb'])
    p.add_argument('--hours', type=int, default=jobs.DEFAULT_RESOURCES['hours'])
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
    elif args.command == 'make':
        resources = {'gpu': not args.cpu, 'gpu_backend': args.gpu_backend,
                     'gpu_type': args.gpu_type, 'cpus': args.cpus, 'mem_gb': args.mem_gb,
                     'hours': args.hours, 'min_gpu_memory_gb': args.min_gpu_memory_gb}
        try:
            job = jobs.make(args.job_dir, args.stage, args.inputs, args.data_root,
                            _params(args.params), resources, args.configs,
                            hash_inputs=not args.no_hash, site=args.site)
        except jobs.JobError as e:
            print(f'NOT MADE: {e}', file=sys.stderr)
            sys.exit(1)
        print(f'-> {job}')
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


if __name__ == '__main__':
    main()
