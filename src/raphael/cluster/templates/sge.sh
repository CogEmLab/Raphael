#!/bin/bash
# Rendered into each job bundle as submit.sge by `raphael-jobs render`, for
# Grid Engine clusters (e.g. Edinburgh's Eddie). Edit this template or
# configs/cluster.yaml, not the rendered copies.
#$ -N @{job_id}
#$ -pe @{pe} @{cpus}
#$ -l h_vmem=@{mem_per_cpu_gb}G
#$ -l h_rt=@{hours}:00:00
#$ -o @{job_dir}/logs/scheduler.out
#$ -j y
@{directives}
set -euo pipefail

@{setup}
@{env}

# The environment is installed once per checkout with
# `uv sync --frozen --extra gpu` (a login node is fine); jobs never resolve
# or install anything, so concurrent jobs cannot race on .venv.
cd "@{repo}"
command -v uv >/dev/null || { echo "uv not on PATH: add it under setup: in configs/cluster.yaml" >&2; exit 1; }
test -d .venv || { echo "no .venv in @{repo}: run 'uv sync --frozen --extra gpu' first" >&2; exit 1; }

@{gpu_check}uv run --frozen --no-sync raphael-jobs run "@{job_dir}" \
    --data-root "@{data}" --work "@{work}"
