#!/bin/bash
# Rendered into each job bundle as submit.slurm by `raphael-jobs render`.
# Edit this template or configs/cluster.yaml, not the rendered copies.
#SBATCH --job-name=@{job_id}
#SBATCH --cpus-per-task=@{cpus}
#SBATCH --mem=@{mem_gb}G
#SBATCH --time=@{hours}:00:00
#SBATCH --output=@{job_dir}/logs/scheduler-%j.out
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

uv run --frozen --no-sync raphael-jobs gpu-check || true
uv run --frozen --no-sync raphael-jobs run "@{job_dir}" \
    --data-root "@{data}" --work "@{work}"
