"""Set the macOS OpenMP workaround before any test module imports torch.

suite2p and Kilosort pull in torch, and on macOS torch and numpy can each
load their own OpenMP runtime, which aborts the process without these.
"""

import os

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import pytest


@pytest.fixture(autouse=True)
def _isolated_cluster_user(tmp_path, monkeypatch):
    """Tests never read the real ~/.config/raphael, and render as `tester`."""
    from raphael.cluster import jobs

    monkeypatch.setattr(jobs, 'USER_CONFIG', tmp_path / 'user_config' / 'cluster.yaml')
    monkeypatch.setenv(jobs.USER_ENV, 'tester')
