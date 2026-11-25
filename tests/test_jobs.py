"""Job bundles and GPU checks, run on the CPU so they pass anywhere."""

import json
from pathlib import Path

import pytest
import yaml

from raphael.cluster import gpu, jobs


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / 'data'
    (root / 'raw').mkdir(parents=True)
    (root / 'raw' / 'movie.bin').write_bytes(b'\x01' * 4096)
    return root


@pytest.fixture
def site(tmp_path):
    cfg = yaml.safe_load(jobs.SITE_CONFIG.read_text())
    cfg['paths'].update(repo='/cluster/social-manifolds', data='/cluster/data',
                        jobs='/cluster/jobs')
    cfg['slurm']['queue'] = '--partition=gpu'
    path = tmp_path / 'cluster.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return path


def make_cpu_job(tmp_path, data_root, site, **kw):
    return jobs.make(tmp_path / 'jobs' / 'smoke-1', 'smoke', ['raw/movie.bin'], data_root,
                     resources={'gpu': False}, site=site, **kw)


def test_make_and_run(tmp_path, data_root, site):
    job = make_cpu_job(tmp_path, data_root, site)
    spec = jobs.load(job)
    assert spec['inputs'][0]['bytes'] == 4096 and len(spec['inputs'][0]['sha256']) == 64
    assert jobs.read_status(job)['state'] == 'PENDING'

    jobs.run(job, data_root)
    assert jobs.read_status(job)['state'] == 'DONE'
    assert json.loads((job / 'outputs' / 'smoke.json').read_text())['device'] == 'cpu'
    env = json.loads((job / 'logs' / 'env.json').read_text())
    assert env['uv_lock_sha256'] == spec['uv_lock_sha256'] and 'pyyaml' in env['packages']
    assert jobs.summary(tmp_path / 'jobs') == [
        ('smoke-1', 'smoke', 'DONE', jobs.read_status(job)['time'])]


def test_changed_input_fails_before_stage(tmp_path, data_root, site):
    job = make_cpu_job(tmp_path, data_root, site)
    (data_root / 'raw' / 'movie.bin').write_bytes(b'\x02' * 4096)  # same size
    with pytest.raises(jobs.JobError, match='checksum'):
        jobs.run(job, data_root)
    jobs.run(job, data_root, hash_inputs=False)  # sizes alone still match

    (data_root / 'raw' / 'movie.bin').write_bytes(b'\x02' * 100)
    with pytest.raises(jobs.JobError, match='partial'):
        jobs.run(job, data_root, hash_inputs=False)
    status = jobs.read_status(job)
    assert status['state'] == 'FAILED' and 'partial' in status['error']
    assert (job / 'logs' / 'traceback.txt').exists()


def test_environment_must_match_lockfile(tmp_path, data_root, site):
    job = make_cpu_job(tmp_path, data_root, site)
    spec = jobs.load(job)
    spec['uv_lock_sha256'] = '0' * 64
    (job / 'job.yaml').write_text(yaml.safe_dump(spec))
    with pytest.raises(jobs.JobError, match='uv.lock'):
        jobs.run(job, data_root)
    jobs.run(job, data_root, allow_env_mismatch=True)


def test_bad_inputs_rejected(tmp_path, data_root, site):
    with pytest.raises(jobs.JobError, match='relative'):
        jobs.make(tmp_path / 'j', 'smoke', [str(data_root / 'raw' / 'movie.bin')],
                  data_root, site=site)
    with pytest.raises(jobs.JobError, match='not found'):
        jobs.make(tmp_path / 'j', 'smoke', ['raw/nope.bin'], data_root, site=site)
    with pytest.raises(jobs.JobError, match='unknown stage'):
        jobs.make(tmp_path / 'j', 'nope', [], data_root, site=site)


def test_render(tmp_path, data_root, site):
    job = jobs.make(tmp_path / 'jobs' / 'gpu-1', 'smoke', ['raw/movie.bin'], data_root,
                    resources={'cpus': 8, 'mem_gb': 30}, site=site)
    slurm = (job / 'submit.slurm').read_text()
    assert '#SBATCH --gres=gpu:a16:1' in slurm and '#SBATCH --partition=gpu' in slurm
    assert '#SBATCH --mem=30G' in slurm
    assert 'raphael-jobs run "/cluster/jobs/gpu-1"' in slurm
    assert '--data-root "/cluster/data"' in slurm and '@{' not in slurm
    assert 'raphael-jobs gpu-check' in slurm                 # GPU jobs report the GPU
    sge = (job / 'submit.sge').read_text()
    assert '#$ -pe sharedmem 8' in sge and '#$ -l h_vmem=4G' in sge  # per core
    assert '#$ -l gpu=1' in sge

    cpu = make_cpu_job(tmp_path, data_root, site)
    assert '--gres' not in (cpu / 'submit.slurm').read_text()


def test_gpu_type(tmp_path, data_root, site):
    job = jobs.make(tmp_path / 'jobs' / 'h100-1', 'smoke', [], data_root,
                    resources={'gpu_type': 'h100'}, site=site)
    slurm = (job / 'submit.slurm').read_text()
    assert '#SBATCH --qos=h100' in slurm and '#SBATCH --account=neurop' in slurm
    assert '#SBATCH --gres=gpu:h100:1' in slurm
    long = jobs.make(tmp_path / 'jobs' / 'long-1', 'smoke', [], data_root,
                     resources={'gpu_type': 'a100_long'}, site=site)
    long_script = (long / 'submit.slurm').read_text()
    assert '--gres=gpu:a100:1' in long_script and '--qos=a100_long' in long_script
    default = jobs.make(tmp_path / 'jobs' / 'default-1', 'smoke', [], data_root, site=site)
    assert '#SBATCH --qos=a16' in (default / 'submit.slurm').read_text()


def test_unfilled_site_config_skips_scripts(tmp_path, data_root, site, capsys):
    cfg = yaml.safe_load(site.read_text())
    cfg['paths']['repo'] = 'TODO'
    site.write_text(yaml.safe_dump(cfg))
    job = jobs.make(tmp_path / 'j', 'smoke', [], data_root, site=site)
    assert 'paths.repo' in capsys.readouterr().out
    assert not (job / 'submit.slurm').exists()


def test_gpu_check_diagnoses(monkeypatch):
    assert gpu.check('cpu')[0] == 'cpu'
    monkeypatch.setattr(gpu, 'report', lambda: {'torch': None})
    with pytest.raises(gpu.GpuUnavailable, match='uv sync'):
        gpu.check('cuda')
    monkeypatch.setattr(gpu, 'report', lambda: {
        'torch': '2.14.1', 'torch_cuda_build': '12.6', 'cuda_available': False,
        'nvidia_smi': None, 'CUDA_VISIBLE_DEVICES': None})
    with pytest.raises(gpu.GpuUnavailable, match='GPU node'):
        gpu.check('cuda')
    monkeypatch.setattr(gpu, 'report', lambda: {
        'torch': '2.14.1', 'torch_cuda_build': '12.6', 'cuda_available': True,
        'devices': [{'free_gb': 4.0}]})
    with pytest.raises(gpu.GpuUnavailable, match='needs 16'):
        gpu.check('cuda', min_memory_gb=16)


def test_cpu_jobs_use_the_cpu_request(tmp_path, data_root, site):
    cfg = yaml.safe_load(site.read_text())
    cfg['slurm']['cpu_request'] = ['--partition=cpu', '--qos=cpu']
    site.write_text(yaml.safe_dump(cfg))
    job = make_cpu_job(tmp_path, data_root, site)
    lines = [l for l in (job / 'submit.slurm').read_text().splitlines()
             if l.startswith('#SBATCH --')]
    assert '#SBATCH --partition=cpu' in lines and '#SBATCH --qos=cpu' in lines
    assert not any('gres' in l or 'partition=gpu' in l for l in lines)
    assert 'gpu-check' not in (job / 'submit.slurm').read_text()    # no GPU to report


def test_project_stages_register_through_entry_points(tmp_path, monkeypatch):
    """A project's stage module, named under `raphael.stages`, is imported
    and the stages it defines join jobs.STAGES."""
    from importlib.metadata import EntryPoint

    (tmp_path / 'fake_project_stages.py').write_text(
        'from raphael.cluster.jobs import stage\n\n'
        "@stage('fake_project_stage')\n"
        'def fake(ctx):\n    pass\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    ep = EntryPoint('fake', 'fake_project_stages', jobs.ENTRY_POINT_GROUP)
    monkeypatch.setattr(jobs, 'entry_points',
                        lambda group: [ep] if group == jobs.ENTRY_POINT_GROUP else [])
    try:
        assert jobs.load_project_stages() == ['fake']
        assert 'fake_project_stage' in jobs.STAGES
    finally:
        jobs.STAGES.pop('fake_project_stage', None)


def test_provenance_uses_the_project_lockfile(tmp_path, monkeypatch):
    from raphael.cluster import provenance

    (tmp_path / 'uv.lock').write_text('lock')
    (tmp_path / 'sub').mkdir()
    monkeypatch.delenv('RAPHAEL_PROJECT_ROOT', raising=False)
    monkeypatch.chdir(tmp_path / 'sub')
    assert provenance.project_root() == tmp_path
    assert provenance.lock_hash() == provenance.sha256(tmp_path / 'uv.lock')
    assert 'version' in provenance.environment()['raphael']


def test_site_paths_are_filled_with_the_users_login(tmp_path, monkeypatch):
    (tmp_path / 'myproject').mkdir()
    (tmp_path / 'myproject' / 'uv.lock').write_text('lock')
    monkeypatch.setenv('RAPHAEL_PROJECT_ROOT', str(tmp_path / 'myproject'))
    monkeypatch.delenv(jobs.USER_ENV)
    with pytest.raises(jobs.JobError, match='cluster-user'):
        jobs.load_site(jobs.DEFAULT_SITE_CONFIG)

    jobs.set_cluster_user('ab12')
    cfg = jobs.load_site(jobs.DEFAULT_SITE_CONFIG)
    assert cfg['paths']['repo'] == '/msc/home/ab12/myproject'
    assert cfg['paths']['jobs'] == '/msc/home/ab12/jobs'
    assert cfg['env']['UV_CACHE_DIR'] == '/msc/home/ab12/.cache/uv'
    assert cfg['slurm']['gpu_request'][1] == '--qos={gpu_type}'   # left for render


def test_personal_overrides_merge_over_the_site(tmp_path, monkeypatch):
    monkeypatch.setenv('RAPHAEL_PROJECT_ROOT', str(tmp_path))
    jobs.USER_CONFIG.parent.mkdir(parents=True)
    jobs.USER_CONFIG.write_text(yaml.safe_dump(
        {'user': 'ignored', 'paths': {'data': '/msc/home/{user}/scratch'}}))
    cfg = jobs.load_site(jobs.DEFAULT_SITE_CONFIG)    # RAPHAEL_CLUSTER_USER wins
    assert cfg['paths']['data'] == '/msc/home/tester/scratch'
    assert cfg['paths']['jobs'] == '/msc/home/tester/jobs'
    assert cfg['slurm']['account'] == '--account=neurop'


def test_recorded_paths_use_forward_slashes(tmp_path, data_root, site):
    """A job made on Windows must run on the (Linux) cluster: every path it
    records uses '/', never '\\'."""
    nested = data_root / 'raw' / 'M1' / 'D2' / 'OF1'
    (nested / 'deeper').mkdir(parents=True)
    (nested / 'deeper' / 'a.bin').write_bytes(b'x' * 10)
    rel = str(Path('raw') / 'M1' / 'D2' / 'OF1')           # backslashes on Windows
    job = jobs.make(tmp_path / 'jobs' / 'nested', 'smoke', [rel],
                    data_root, resources={'gpu': False}, site=site)
    spec = jobs.load(job)
    paths = [spec['inputs'][0]['path']] + [f['path'] for f in spec['inputs'][0]['files']]
    assert paths == ['raw/M1/D2/OF1', 'deeper/a.bin']
    jobs.verify_inputs(spec, data_root)
