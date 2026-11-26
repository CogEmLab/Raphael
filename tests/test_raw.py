"""Raw recordings recognised from their files, and the plan, file lists and
job bundles made from them (raphael.raw, raphael.cluster.batch). The raw
tree is the toy dataset's (docs/data_server.md layout) plus small
hand-written Inscopix and SpikeGLX recordings."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from test_imaging_1p import write_isxd

from raphael import raw, toy
from raphael.cluster import batch, jobs
from raphael.ephys.layout import Session


@pytest.fixture(scope='module')
def tree(tmp_path_factory):
    """raw/ and metadata/ of one mouse: day 1 with Neuropixels (Open Ephys),
    two-photon and photometry; day 2 a miniscope session; day 3 a SpikeGLX
    session."""
    root = tmp_path_factory.mktemp('lab')
    toy.raw(root / 'raw', Session('1', '1', 'OF1'), duration=40.0, recording_duration_s=5.0,
            imaging_duration_s=2.0, n_place=2, n_hd=1, n_speed=1, n_theta=1, n_event=1,
            n_untuned=1, n_rois=2)
    ophys = root / 'raw/M1/D2/HBT/sub-M1_ses-D2_typ-HBT_ophys'
    ophys.mkdir(parents=True)
    movie = np.random.default_rng(0).integers(0, 4000, (12, 8, 8)).astype(np.uint16)
    write_isxd(ophys / 'video.isxd', movie)                        # 59.99 Hz
    (ophys / 'video.gpio').write_bytes(b'g')
    (ophys / 'session.json').write_text('{}')
    probe = root / 'raw/M1/D3/VR/sub-M1_ses-D3_typ-VR_ephys/run_g0/run_g0_imec0'
    probe.mkdir(parents=True)
    (probe / 'run_g0_t0.imec0.ap.meta').write_text('fileCreateTime=2026-11-03T10:00:05\n')
    (probe / 'run_g0_t0.imec0.ap.bin').write_bytes(b'\0' * 64)
    # The describing files at each level (docs/data_server.md).
    (root / 'raw/M1/D1/OF1/sub-M1_ses-D1_typ-OF1_session.yaml').write_text(
        'start: 2026-11-01T09:30:00\ntask: OF\nexperimenter: [hclark]\n')
    (root / 'raw/M1/D3/VR/sub-M1_ses-D3_typ-VR_session.yaml').write_text(
        'start: 2026-11-03T10:00:00\ntask: VR\n')
    (root / 'raw/M1/sub-M1_subject.yaml').write_text(
        'mouse: 1\nsex: F\ndate_of_birth: 2026-06-01\nspecies: Mus musculus\n'
        'strain: C57BL/6J\ngenotype: wild type\n')
    (root / 'raw/M1/sub-M1_procedures.yaml').write_text(
        '- {date: 2026-09-01, procedure: probe implant, label: ephys, target: CA1}\n')
    return root


def test_scan_recognises_every_modality(tree):
    found = {(r.modality, r.format): r for r in raw.scan(tree / 'raw')}
    assert set(found) == {('neuropixels', 'openephys'), ('neuropixels', 'spikeglx'),
                          ('two_photon', 'tiff'), ('photometry', 'doric'),
                          ('one_photon', 'inscopix')}
    ephys = found['neuropixels', 'openephys']
    assert ephys.path == 'M1/D1/OF1/sub-M1_ses-D1_typ-OF1_ephys'     # the wrapping folder
    assert (ephys.mouse, ephys.day, ephys.session_type) == ('1', 1, 'OF1')
    assert ephys.cohort is None                       # raw/M1/...: no cohort level
    assert ephys.stages == ('kilosort4', 'lfp') and ephys.start is None   # toy: no date
    assert found['neuropixels', 'spikeglx'].start == '2026-11-03T10:00:05'
    tif = found['two_photon', 'tiff']
    assert tif.path.endswith('_ophys/sub-M1_ses-D1_typ-OF1_ophys_00001.tif')
    assert tif.rate_hz == 30.0 and tif.session_type == 'OF1'
    assert found['photometry', 'doric'].start.startswith('2025-08-28T14:08:22')
    one = found['one_photon', 'inscopix']
    assert one.path == 'M1/D2/HBT/sub-M1_ses-D2_typ-HBT_ophys/video.isxd'
    assert one.session_type == 'HBT' and one.rate_hz == pytest.approx(59.988, abs=0.01)
    assert one.start.startswith('2024-06-05T17:13:10')


def test_what_is_not_a_recording(tmp_path):
    """Inscopix movies without nVista's session.json (conversions, IMU as
    .isxd), Inscopix TIFF exports and stray TIFFs outside _ophys folders."""
    movie = np.zeros((3, 4, 4), np.uint16)
    write_isxd(tmp_path / 'converted_video.isxd', movie)
    import tifffile
    tifffile.imwrite(tmp_path / 'image.tif', movie)
    assert raw.scan(tmp_path) == []
    (tmp_path / 'session.json').write_text('{}')
    tifffile.imwrite(tmp_path / 'converted_video.tif', movie)    # Inscopix's export
    assert [r.format for r in raw.scan(tmp_path)] == ['inscopix']


def test_unclosed_inscopix_movie_uses_its_recovered_copy(tmp_path):
    (tmp_path / 'session.json').write_text('{}')
    (tmp_path / 'a_video.isxd').write_bytes(b'\0' * 100)          # never closed
    write_isxd(tmp_path / 'a_video_recovered.isxd', np.zeros((3, 4, 4), np.uint16))
    [rec] = raw.scan(tmp_path)
    assert rec.path == 'a_video_recovered.isxd' and rec.start
    assert 'using a_video_recovered.isxd' in rec.note


def test_openephys_start_from_sync_messages(tmp_path):
    node = tmp_path / 'rec' / 'Record Node 101'
    node.mkdir(parents=True)
    (node / 'sync_messages.txt').write_text(
        'Software Time (milliseconds since midnight Jan 1st 1970 UTC): 1762162205000\n')
    rec = raw.detect(tmp_path / 'rec', tmp_path)
    assert rec.format == 'openephys' and rec.start == '2025-11-03T10:30:05+01:00'


def test_plan_files_and_jobs(tree, tmp_path):
    root = tree / 'raw'
    table, skipped = batch.plan(raw.scan(root), root, batch_size=2)
    assert skipped == []
    by_id = table.set_index('job_id')
    assert list(table['job_id']) == [
        'kilosort4-b01-M1-D1', 'lfp-b01-M1-D1', 'suite2p-b01-M1-D1-OF1',
        'photometry-b02-M1-D1-OF1', 'cnmfe-b02-M1-D2-HBT', 'kilosort4-b03-M1-D3',
        'lfp-b03-M1-D3']
    ks = by_id.loc['kilosort4-b01-M1-D1']
    assert ks['session_start'] == '2026-11-01T09:30:00'         # from the session file
    assert ks['gpu'] and ks['gpu_type'] == 'a16' and ks['ready']
    assert by_id.loc['kilosort4-b03-M1-D3', 'session_start'] == '2026-11-03T10:00:05'  # file
    s2p = by_id.loc['suite2p-b01-M1-D1-OF1']
    assert s2p['fs'] == 30.0 and s2p['ready']
    cn = by_id.loc['cnmfe-b02-M1-D2-HBT']
    assert not cn['gpu'] and cn['mem_gb'] == 16 and cn['sex'] == 'F'

    path = tmp_path / 'plan.csv'
    table.to_csv(path, index=False)
    plan = batch.read_plan(path)
    files = batch.batch_files(plan, 2, root)
    assert 'M1/D2/HBT/sub-M1_ses-D2_typ-HBT_ophys/video.gpio' in files
    assert 'M1/D2/HBT/sub-M1_ses-D2_typ-HBT_ophys/session.json' in files
    ephys_files = batch.batch_files(plan, 3, root)
    probe = 'M1/D3/VR/sub-M1_ses-D3_typ-VR_ephys/run_g0/run_g0_imec0'
    assert ephys_files == [f'{probe}/run_g0_t0.imec0.ap.bin', f'{probe}/run_g0_t0.imec0.ap.meta']

    site = yaml.safe_load(jobs.SITE_CONFIG.read_text())
    site['paths'].update(repo='/cluster/repo', data='/cluster/data', jobs='/cluster/jobs')
    (tmp_path / 'cluster.yaml').write_text(yaml.safe_dump(site))
    made, _, not_ready = batch.make_jobs(plan, 1, root, tmp_path / 'jobs',
                                         hash_inputs=False, site=tmp_path / 'cluster.yaml')
    assert [j.name for j in made] == ['kilosort4-b01-M1-D1', 'lfp-b01-M1-D1',
                                      'suite2p-b01-M1-D1-OF1'] and not not_ready
    spec = jobs.load(made[0])
    assert spec['stage'] == 'kilosort4'
    assert spec['params'] == {'mouse': '1', 'day': 1, 'session_type': 'OF1',
                              'session_start': '2026-11-01T09:30:00',
                              'subject': {'sex': 'F', 'date_of_birth': '2026-06-01',
                                          'strain': 'C57BL/6J', 'genotype': 'wild type',
                                          'species': 'Mus musculus'}}
    assert spec['resources']['gpu_type'] == 'a16'
    assert (made[0] / 'config' / 'ephys.yaml').exists()
    assert (made[2] / 'config' / 'imaging.yaml').exists()
    assert jobs.load(made[2])['params']['fs'] == 30.0
    again = batch.make_jobs(plan, 1, root, tmp_path / 'jobs', hash_inputs=False,
                            site=tmp_path / 'cluster.yaml')
    assert again[0] == [] and len(again[1]) == 3                  # made once only


def test_plan_without_start_times_marks_jobs_not_ready(tree, tmp_path):
    """A session with no start time in its files or its session file."""
    import shutil
    copy = tmp_path / 'raw'
    shutil.copytree(tree / 'raw' / 'M1' / 'D1', copy / 'M1' / 'D1')
    (copy / 'M1/D1/OF1/sub-M1_ses-D1_typ-OF1_session.yaml').unlink()
    table, _ = batch.plan(raw.scan(copy), copy)
    ks = table.set_index('job_id').loc['kilosort4-b01-M1-D1']
    assert not ks['ready'] and 'needs session_start' in ks['note']


def test_describing_files_at_each_level(tree):
    rec = raw.detect(tree / 'raw/M1/D1/OF1/sub-M1_ses-D1_typ-OF1_ephys', tree / 'raw')
    info = raw.describing(tree / 'raw' / rec.path, rec)
    assert info['session']['task'] == 'OF' and info['subject']['sex'] == 'F'
    assert info['procedures'][0]['label'] == 'ephys' and info['cohort'] is None
    assert set(info['paths']) == {'session', 'subject', 'procedures'}
    assert raw.describing(tree / 'raw/M1/D1/OF1/sub-M1_ses-D1_typ-OF1_beh-rig.csv') == {}


def test_session_table_names_recordings_outside_the_layout(tmp_path):
    """The helping-behaviour case: raw folders as recorded, names from a
    table (the port's: cohort + raw_recording, type)."""
    folder = tmp_path / 'May_2024' / 'some_session'
    folder.mkdir(parents=True)
    write_isxd(folder / 'm_video.isxd', np.zeros((6, 4, 4), np.uint16),
               period=(25000, 1000000))                              # 40 Hz
    (folder / 'session.json').write_text('{}')
    found = raw.scan(tmp_path)
    assert found[0].mouse is None
    sessions = pd.DataFrame({'cohort': ['May_2024'], 'raw_recording': ['some_session/m_video.isxd'],
                             'mouse': ['8472'], 'day': [6], 'type': ['HBTD3']})
    table, skipped = batch.plan(found, tmp_path, sessions)
    assert skipped == [] and list(table['job_id']) == ['cnmfe-b01-M8472-D6-HBTD3']
    assert table['cohort'][0] == 'May_2024'                       # from the table
    assert json.loads(table['params'][0]) == {'cnmfe': {'temporal_downsample': 1}}
    assert '40.00 Hz kept at its own rate' in table['note'][0]
    assert batch.plan(found, tmp_path)[1][0][0] == 'May_2024/some_session/m_video.isxd'


def test_make_without_a_stage_reads_the_inputs(tree):
    stage, params, resources, config, _ = batch.auto_job(
        ['M1/D1/OF1/sub-M1_ses-D1_typ-OF1_photometry.doric'], tree / 'raw')
    assert stage == 'photometry' and config.name == 'photometry.yaml'
    assert params['session_type'] == 'OF1' and params['subject']['sex'] == 'F'
    assert resources['gpu'] is False
    stage, params, *_ = batch.auto_job(['M1/D1/OF1/sub-M1_ses-D1_typ-OF1_ephys'], tree / 'raw')
    assert stage == 'kilosort4' and params['session_start'] == '2026-11-01T09:30:00'
    assert batch.auto_job(['M1/D1/OF1/sub-M1_ses-D1_typ-OF1_ephys'], tree / 'raw',
                          stage='lfp')[0] == 'lfp'
    assert batch.auto_job(['M1/D1/OF1/sub-M1_ses-D1_typ-OF1_beh-rig.csv'], tree / 'raw') is None


def test_cohort_level_and_collecting_outputs(tmp_path):
    """raw/{cohort}/M{mouse}/D{day}/{type}/: the cohort comes from the path,
    goes into the plan, and a batch's outputs are copied into
    processed/{cohort}/."""
    ophys = tmp_path / 'raw/2024-05_dHPC/M7/D1/HBT1/sub-M7_ses-D1_typ-HBT1_ophys'
    ophys.mkdir(parents=True)
    write_isxd(ophys / 'v.isxd', np.zeros((6, 4, 4), np.uint16))
    (ophys / 'session.json').write_text('{}')
    for root in (tmp_path / 'raw', tmp_path / 'raw' / '2024-05_dHPC'):   # whole tree or one cohort
        [rec] = raw.scan(root)
        assert (rec.cohort, rec.mouse, rec.day, rec.session_type) == ('2024-05_dHPC', '7', 1,
                                                                       'HBT1')
    table, _ = batch.plan(raw.scan(tmp_path / 'raw'), tmp_path / 'raw')
    assert table['cohort'][0] == '2024-05_dHPC'

    outputs = tmp_path / 'jobs' / table['job_id'][0] / 'outputs'
    (outputs / 'M7' / 'D1' / 'HBT1').mkdir(parents=True)
    (outputs / 'cnmfe_summary.json').write_text('{}')            # not part of the tree
    [(job_id, dest, command)] = batch.collect_commands(table, 1, tmp_path / 'jobs',
                                                        tmp_path / 'processed')
    assert dest == tmp_path / 'processed' / '2024-05_dHPC'
    assert command == ['rsync', '-a', (outputs / 'M7').as_posix(), f'{dest.as_posix()}/']
    [(_, _, remote)] = batch.collect_commands(table, 1, 'me@host:jobs', tmp_path / 'processed')
    assert remote[2] == f'me@host:jobs/{job_id}/outputs/M*'


def test_the_data_template_matches_the_code():
    """templates/data (the tree to copy onto the server): its example is
    found and named as the docs say, its describing files are read, and the
    templates hold every field Raphael reads."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / 'templates' / 'data'
    for path in root.rglob('*.yaml'):
        yaml.safe_load(path.read_text())                       # every file parses
    [rec] = raw.scan(root / 'raw')
    assert (rec.cohort, rec.mouse, rec.day, rec.session_type) == ('2027-01_2P-Testing', '1', 1,
                                                                   'OF1')
    assert rec.modality == 'two_photon' and 'no movie' in rec.note   # an empty example folder
    info = raw.describing(root / 'raw' / rec.path, rec)
    assert info['cohort']['cohort'] == rec.cohort
    assert info['session']['start'] and info['subject']['sex']
    assert [p['label'] for p in info['procedures'] if p.get('label')] == ['ophys', 'ophys']
    templates = root / 'lab' / 'templates'
    subject = yaml.safe_load((templates / 'subject.yaml').read_text())
    assert set(batch.SUBJECT_KEYS) <= set(subject)
    session = yaml.safe_load((templates / 'session.yaml').read_text())
    assert {'start', 'task', 'experimenter', 'project'} <= set(session)
