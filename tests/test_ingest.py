"""Ingest: hand-ins in data/incoming/ checked, filed into data/raw/ with
their recordings renamed, a plan per person, the catalogue, and feedback.
The recordings are small hand-made files in the formats Raphael reads."""

import csv
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import yaml
from test_imaging_1p import write_isxd

from raphael import ingest
from raphael.photometry import benchmark

NOW = datetime(2026, 3, 20, 9, 0, 0)
COHORT = '2026-03_dHPC'


@pytest.fixture
def root(tmp_path):
    root = tmp_path / 'data'
    meta = root / 'metadata'
    (meta / 'rigs').mkdir(parents=True)
    (root / 'raw').mkdir()
    (root / 'processed').mkdir()
    (root / 'incoming' / 'jsmith').mkdir(parents=True)
    dump = lambda p, d: p.write_text(yaml.safe_dump(d))
    dump(meta / 'people.yaml', {'jsmith': {'name': 'Smith, Jo', 'email': 'jo@example.org'},
                                'dm': {'name': 'Manager, Data', 'email': 'dm@example.org'}})
    dump(meta / 'sessions.yaml', {'OF': {'name': 'open field'}, 'VR': {'name': 'VR'},
                                  'HAB': {'name': 'habituation'}})
    dump(meta / 'projects.yaml', {'place-cells': {'owner': 'jsmith'}})
    dump(meta / 'rigs' / 'ephys-rig-1.yaml', {'rig': 'ephys-rig-1',
                                              'cameras': [{'label': 'top'}, {'label': 'side'}]})
    dump(meta / 'lab.yaml', {'lab': 'Test lab', 'timezone': 'Europe/Vienna'})
    return root


def describing(user_dir, mouse=10432, procedures=None, sex='F', cohort=COHORT):
    (user_dir / f'{cohort}_cohort.yaml').write_text(yaml.safe_dump(
        {'cohort': cohort, 'project': 'place-cells', 'owner': 'jsmith'}))
    (user_dir / f'sub-M{mouse}_subject.yaml').write_text(yaml.safe_dump(
        {'mouse': mouse, 'cohort': cohort, 'sex': sex, 'date_of_birth': '2025-12-01',
         'species': 'Mus musculus', 'strain': 'C57BL/6J'}))
    (user_dir / f'sub-M{mouse}_procedures.yaml').write_text(
        yaml.safe_dump(procedures) if procedures else '[]\n')


def openephys(folder, when):
    """An Open Ephys folder whose software time is `when` (Vienna)."""
    node = folder / 'Record Node 101'
    node.mkdir(parents=True)
    ms = int(when.replace(tzinfo=ZoneInfo('Europe/Vienna')).timestamp() * 1000)
    (node / 'sync_messages.txt').write_text(
        f'Software Time (milliseconds since midnight Jan 1st 1970 UTC): {ms}\n')
    (node / 'continuous.dat').write_bytes(b'\0' * 64)


def session(user_dir, when, mouse=10432, day=1, kind='OF1', ephys=True, extra=None,
            session_file=None):
    """A hand-in session folder named for `when`, with an Open Ephys
    recording started then, a camera file and a rig log."""
    name = f'sub-M{mouse}_day-{day:02d}_ses-{kind}_{when:%Y-%m-%d_%H-%M-%S}'
    folder = user_dir / name
    folder.mkdir()
    if ephys:
        openephys(folder / f'{when:%Y-%m-%d_%H-%M-%S}', when)
    (folder / 'cam0.mp4').write_bytes(b'video')
    (folder / 'rig_log.csv').write_text('t,x\n0,1\n')
    stem = f'sub-M{mouse}_day-{day:02d}_ses-{kind}'
    (folder / f'{stem}_session.yaml').write_text(yaml.safe_dump(
        session_file or {'task': ''.join(c for c in kind if c.isalpha()),
                         'experimenter': ['jsmith'], 'project': 'place-cells',
                         'rig': 'ephys-rig-1', 'weight_g': 24.1, 'notes': 'cable twisted'}))
    if extra:
        extra(folder)
    return folder


def files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file())


def test_a_session_is_filed_with_a_plan_and_a_catalogue_row(root):
    user = root / 'incoming' / 'jsmith'
    describing(user)
    session(user, datetime(2026, 3, 14, 10, 5, 12))
    [report] = ingest.ingest(root, now=NOW, runner='dm')
    assert report.problems == [] and report.filed == [f'raw/{COHORT}/sub-M10432/day-01/ses-OF1']
    target = root / report.filed[0]
    stem = 'sub-M10432_day-01_ses-OF1'
    assert sorted(p.name for p in target.iterdir()) == [
        f'{stem}_beh.csv', f'{stem}_ephys', f'{stem}_manifest.tsv', f'{stem}_session.yaml',
        f'{stem}_video.mp4']
    assert (target / f'{stem}_ephys' / 'Record Node 101' / 'sync_messages.txt').exists()
    described = yaml.safe_load((target / f'{stem}_session.yaml').read_text())
    assert described['start'] == '2026-03-14T10:05:12' and described['ingested_by'] == 'dm'
    assert described['notes'] == 'cable twisted' and described['task'] == 'OF'
    with open(target / f'{stem}_manifest.tsv') as f:
        manifest = {r['path']: r for r in csv.DictReader(f, delimiter='\t')}
    assert manifest[f'{stem}_video.mp4']['original_name'] == 'cam0.mp4'
    assert len(manifest[f'{stem}_video.mp4']['sha256']) == 64
    if os.name == 'posix':                                   # recordings locked
        assert not os.access(target / f'{stem}_video.mp4', os.W_OK)
    for name in ('cohort.yaml', 'sub-M10432/sub-M10432_subject.yaml',
                 'sub-M10432/sub-M10432_procedures.yaml'):
        assert (root / 'raw' / COHORT / name).exists()
    assert files(root / 'incoming') == []                    # the hand-in is emptied
    plan = root / 'metadata' / 'plans' / 'plan_2026-03-20_09-00-00_jsmith.csv'
    assert report.plan == plan.relative_to(root).as_posix()
    with open(plan) as f:
        rows = list(csv.DictReader(f))
    assert {r['stage'] for r in rows} == {'kilosort4', 'lfp'}
    assert all(r['inputs'] == f'{COHORT}/sub-M10432/day-01/ses-OF1/{stem}_ephys' for r in rows)
    assert rows[0]['ready'] == 'True'                         # start time from the recording
    with open(root / 'metadata' / 'catalogue.csv') as f:
        f.readline()
        [row] = list(csv.DictReader(f))
    assert row['plan'] == plan.stem and row['status'] == 'to do'
    assert row['ingested'].startswith('2026-03-20')


def test_check_and_dry_run_change_nothing(root):
    user = root / 'incoming' / 'jsmith'
    describing(user)
    session(user, datetime(2026, 3, 14, 10, 5, 12))
    before = files(root)
    [report] = ingest.ingest(root, dry_run=True, now=NOW)
    assert report.filed == [f'raw/{COHORT}/sub-M10432/day-01/ses-OF1'] and not report.plan
    assert 'would be added' in report.text(dry_run=True)
    assert files(root) == before


def test_problems_are_reported_and_nothing_is_filed(root):
    user = root / 'incoming' / 'jsmith'
    describing(user)
    session(user, datetime(2026, 3, 14, 10, 5, 12))
    ingest.ingest(root, now=NOW)                              # day 1 is 2026-03-14
    describing(user)
    session(user, datetime(2026, 3, 16, 10, 0, 0), day=3)     # should be day-02
    session(user, datetime(2026, 3, 17, 10, 0, 0), day=3, kind='XX1')   # unregistered code
    late = session(user, datetime(2026, 3, 18, 12, 0, 0), day=4, ephys=False)
    openephys(late / 'rec', datetime(2026, 3, 18, 10, 5, 0))  # started 2 h before the name
    (user / 'sub-M10432_procedures.yaml').unlink()
    before = files(root / 'incoming')
    [report] = ingest.ingest(root, now=NOW)
    assert report.filed == []
    text = ' '.join(f'{i} {p} {f}' for i, p, f in report.problems)
    assert 'is day-02, not day-03' in text and 'rename the folder sub-M10432_day-02' in text
    assert 'session code XX is not in metadata/sessions.yaml' in text
    assert 'started at 2026-03-18 10:05:00, the folder name says 2026-03-18 12:00:00' in text
    assert 'no sub-M10432_procedures.yaml in the hand-in' in text
    problems_md = user / 'PROBLEMS.md'
    assert problems_md.exists() and 'Nothing was added' in problems_md.read_text()
    assert files(root / 'incoming') == sorted([*before, 'jsmith/PROBLEMS.md'])


def test_days_and_runs(root):
    user = root / 'incoming' / 'jsmith'
    describing(user)
    session(user, datetime(2026, 3, 14, 10, 0, 0), kind='OF1')
    session(user, datetime(2026, 3, 14, 11, 0, 0), kind='VR1')   # same day, one hand-in
    [report] = ingest.ingest(root, now=NOW)
    assert len(report.filed) == 2 and report.problems == []
    describing(user)
    session(user, datetime(2026, 3, 14, 14, 0, 0), kind='OF2')   # a second OF that day
    session(user, datetime(2026, 3, 15, 10, 0, 0), day=2)        # the next day
    taken = session(user, datetime(2026, 3, 14, 15, 0, 0), kind='VR1')   # VR1 is taken
    [report] = ingest.ingest(root, now=NOW)
    assert report.filed == []                                    # nothing until it's clarified
    assert any('already in raw/' in p and 'ses-VR2' in f for _, p, f in report.problems)
    import shutil
    shutil.rmtree(taken)                                         # clarified: it was a duplicate
    [report] = ingest.ingest(root, now=NOW)
    assert report.problems == []
    assert report.filed == [f'raw/{COHORT}/sub-M10432/day-01/ses-OF2',
                            f'raw/{COHORT}/sub-M10432/day-02/ses-OF1']
    assert not (user / 'PROBLEMS.md').exists()


def test_describing_files_update_with_history_and_identity_needs_the_data_manager(root):
    user = root / 'incoming' / 'jsmith'
    describing(user)
    session(user, datetime(2026, 3, 14, 10, 0, 0))
    ingest.ingest(root, now=NOW)
    surgery = [{'date': '2026-03-16', 'procedure': 'fibre implant', 'label': 'dHPC'}]
    describing(user, procedures=surgery)                         # no session: a surgery
    [report] = ingest.ingest(root, now=datetime(2026, 3, 21, 9, 0, 0))
    mouse_dir = root / 'raw' / COHORT / 'sub-M10432'
    assert yaml.safe_load((mouse_dir / 'sub-M10432_procedures.yaml').read_text()) == surgery
    assert [p.name for p in (mouse_dir / '.history').iterdir()] == [
        'sub-M10432_procedures.2026-03-21_09-00-00.yaml']
    assert files(root / 'incoming') == []
    describing(user, procedures=surgery, sex='M')                # an identity change
    [report] = ingest.ingest(root, now=datetime(2026, 3, 22, 9, 0, 0))
    assert yaml.safe_load((mouse_dir / 'sub-M10432_subject.yaml').read_text())['sex'] == 'F'
    assert any('sex differ' in p for _, p, _ in report.problems)
    assert (user / 'sub-M10432_subject.yaml').exists()            # kept for the data manager


def test_photometry_files_are_named_by_their_labels(root):
    user = root / 'incoming' / 'jsmith'
    labels = [{'label': 'dHPC'}, {'label': 'mPFC'}]
    describing(user, procedures=labels)

    def two_fibres(folder):
        for region in ('dHPC', 'mPFC'):
            benchmark.simulate(folder / f'fibre_{region}.doric', duration_s=2.0,
                               start='Sat Mar 14 10:05:12 2026')
    session(user, datetime(2026, 3, 14, 10, 5, 12), ephys=False, extra=two_fibres)
    [report] = ingest.ingest(root, now=NOW)
    assert report.problems == []
    names = sorted(p.name for p in (root / report.filed[0]).iterdir())
    assert 'sub-M10432_day-01_ses-OF1_photometry-dHPC.doric' in names
    assert 'sub-M10432_day-01_ses-OF1_photometry-mPFC.doric' in names

    describing(user, procedures=labels)

    def unclear(folder):
        for name in ('fibre_a', 'fibre_b'):
            benchmark.simulate(folder / f'{name}.doric', duration_s=2.0,
                               start='Sun Mar 15 10:05:12 2026')
    session(user, datetime(2026, 3, 15, 10, 5, 12), day=2, ephys=False, extra=unclear)
    [report] = ingest.ingest(root, now=NOW)
    assert report.filed == [] and any('procedure label' in p for _, p, _ in report.problems)


def test_loose_inscopix_files_go_in_one_ophys_folder(root):
    user = root / 'incoming' / 'jsmith'
    describing(user)
    when = datetime(2026, 3, 14, 17, 13, 10)

    def miniscope(folder):
        # nVista stores the local clock time as if it were UTC.
        ms = int(when.replace(tzinfo=ZoneInfo('UTC')).timestamp() * 1000)
        write_isxd(folder / 'movie_video.isxd', np.zeros((12, 8, 8), np.uint16), start_ms=ms)
        (folder / 'movie_video.gpio').write_bytes(b'g')
        (folder / 'session.json').write_text('{}')
    session(user, when, ephys=False, extra=miniscope)
    [report] = ingest.ingest(root, now=NOW)
    assert report.problems == []
    ophys = root / report.filed[0] / 'sub-M10432_day-01_ses-OF1_ophys'
    assert sorted(p.name for p in ophys.iterdir()) == ['movie_video.gpio', 'movie_video.isxd',
                                                       'session.json']
    with open(root / report.plan) as f:
        assert [r['stage'] for r in csv.DictReader(f)] == ['cnmfe']


def test_a_damaged_copy_leaves_the_hand_in_and_raw_untouched(root, monkeypatch):
    user = root / 'incoming' / 'jsmith'
    describing(user)
    session(user, datetime(2026, 3, 14, 10, 5, 12))
    before = files(root / 'incoming')
    monkeypatch.setattr(ingest, '_sha256', lambda path: '0' * 64)   # every copy "differs"
    [report] = ingest.ingest(root, now=NOW)
    assert report.filed == [] and any('filing failed' in p for _, p, _ in report.problems)
    assert not (root / 'raw' / COHORT / 'sub-M10432' / 'day-01').exists()
    assert not (root / 'raw' / '.ingesting').exists()
    assert [f for f in files(root / 'incoming') if not f.endswith('PROBLEMS.md')] == before
    monkeypatch.undo()                                        # a rerun then files it
    [report] = ingest.ingest(root, now=NOW)
    assert report.filed and report.problems == []
    assert files(root / 'incoming') == []


def test_email_goes_to_the_person_and_the_data_managers(root, monkeypatch):
    lab = root / 'metadata' / 'lab.yaml'
    lab.write_text(yaml.safe_dump({'lab': 'Test lab', 'data_managers': ['dm'],
                                   'mail': {'host': 'smtp.example.org', 'from': 'data@x'}}))
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            assert host == 'smtp.example.org'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def send_message(self, msg):
            sent.append(msg)

    import smtplib
    monkeypatch.setattr(smtplib, 'SMTP', FakeSMTP)
    user = root / 'incoming' / 'jsmith'
    describing(user)
    session(user, datetime(2026, 3, 14, 10, 5, 12))
    [report] = ingest.ingest(root, now=NOW)
    assert len(sent) == 1 and sent[0]['To'] == 'jo@example.org, dm@example.org'
    assert '1 added' in sent[0]['Subject']
    assert ('email', 'emailed jo@example.org, dm@example.org') in report.notes


def test_one_problem_holds_back_the_whole_hand_in(root):
    user = root / 'incoming' / 'jsmith'
    describing(user)
    session(user, datetime(2026, 3, 14, 10, 0, 0))                    # fine on its own
    session(user, datetime(2026, 3, 15, 10, 0, 0), day=2, kind='XX1')   # unregistered code
    before = files(root / 'incoming')
    [report] = ingest.ingest(root, now=NOW)
    assert report.filed == [] and len(report.problems) == 1
    assert 'Nothing was added: 1 thing(s) to clarify' in report.text()
    assert files(root / 'raw') == [] and not (root / 'metadata' / 'plans').exists()
    assert files(root / 'incoming') == sorted([*before, 'jsmith/PROBLEMS.md'])


def test_the_command_stops_and_says_what_to_clarify(root, capsys, monkeypatch):
    import sys

    from raphael.scripts import data
    user = root / 'incoming' / 'jsmith'
    describing(user)
    session(user, datetime(2026, 3, 14, 10, 5, 12), day=2)      # wrong: the first day is 1
    monkeypatch.setattr(sys, 'argv', ['raphael-data', 'ingest', str(root)])
    with pytest.raises(SystemExit) as stop:
        data.main()
    assert stop.value.code == 1
    out = capsys.readouterr().out
    assert 'Nothing was added' in out and 'is day-01, not day-02' in out
    assert (user / 'PROBLEMS.md').exists() and not any(Path(root / 'raw').iterdir())


def test_a_session_folder_that_is_the_open_ephys_folder(root):
    """Open Ephys set to name its own folder: Record Node is directly inside
    the session folder, with the camera file beside it. Its software time is
    the local clock written as UTC (as on the clark2025 rig): an hour off if
    read as UTC, so the start comes from settings.xml's local DATE."""
    user = root / 'incoming' / 'jsmith'
    describing(user)
    when = datetime(2024, 11, 13, 13, 32, 25)

    def oe_is_the_folder(folder):
        node = folder / 'Record Node 109'
        (node / 'experiment1' / 'recording1').mkdir(parents=True)
        (node / 'settings.xml').write_text(
            '<SETTINGS><INFO><DATE>13 Nov 2024 13:32:25</DATE></INFO></SETTINGS>')
        ms = int(when.replace(tzinfo=ZoneInfo('UTC')).timestamp() * 1000)
        (node / 'experiment1' / 'recording1' / 'sync_messages.txt').write_text(
            f'Software Time (milliseconds since midnight Jan 1st 1970 UTC): {ms}\n')
        (node / 'experiment1' / 'recording1' / 'continuous.dat').write_bytes(b'\0' * 64)
        (folder / 'cam0.mp4').unlink()
        (folder / 'rig_log.csv').unlink()
        (folder / 'sub-25_day-25_ses-OF1_video.avi').write_bytes(b'video')
    session(user, when, ephys=False, extra=oe_is_the_folder)
    [report] = ingest.ingest(root, now=NOW)
    assert report.problems == [], report.problems
    target = root / report.filed[0]
    stem = 'sub-M10432_day-01_ses-OF1'
    assert (target / f'{stem}_ephys' / 'Record Node 109' / 'settings.xml').exists()
    assert (target / f'{stem}_ephys' / 'Record Node 109' / 'experiment1' / 'recording1'
            / 'continuous.dat').exists()
    assert (target / f'{stem}_video.avi').exists()
    with open(root / report.plan) as f:
        rows = list(csv.DictReader(f))
    assert {r['stage'] for r in rows} == {'kilosort4', 'lfp'}
    assert rows[0]['session_start'] == '2024-11-13T13:32:25'
