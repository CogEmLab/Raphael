"""TEMPORARY: the helping-behaviour analysis's processed data, ported into
Raphael's processed tree so analysis can start before Raphael reprocesses it.

**This module will be removed.** Every helping-behaviour recording is to be
reprocessed through Raphael's own pipeline (job stage `cnmfe`, then the
behaviour stages), which writes the same tree under `src-cnmfe`. Until
then, this reads what the earlier analysis
(CogEmLab/helping-behaviour-calcium-imaging) produced and writes it in the
layout of docs/data_server.md, tagged `src-cnmfehb` so the two never mix:

    {out}/processed/M{mouse}/D{day}/{type}/
        sub-…_src-cnmfehb_ophys.nwb     every CaImAn component (accepted and
                                        rejected), C, C + YrA, S, detrended
        sub-…_src-cnmfehb_traces.npz    C of every component (TsdFrame)
        sub-…_src-cnmfehb_rois.csv      accepted, snr, r_value, centroid
        sub-…_beh.nwb                   P_x, P_y (cm), H (rad), S (cm/s), the
                                        door angle, running bouts, trial
                                        events, DLC pose (pixels)
        qc/plane1_rois.png
        provenance/port-helping-behaviour.json
    {out}/metadata/sessions.csv, subjects.csv

What is carried over as it was, not fixed (the reprocessing fixes these):
- the traces as the earlier run made them, including the sessions cut at
  32,767 frames (`truncated` in sessions.csv and the NWB notes);
- its timing: frames at the session table's fps (30) from the first frame,
  where the recording ran at 59.988 / 2 = 29.994 Hz; behaviour on the video's
  30 fps from the video's first frame, assumed to start with the imaging,
  as the earlier analysis assumed;
- the behaviour table from the start of the session to the victim's head
  crossing (HBTD sessions), as that analysis cut it.

Days are each mouse's recording days (`D1` its first recording date,
habituation included), from the raw recordings' own start times where they
are on the share, else the session table's dates.

    raphael-imaging port-helping-behaviour --analysis ANALYSIS --raw RAW --out OUT
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from raphael.imaging.layout import Session

PIPELINE = 'cnmfehb'
SERVER_PREFIX = ('/home/aagafono/linstore/CogEm/Projects/helping_behavior/Analysis/'
                 'Calcium-Imaging-Analysis/')
EVENTS = {                                  # session table column -> event name
    'timevictiminsertion(s)': 'victim_insertion',
    'timeonsetdooropening (s)': 'door_opening',
    'timevictimheadcrossing(s)': 'victim_head_crossing',
    'timevictimbodycrossing(s)': 'victim_body_crossing',
    'timevideoend(s)': 'video_end',
}
COHORTS_WITHOUT_RAW = ('Feb_2025',)


def _log(message):
    print(f'[port] {message}', flush=True)


def local_path(analysis, value):
    """A path in the session table (relative to the analysis folder, or the
    analysis server's absolute path) on this machine, or None."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace(SERVER_PREFIX, '')
    path = Path(text)
    return path if path.is_absolute() else Path(analysis) / path


def session_type(name):
    return re.sub(r'[^0-9A-Za-z]+', '', str(name))


def _mouse_id(*texts, known=()):
    """The mouse a raw recording belongs to: the first number in its animal
    ID, folder or file name that is one of the `known` mice."""
    for text in texts:
        for number in re.findall(r'\d+', str(text or '')):
            if str(int(number)) in known:
                return str(int(number))
    return None


def raw_recordings(raw_tables, known_mice):
    """Readable raw recordings (from `raphael-imaging inscopix-sessions`
    tables) with their mouse and the video stem the session table names
    (`_recovered`, `_fixed` copies matched to their original's name)."""
    raw = pd.concat([pd.read_csv(t, dtype={'mouse': str, 'animal_id': str}) for t in raw_tables],
                    ignore_index=True)
    raw = raw[raw['error'].isna()].copy()
    raw['mouse'] = [_mouse_id(a, f, n, known=known_mice) for a, f, n in
                    zip(raw['animal_id'], raw['folder'], raw['file'], strict=True)]
    raw['stem'] = raw['file'].str.replace(r'\.isxd$', '', regex=True).str.replace(
        r'_(recovered\d*|fixed)$', '', regex=True)
    raw['copy'] = raw['file'].str.contains(r'_(recovered\d*|fixed)\.isxd$', regex=True)
    return raw.sort_values(['copy', 'file'])


def session_table(analysis, raw_tables=(), table='data/SESSION_FILE_batch_cnmfe_main.csv'):
    """Every session the earlier analysis processed, one tidy row each, with
    its day, start time and source files."""
    analysis = Path(analysis)
    t = pd.read_csv(analysis / table)
    t.columns = [c.strip() for c in t.columns]
    t = t[t['excluded'].astype(str).str.upper() != 'TRUE'].copy()
    t['mouse'] = t['mouse'].astype(int).astype(str)
    t['session_type'] = t['session'].map(session_type)
    t['table_date'] = pd.to_datetime(t['date'], dayfirst=True, errors='coerce').dt.date
    raw = raw_recordings(raw_tables, set(t['mouse'])) if raw_tables else pd.DataFrame(
        columns=['mouse', 'stem', 'date', 'start_local', 'path', 'animal_dob'])
    first = raw.drop_duplicates('stem')
    t = t.merge(first[['stem', 'start_local', 'path', 'animal_dob']].rename(
        columns={'stem': 'miniscope video', 'path': 'raw_path'}), on='miniscope video', how='left')
    t['date'] = [pd.Timestamp(s).date() if isinstance(s, str) else d
                 for s, d in zip(t['start_local'], t['table_date'], strict=True)]
    dates = pd.concat([raw[['mouse', 'date']].assign(date=pd.to_datetime(raw['date']).dt.date),
                       t[['mouse', 'date']]]).dropna().drop_duplicates()
    dates['day'] = dates.groupby('mouse')['date'].rank(method='dense').astype(int)
    t = t.merge(dates, on=['mouse', 'date'], how='left')
    t['day_source'] = np.where(t['start_local'].notna(), 'raw recordings',
                               'session table dates (no raw recordings on the share)')
    return t


def read_behaviour(path, fps):
    """The earlier analysis's tracking table (results.csv: one row per video
    frame at `fps`, arena cm) as Raphael's behaviour series."""
    b = pd.read_csv(path)
    t = b['Frame'].to_numpy(float) / fps
    num = lambda c: pd.to_numeric(b[c], errors='coerce').to_numpy(np.float64)
    series = {
        'P_x': (num('X Center'), 'cm', t, ('Body centre x (weighted centroid of 7 body parts, '
                'smoothed), arena cm.')),
        'P_y': (num('Y Center'), 'cm', t, 'Body centre y, arena cm.'),
        'H': (np.deg2rad(num('Allocentric orientation') % 360), 'radians', t,
              'Allocentric head direction, back-to-front body axis.'),
        'S': (num('Instantaneous Velocity'), 'cm/s', t,
              'Speed of the body centre (displacement over 10 frames x fps).'),
        'head_x': (num('X Front'), 'cm', t, 'Front-body centroid x, arena cm.'),
        'head_y': (num('Y Front'), 'cm', t, 'Front-body centroid y, arena cm.'),
        'door_angle': (num('Egocentric orientation (to door)'), 'degrees', t,
                       "Egocentric angle to the door, as the earlier analysis computed it."),
    }
    running = b['Running'].astype(str).str.upper().eq('TRUE').to_numpy()
    edges = np.diff(np.r_[0, running.astype(int), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    bouts = pd.DataFrame({'start_time': t[starts] if len(t) else [],
                          'stop_time': t[stops] if len(t) else []})
    return series, bouts


def read_pose(path, fps):
    """DLC output (pixels of the cropped 30 fps video) as Raphael's pose table."""
    from raphael.behavioural_tracking.io import read_dlc_h5

    pose = read_dlc_h5(path)
    pose.index = pd.Index(np.arange(len(pose)) / fps, name='time')
    pose.attrs = {'fps': fps, 'arena': {'units': 'px'}}
    return pose


def trials_table(row, stop):
    """The session as one NWB trial (from the video's first frame to
    `stop`, s), with the session table's event times as columns (NaN when
    the event didn't happen or wasn't scored)."""
    times = {f'{name}_time': float(pd.to_numeric(pd.Series([row.get(column)]),
                                                  errors='coerce').iloc[0])
             for column, name in EVENTS.items()}
    return pd.DataFrame({'start_time': [0.0], 'stop_time': [float(stop)], **{
        k: [v] for k, v in times.items()}})


def _tracking_files(analysis, row):
    results = local_path(analysis, row.get('DLC results'))
    if results is None or not results.exists():
        return None, None
    h5 = results.parent.parent.parent / f'{results.parent.name}.h5'
    return results, (h5 if h5.exists() else None)


def port_session(row, analysis, out, settings, environment):
    """Write one session; returns its row for metadata/sessions.csv."""
    import tifffile

    from raphael.cluster import provenance
    from raphael.imaging import cnmfe, nwb, qc

    session = Session(row['mouse'], str(int(row['day'])), row['session_type'])
    processed = Path(out) / 'processed'
    fps = float(row['fps (ci downsampled)'])
    model = local_path(analysis, row['cnmfe model path'])
    result = cnmfe.from_caiman_hdf5(model)
    for name, column in (('correlation', 'correlation image path'),
                         ('pnr', 'peak-to-noise ratio path')):
        image = local_path(analysis, row.get(column))
        result[name] = tifffile.imread(image) if image and image.exists() else None
    frames = result['C'].shape[1]
    truncated = frames == 32767
    plane = cnmfe.to_plane(result, np.arange(frames) / fps, fps,
                           description=f'CaImAn CNMF-E (helping-behaviour run) on {model.name}.')
    start = pd.Timestamp(row['start_local']).to_pydatetime() if isinstance(
        row.get('start_local'), str) else nwb.parse_start_time(str(row['date']),
                                                               settings['nwb']['timezone'])
    subject = {'sex': str(row['sex']).upper()[:1] if str(row['sex']).upper()[:1] in 'MF' else 'U'}
    dob = pd.to_datetime(row.get('animal_dob'), dayfirst=True, errors='coerce')
    if pd.notna(dob):
        subject['date_of_birth'] = nwb.parse_start_time(dob.date().isoformat(),
                                                        settings['nwb']['timezone'])
    notes = {
        'pipeline': PIPELINE, 'ported_by': 'raphael.imaging.helping_behaviour (temporary)',
        'source': 'CogEmLab/helping-behaviour-calcium-imaging, batch_cnmfe.py',
        'model': str(model), 'raw_recording': row.get('raw_path'),
        'fps_assumed': fps, 'truncated_at_32767_frames': bool(truncated),
        'day_source': row['day_source'],
        'start_time_source': 'raw .isxd' if isinstance(row.get('start_local'), str)
        else 'session table date only (time unknown)',
    }
    ophys = nwb.new_nwbfile(settings, session,
                            f'One-photon imaging of {session.stem}, processed by the '
                            'helping-behaviour analysis (CaImAn CNMF-E), ported to Raphael.',
                            session_start_time=start, subject=subject, notes=notes)
    nwb.build_ophys_nwb(ophys, [plane], settings,
                        description='Processed one-photon (miniscope) imaging data.')
    nwb.write(ophys, session.ophys(processed, PIPELINE))
    nwb.export_tables([plane], session.traces(processed, PIPELINE),
                      session.rois(processed, PIPELINE), trace='denoised')
    qc.plane_figure(plane, 1, session.qc(processed))

    inputs = [model] + [p for p in (local_path(analysis, row.get('correlation image path')),
                                    local_path(analysis, row.get('peak-to-noise ratio path')))
                        if p and p.exists()]
    results, pose_file = _tracking_files(analysis, row)
    if results is not None:
        series, bouts = read_behaviour(results, fps)
        video_end = trials_table(row, 0)['video_end_time'].iloc[0]
        stop = video_end if np.isfinite(video_end) else series['P_x'][2][-1]
        intervals = {'trials': (trials_table(row, stop), (
                         'The session, with event times from the session table (s from the '
                         'first video frame): victim insertion, door opening, victim head and '
                         'body crossing, video end.')),
                     'running': (bouts, 'Running bouts, as the earlier analysis detected them.')}
        pose = read_pose(pose_file, fps) if pose_file else None
        video = local_path(analysis, f"data/{row['cohort']}/{row['session']}/bottom_camera/"
                           f"{pose_file.name.split('DLC_')[0]}.mp4") if pose_file else None
        beh = nwb.new_nwbfile(settings, session, f'Behaviour of {session.stem}, from the '
                              'helping-behaviour analysis\'s tracking, ported to Raphael.',
                              session_start_time=start, subject=subject,
                              notes={**notes, 'tracking': str(results), 'pose': str(pose_file)})
        nwb.build_beh_nwb(beh, series, intervals, settings, pose=pose,
                          scorer=pose_file.name.split('DLC_')[1].split('_el')[0]
                          if pose_file else None,
                          video=video, camera='BottomCamera')
        nwb.write(beh, session.beh(processed))
        inputs += [p for p in (results, pose_file) if p]

    record = {'stage': 'port-helping-behaviour', 'temporary': True, 'raphael': environment['raphael'],
              'environment': {k: environment[k] for k in ('python', 'platform', 'packages')},
              'params': {'pipeline': PIPELINE, 'fps': fps},
              'inputs': [{'path': str(p), 'bytes': p.stat().st_size,
                          'sha256': provenance.sha256(p)} for p in inputs]}
    provenance.write_records(session.folder(processed), record, 'port-helping-behaviour')
    _log(f"{row['cohort']} {row['session']} M{row['mouse']} -> {session.stem}: "
         f"{len(plane.masks)} components ({int(plane.rois['accepted'].sum())} accepted), "
         f"{frames} frames{' TRUNCATED' if truncated else ''}"
         f"{', behaviour' if results is not None else ', no tracking'}")
    return {'n_components': len(plane.masks), 'n_accepted': int(plane.rois['accepted'].sum()),
            'frames': frames, 'truncated': truncated, 'has_behaviour': results is not None,
            'has_pose': pose_file is not None}


def port(analysis, out, raw_tables=(), settings=None, sessions=None):
    """Port every session (or the `(mouse, session)` pairs in `sessions`),
    and write metadata/sessions.csv and subjects.csv."""
    from raphael.cluster import provenance
    from raphael.imaging.layout import load_settings

    settings = settings or load_settings(provenance.PACKAGE_ROOT / 'configs' / 'imaging_1p.yaml')
    table = session_table(analysis, raw_tables)
    if sessions:
        table = table[[(m, s) in sessions for m, s in zip(table['mouse'], table['session'],
                                                          strict=True)]]
    environment = provenance.environment()
    rows = []
    for row in table.to_dict('records'):
        try:
            rows.append({**row, **port_session(row, analysis, out, settings, environment)})
        except (OSError, KeyError, ValueError) as e:
            _log(f"skipped {row['cohort']} {row['session']} M{row['mouse']}: {e}")
            rows.append({**row, 'error': str(e)})
    done = pd.DataFrame(rows)
    keep = {'mouse': 'mouse', 'day': 'day', 'session_type': 'type', 'session': 'session',
            'cohort': 'cohort', 'date': 'date', 'start_local': 'start_time', 'sex': 'sex',
            'pair': 'pair', 'dayofhelping': 'day_of_helping', 'session outcome': 'session_outcome',
            'final outcome': 'final_outcome', 'latency': 'latency_s',
            'fps (ci downsampled)': 'fps', **{c: f'{n}_s' for c, n in EVENTS.items()},
            'n_components': 'n_components', 'n_accepted': 'n_accepted', 'frames': 'frames',
            'truncated': 'truncated', 'has_behaviour': 'has_behaviour', 'has_pose': 'has_pose',
            'day_source': 'day_source', 'miniscope video': 'video', 'raw_path': 'raw_recording',
            'error': 'error'}
    meta = Path(out) / 'metadata'
    meta.mkdir(parents=True, exist_ok=True)
    sessions_out = done[[c for c in keep if c in done]].rename(columns=keep)
    sessions_out.sort_values(['mouse', 'day', 'type']).to_csv(meta / 'sessions.csv', index=False)
    subjects = done.groupby('mouse').agg(sex=('sex', 'first'), date_of_birth=(
        'animal_dob', lambda s: next((v for v in s if isinstance(v, str)), None)),
        cohort=('cohort', 'first')).reset_index()
    subjects['date_of_birth'] = pd.to_datetime(subjects['date_of_birth'], dayfirst=True,
                                               errors='coerce').dt.date
    subjects['strain'] = settings['subject'].get('strain')
    subjects.to_csv(meta / 'subjects.csv', index=False)
    (meta / 'README.md').write_text(
        '# Helping-behaviour sessions (temporary port)\n\n'
        'Ported by raphael.imaging.helping_behaviour from the earlier analysis. Replaced '
        'when the recordings are reprocessed through Raphael (src-cnmfe). See that '
        "module's docstring for what is carried over unchanged.\n")
    _log(f'{len(done)} sessions, {int(done.get("error", pd.Series(dtype=str)).notna().sum())} '
         f'skipped -> {out}')
    return sessions_out

