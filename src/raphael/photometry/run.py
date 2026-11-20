"""One Doric file -> processed files for one session. Used by
`raphael-photometry process` and the `photometry` job stage."""

from raphael.imaging import nwb as imaging_nwb
from raphael.photometry import doric, processing, qc
from raphael.photometry import nwb as fp_nwb
from raphael.photometry.layout import Session


def roles_from(settings, rec):
    """The configured channels present in this recording; error if a
    configured signal or its reference is missing."""
    roles = {}
    for name, spec in (settings.get('channels') or {}).items():
        if name in rec.channels:
            roles[name] = dict(spec)
        elif spec.get('role') == 'signal':
            raise ValueError(f'{rec.path.name}: configured signal channel {name!r} is not in '
                             f'the recording (has {sorted(rec.channels)})')
    if not any(s.get('role') == 'signal' for s in roles.values()):
        raise ValueError(f'{rec.path.name}: none of the configured channels '
                         f'{sorted(settings.get("channels") or {})} are in the recording '
                         f'(has {sorted(rec.channels)}); set `channels` in photometry.yaml')
    return roles


def process_file(path, root, session: Session, settings, subject=None, start_time=None,
                 notes=None):
    """Read, process and write one session; returns a summary dict."""
    rec = doric.read(path)
    roles = roles_from(settings, rec)
    proc = {**processing.DEFAULTS, **(settings.get('processing') or {})}
    # Channels of one console share a clock; use the first signal's times.
    first = next(n for n, s in roles.items() if s['role'] == 'signal')
    times = rec.channels[first].times
    channels = {}
    for name in roles:
        c = rec.channels[name]
        if len(c.times) != len(times) or abs(c.times[0] - times[0]) > 1e-6:
            raise ValueError(f'{name} and {first} are sampled at different times')
        channels[name] = c.values
    kept, out, fits = processing.process(times, channels, roles, proc)
    table = fp_nwb.channel_table(fits, roles, rec)

    timezone = settings['nwb'].get('timezone', 'UTC')
    start = start_time or rec.start_time
    if start is None:
        raise ValueError(f'{rec.path.name} has no start time; pass one')
    start = imaging_nwb.parse_start_time(start, timezone)
    file_notes = {'pipeline': 'raphael.photometry', 'input': rec.path.name,
                  'software': rec.software, 'processing': proc,
                  'ttl': {line: {'frequency_hz': m.get('Frequency'), 'period_ms': m.get('Period')}
                          for line, m in rec.digital_outputs.items()},
                  **(notes or {})}
    nwbfile = imaging_nwb.new_nwbfile(
        settings, session, f'Fibre photometry for {session.stem}.', session_start_time=start,
        subject=imaging_nwb.subject_fields(subject, timezone), notes=file_notes)
    fp_nwb.build_photometry_nwb(nwbfile, rec, kept, out, table)
    imaging_nwb.write(nwbfile, session.photometry(root))
    fp_nwb.export_tables(kept, out, table, session.photometry_traces(root),
                         session.photometry_channels(root))
    figure = qc.session_figure(rec, kept, out, fits, proc, session.qc(root), title=session.stem)
    return {'input': rec.path.name, 'session': session.stem, 'duration_s': round(
        float(times[-1] - times[0]), 1), 'rate_hz': round(rec.channels[first].rate, 2),
            'signals': {n: {'reference': fits[n].get('reference'),
                            'motion_r': fits[n].get('motion_r'),
                            'dff_sd': round(float(out[n]['dff'].std()), 3)} for n in out},
            'qc': str(figure)}
