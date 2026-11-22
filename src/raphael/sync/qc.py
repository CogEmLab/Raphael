"""The sync QC figure, written beside every synced session
(qc/sub-..._sync.png):

a  every pulse on the recording clock: paired, explained (one device not
   recording) and unpaired (red: what fails counts_agree);
b  fit residuals against the limit: a trend means drift the line missed, a
   jump means a pulse paired with the wrong partner;
c  inter-pulse intervals, behaviour against recording: on the diagonal;
   the checks and their verdict, top right;
d-f the behaviour CSV's TTL column, moved onto the recording clock, drawn
   over the recording's pulses at the start, middle and end of the session
   (or at an unpaired pulse): after a good sync the edges coincide to
   within one behaviour sample.
"""

import matplotlib.pyplot as plt
import numpy as np

from raphael import style


@style.styled
def figure(result, title='', behaviour_trace=None):
    """`behaviour_trace`: (behaviour times, TTL values) from the CSV, for d-f."""
    b_on = np.asarray(result.behaviour.onsets)
    r_on = np.asarray(result.recording.onsets)
    ia, ib = result.pairs[:, 0], result.pairs[:, 1]
    bad_a = result.unpaired.get('behaviour', np.array([], int))
    bad_b = result.unpaired.get('recording', np.array([], int))
    ok_a = np.setdiff1d(np.setdiff1d(np.arange(len(b_on)), ia), bad_a)    # explained
    ok_b = np.setdiff1d(np.setdiff1d(np.arange(len(r_on)), ib), bad_b)
    limit = result.settings['residual_ms']

    rows = 3 if behaviour_trace is not None else 2
    fig = plt.figure(figsize=style.figsize('double', height_mm=45 * rows + 5))
    grid = fig.add_gridspec(rows, 3, width_ratios=[2.2, 1, 1.1], hspace=0.75, wspace=0.45)

    ax = fig.add_subplot(grid[0, :2])
    mapped = result.to_recording(b_on)
    ax.eventplot([mapped[ia], r_on[ib]], lineoffsets=[1, 0], linelengths=0.7, colors='0.35',
                 linewidths=0.5)
    ax.eventplot([mapped[ok_a], r_on[ok_b]], lineoffsets=[1, 0], linelengths=0.7,
                 colors=style.GREYS['light'], linewidths=0.5)
    if len(bad_a) or len(bad_b):
        ax.eventplot([mapped[bad_a], r_on[bad_b]], lineoffsets=[1, 0], linelengths=0.9,
                     colors=style.CYCLE[1], linewidths=1.2)
    ax.set(yticks=[0, 1], yticklabels=['Recording', 'Behaviour'],
           xlabel='Recording time (s)',
           title=f'Pulses: {len(ia)} paired, {len(ok_a) + len(ok_b)} outside the overlap '
                 f'(light), {len(bad_a) + len(bad_b)} unpaired (red)')
    style.panel_label(ax, 'a', x=-0.1)

    ax = fig.add_subplot(grid[1, :2])
    ax.axhspan(-limit, limit, color=style.GREYS['faint'], linewidth=0)
    ax.plot(r_on[ib], result.residuals * 1e3, '.', ms=2.5, color=style.CYCLE[0])
    ax.set(xlabel='Recording time (s)', ylabel='Residual (ms)',
           title=f'Fit residuals (limit +/-{limit:.1f} ms, grey)')
    style.panel_label(ax, 'b', x=-0.1)

    ax = fig.add_subplot(grid[1, 2])
    consecutive = (np.diff(ia) == 1) & (np.diff(ib) == 1)
    da = np.diff(b_on[ia])[consecutive]
    db = np.diff(r_on[ib])[consecutive]
    ax.plot(da, db, '.', ms=2.5, color=style.CYCLE[0])
    lims = [min(da.min(), db.min()), max(da.max(), db.max())] if len(da) else [0, 1]
    ax.plot(lims, lims, color=style.GREYS['mid'], lw=0.5)
    ax.set(xlabel='Behaviour interval (s)', ylabel='Recording interval (s)',
           title='Inter-pulse intervals')
    ax.set_aspect('equal')
    style.panel_label(ax, 'c', x=-0.5)

    ax = fig.add_subplot(grid[0, 2])
    ax.set_axis_off()
    lines = []
    for name, c in result.checks.items():
        mark = {True: 'pass', False: 'FAIL', None: '-'}[c['passed']]
        lines.append((f'{mark:>4}  {name}', c['passed']))
    verdict = ('PASSED' if result.passed else
               'FAILED, waived' if result.waived else 'FAILED')
    ax.text(0, 1, f'Sync {verdict}', fontweight='bold', va='top', transform=ax.transAxes,
            color=style.CYCLE[2] if result.passed else style.CYCLE[1])
    for k, (text, ok) in enumerate(lines):
        ax.text(0, 0.86 - k * 0.11, text, va='top', transform=ax.transAxes, family='monospace',
                fontsize=style.FONT['small'], color=style.CYCLE[1] if ok is False else '0.15')
    ax.text(0, 0.86 - len(lines) * 0.11 - 0.04,
            f'drift {result.drift_ppm:.1f} +/- {result.drift_se_ppm:.1f} ppm\n'
            f'offset {result.intercept:.4f} s', va='top', transform=ax.transAxes,
            fontsize=style.FONT['small'])
    if behaviour_trace is not None:
        _overlays(fig, grid[2, :], result, behaviour_trace, bad_b)
    if title:
        fig.suptitle(title)
    return fig


def _overlays(fig, cell, result, behaviour_trace, bad_b, half_window=1.5):
    t_beh, ttl = (np.asarray(x, float) for x in behaviour_trace)
    t_rec = result.to_recording(t_beh)
    ttl = (ttl - np.nanmin(ttl)) / max(np.nanmax(ttl) - np.nanmin(ttl), 1e-12)
    r_on = np.asarray(result.recording.onsets)
    r_w = np.nan_to_num(np.asarray(result.recording.widths), nan=0.0)
    paired = r_on[np.sort(result.pairs[:, 1])]
    centres = [('start', paired[int(0.05 * (len(paired) - 1))]),
               ('middle', paired[len(paired) // 2]),
               ('end', paired[int(0.95 * (len(paired) - 1))])]
    if len(bad_b):
        centres[1] = ('unpaired pulse', r_on[bad_b[0]])
    sub = cell.subgridspec(1, 3, wspace=0.3)
    for k, (label, centre) in enumerate(centres):
        ax = fig.add_subplot(sub[0, k])
        lo, hi = centre - half_window, centre + half_window
        for on, w in zip(r_on, r_w, strict=True):
            if on + w >= lo and on <= hi:
                ax.axvspan(on, on + max(w, 0.002), ymax=0.85, color=style.GREYS['light'],
                           linewidth=0)
        keep = (t_rec >= lo - 0.1) & (t_rec <= hi + 0.1)
        ax.step(t_rec[keep], 0.85 * ttl[keep], where='post', color=style.CYCLE[0], lw=0.8)
        ax.set(xlim=(lo, hi), ylim=(-0.05, 1), yticks=[], xlabel='Recording time (s)',
               title=label.capitalize())
        ax.spines['left'].set_visible(False)
        if k == 0:
            ax.set_ylabel('Behaviour TTL (line)\non recording (grey)')
            style.panel_label(ax, 'd', x=-0.12)
