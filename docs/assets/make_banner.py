"""Raphael README banners: a red bandana mask over a head drawn in spikes.

    python docs/assets/make_banner.py          # writes both versions next to this file

raphael-banner.svg         the mask and eyes drawn as shapes
raphael-banner-spikes.svg  the mask drawn in red spikes too, eyes left blank

The head is a spike raster: high firing rates inside it, low outside, with
the mouth and nostrils as silent gaps. Fibre photometry and dF/F traces run
behind it, and the mask's tails unravel into extracellular spike traces.
"""

from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon
from scipy.spatial import cKDTree

W, H = 1200, 430
RED, DARK, SHADE = '#d62828', '#7a0f14', '#a61b1f'
SPIKES, PHOTOMETRY, DFF = '#2d6a4f', '#40916c', '#52b788'
GREENS = [(0.0, 0.5, '#40916c'), (0.5, 0.85, '#2d6a4f'), (0.85, 1.0, '#1b4332'),
          (1.0, 9.0, '#2d6a4f')]
REDS = ['#ef3b36', RED, SHADE]          # mask top to bottom
ROW = 4.5
FADE = 120          # px over which everything behind the mask fades out at the sides


def side_fade(x):
    """1 in the middle, easing to 0 at the left and right edges, so the
    banner has no visible start or end."""
    u = np.clip(np.minimum(x, W - x) / FADE, 0, 1)
    return u * u * (3 - 2 * u)


def path(points, close=False):
    d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in points)
    return d + (' Z' if close else '')


def bezier(p0, p1, p2, p3, n=40):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


# ------------------------------------------------------------------ geometry

CX, CY = 400, 180
KX, KY = 735, 182
TAILS = [(-18, 40, 3, 0.4, 30), (22, 34, 11, 2.2, 38)]   # dy, spike amp, seed, phase, swing


def band_polygon():
    """The band across the face: curved, its left end wrapping round the head."""
    xs = np.linspace(90, 720, 60)
    top = np.c_[xs, CY - 62 + 0.00035 * (xs - CX) ** 2 - 10 * np.sin((xs - 90) / 640 * np.pi)]
    xb = xs[::-1]
    bottom = np.c_[xb, CY + 58 + 0.00030 * (xb - CX) ** 2 - 6 * np.sin((xb - 90) / 640 * np.pi)]
    left = bezier(bottom[-1], (90 - 28, bottom[-1][1] - 30), (90 - 28, top[0][1] + 30), top[0])
    return np.vstack([top, bottom, left[1:]])


def eye_polygon(ex, ey, flip):
    """Almond eyehole, its inner corner pointed and lowered: a determined look."""
    s = -1 if flip else 1
    inner, outer = (ex + s * 66, ey + 12), (ex - s * 74, ey - 22)
    upper = bezier(outer, (ex - s * 30, ey - 30), (ex + s * 30, ey - 12), inner)
    lower = bezier(inner, (ex + s * 36, ey + 40), (ex - s * 50, ey + 36), outer)
    return np.vstack([upper, lower[1:]])


def knot_polygon():
    a = np.linspace(0, 2 * np.pi, 80)
    return np.c_[KX + 34 * np.cos(a), KY + 44 * np.sin(a)]


def tail_centre(x, dy, phase, swing):
    """Leaves the knot, falls away from its partner, then flutters."""
    u = (x - KX) / 250
    return (KY + dy + np.sign(dy) * 70 * np.sin(np.clip(u, 0, 1) * np.pi / 2)
            + swing * np.sin(u * 2.6 * np.pi + phase) * np.clip(u * 2, 0, 1))


def tail_polygon(dy, phase, swing):
    xs = np.linspace(KX + 20, KX + 250, 60)
    width = 30 * (1 - np.linspace(0, 1, len(xs)) ** 1.4) + 3
    c = tail_centre(xs, dy, phase, swing)
    return np.vstack([np.c_[xs, c - width / 2], np.c_[xs[::-1], (c + width / 2)[::-1]]])


BAND = band_polygon()
EYES = [eye_polygon(300, 178, False), eye_polygon(505, 178, True)]
KNOT = knot_polygon()
TAIL_POLYS = [tail_polygon(dy, ph, sw) for dy, _, _, ph, sw in TAILS]
MASK_PARTS = [BAND, KNOT, *TAIL_POLYS]


def inside(polygons, x, y):
    pts = np.c_[x, np.full_like(x, y)]
    hit = np.zeros(len(x), bool)
    for poly in polygons:
        hit |= Polygon(poly).contains_points(pts)
    return hit


# Outline points, to draw the mask's edge in darker spikes.
OUTLINE = cKDTree(np.vstack([np.vstack([np.linspace(p, q, 6) for p, q in zip(poly, np.roll(poly, -1, 0))])
                             for poly in [*MASK_PARTS, *EYES]]))

# ---------------------------------------------------------------- the head

FACE_C, FACE_R = (410.0, 225.0), (350.0, 215.0)


def face_depth(x, y):
    """0 at the centre of the head, 1 at its edge, >1 outside."""
    u = np.abs((x - FACE_C[0]) / FACE_R[0])
    v = np.abs((y - FACE_C[1]) / FACE_R[1])
    return (u ** 2.4 + v ** 2.4) ** (1 / 2.4)


def features(x, y):
    """True on the mouth and nostrils, where the rate drops."""
    dx = x - FACE_C[0]
    mouth = (np.abs(y - (342 - 0.0009 * dx ** 2 + 0.000000032 * dx ** 4)) < 6.5) & (np.abs(dx) < 175)
    nostrils = ((x - 380) ** 2 + (y - 284) ** 2 < 10 ** 2) | ((x - 440) ** 2 + (y - 284) ** 2 < 10 ** 2)
    return mouth | nostrils


def field(seed, scale):
    """A smooth random 2D field (sum of a few random plane waves), ~[-1, 1]."""
    r = np.random.default_rng(seed)
    waves = [(r.uniform(0.5, 1.5) / scale, r.uniform(0, 2 * np.pi), r.uniform(0, 2 * np.pi))
             for _ in range(6)]

    def f(x, y):
        return sum(np.sin(k * (x * np.cos(a) + y * np.sin(a)) + ph) for k, a, ph in waves) / 2.5
    return f


def smooth_noise(n, width, scale, seed):
    r = np.random.default_rng(seed)
    k = np.exp(-0.5 * (np.arange(-3 * width, 3 * width + 1) / width) ** 2)
    return np.convolve(r.normal(0, 1, n + len(k)), k / k.sum(), mode='same')[:n] * scale


def spike_trace(x0, x1, n_spikes, amp, seed, drift):
    """Extracellular-looking trace: noise plus biphasic spikes (sharp trough, slower peak)."""
    r = np.random.default_rng(seed)
    x = np.linspace(x0, x1, int((x1 - x0) * 2))
    y = np.convolve(r.normal(0, 0.9, len(x)), np.ones(3) / 3, mode='same')
    for c in np.sort(r.uniform(x0 + 15, x1 - 10, n_spikes)):
        a = amp * r.uniform(0.6, 1.0)
        dx = x - c
        y += a * np.exp(-(dx / 1.6) ** 2) - 0.38 * a * np.exp(-((dx - 4.5) / 3.2) ** 2)
    taper = np.clip((x - x0) / 40, 0, 1)                   # grows out of the ribbon
    return list(zip(x, drift(x) + y * (0.35 + 0.65 * taper)))


# ----------------------------------------------------------------- drawing

def raster(spike_mask):
    """{colour: [tick path strings]}: the head, and with `spike_mask` the
    mask in red, its edge darker, and the eyes silent."""
    rng = np.random.default_rng(7)
    extra = np.random.default_rng(8)       # the mask's own spikes, so the head is the same
    edge, patches = field(41, 40), field(42, 70)
    ticks = {c: [] for c in [g[2] for g in GREENS] + REDS + [DARK]}

    def tick(colour, x, yy, r):
        h = r.uniform(2.2, 6.0)
        ticks[colour].append(f'M{x:.0f} {yy - h / 2 + r.normal(0, 1.0):.1f}v{h:.1f}')

    top, bottom = BAND[:, 1].min(), BAND[:, 1].max()
    for yy in np.arange(ROW / 2, H, ROW):
        # Bursty spike trains: Poisson burst centres, each with a few spikes.
        centres = rng.uniform(0, W, rng.poisson(0.21 * W))
        sizes = rng.geometric(0.45, len(centres))
        xs = np.concatenate([c + np.cumsum(rng.exponential(1.6, k))
                             for c, k in zip(centres, sizes, strict=True)])
        xs = xs[(xs > 0) & (xs < W)]
        depth = face_depth(xs, yy) + 0.07 * edge(xs, yy)
        rate = np.where(depth <= 1,
                        0.6 + 0.25 * (1 - np.clip(depth, 0, 1) ** 3) + 0.2 * patches(xs, yy),
                        np.maximum(0.5 * np.exp(-(depth - 1) / 0.06), 0.03))
        rate = np.where(features(xs + rng.normal(0, 2, len(xs)), yy), 0.04, rate)
        keep = rng.random(len(xs)) < np.clip(rate * side_fade(xs), 0, 1)
        if spike_mask:
            covered = inside(MASK_PARTS, xs, yy) | inside(EYES, xs, yy)
            keep &= ~covered                  # the mask replaces the head there
        for lo, hi, colour in GREENS:
            for x in xs[keep & (depth >= lo) & (depth < hi)]:
                tick(colour, x, yy, rng)
        if not spike_mask:
            continue
        # The mask: dense red firing, darker at its edge; the eyes are silent.
        mx = extra.uniform(0, W, extra.poisson(1.1 * W))
        on = inside(MASK_PARTS, mx, yy) & ~inside(EYES, mx, yy)
        on &= extra.random(len(mx)) < 0.8 + 0.2 * patches(mx, yy + 500)
        near_edge = OUTLINE.query(np.c_[mx, np.full_like(mx, yy)])[0] < 3
        shade = REDS[min(2, int(3 * np.clip((yy - top) / (bottom - top), 0, 0.999)))]
        for x, rim in zip(mx[on], near_edge[on], strict=True):
            tick(DARK if rim else shade, x, yy, extra)
    return ticks


def traces(spike_mask):
    """Fibre photometry and dF/F behind the head; in the spike version they
    stop at the mask instead of being hidden by it."""
    out = []

    def draw(xs, ys, colour, opacity, width):
        pieces = [np.ones(len(xs), bool)]
        if spike_mask:
            pts = np.c_[xs, ys]
            hidden = np.zeros(len(xs), bool)
            for poly in MASK_PARTS:
                hidden |= Polygon(poly).contains_points(pts)
            pieces = [~hidden]
        visible = pieces[0]
        runs = np.split(np.arange(len(xs)), np.flatnonzero(np.diff(visible.astype(int))) + 1)
        for run in runs:
            if visible[run[0]] and len(run) > 1:
                out.append(f'<path d="{path(list(zip(xs[run], ys[run])))}" stroke="url(#{colour[1:]})" '
                           f'stroke-opacity="{opacity}" stroke-width="{width}" fill="none" '
                           'stroke-linejoin="round"/>')

    xs = np.linspace(0, W, 600)
    for yy, seed in [(105, 21), (262, 22)]:
        draw(xs, yy + 6 * np.exp(-xs / 300) + smooth_noise(len(xs), 18, 260, seed),
             PHOTOMETRY, 0.3, 2.5)
    xs = np.linspace(0, W, 1200)
    for yy, seed in [(150, 31), (205, 32), (250, 33)]:
        r = np.random.default_rng(seed)
        y = r.normal(0, 0.5, len(xs))
        for onset in r.uniform(0, W, 6):
            dt = np.clip(xs - onset, 0, None)
            y -= r.uniform(22, 40) * (1 - np.exp(-dt / 2.0)) * np.exp(-dt / 45) * (xs > onset)
        draw(xs, yy + y, DFF, 0.45, 2.4)
    return out


def mask_shapes():
    """The mask as filled shapes, with eyes and pupils."""
    out = []
    eyes = ' '.join(path(e, close=True) for e in EYES)
    out.append(f'<path d="{path(BAND, close=True)} {eyes}" fill="url(#band)" fill-rule="evenodd" '
               f'stroke="{DARK}" stroke-width="5" stroke-linejoin="round"/>')
    for e, px in zip(EYES, (312, 493), strict=True):
        out.append(f'<path d="{path(e, close=True)}" fill="#ffffff" stroke="{DARK}" '
                   'stroke-width="5" stroke-linejoin="round"/>')
        out.append(f'<circle cx="{px}" cy="186" r="14" fill="#1b1b1b"/>'
                   f'<circle cx="{px + 4}" cy="181" r="4.5" fill="#ffffff"/>')
    xs = np.linspace(130, 690, 40)
    fold = np.c_[xs, CY + 40 + 0.0003 * (xs - CX) ** 2 - 6 * np.sin((xs - 90) / 640 * np.pi)]
    out.append(f'<path d="{path(fold)}" stroke="{DARK}" stroke-opacity="0.35" stroke-width="3" '
               'fill="none"/>')
    out.append(f'<ellipse cx="{KX}" cy="{KY}" rx="34" ry="44" fill="url(#band)" stroke="{DARK}" '
               'stroke-width="5"/>')
    out.append(f'<path d="M{KX - 14},{KY - 28} Q{KX + 8},{KY} {KX - 12},{KY + 30}" '
               f'stroke="{DARK}" stroke-opacity="0.45" stroke-width="3" fill="none"/>')
    for poly in TAIL_POLYS:
        out.append(f'<path d="{path(poly, close=True)}" fill="url(#band)" stroke="{DARK}" '
                   'stroke-width="4" stroke-linejoin="round"/>')
    return out


def tail_traces():
    """Where each tail ends, its ribbon unravels into a spike trace."""
    out = []
    for i, (dy, amp, seed, phase, swing) in enumerate(TAILS):
        x_end = KX + 250
        level = tail_centre(x_end, dy, phase, swing) + np.sign(dy) * 8

        def settle(x, x_end=x_end, level=level, dy=dy, phase=phase, swing=swing):
            # leaves the ribbon on its curve, then flattens onto a baseline
            w = np.clip((x - x_end) / 70, 0, 1)
            w = w * w * (3 - 2 * w)
            return (1 - w) * tail_centre(np.minimum(x, x_end + 10), dy, phase, swing) + w * level
        trace = spike_trace(x_end - 10, W - 20, 11 - 2 * i, amp, seed, settle)
        out.append(f'<path d="{path(trace)}" stroke="url(#fade)" stroke-width="3" '
                   'stroke-linejoin="round" stroke-linecap="round" fill="none"/>')
    return out


def side_gradient(colour):
    """A stroke colour that fades out over FADE px at both sides."""
    f = FADE / W
    stops = [(0, 0), (f, 1), (1 - f, 1), (1, 0)]
    return (f'<linearGradient id="{colour[1:]}" gradientUnits="userSpaceOnUse" x1="0" y1="0" '
            f'x2="{W}" y2="0">' + ''.join(f'<stop offset="{o:.4f}" stop-color="{colour}" '
                                          f'stop-opacity="{a}"/>' for o, a in stops)
            + '</linearGradient>')


def banner(spike_mask):
    label = ('A bandana mask in red spikes, its eyes blank,' if spike_mask
             else 'A red bandana mask') + (' on a head drawn in green spikes; its tails turn into '
                                           'spike traces, over fibre photometry and dF/F traces')
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
           f'height="{H}" role="img" aria-label="{label}">',
           '<defs><linearGradient id="band" x1="0" y1="0" x2="0" y2="1">'
           f'<stop offset="0" stop-color="#ef3b36"/><stop offset="0.55" stop-color="{RED}"/>'
           f'<stop offset="1" stop-color="{SHADE}"/></linearGradient>'
           '<linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">'
           f'<stop offset="0" stop-color="{RED}"/>'
           f'<stop offset="1" stop-color="{RED}" stop-opacity="0.15"/></linearGradient>'
           + ''.join(side_gradient(c) for c in (PHOTOMETRY, DFF)) + '</defs>']
    ticks = raster(spike_mask)
    for _, lo_hi_colour in enumerate(GREENS):
        lo, _, colour = lo_hi_colour
        if ticks[colour]:
            out.append(f'<path d="{" ".join(ticks[colour])}" stroke="{colour}" '
                       f'stroke-opacity="{0.35 if lo >= 1.0 else 0.9}" stroke-width="2" '
                       'stroke-linecap="butt" fill="none"/>')
    out += traces(spike_mask)
    if spike_mask:
        for colour in [*REDS, DARK]:
            if ticks[colour]:
                out.append(f'<path d="{" ".join(ticks[colour])}" stroke="{colour}" '
                           'stroke-opacity="0.95" stroke-width="2" stroke-linecap="butt" '
                           'fill="none"/>')
    else:
        out += mask_shapes()
    out += tail_traces()
    out.append('</svg>')
    return '\n'.join(out)


if __name__ == '__main__':
    here = Path(__file__).parent
    for name, spike_mask in (('raphael-banner.svg', False), ('raphael-banner-spikes.svg', True)):
        (here / name).write_text(banner(spike_mask))
        print(f'-> {here / name}')
