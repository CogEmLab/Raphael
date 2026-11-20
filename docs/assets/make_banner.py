"""Raphael README banner: a red bandana eye mask whose tails become spike traces."""
import numpy as np

W, H = 1200, 430
rng = np.random.default_rng(7)
RED, DARK, SHADE = '#d62828', '#7a0f14', '#a61b1f'
out = []
add = out.append

def path(points, close=False):
    d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in points)
    return d + (' Z' if close else '')

def spike_trace(x0, x1, y0, n_spikes, amp, seed, drift=None):
    """Extracellular-looking trace: noise plus biphasic spikes (sharp trough, slower peak)."""
    r = np.random.default_rng(seed)
    x = np.linspace(x0, x1, int((x1 - x0) * 2))
    y = r.normal(0, 0.9, len(x))
    y = np.convolve(y, np.ones(3) / 3, mode='same')
    for c in np.sort(r.uniform(x0 + 15, x1 - 10, n_spikes)):
        a = amp * r.uniform(0.6, 1.0)
        dx = x - c
        y += a * np.exp(-(dx / 1.6) ** 2) - 0.38 * a * np.exp(-((dx - 4.5) / 3.2) ** 2)
    taper = np.clip((x - x0) / 40, 0, 1)                   # grows out of the ribbon
    base = y0 + (drift(x) if drift else 0)
    return list(zip(x, base + y * (0.35 + 0.65 * taper)))

add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    'role="img" aria-label="A red bandana mask whose tails turn into spike traces, '
    'over spikes, fibre photometry and dF/F traces in greens">')
add('<defs>'
    f'<linearGradient id="band" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ef3b36"/>'
    f'<stop offset="0.55" stop-color="{RED}"/><stop offset="1" stop-color="{SHADE}"/></linearGradient>'
    '<linearGradient id="fade" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#d62828"/>'
    '<stop offset="1" stop-color="#d62828" stop-opacity="0.15"/></linearGradient>'
    '</defs>')

# Background, in greens: a spike raster, fibre photometry and dF/F traces.
SPIKES, PHOTOMETRY, DFF = '#2d6a4f', '#40916c', '#52b788'


def smooth_noise(n, width, scale, seed):
    r = np.random.default_rng(seed)
    k = np.exp(-0.5 * (np.arange(-3 * width, 3 * width + 1) / width) ** 2)
    return np.convolve(r.normal(0, 1, n + len(k)), k / k.sum(), mode='same')[:n] * scale


xs_bg = np.linspace(0, W, 600)
# The face: a spike raster whose firing rate is high inside a rounded head
# behind the mask and low outside it, so the head appears from spike
# density alone. The mouth and nostrils are low-rate gaps; shading comes
# from rate and colour (darker towards the rim).
FACE_C, FACE_R = (410.0, 225.0), (350.0, 215.0)


def face_depth(x, y):
    """0 at the centre of the head, 1 at its edge, >1 outside."""
    u = np.abs((x - FACE_C[0]) / FACE_R[0])
    v = np.abs((y - FACE_C[1]) / FACE_R[1])
    return (u ** 2.4 + v ** 2.4) ** (1 / 2.4)


def features(x, y):
    """True on the mouth and nostrils, where the rate drops."""
    mouth = ((np.abs(y - (342 - 0.0009 * (x - FACE_C[0]) ** 2 + 0.0000032 * (x - FACE_C[0]) ** 4 / 100))
              < 6.5) & (np.abs(x - FACE_C[0]) < 175))
    nostrils = ((x - 380) ** 2 + (y - 284) ** 2 < 10 ** 2) | ((x - 440) ** 2 + (y - 284) ** 2 < 10 ** 2)
    return mouth | nostrils


SHADES = [(0.0, 0.5, '#40916c'), (0.5, 0.85, '#2d6a4f'), (0.85, 1.0, '#1b4332'),
          (1.0, 9.0, '#2d6a4f')]
ROW = 4.5


def field(seed, scale):
    """A smooth random 2D field (sum of a few random plane waves), ~[-1, 1]."""
    r = np.random.default_rng(seed)
    waves = [(r.uniform(0.5, 1.5) / scale, r.uniform(0, 2 * np.pi), r.uniform(0, 2 * np.pi))
             for _ in range(6)]

    def f(x, y):
        return sum(np.sin(k * (x * np.cos(a) + y * np.sin(a)) + ph) for k, a, ph in waves) / 2.5
    return f


edge = field(41, 40)        # ragged head outline
patches = field(42, 70)     # patchy firing across the face
ticks = {colour: [] for _, _, colour in SHADES}
for yy in np.arange(ROW / 2, H, ROW):
    # Bursty spike trains: Poisson burst centres, each with a few spikes.
    centres = rng.uniform(0, W, rng.poisson(0.21 * W))
    sizes = rng.geometric(0.45, len(centres))
    xs = np.concatenate([c + np.cumsum(rng.exponential(1.6, k)) for c, k in zip(centres, sizes)])
    xs = xs[(xs > 0) & (xs < W)]
    depth = face_depth(xs, yy) + 0.07 * edge(xs, yy)
    inside = 0.6 + 0.25 * (1 - np.clip(depth, 0, 1) ** 3) + 0.2 * patches(xs, yy)
    halo = 0.5 * np.exp(-(depth - 1) / 0.06)               # spikes bleeding past the edge
    rate = np.where(depth <= 1, inside, np.maximum(halo, 0.03))
    rate = np.where(features(xs + rng.normal(0, 2, len(xs)), yy), 0.04, rate)
    keep = rng.random(len(xs)) < np.clip(rate, 0, 1)
    for lo, hi, colour in SHADES:
        sel = keep & (depth >= lo) & (depth < hi)
        for x in xs[sel]:
            h = rng.uniform(2.2, 6.0)
            y0 = yy - h / 2 + rng.normal(0, 1.0)
            ticks[colour].append(f'M{x:.0f} {y0:.1f}v{h:.1f}')
for (lo, _, colour) in SHADES:
    opacity = 0.35 if lo >= 1.0 else 0.9
    add(f'<path d="{" ".join(ticks[colour])}" stroke="{colour}" stroke-opacity="{opacity}" '
        'stroke-width="2" stroke-linecap="butt" fill="none"/>')
# Fibre photometry: slow, smooth fluctuations with a gentle bleaching decay.
for yy, seed in [(105, 21), (262, 22)]:
    y = yy + 6 * np.exp(-xs_bg / 300) + smooth_noise(len(xs_bg), 18, 260, seed)
    add(f'<path d="{path(list(zip(xs_bg, y)))}" stroke="{PHOTOMETRY}" stroke-opacity="0.3" '
        'stroke-width="2.5" fill="none" stroke-linejoin="round"/>')
# dF/F: calcium-like transients, fast rise and slow decay, on a quiet baseline.
xs_ca = np.linspace(0, W, 1200)
for yy, seed in [(150, 31), (205, 32), (250, 33)]:
    r = np.random.default_rng(seed)
    y = r.normal(0, 0.5, len(xs_ca))
    for onset in r.uniform(0, W, 6):
        dt = np.clip(xs_ca - onset, 0, None)
        y -= r.uniform(22, 40) * (1 - np.exp(-dt / 2.0)) * np.exp(-dt / 45) * (xs_ca > onset)
    add(f'<path d="{path(list(zip(xs_ca, yy + y)))}" stroke="{DFF}" stroke-opacity="0.45" '
        'stroke-width="2.4" fill="none" stroke-linejoin="round"/>')

# The mask: a band across, slightly curved like it wraps a head, with eyeholes.
cx, cy = 400, 180
top = [(x, cy - 62 + 0.00035 * (x - cx) ** 2 - 10 * np.sin((x - 90) / 640 * np.pi))
       for x in np.linspace(90, 720, 60)]
bottom = [(x, cy + 58 + 0.00030 * (x - cx) ** 2 - 6 * np.sin((x - 90) / 640 * np.pi))
          for x in np.linspace(720, 90, 60)]
# Left end wraps round the head: pinch it in and round it off.
lx = 90
left_end = (f' C{lx - 28},{bottom[-1][1] - 30} {lx - 28},{top[0][1] + 30} {top[0][0]},{top[0][1]}')
band = path(top + bottom) + left_end + ' Z'

def eye(ex, ey, flip):
    """Almond eyehole, pointed and lowered at the inner corner for a determined look."""
    s = -1 if flip else 1
    inner, outer = (ex + s * 66, ey + 12), (ex - s * 74, ey - 22)
    return (f'M{outer[0]},{outer[1]} '
            f'C{ex - s * 30},{ey - 30} {ex + s * 30},{ey - 12} {inner[0]},{inner[1]} '
            f'C{ex + s * 36},{ey + 40} {ex - s * 50},{ey + 36} {outer[0]},{outer[1]} Z')

left, right = eye(300, 178, False), eye(505, 178, True)
add(f'<path d="{band} {left} {right}" fill="url(#band)" fill-rule="evenodd" '
    f'stroke="{DARK}" stroke-width="5" stroke-linejoin="round"/>')
# Eyes seen through the holes, with a pupil each.
for d, px in ((left, 312), (right, 493)):
    add(f'<path d="{d}" fill="#ffffff" stroke="{DARK}" stroke-width="5" stroke-linejoin="round"/>')
    add(f'<circle cx="{px}" cy="186" r="14" fill="#1b1b1b"/>'
        f'<circle cx="{px + 4}" cy="181" r="4.5" fill="#ffffff"/>')
# A fold line along the band, for cloth.
fold = [(x, cy + 40 + 0.0003 * (x - cx) ** 2 - 6 * np.sin((x - 90) / 640 * np.pi))
        for x in np.linspace(130, 690, 40)]
add(f'<path d="{path(fold)}" stroke="{DARK}" stroke-opacity="0.35" stroke-width="3" fill="none"/>')

# Knot on the right of the band.
kx, ky = 735, 182
add(f'<ellipse cx="{kx}" cy="{ky}" rx="34" ry="44" fill="url(#band)" stroke="{DARK}" stroke-width="5"/>')
add(f'<path d="M{kx - 14},{ky - 28} Q{kx + 8},{ky} {kx - 12},{ky + 30}" stroke="{DARK}" '
    'stroke-opacity="0.45" stroke-width="3" fill="none"/>')

# Two tails: ribbons that taper off the knot and unravel into spike traces.
for i, (dy, amp, seed, phase, swing) in enumerate([(-18, 40, 3, 0.4, 30), (22, 34, 11, 2.2, 38)]):
    def centre(x, dy=dy, phase=phase, swing=swing):
        u = (x - kx) / 250
        # leaves the knot, falls away from its partner, then flutters
        return (ky + dy + np.sign(dy) * 70 * np.sin(np.clip(u, 0, 1) * np.pi / 2)
                + swing * np.sin(u * 2.6 * np.pi + phase) * np.clip(u * 2, 0, 1))
    xs = np.linspace(kx + 20, kx + 250, 60)
    width = 30 * (1 - np.linspace(0, 1, len(xs)) ** 1.4) + 3
    upper = [(x, centre(x) - w / 2) for x, w in zip(xs, width)]
    lower = [(x, centre(x) + w / 2) for x, w in zip(xs[::-1], width[::-1])]
    add(f'<path d="{path(upper + lower, close=True)}" fill="url(#band)" stroke="{DARK}" '
        'stroke-width="4" stroke-linejoin="round"/>')
    x_end = kx + 250
    level = centre(x_end) + np.sign(dy) * 8

    def settle(x, x_end=x_end, level=level, centre=centre):
        # leaves the ribbon on its curve, then flattens onto a baseline
        w = np.clip((x - x_end) / 70, 0, 1)
        w = w * w * (3 - 2 * w)
        return (1 - w) * centre(np.minimum(x, x_end + 10)) + w * level
    trace = spike_trace(x_end - 10, W - 20, 0, 11 - 2 * i, amp, seed, drift=settle)
    add(f'<path d="{path(trace)}" stroke="url(#fade)" stroke-width="3" stroke-linejoin="round" '
        'stroke-linecap="round" fill="none"/>')

add('</svg>')
open(__import__('pathlib').Path(__file__).with_name('raphael-banner.svg'), 'w').write('\n'.join(out))
print('ok')
