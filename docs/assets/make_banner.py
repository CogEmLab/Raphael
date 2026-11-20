"""Raphael README banner: a red bandana eye mask whose tails become spike traces."""
import numpy as np

W, H = 1200, 360
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
    'role="img" aria-label="A red bandana mask whose tails turn into spike trains">')
add('<defs>'
    f'<linearGradient id="band" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ef3b36"/>'
    f'<stop offset="0.55" stop-color="{RED}"/><stop offset="1" stop-color="{SHADE}"/></linearGradient>'
    '<linearGradient id="fade" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#d62828"/>'
    '<stop offset="1" stop-color="#d62828" stop-opacity="0.15"/></linearGradient>'
    '</defs>')

# Faint raster in the background: spike trains of a small population.
for row in range(9):
    yy = 40 + row * 32
    times = np.sort(rng.uniform(40, 1160, rng.integers(10, 26)))
    ticks = ' '.join(f'M{t:.1f},{yy - 6} L{t:.1f},{yy + 6}' for t in times)
    add(f'<path d="{ticks}" stroke="#d62828" stroke-opacity="0.13" stroke-width="2" '
        'stroke-linecap="round" fill="none"/>')

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
