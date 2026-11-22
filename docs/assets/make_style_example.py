"""docs/assets/style-example.png: the same panels in matplotlib's defaults
and in Raphael's style.   python docs/assets/make_style_example.py"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from raphael import style

rng = np.random.default_rng(1)
x = np.linspace(0, 200, 80)
curves = {r: (np.exp(-((x - c) / 25) ** 2) * a + 0.15 + rng.normal(0, .02, x.size),
              0.05 + 0.03 * rng.random(x.size))
          for r, c, a in (('MEC', 70, 1.0), ('VIS', 130, 0.7))}
sx = rng.normal(0, 1, 60)
sy = 0.6 * sx + rng.normal(0, 0.6, 60)
groups = [rng.normal(m, .5, 12) for m in (1.0, 1.6, 1.2)]
yy, xx = np.mgrid[0:30, 0:30]
ratemap = np.exp(-((xx - 18) ** 2 + (yy - 11) ** 2) / 40) + 0.1 * rng.random((30, 30))
spikes = [np.sort(rng.uniform(0, 2, rng.integers(8, 30))) for _ in range(15)]


def panels(axes, styled):
    a, b, c, d, e, f = axes
    for region, (mean, sem) in curves.items():
        color = style.REGIONS[region] if styled else None
        if styled:
            style.mean_with_error(a, x, mean, sem, color=color, label=region)
        else:
            (line,) = a.plot(x, mean, label=region)
            a.fill_between(x, mean - sem, mean + sem, color=line.get_color(), alpha=.25)
    a.axhline(0.15, ls='--', color='0.5')
    a.set(xlabel='Position (cm)', ylabel='Rate (Hz)', title='Tuning')
    a.legend()

    b.scatter(sx, sy, s=9 if styled else None, color=style.CYCLE[0] if styled else None)
    fit = np.polyfit(sx, sy, 1)
    b.plot(np.sort(sx), np.polyval(fit, np.sort(sx)), color='0.2' if styled else None)
    b.set(xlabel='Speed (z)', ylabel='Rate (z)', title='Correlation')

    means = [g.mean() for g in groups]
    sems = [g.std(ddof=1) / np.sqrt(g.size) for g in groups]
    c.bar(range(3), means, yerr=sems, color=[style.GREYS['light'], style.CYCLE[1],
                                             style.GREYS['light']] if styled else None)
    for i, g in enumerate(groups):
        c.scatter(np.full(g.size, i) + rng.uniform(-.15, .15, g.size), g,
                  s=4 if styled else None, color='0.2' if styled else None, zorder=3)
    c.set(xticks=range(3), xticklabels=['A', 'B', 'C'], ylabel='Score', title='Groups')
    if styled:
        style.significance(c, 0, 1, max(groups[1].max(), groups[0].max()) + .1, p=0.004)

    d.imshow(ratemap, cmap=style.CMAPS['rate'] if styled else None, origin='lower')
    d.set_title('Rate map')
    if styled:
        style.bare(d)

    d_ = e
    d_.eventplot(spikes, color='0.15' if styled else None, linelengths=0.7,
                 linewidths=0.6 if styled else None)
    d_.set(xlabel='Time (s)', ylabel='Unit', title='Raster')
    if styled:
        style.tidy(d_)

    f.hist(np.concatenate(groups), bins=15, color=style.CYCLE[3] if styled else None)
    f.set(xlabel='Score', ylabel='Count', title='Distribution')

    if styled:
        for ax, letter in zip(axes, 'abcdef', strict=True):
            style.panel_label(ax, letter)


here = Path(__file__).parent
fig = plt.figure(figsize=style.figsize('double', height_mm=118), dpi=200)
sub = fig.subfigures(2, 1, hspace=0.06)
with plt.style.context('default'):
    axes = sub[0].subplots(1, 6)
    panels(axes, styled=False)
    sub[0].suptitle('matplotlib defaults', fontsize=8, x=0.01, ha='left', color='0.4')
with style.context():
    axes = sub[1].subplots(1, 6, gridspec_kw={'wspace': 0.75})
    panels(axes, styled=True)
    sub[1].suptitle('raphael.style', fontsize=8, x=0.01, ha='left', color='0.4')
for s in sub:
    s.subplots_adjust(left=0.06, right=0.99, bottom=0.2, top=0.8, wspace=0.75)
style.save(fig, here / 'style-example', formats=('png',), dpi=200)
print(here / 'style-example.png')
