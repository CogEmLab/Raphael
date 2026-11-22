"""docs/assets/plots-gallery.png: every raphael.plots plot, on example_data().
    python docs/assets/make_plots_gallery.py"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from raphael import plots, style

style.use()
d = plots.example_data()
spikes, beh, arena = d['spikes'], d['behaviour'], d['arena']

fig = plt.figure(figsize=style.figsize('double', height_mm=150), dpi=200)
grid = fig.add_gridspec(4, 6, hspace=0.9, wspace=1.0, height_ratios=[1, 1, 0.55, 1])

ax = fig.add_subplot(grid[0, 0]); plots.rate_map(ax, spikes[0], beh, arena); ax.set_title('Place cell')
ax = fig.add_subplot(grid[0, 1]); plots.rate_map(ax, spikes[1], beh, arena); ax.set_title('Place cell')
ax = fig.add_subplot(grid[0, 2]); plots.rate_map(ax, d['dff'][:, 0], beh, arena); ax.set_title('ROI dF/F')
ax = fig.add_subplot(grid[0, 3], projection='polar'); plots.head_direction(ax, spikes[2], beh)
ax = fig.add_subplot(grid[0, 4:]); plots.phase_histogram(ax, spikes[3], d['theta_phase'])
ax.set_title('Theta phase')

ax = fig.add_subplot(grid[1, 0:2]); plots.tuning_curve(ax, spikes[6], beh['S'], minmax=(2, 38),
                                                        xlabel='Speed (cm/s)', label='speed cell')
plots.tuning_curve(ax, spikes[5], beh['S'], minmax=(2, 38), label='untuned'); ax.legend()
ax.set_title('Speed tuning')
ax = fig.add_subplot(grid[1, 2:4]); plots.autocorrelogram(ax, spikes[3]); ax.set_title('Autocorrelogram')
ax = fig.add_subplot(grid[1, 4:]); plots.isi_histogram(ax, spikes[4]); ax.set_title('ISI')

top = fig.add_subplot(grid[2, 0:2]); bottom = fig.add_subplot(grid[3, 0:2], sharex=top)
plots.peth(top, bottom, spikes[4], d['events'], window=(-0.5, 1.0), binsize=0.025)
top.set_title('Peri-event (spikes)')
top = fig.add_subplot(grid[2, 2:4]); bottom = fig.add_subplot(grid[3, 2:4], sharex=top)
plots.peth(top, bottom, d['photometry'], d['events'], window=(-1, 4), color=style.CYCLE[2])
top.set_title('Peri-event (photometry)')

ax = fig.add_subplot(grid[2:, 4]); plots.raster(ax, spikes, 100, 104); ax.set_title('Raster')
ax = fig.add_subplot(grid[2:, 5]); plots.traces(ax, d['dff'], 100, 160, scale_label='0.5 dF/F',
                                                scale=0.5); ax.set_title('dF/F')
fig.text(0.5, 0.005, 'raphael.plots on raphael.plots.example_data()', ha='center', color='0.5',
         fontsize=6)
out = Path(__file__).with_name('plots-gallery')
style.save(fig, out, formats=('png',), dpi=200)
print(out.with_suffix('.png'))
