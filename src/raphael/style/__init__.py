"""The labs' figure style: one look for every project's plots.

    from raphael import style
    style.use()                                    # once, before plotting

    fig, axes = plt.subplots(1, 3, figsize=style.figsize('double', height_mm=45))
    style.mean_with_error(axes[0], x, mean, sem, color=style.REGIONS['MEC'])
    style.panel_label(axes[0], 'a')
    style.save(fig, 'figures/fig1')                # fig1.pdf and fig1.png

`use()` applies raphael.mplstyle: Arial (with metric-compatible fallbacks),
6.5-8 pt text, open axes (no top or right spine), thin lines, short outward
ticks, frameless legends, shaded bands and markers without outlines, viridis,
and PDFs with editable embedded fonts. Projects built from starter_project
apply it when their package is imported; override any setting afterwards
with `style.use({'font.size': 8})` or plain `plt.rcParams`.

The rest replaces helpers that every figure script used to redefine: panel
letters, axis tidying, significance stars and brackets, mean +/- error bands,
figure sizes in mm for journal columns, shared colours, and saving.
Distilled from the AnchorDynamics2026 figures (spatial-manifolds); see
docs/style.md.
"""

import contextlib
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

STYLE_FILE = Path(__file__).with_name('raphael.mplstyle')

# ----------------------------------------------------------------- sizes

MM = 1 / 25.4                       # inches per mm: figsize=(120 * MM, 60 * MM)

#: Printed column widths (mm) of common journals.
WIDTHS_MM = {
    'single': 89, 'onehalf': 120, 'double': 183,        # Nature family
    'cell_single': 85, 'cell_onehalf': 114, 'cell_double': 174,
    'neuron_single': 85, 'neuron_double': 174,
    'elife_single': 86, 'elife_double': 178,
}

#: Text sizes (pt) used by the style, for code that sets sizes explicitly.
FONT = {'panel': 10, 'title': 7.5, 'label': 7, 'tick': 6.5, 'legend': 6.5, 'small': 6}

# ---------------------------------------------------------------- colours

#: Brain regions: the same colour for a region in every project's figures.
REGIONS = {
    'MEC': '#7b4173', 'LEC': '#8c6d31', 'VIS': '#3f9b4f', 'SUB': '#c0723a',
    'CB': '#4c72b0', 'CA1': '#c04744', 'CA3': '#3f9b8f', 'DG': '#b8860b',
    'HPC': '#a8559e', 'PFC': '#3171ae',
}
#: Neutral greys for data that should recede (context, individual animals).
GREYS = {'dark': '0.25', 'mid': '0.5', 'light': '0.75', 'faint': '0.9'}
#: The default colour cycle (also axes.prop_cycle in raphael.mplstyle).
CYCLE = ['#3171ae', '#c04744', '#3f9b4f', '#7b4173', '#d95f02', '#b8860b', '#3f9b8f',
         '#888888']
#: Colormaps by what they show. Avoid jet and hsv: they invent edges and are
#: not readable in greyscale or by colour-blind readers.
CMAPS = {'rate': 'viridis', 'density': 'magma', 'diverging': 'RdBu_r',
         'correlation': 'RdBu_r', 'phase': 'twilight', 'binary': 'Greys'}

_warned_font = False


def font_in_use():
    """The file of the font the style's sans-serif family resolves to."""
    return Path(font_manager.findfont(font_manager.FontProperties(family=['sans-serif']))).name


def _check_font():
    global _warned_font
    name = font_in_use()
    if not _warned_font and 'arial' not in name.lower():
        _warned_font = True
        warnings.warn(f'Arial is not installed, so figures use {name}. For publication '
                      'figures install Arial (macOS and Windows have it; on Linux: '
                      'ttf-mscorefonts-installer). Liberation Sans/Arimo are metric-'
                      'compatible stand-ins.', stacklevel=3)
    return name


def use(overrides=None):
    """Apply the style globally, then `overrides` (a dict of rcParams)."""
    plt.style.use(STYLE_FILE)
    if overrides:
        mpl.rcParams.update(overrides)
    _check_font()


@contextlib.contextmanager
def context(overrides=None):
    """The style inside a `with` block only, leaving global settings alone."""
    with plt.style.context(STYLE_FILE), mpl.rc_context(overrides or {}):
        yield


def styled(function):
    """Decorator: draw `function`'s figure in the style without changing the
    caller's settings (Raphael's own QC figures use this)."""
    import functools

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with context():
            return function(*args, **kwargs)
    return wrapper


def figsize(width='double', height_mm=None, aspect=None):
    """(width, height) in inches for a figure `width` wide: a key of
    WIDTHS_MM or a number of mm. Height from `height_mm`, or width * aspect
    (default 0.6)."""
    w = WIDTHS_MM[width] if isinstance(width, str) else float(width)
    h = height_mm if height_mm is not None else w * (0.6 if aspect is None else aspect)
    return w * MM, h * MM


# ------------------------------------------------------------------ axes

def panel_label(ax, letter, x=-0.18, y=1.02, **kwargs):
    """A bold panel letter at the top left of `ax`, outside the axes
    (positions in axes fractions). Nature uses lower case (a), Cell upper (A)."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FONT['panel'],
            fontweight='bold', va='bottom', ha='right', **kwargs)


def tidy(ax, left=True, bottom=True):
    """Open axes (no top or right spine); `left=False` / `bottom=False` also
    remove that spine and its ticks, e.g. for rasters and images."""
    ax.spines[['top', 'right']].set_visible(False)
    if not left:
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False, labelleft=False)
    if not bottom:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)
    return ax


def bare(ax):
    """No spines, ticks or labels: for images, rate maps, example traces."""
    ax.set_axis_off()
    return ax


def offset_spines(ax, points=3):
    """Move the left and bottom spines `points` away from the data."""
    for side in ('left', 'bottom'):
        ax.spines[side].set_position(('outward', points))
    return ax


def scale_bar(ax, length, label, where='lower right', vertical=False, pad=0.04,
              linewidth=1.2, color='0.1'):
    """A labelled scale bar in data units, for traces and images drawn
    without axes (e.g. `scale_bar(ax, 1, '1 s')`)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    right, top = 'right' in where, 'upper' in where
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    if vertical:
        x = x1 - px if right else x0 + px
        ya = y1 - py - length if top else y0 + py
        ax.plot([x, x], [ya, ya + length], color=color, lw=linewidth, solid_capstyle='butt',
                clip_on=False)
        ax.text(x + (-px / 2 if right else px / 2), ya + length / 2, label, rotation=90,
                ha='right' if right else 'left', va='center', fontsize=FONT['small'],
                color=color)
    else:
        xa = x1 - px - length if right else x0 + px
        y = y1 - py if top else y0 + py
        ax.plot([xa, xa + length], [y, y], color=color, lw=linewidth, solid_capstyle='butt',
                clip_on=False)
        ax.text(xa + length / 2, y + (-py / 2 if top else py / 2), label, ha='center',
                va='top' if top else 'bottom', fontsize=FONT['small'], color=color)


# -------------------------------------------------------------- plotting

def mean_with_error(ax, x, mean, error, color=None, label=None, alpha=0.25, lw=None,
                    **line_kwargs):
    """A line with a shaded error band (e.g. +/- SEM) of the same colour.
    `error` is one array (symmetric) or (lower, upper) offsets. The band has
    no outline, so it reads as one colour."""
    x, mean = np.asarray(x), np.asarray(mean)
    if isinstance(error, (tuple, list)) and len(error) == 2:
        low, high = mean - np.asarray(error[0]), mean + np.asarray(error[1])
    else:
        low, high = mean - np.asarray(error), mean + np.asarray(error)
    (line,) = ax.plot(x, mean, color=color, label=label, lw=lw, **line_kwargs)
    ax.fill_between(x, low, high, color=line.get_color(), alpha=alpha, linewidth=0)
    return line


def stars(p, ns='n.s.'):
    """'***' p < 0.001, '**' p < 0.01, '*' p < 0.05, else `ns`."""
    if p is None or not np.isfinite(p) or p >= 0.05:
        return ns
    return '***' if p < 0.001 else '**' if p < 0.01 else '*'


def significance(ax, x1, x2, y, p, height=None, text=None, color='0.1', lw=0.6):
    """A bracket from x1 to x2 at height y (data units), labelled with
    stars(p) or `text`."""
    lo, hi = ax.get_ylim()
    h = height if height is not None else 0.02 * (hi - lo)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, lw=lw, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text or stars(p), ha='center', va='bottom',
            fontsize=FONT['small'], color=color)


# ---------------------------------------------------------------- saving

def save(fig, path, formats=('pdf', 'png'), dpi=300, close=False, **kwargs):
    """Save `fig` as each of `formats` beside `path` (any suffix is replaced).
    PDFs embed editable TrueType fonts and carry no timestamp, so re-running
    a script reproduces the file byte for byte. Returns the paths written."""
    from raphael import __version__

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        target = path.with_suffix(f'.{fmt}')
        metadata = {'Creator': f'raphael {__version__} (matplotlib {mpl.__version__})'}
        if fmt == 'pdf':
            metadata['CreationDate'] = None
        elif fmt == 'svg':
            metadata['Date'] = None
        elif fmt == 'png':
            metadata = {'Software': metadata['Creator']}
        else:
            metadata = None
        fig.savefig(target, dpi=dpi, metadata=metadata, **kwargs)
        written.append(target)
    if close:
        plt.close(fig)
    return written
