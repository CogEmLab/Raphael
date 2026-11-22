"""The figure style: settings applied, shaded bands without outlines,
helpers, and reproducible saving."""

import matplotlib
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from raphael import style


@pytest.fixture(autouse=True)
def _restore_rcparams():
    with matplotlib.rc_context():
        yield
    plt.close('all')


def test_use_applies_the_house_style():
    style.use()
    rc = plt.rcParams
    assert rc['font.sans-serif'][0] == 'Arial' and rc['font.family'] == ['sans-serif']
    assert rc['pdf.fonttype'] == 42 and rc['svg.fonttype'] == 'none'
    assert not rc['axes.spines.top'] and not rc['axes.spines.right']
    assert not rc['legend.frameon'] and rc['font.size'] == 7
    assert rc['patch.linewidth'] == 0 and rc['image.cmap'] == 'viridis'
    style.use({'font.size': 9})                     # projects can override
    assert rc['font.size'] == 9


def test_shaded_bands_and_markers_have_no_outline():
    with style.context():
        _, ax = plt.subplots()
        x = np.linspace(0, 1, 10)
        band = ax.fill_between(x, x - .1, x + .1, alpha=.25)
        bars = ax.bar([0, 1], [1, 2])
        points = ax.scatter(x, x)
        line = style.mean_with_error(ax, x, x, 0.1, color='#c04744')
    assert list(band.get_linewidths()) == [0]
    assert bars[0].get_linewidth() == 0 and len(points.get_edgecolors()) == 0
    shaded = ax.collections[-1]
    assert list(shaded.get_linewidths()) == [0]
    assert np.allclose(shaded.get_facecolor()[0][:3], matplotlib.colors.to_rgb(line.get_color()))


def test_context_and_decorator_leave_global_settings_alone():
    before = plt.rcParams['font.size']
    with style.context():
        assert plt.rcParams['font.size'] == 7

    @style.styled
    def size():
        return plt.rcParams['font.size']
    assert size() == 7 and plt.rcParams['font.size'] == before


def test_figsize_and_stars():
    w, h = style.figsize('double', height_mm=50)
    assert w == pytest.approx(183 / 25.4) and h == pytest.approx(50 / 25.4)
    assert style.figsize(100, aspect=0.5) == pytest.approx((100 / 25.4, 50 / 25.4))
    assert [style.stars(p) for p in (0.2, 0.04, 0.009, 0.0001, np.nan)] == \
        ['n.s.', '*', '**', '***', 'n.s.']


def test_axis_helpers():
    with style.context():
        _, (a, b, c) = plt.subplots(1, 3)
        a.plot([0, 1], [0, 1])
        style.tidy(a, left=False)
        style.panel_label(a, 'a')
        style.bare(b)
        c.plot([0, 2], [0, 1])
        style.significance(c, 0, 2, 1.0, p=0.003)
        style.scale_bar(c, 1, '1 s')
    assert not a.spines['left'].get_visible() and not a.spines['top'].get_visible()
    assert any(t.get_text() == 'a' and t.get_fontweight() == 'bold' for t in a.texts)
    assert not b.axison
    assert [t.get_text() for t in c.texts] == ['**', '1 s']


def test_save_is_reproducible_and_embeds_fonts(tmp_path):
    with style.context():
        fig, ax = plt.subplots(figsize=style.figsize('single', height_mm=40))
        ax.plot([0, 1], [0, 1])
        ax.set_title('Rate (Hz)')
        first = style.save(fig, tmp_path / 'fig.png')      # suffix replaced
        a = (tmp_path / 'fig.pdf').read_bytes()
        style.save(fig, tmp_path / 'fig')
        b = (tmp_path / 'fig.pdf').read_bytes()
    assert [p.name for p in first] == ['fig.pdf', 'fig.png']
    assert a == b                                   # no timestamp
    assert b'FontFile2' in a and b'/Type3' not in a  # editable TrueType text


def test_qc_figures_use_the_style(tmp_path):
    """Raphael's QC figures draw in the style without changing the caller's."""
    from raphael.photometry import benchmark, doric, processing, qc

    t, _ = benchmark.simulate(tmp_path / 'r.doric', duration_s=60)
    rec = doric.read(tmp_path / 'r.doric')
    roles = {'green': {'role': 'signal', 'reference': 'iso'}, 'iso': {'role': 'reference'}}
    kept, out, fits = processing.process(t, {k: c.values for k, c in rec.channels.items()},
                                         roles)
    plt.rcParams['font.size'] = 12
    qc.session_figure(rec, kept, out, fits, {}, tmp_path, title='s')
    assert plt.rcParams['font.size'] == 12
