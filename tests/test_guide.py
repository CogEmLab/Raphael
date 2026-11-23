"""The guide for coding agents (raphael/AGENTS.md, `python -m raphael`)
names only things that exist, and ships with the package."""

import importlib
import re

import pytest

import raphael
from raphael.__main__ import GUIDE, api, main, modules


def resolve(dotted):
    """raphael.a.b.c -> the module or attribute it names, or raise."""
    parts = dotted.split('.')
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module('.'.join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(dotted)


def test_every_name_in_the_guide_exists():
    text = GUIDE.read_text()
    for dotted in set(re.findall(r'`(raphael(?:\.\w+)+)', text)):
        resolve(dotted)
    rows = [line for line in text.splitlines() if line.startswith('| ') and 'raphael' in line]
    assert len(rows) >= 10
    for row in rows:
        use = row.split('|')[2]
        module = re.search(r'`(raphael(?:\.\w+)+)`', use)
        if not module or ':' not in use:
            continue
        base = resolve(module.group(1))
        after = use.split(':', 1)[1]
        for name in re.findall(r'`([A-Za-z_]\w*(?:\.\w+)*)`', after):
            if name.startswith('raphael') or name in ('bin_config',):
                continue
            try:
                resolve(f'{module.group(1)}.{name}')
            except (AttributeError, ImportError):
                pytest.fail(f'guide names {module.group(1)}.{name}, which does not exist '
                            f'({base})')


def test_guide_ships_with_the_package(capsys):
    assert GUIDE.parent == __import__('pathlib').Path(raphael.__file__).parent
    assert main([]) == 0
    out = capsys.readouterr().out
    assert 'guide for coding agents' in out and raphael.__version__ in out


def test_api_index_covers_the_modules():
    text = api()
    for module in ('raphael.population', 'raphael.plots', 'raphael.style', 'raphael.ml',
                   'raphael.sync', 'raphael.toy', 'raphael.tuning_scores'):
        assert module in modules()
        assert f'\n{module}' in text or f'\n{module}.' in text
    assert 'def rate_map(' in api('plots')
    assert 'raphael.sync' not in api('plots')
    assert 'Unspecified run-time error' not in text


def test_every_module_is_in_the_guide():
    text = GUIDE.read_text()
    top = {name.split('.')[1] for name in modules() if name.count('.') >= 1}
    top -= {'util', 'scripts'}
    missing = sorted(m for m in top if f'raphael.{m}' not in text)
    assert not missing, f'add a row for these modules to src/raphael/AGENTS.md: {missing}'
