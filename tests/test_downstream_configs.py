"""The CI's list of projects that use Raphael (.github/downstream/*.yml):
every file is valid, and mistakes in one are caught rather than dropping
the project from CI silently."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / '.github' / 'scripts' / 'downstream_matrix.py'
spec = importlib.util.spec_from_file_location('downstream_matrix', SCRIPT)
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)


def test_the_listed_projects_are_valid():
    specs = {s['name']: s for s in dm.load_all()}
    assert specs['starter_project']['enabled'] and not specs['starter_project']['private']
    assert specs['starter_project']['extras_flags'] == '--extra dev'
    assert specs['social_manifolds']['private']


@pytest.mark.parametrize(('text', 'message'), [
    ('repo: a/b\nextra: [dev]\n', 'unknown fields'),
    ('branch: main\n', 'repo'),
    ('repo: not-a-repo\n', 'owner/name'),
    ('repo: a/b\nprivate: "no"\n', 'private'),
    ('repo: a/b\nextras: ["dev; rm -rf /"]\n', 'extras'),
])
def test_mistakes_are_caught(tmp_path, text, message):
    (tmp_path / 'project.yml').write_text(text)
    with pytest.raises(dm.ConfigError, match=message):
        dm.load_all(tmp_path)


def test_defaults_and_duplicates(tmp_path):
    (tmp_path / 'one.yml').write_text('repo: lab/one\n')
    spec, = dm.load_all(tmp_path)
    assert spec['enabled'] and not spec['private'] and spec['extras'] == ['dev']
    assert spec['test'] == 'uv run pytest -q -rs'
    (tmp_path / 'two.yml').write_text('repo: lab/one\n')
    with pytest.raises(dm.ConfigError, match='more than once'):
        dm.load_all(tmp_path)
