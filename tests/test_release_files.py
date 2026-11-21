"""Releases: the version is the same everywhere, the changelog has a dated
section for it, and the release tool prepares and checks releases."""

import importlib.util
import shutil
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / '.github' / 'scripts' / 'release_tools.py'
spec = importlib.util.spec_from_file_location('release_tools', SCRIPT)
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


def test_raphael_versions_agree():
    found = rt.versions(ROOT)
    assert set(found) == {'pyproject.toml', 'CITATION.cff', 'src/raphael/__init__.py'}
    assert len(set(found.values())) == 1, found
    version = found['pyproject.toml']
    rt.check(f'v{version}', ROOT)                  # dated changelog section, citation date


@pytest.fixture
def repo(tmp_path):
    for name in ('pyproject.toml', 'CHANGELOG.md', 'CITATION.cff'):
        shutil.copy(ROOT / name, tmp_path / name)
    (tmp_path / 'src' / 'raphael').mkdir(parents=True)
    shutil.copy(ROOT / 'src' / 'raphael' / '__init__.py', tmp_path / 'src' / 'raphael')
    log = tmp_path / 'CHANGELOG.md'
    log.write_text(log.read_text().replace('## [Unreleased]\n',
                                           '## [Unreleased]\n\n### Fixed\n- A test fix.\n', 1))
    return tmp_path


def test_prepare_then_check(repo):
    current = rt.versions(repo)['pyproject.toml']
    major, minor, patch = map(int, current.split('.'))
    new = f'{major}.{minor}.{patch + 1}'
    rt.prepare(new, repo, today=date(2027, 1, 2), lock=False)
    assert set(rt.versions(repo).values()) == {new}
    assert rt.check(f'v{new}', repo) == (new, {})
    assert 'A test fix.' in rt.notes(new, repo)
    sections = rt.changelog_sections(repo)
    assert sections[0][:2] == ('Unreleased', None) and not sections[0][2]   # emptied
    assert sections[1][:2] == (new, '2027-01-02')
    assert f'[{new}]: https://github.com/HDClark94/Raphael/compare/v{current}...v{new}' in \
        (repo / 'CHANGELOG.md').read_text()
    with pytest.raises(rt.ReleaseError, match='not after'):
        rt.prepare(current, repo, lock=False)
    with pytest.raises(rt.ReleaseError, match='nothing under'):
        rt.prepare(f'{major}.{minor}.{patch + 2}', repo, lock=False)   # changelog now empty


def test_check_catches_mismatches(repo):
    with pytest.raises(rt.ReleaseError, match='says'):
        rt.check('v9.9.9', repo)
    with pytest.raises(rt.ReleaseError, match='v1.2.3'):
        rt.check('0.2.0', repo)


@pytest.mark.parametrize(('source', 'ok'), [
    ('{ git = "https://github.com/HDClark94/Raphael?tag=v0.2.0#abc" }', True),
    ('{ git = "https://github.com/HDClark94/Raphael?branch=main#abc" }', False),
    ('{ editable = "../Raphael" }', False),
    ('{ directory = "../Raphael" }', False),
])
def test_projects_must_pin_a_released_raphael(tmp_path, source, ok):
    (tmp_path / 'uv.lock').write_text(f'version = 1\n\n[[package]]\nname = "raphael"\n'
                                      f'version = "0.2.0"\nsource = {source}\n')
    if ok:
        assert rt.pinned_tag('raphael', tmp_path) == 'v0.2.0'
    else:
        with pytest.raises(rt.ReleaseError, match='release|tag'):
            rt.pinned_tag('raphael', tmp_path)
