"""Read .github/downstream/*.yml into the CI matrix of downstream projects.

    python .github/scripts/downstream_matrix.py            # print the matrix (JSON)
    python .github/scripts/downstream_matrix.py --github   # also write GitHub outputs

Each file describes one project that uses Raphael; see docs/ci.md. Invalid
files fail loudly, so a typo cannot silently drop a project from CI.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

FOLDER = Path(__file__).resolve().parents[1] / 'downstream'
FIELDS = {'repo': str, 'branch': str, 'private': bool, 'enabled': bool, 'extras': list,
          'test': str, 'contact': str}
DEFAULTS = {'branch': '', 'private': False, 'enabled': True, 'extras': ['dev'],
            'test': 'uv run pytest -q -rs', 'contact': ''}


class ConfigError(ValueError):
    pass


def load(path):
    spec = yaml.safe_load(Path(path).read_text()) or {}
    name = Path(path).stem
    if not isinstance(spec, dict):
        raise ConfigError(f'{path}: expected a mapping of fields')
    unknown = set(spec) - set(FIELDS)
    if unknown:
        raise ConfigError(f'{path}: unknown fields {sorted(unknown)}; allowed {sorted(FIELDS)}')
    if 'repo' not in spec:
        raise ConfigError(f'{path}: `repo` (owner/name) is required')
    for key, kind in FIELDS.items():
        if key in spec and spec[key] is not None and not isinstance(spec[key], kind):
            raise ConfigError(f'{path}: `{key}` must be a {kind.__name__}')
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', spec['repo']):
        raise ConfigError(f'{path}: `repo` must look like owner/name, got {spec["repo"]!r}')
    spec = {**DEFAULTS, **{k: v for k, v in spec.items() if v is not None}}
    if any(not re.fullmatch(r'[\w-]+', e) for e in spec['extras']):
        raise ConfigError(f'{path}: `extras` must be a list of extra names, e.g. [dev]')
    return {'name': name, **spec,
            'extras_flags': ' '.join(f'--extra {e}' for e in spec['extras'])}


def load_all(folder=FOLDER):
    specs = [load(p) for p in sorted(Path(folder).glob('*.yml'))]
    repos = [s['repo'] for s in specs]
    duplicated = {r for r in repos if repos.count(r) > 1}
    if duplicated:
        raise ConfigError(f'listed more than once: {sorted(duplicated)}')
    return specs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--github', action='store_true')
    args = p.parse_args(argv)
    try:
        specs = load_all()
    except ConfigError as e:
        print(f'::error::{e}' if args.github else e, file=sys.stderr)
        return 1
    enabled = [s for s in specs if s['enabled']]
    print(json.dumps(enabled, indent=2))
    if args.github:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f'matrix={json.dumps(enabled)}\ncount={len(enabled)}\n')
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write('### Downstream projects\n\n| project | repo | status |\n|---|---|---|\n')
            f.writelines(f'| {s["name"]} | {s["repo"]} | '
                        f'{"tested" if s["enabled"] else "disabled"} |\n' for s in specs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
