"""`python -m raphael`: what Raphael provides, for people and coding agents.

    uv run python -m raphael             # the guide: rules, and which tool for which task
    uv run python -m raphael api         # every public function, one line each
    uv run python -m raphael api plots   # one module
    uv run python -m raphael path        # where the installed source is

Everything comes from the installed Raphael, so it describes exactly the
version this project pins.
"""

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

import raphael

GUIDE = Path(raphael.__file__).with_name('AGENTS.md')
SKIP = ('raphael.scripts', 'raphael.__main__')


def modules():
    """Importable raphael modules, skipping command-line entry points."""
    names = ['raphael']
    for info in pkgutil.walk_packages(raphael.__path__, 'raphael.'):
        if not info.name.startswith(SKIP) and '._' not in info.name:
            names.append(info.name)
    return sorted(names)


def _summary(obj):
    # A class's own docstring only (not one inherited from, e.g., RuntimeError).
    doc = (inspect.cleandoc(obj.__dict__.get('__doc__') or '') if inspect.isclass(obj)
           else inspect.getdoc(obj) or '')
    return doc.split('\n\n')[0].replace('\n', ' ').strip()


def public(module):
    """[(name, kind, signature, first docstring paragraph)] defined in `module`."""
    out = []
    names = getattr(module, '__all__', None) or [n for n in dir(module) if not n.startswith('_')]
    for name in names:
        obj = getattr(module, name, None)
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        if getattr(obj, '__module__', '') != module.__name__ and not hasattr(module, '__all__'):
            continue                                  # imported from elsewhere
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            sig = '(...)'
        out.append((name, 'class' if inspect.isclass(obj) else 'def', sig, _summary(obj)))
    return out


def api(prefix=None):
    lines = [f'Raphael {raphael.__version__}: public API'
             + (f' of raphael.{prefix}' if prefix else '')]
    for name in modules():
        if prefix and not (name == f'raphael.{prefix}' or name.startswith(f'raphael.{prefix}.')):
            continue
        try:
            module = importlib.import_module(name)
        except (ImportError, OSError) as e:           # an optional extra not installed
            lines += ['', f'{name}  (not importable here: {type(e).__name__}: {e})']
            continue
        items = public(module)
        if not items:
            continue
        lines += ['', f'{name}: {_summary(module)}']
        for item, kind, sig, summary in items:
            sig = sig if len(sig) <= 70 else sig[:67] + '...)'
            lines.append(f'  {kind} {item}{sig}')
            if summary:
                lines.append(f'      {summary[:160]}')
    return '\n'.join(lines)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(GUIDE.read_text())
        print(f'(Raphael {raphael.__version__}, installed at {Path(raphael.__file__).parent})')
    elif args[0] == 'api':
        print(api(args[1] if len(args) > 1 else None))
    elif args[0] == 'path':
        print(Path(raphael.__file__).parent)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
