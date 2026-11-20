"""Figure: fraction of units tuned to position, head direction and speed.

    uv run python scripts/figure_tuning.py PROCESSED_ROOT M25 D18 OF1 --out figures/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from example_project.analysis import score_session

from raphael.ephys.layout import Session


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('mouse')
    p.add_argument('day')
    p.add_argument('session_type')
    p.add_argument('--out', type=Path, default=Path('figures'))
    p.add_argument('--n-shuffles', type=int, default=100)
    args = p.parse_args(argv)

    session = Session(args.mouse.removeprefix('M'), args.day.removeprefix('D'),
                      args.session_type)
    table = score_session(args.root, session, n_shuffles=args.n_shuffles)
    fractions = table[['si_sig', 'hd_sig', 'speed_sig']].mean()

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar(['position', 'head direction', 'speed'], fractions.values, color='0.3')
    ax.set_ylabel('fraction of units')
    ax.set_ylim(0, 1)
    ax.set_title(session.stem, fontsize=7)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f'{session.stem}_tuning_fractions.pdf'
    fig.savefig(path)
    table.to_csv(path.with_suffix('.csv'), index=False)
    print(f'-> {path}')
    return path


if __name__ == '__main__':
    main()
