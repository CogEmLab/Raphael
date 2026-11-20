"""Where processed imaging files go, named as in clark2025."""

from dataclasses import dataclass
from pathlib import Path

import yaml

# Raphael's defaults. A project overrides them by passing its own file
# (`load_settings(path)`, or `raphael-jobs make --config`).
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'imaging.yaml'


def load_settings(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Session:
    mouse: str
    day: str
    session_type: str

    @property
    def stem(self):
        return f'sub-M{self.mouse}_ses-D{self.day}_typ-{self.session_type}'

    def folder(self, root):
        return Path(root) / f'M{self.mouse}' / f'D{self.day}' / self.session_type

    def beh(self, root):
        return self.folder(root) / f'{self.stem}_beh.nwb'

    def ophys(self, root, pipeline):
        return self.folder(root) / f'{self.stem}_src-{pipeline}_ophys.nwb'

    def traces(self, root, pipeline):
        return self.folder(root) / f'{self.stem}_src-{pipeline}_traces.npz'

    def rois(self, root, pipeline):
        return self.folder(root) / f'{self.stem}_src-{pipeline}_rois.csv'

    def qc(self, root):
        return self.folder(root) / 'qc'
