"""Where processed files go: the session folders and BIDS-style names
shared by every modality (ephys and photometry subclass `Session`)."""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

# Raphael's defaults. A project overrides them by passing its own file
# (`load_settings(path)`, or `raphael-jobs make --config`).
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'imaging.yaml'


def load_settings(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


#: A session's name stem, now and before 2026-11 (clark2025, mecpy).
STEM = re.compile(r'sub-M(?P<mouse>[^_/]+)_day-(?P<day>\d+)_ses-(?P<type>[^_/]+)')
LEGACY_STEM = re.compile(r'sub-M(?P<mouse>[^_/]+)_ses-D(?P<day>\d+)_typ-(?P<type>[^_/]+)')


def day_label(day):
    """`day-01`: the day, two digits (three from day 100)."""
    text = str(day)
    return f'day-{int(text):02d}' if text.isdigit() else f'day-{text}'


@dataclass(frozen=True)
class Session:
    """One session's folder and file names in the processed (and raw) tree:

        {root}/sub-M{mouse}/day-{dd}/ses-{type}/sub-M{mouse}_day-{dd}_ses-{type}_...

    With `legacy=True`, the names used before November 2026 (clark2025,
    mecpy): `{root}/M{mouse}/D{day}/{type}/sub-M{mouse}_ses-D{day}_typ-{type}_...`.
    `Session.find` picks whichever exists; outputs are always written with
    the current names."""

    mouse: str
    day: str
    session_type: str
    legacy: bool = False

    @property
    def stem(self):
        if self.legacy:
            return f'sub-M{self.mouse}_ses-D{self.day}_typ-{self.session_type}'
        return f'sub-M{self.mouse}_{day_label(self.day)}_ses-{self.session_type}'

    def folder(self, root):
        if self.legacy:
            return Path(root) / f'M{self.mouse}' / f'D{self.day}' / self.session_type
        return (Path(root) / f'sub-M{self.mouse}' / day_label(self.day)
                / f'ses-{self.session_type}')

    @classmethod
    def find(cls, root, mouse, day, session_type):
        """The session as it is stored under `root`: current names if that
        folder exists, else the legacy names if that one does, else current."""
        current = cls(str(mouse), str(day), str(session_type))
        old = cls(str(mouse), str(day), str(session_type), legacy=True)
        if not current.folder(root).exists() and old.folder(root).exists():
            return old
        return current

    @classmethod
    def parse(cls, name):
        """The Session a file name, stem or path names (either scheme), or None."""
        text = str(name).replace('\\', '/').rsplit('/', 1)[-1]
        for pattern, legacy in ((STEM, False), (LEGACY_STEM, True)):
            m = pattern.search(text)
            if m:
                day = str(int(m['day']))
                return cls(m['mouse'], day, m['type'], legacy=legacy)
        return None

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
