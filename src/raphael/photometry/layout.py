"""Where processed photometry files go: the same tree as ephys and imaging."""

from pathlib import Path

from raphael.imaging import layout as imaging_layout

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'photometry.yaml'
load_settings = imaging_layout.load_settings


class Session(imaging_layout.Session):
    def photometry(self, root):
        return self.folder(root) / f'{self.stem}_photometry.nwb'

    def photometry_traces(self, root):
        return self.folder(root) / f'{self.stem}_photometry.npz'

    def photometry_channels(self, root):
        return self.folder(root) / f'{self.stem}_photometry.csv'
