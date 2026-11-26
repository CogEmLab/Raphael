"""Where processed ephys files go: the shared session layout (imaging.layout)
with the sorter as the `srt-` tag."""

from pathlib import Path

from raphael.imaging import layout as imaging_layout

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'ephys.yaml'
load_settings = imaging_layout.load_settings


class Session(imaging_layout.Session):
    """A session's file names; `sorter` becomes the `srt-` tag."""

    def _sorted(self, root, sorter, suffix):
        return self.folder(root) / f'{self.stem}_srt-{sorter}_{suffix}'

    def ecephys(self, root, sorter):
        return self._sorted(root, sorter, 'ecephys.nwb')

    def clusters(self, root, sorter):
        return self._sorted(root, sorter, 'clusters.npz')

    def analyzer(self, root, sorter):
        return self._sorted(root, sorter, 'analyzer.zarr')

    def lfp(self, root):
        return self.folder(root) / f'{self.stem}_lfp.nwb'

    def curation(self, root, sorter):
        return self._sorted(root, sorter, 'bombcell_curation.json')
