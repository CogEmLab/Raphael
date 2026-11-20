"""One way to load a session's neurons, whichever pipeline recorded them.

Each pipeline ends in a table (one row per neuron) and the activity:

                  table                               activity
    ephys         metadata of ..._clusters.npz        the same file: TsGroup, spikes
    imaging       ..._src-suite2p_rois.csv            ..._traces.npz: TsdFrame, dF/F

Each table has an `accepted` column and an id column (`unit` or `roi`)
that keys the activity. `load` takes the shared prefix of the files:

    pop = population.load('.../sub-M25_ses-D18_typ-OF1_srt-kilosort4')
    pop = pop.accepted()
    rates = pop.binned(0.1)       # TsdFrame, time x neuron, either modality

so analysis written against `binned` (or tuning curves from pynapple, which
take either a TsGroup or a TsdFrame) runs unchanged on both.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pynapple as nap

FILES = {'spikes': (None, 'clusters.npz', 'unit'),     # table = the TsGroup's metadata
         'dff': ('rois.csv', 'traces.npz', 'roi')}


@dataclass
class Population:
    kind: str                    # 'spikes' or 'dff'
    data: nap.TsGroup | nap.TsdFrame
    table: pd.DataFrame          # one row per neuron, in `ids` order
    id_column: str

    @property
    def ids(self):
        return self.table[self.id_column].to_numpy()

    def select(self, ids):
        ids = list(ids)
        table = self.table.set_index(self.id_column).loc[ids].reset_index()
        data = self.data[ids] if self.kind == 'spikes' else self.data.loc[ids]
        return Population(self.kind, data, table, self.id_column)

    def accepted(self):
        """Only neurons that passed curation."""
        return self.select(self.ids[self.table['accepted'].astype(bool).to_numpy()])

    def binned(self, bin_size):
        """Time x neuron TsdFrame: firing rate (Hz) for spikes, mean dF/F
        per bin for imaging. Columns are the neuron ids."""
        if self.kind == 'spikes':
            return self.data.count(bin_size) / bin_size
        return self.data.bin_average(bin_size)


def load(prefix):
    """`prefix`: a session's file path up to the pipeline tag, e.g.
    `.../sub-M25_ses-D18_typ-OF1_srt-kilosort4`."""
    prefix = str(Path(prefix).expanduser())
    for kind, (table_name, data_name, id_column) in FILES.items():
        data_path = Path(f'{prefix}_{data_name}')
        if not data_path.exists():
            continue
        data = nap.load_file(str(data_path))
        if table_name is None:
            table = data.metadata.rename_axis(id_column).reset_index()
        elif Path(f'{prefix}_{table_name}').exists():
            table = pd.read_csv(f'{prefix}_{table_name}')
        else:
            continue
        return Population(kind, data, table, id_column)
    raise FileNotFoundError(f'no clusters.npz, or rois.csv + traces.npz, at {prefix}_*')
