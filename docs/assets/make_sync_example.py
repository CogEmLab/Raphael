"""docs/assets/sync-example.png: the sync QC figure for a simulated session
whose behaviour CSV missed one pulse.   python docs/assets/make_sync_example.py"""

import shutil
import tempfile
from pathlib import Path

from raphael.ephys.layout import Session
from raphael.sync.run import sync_session
from raphael.sync.simulate import simulate_session

with tempfile.TemporaryDirectory() as tmp:
    sim = simulate_session(Path(tmp) / 'sim', duration_s=90, n_channels=1,
                           drop_behaviour_pulses=[30])
    _, written = sync_session(sim['behaviour'], sim['recording'], Path(tmp) / 'processed',
                              Session('1', '1', 'OF1'))
    shutil.copy(written['figure'], Path(__file__).with_name('sync-example.png'))
