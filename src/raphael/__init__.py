"""Raphael: the shared core for neural data processing and analysis.

Raw recordings to processed, publishable per-session files (Neuropixels
spike sorting and LFP in `ephys`, two-photon imaging in `imaging`), one
loader for either (`population`), shared tuning scores, and job bundles that
run the heavy stages on a cluster (`cluster`). Project repositories depend
on this package and keep their own data conversion, behaviour and figures.
"""

__version__ = '0.1.0'
