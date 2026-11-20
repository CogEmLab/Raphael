"""Single-neuron tuning scores, each against a shift-shuffled null.

Every function takes a session dict of pynapple objects (position, speed,
head direction, a `moving` IntervalSet, ...), one neuron's spikes, a
significance level and a number of shuffles. Spatial binning is passed in
as `bin_config`, because arenas differ between projects.

Ported from spatial-manifolds. Grid score, ramp class, stability and theta
index are not here yet: the first two are MEC/VR-specific and stability
assumes a VR trial structure. Theta index is a candidate now that the `lfp`
stage exists.
"""

from .hd_mean_vector_length import (
    compute_hd_mean_vector_length,
    mean_vector_length,
)
from .spatial_information import compute_spatial_information
from .speed_correlation import compute_speed_correlation, correlation

__all__ = [
    'compute_hd_mean_vector_length',
    'compute_spatial_information',
    'compute_speed_correlation',
    'correlation',
    'mean_vector_length',
]
