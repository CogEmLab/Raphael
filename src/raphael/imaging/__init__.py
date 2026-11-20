"""In vivo two-photon imaging: raw movie -> processed, publishable NWB.

The output tree mirrors clark2025 (see `layout`), with the spike-sorting
outputs replaced by their imaging equivalents:

    {root}/M{mouse}/D{day}/{type}/
        sub-M{mouse}_ses-D{day}_typ-{type}_beh.nwb              behaviour
        sub-..._src-{pipeline}_ophys.nwb                        processed ophys
        sub-..._src-{pipeline}_traces.npz                       dF/F, TsdFrame
        sub-..._src-{pipeline}_rois.csv                         one row per ROI
        qc/                                                     figures

Settings and publication metadata: configs/imaging.yaml.
"""
