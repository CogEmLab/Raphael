"""In vivo calcium imaging, two-photon (suite2p) and one-photon (Inscopix
movies, CaImAn CNMF-E): raw movie -> processed, publishable NWB.

The output tree mirrors clark2025 (see `layout`), with the spike-sorting
outputs replaced by their imaging equivalents:

    {root}/sub-M{mouse}/day-{dd}/ses-{type}/
        sub-M{mouse}_day-{dd}_ses-{type}_beh.nwb              behaviour
        sub-..._src-{pipeline}_ophys.nwb                        processed ophys
        sub-..._src-{pipeline}_traces.npz                       dF/F, TsdFrame
        sub-..._src-{pipeline}_rois.csv                         one row per ROI
        qc/                                                     figures

Settings and publication metadata: configs/imaging.yaml (two-photon),
configs/imaging_1p.yaml (one-photon).
"""
