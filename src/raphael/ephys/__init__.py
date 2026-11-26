"""In vivo extracellular electrophysiology: Neuropixels 2.0 recording ->
Kilosort 4 sorting (and LFP) -> processed per-session files.

The processed dataset per session, in clark2025's tree:

    {root}/sub-M{mouse}/day-{dd}/ses-{type}/
        sub-M{mouse}_day-{dd}_ses-{type}_beh.nwb      all behaviour (planned)
        sub-..._srt-{sorter}_clusters.npz               TsGroup: spike times, with
                                                        curation, metrics and
                                                        waveforms as unit metadata
        sub-..._lfp.nwb                                 LFP and theta phase (`lfp` stage)
        qc/                                             figures

Sorting analysers (zarr) and bombcell curation files are working files and
stay out of this tree (`keep_analyzer`); `_ecephys.nwb` is optional
(`write_ecephys_nwb`). `raphael.population` loads `_clusters.npz`
and imaging's `_traces.npz` alike.

Settings and publication metadata: configs/ephys.yaml.
"""
