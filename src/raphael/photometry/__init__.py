"""Fibre photometry: Doric recordings -> dF/F per fibre channel, as NWB and
a pynapple table in the processed tree.

    {root}/sub-M{mouse}/day-{dd}/ses-{type}/
        sub-M{mouse}_day-{dd}_ses-{type}_photometry.nwb   raw channels (acquisition)
                                                             and dF/F, z-score (processing)
        sub-..._photometry.npz                               TsdFrame: {channel}_dff, {channel}_zscore
        sub-..._photometry.csv                               one row per channel: role, reference,
                                                             indicator, fit parameters
        qc/

Steps (`processing`), following Simpson et al. 2024, Neuron ("Lights,
fiber, action! A primer on in vivo fiber photometry"): trim the LED-on
transient, low-pass filter, fit and remove photobleaching (double
exponential), regress the reference (isosbestic) channel out of each
signal, then dF/F relative to the bleaching fit, and z-score.

Settings and publication metadata: configs/photometry.yaml. Run with
`raphael-photometry process` or the `photometry` job stage.
"""
