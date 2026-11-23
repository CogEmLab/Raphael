"""Behavioural tracking: video -> DeepLabCut pose -> behaviour and VAME motifs.

    videos     inventory, day-held-out split, crop and transcode (ffmpeg)
    dlc        a DeepLabCut project from scratch: create, extract, split, train, analyse
    zoo        one project from existing labels, several models on the same split,
               an optional benchmark model, training, k-fold day splits
    multipeak  top-k heatmap peaks from DLC models (what DLC throws away)
    sweep      re-track a capture at any likelihood / detector threshold
    skeleton   what a real skeleton looks like, learned from labels
    identity   fix bodyparts DLC gives to the wrong animal (two animals)
    arena      pixels -> arena cm from tracked corner landmarks
    io         read DLC output; pose -> Raphael behaviour (P_x, P_y, H, S, moving);
               pose in the processed tree (`_src-dlc_pose.h5`)
    vame       VAME motifs from pose, run in VAME's own locked environment

DeepLabCut comes with the `tracking` extra (uv sync --extra tracking). Only
`dlc`, `zoo`, `multipeak` and `sweep.track` import it, lazily, so the rest
(io, arena, skeleton, identity, vame) works without it. VAME is never
imported here: it runs in a separate environment (see `vame`).

Settings for a project: a copy of raphael/configs/tracking.yaml. Commands:
`raphael-tracking`. Job stages: dlc_train, dlc_capture, vame.
See docs/behavioural_tracking.md.

Moved here from social-manifolds (pose/), where it was developed.
"""

import os

# On macOS the conda env and PyTorch can each bring their own libomp;
# whichever loads second aborts the process (or segfaults during DLC
# training) unless both of these are set BEFORE torch, cv2 or deeplabcut is
# imported. Setting them here covers every entry point through this package.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
