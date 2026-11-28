# processed/_dlcmodels/

The lab's trained DeepLabCut models, one folder per model. A session whose
task names a model in `metadata/sessions.yaml` (`dlc_model:`) gets a
DeepLabCut job for each of its videos in the plan ingest writes.

```
_dlcmodels/openfield-top-v2/
├── model.yaml                  how to run it (below)
├── train/
│   ├── pytorch_config.yaml     the model folder as DeepLabCut wrote it
│   └── snapshot-200.pt
└── skeleton.json               optional: two-animal identity fixes
```

```yaml
# model.yaml
snapshot: snapshot-200.pt       # in train/ (required)
detector_snapshot:              # top-down models only
skeleton: skeleton.json         # optional, in this folder
bodyparts: [nose, ear_left, ear_right, tailbase]
trained_by: hclark
trained_on: 2026-11-10
description: open field, top camera, one mouse
```

**A model is never changed once used.** A retrained model gets a new name
(`openfield-top-v3`) and its own folder, and the task's `dlc_model` is
changed to it; the old one stays, so every output can be traced to the
model that made it. Each plan records the model folder's checksum, and a
job refuses a copy that differs.

**On the cluster**, the models are copied once (again after adding one):

```bash
rsync -a /path/to/data/processed/_dlcmodels/ $CLUSTER:models/
```

Outputs go into the session's folder in `processed/`, named for the
video and the model: `sub-M…_day-…_ses-OF1_video-top_src-dlc-openfield-top-v2_capture.npz`.
