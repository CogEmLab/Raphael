# rigs/

One file per rig, `{rig}.yaml` (copy `../templates/rig.yaml`), named as in
the `rig` of session files: e.g. `ephys-rig-1.yaml`. Devices, cameras,
probes, photometry and microscope settings, and how TTL lines are wired, so
syncing and the NWB device descriptions don't depend on anyone's memory.
Probe channel maps or other large files go in a folder of the same name.
