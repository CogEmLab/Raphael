# incoming/

Where new sessions wait to be added to `raw/`. Each person has their own
folder, `incoming/{your username}/`, that only they can write to.

1. **Write the describing files** from `lab/templates/`: the session file
   for every session, and the subject, procedures or cohort file if the
   mouse or cohort is new. (Your entry in `lab/people.yaml`, and the task
   code in `lab/tasks.yaml`, must exist already.)
2. **Copy the session here** with the folders and names it will have in
   `raw/` (see `raw/README.md`):
   ```
   incoming/hclark/2026-03_dHPC-place/M10432/D4/OF1/
       sub-M10432_ses-D4_typ-OF1_session.yaml
       sub-M10432_ses-D4_typ-OF1_ephys/Record Node 101/...      (as recorded inside)
       sub-M10432_ses-D4_typ-OF1_video-top.mp4
   ```
   A new mouse or cohort comes with its files one level up
   (`M10432/sub-M10432_subject.yaml`, `2026-03_dHPC-place/cohort.yaml`).
3. **Tell the data manager.** They check every required field and that
   the task, project, rig and people exist in `lab/`, write the manifest
   (checksums and original names), move the session into `raw/`, and lock
   the recordings. Then it disappears from here.

Keep your own copy until it is in `raw/`.
