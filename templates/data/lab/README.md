# lab/

Reference lists that aren't about any one animal. Everything about a
cohort, a mouse or a session lives in that cohort's, mouse's or session's
own folder under `raw/` (see `raw/README.md`).

| file | what | edited by |
|---|---|---|
| `lab.yaml` | lab name, institution, PI, timezone, default licence | data managers |
| `people.yaml` | everyone named in the tree: full name, email, ORCID, roles | each person adds themselves |
| `projects.yaml` | projects: description, members, animal licence, funding, publications | project owners |
| `tasks.yaml` | task codes (`OF`, `VR`, `HBT`): what each task is | task owners; a code before its first session |
| `rigs/{rig}.yaml` | each rig's devices, cameras, probes, TTL wiring | rig owners |
| `templates/` | blank files to copy when adding a cohort, mouse, procedure, session or rig | |

These, together with the per-level files in `raw/`, hold everything an NWB
file and a DANDI upload need: lab and institution; experimenters with
ORCIDs; the experiment and session descriptions; subject species, sex,
date of birth, strain and genotype; surgeries, viruses and implant
locations; the animal licence (NWB `protocol`); keywords, funding and
publications.

All files are YAML: `key: value`, lists in `[ ]` or `- ` lines, `#` starts
a comment. Dates are `YYYY-MM-DD`, times are local (`2026-03-14T10:05:00`,
read in `lab.yaml`'s timezone).
