# metadata/

What's in `data/`, and the lab's reference lists. Everything about a
particular cohort, mouse or session lives in that cohort's, mouse's or
session's own folder under `raw/` (see `raw/README.md`).

| file | what | edited by |
|---|---|---|
| `catalogue.csv` | **every session in `raw/`, one row each**: what it is, what it recorded, when it was added, and for each processing stage it needs, when it was processed and by which Raphael version (or `to do`); `needs_processing`, `outdated` | **generated**: rebuilt by ingest and `raphael-jobs collect` (or `raphael-data catalogue`); never edit |
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

`catalogue.csv` starts with a `#` line saying when it was built: read it
with `pd.read_csv('catalogue.csv', comment='#')`, or skip the first row in
Excel.
