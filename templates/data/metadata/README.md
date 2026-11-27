# metadata/

What's in `data/`, and the lab's reference lists. Everything about a
particular cohort, mouse or session lives in that cohort's, mouse's or
session's own folder under `raw/` (see `raw/README.md`).

| file | what | edited by |
|---|---|---|
| `catalogue.csv` | **every session in `raw/`, one row each**: what it is, what it recorded, when it was added, its processing status, when it was processed, with which software and Raphael release, and whether it is outdated | **generated**: rebuilt by ingest and `raphael-jobs collect` (or `raphael-data catalogue`); never edit |
| `plans/` | processing plans, one per person per ingest run: `plan_{date}_{time}_{username}.csv`; the catalogue’s `plan` column names each session’s | **written by ingest**; may be edited (memory, a missing value) until jobs are made from it |
| `lab.yaml` | lab name, institution, PI, timezone, default licence | data managers |
| `people.yaml` | everyone named in the tree: full name, email, ORCID, roles | each person adds themselves |
| `projects.yaml` | projects: description, members, animal licence, funding, publications | project owners |
| `sessions.yaml` | the kinds of session, by code (`OF`, `VR`, `HBT`): what each is, its protocol, and a `discussion` of anything worth knowing about it | their owners; a code before its first session |
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
