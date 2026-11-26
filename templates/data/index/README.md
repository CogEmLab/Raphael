# index/

Tables of everything in `raw/`, **built from the describing files**, for
searching and filtering. Never edited by hand: correct the `.yaml` file in
`raw/` (or `lab/`) instead, and the tables follow.

| file | one row per | built from |
|---|---|---|
| `cohorts.tsv` | cohort | `raw/{cohort}/cohort.yaml` |
| `subjects.tsv` | mouse | `sub-M…_subject.yaml` (BIDS `participants.tsv`) |
| `procedures.tsv` | procedure | `sub-M…_procedures.yaml` |
| `sessions.tsv` | session | `sub-M…_ses-…_session.yaml`, with the path of each recording (BIDS `sessions.tsv`) |

Rebuilt by ingest and nightly. **(To be written:** the script that builds
them, planned as a Raphael command.)
