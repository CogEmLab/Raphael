# _example/

Two hand-ins, exactly as they should look in your `incoming/{username}/`.
Recording files here are **empty placeholders**; the `.yaml` files are
filled in as you would fill yours.

```
incoming/{username}/
├── 2027-01_2P-Testing_cohort.yaml                          the cohort
├── sub-M1_subject.yaml                                     mouse 1
├── sub-M1_procedures.yaml                                  its surgeries
├── sub-M1_day-02_ses-OF1_2027-01-13_10-20-00/              mouse 1, day 2, open field run 1
│   ├── sub-M1_day-02_ses-OF1_session.yaml
│   ├── scanimage/file_00001.tif                            what the 2P rig wrote, not renamed
│   └── cam0_2027-01-13.mp4                                 what the camera wrote, not renamed
├── sub-M2_subject.yaml                                     mouse 2, new
├── sub-M2_procedures.yaml                                  blank: no surgery yet
└── sub-M2_day-01_ses-HAB1_2027-01-13_14-05-00/             mouse 2, day 1, habituation run 1
    ├── sub-M2_day-01_ses-HAB1_session.yaml
    └── cam0_2027-01-13_2.mp4                               behaviour only
```

## What every hand-in has

- **The cohort file**, `{cohort}_cohort.yaml`, for every cohort in it.
- **Each mouse's subject and procedures files**, `sub-M{mouse}_subject.yaml`
  and `sub-M{mouse}_procedures.yaml`, every time. The procedures file is
  blank (`[]`) until the mouse has a surgery; then add an entry and hand it
  in with the next session.
- **One folder per session**, named
  `sub-M{mouse}_day-{dd}_ses-{task}{run}_{YYYY-MM-DD}_{HH-MM-SS}`
  (the session's start), holding its session file
  `sub-M{mouse}_day-{dd}_ses-{task}{run}_session.yaml` and everything the
  rig wrote, **not renamed**.

The session folder's name says which mouse, day and session it is, and
when it started, so you don't build any folder tree: the data manager's
ingest works out where everything goes, checks the name against the
recording's own start time, the session file and the mouse's earlier days,
and renames the recordings. If something doesn't match, it tells you.

- **Day** (`day-02`): the mouse's recording days in date order. M1 already
  has day 1, so a session on a new date is day 2. Two sessions on the same
  date share the day.
- **Run** (`ses-OF1`): the first open field of that day is `OF1`, a second
  one `OF2`. Task codes (`OF`, `HAB`) must be in `metadata/sessions.yaml`.

The cohort, subject and procedures files are the same ones each time: keep
your copies and hand them in again. Change them when something changes
(a surgery, a correction); ingest updates the server's copy and keeps the
old one.
