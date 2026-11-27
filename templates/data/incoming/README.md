# incoming/

Where new sessions wait to be added to `raw/`. Each person has their own
folder, `incoming/{your username}/`, that only they can write to.

**`_example/` shows a hand-in exactly as it should look.** In short, drop
into your folder:

- the cohort's `{cohort}_cohort.yaml`, the mouse's
  `sub-M{mouse}_subject.yaml` and `sub-M{mouse}_procedures.yaml` (`[]`
  until its first surgery), every time;
- one folder per session, named `sub-M{mouse}_day-{dd}_ses-{type}_{date}_{time}`,
  holding everything the rig wrote, not renamed (and a `…_session.yaml`
  with notes, if there's anything to note).

Then tell the data manager. Their ingest checks it, renames the recordings,
files it in `raw/` and empties your folder, or tells you what to fix.
Keep your own copy until it is in `raw/`.

The full guide, with more examples: "Adding data" in [../README.md](../README.md).
