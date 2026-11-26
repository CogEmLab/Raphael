# _example/

What a hand-in looks like, for the example cohort `2027-01_2P-Testing`.
Your own goes in `incoming/{your username}/`, laid out the same way. The
recording files here are **empty placeholders** with the names real ones
would have; the `.yaml` files are filled in as you would fill yours.

```
_example/2027-01_2P-Testing/
├── M1/                                   a mouse already in raw/: just the new session
│   └── D2/OF1/
│       ├── sub-M1_ses-D2_typ-OF1_session.yaml
│       ├── sub-M1_ses-D2_typ-OF1_ophys/          ScanImage's folder, renamed outside only
│       │   └── file_00001.tif
│       └── sub-M1_ses-D2_typ-OF1_video-top.mp4
└── M2/                                   a new mouse: its files, then its first session
    ├── sub-M2_subject.yaml
    ├── sub-M2_procedures.yaml
    └── D1/OF1/
        ├── sub-M2_ses-D1_typ-OF1_session.yaml
        ├── sub-M2_ses-D1_typ-OF1_ophys/
        │   └── file_00001.tif
        └── sub-M2_ses-D1_typ-OF1_video-top.mp4
```

- **M1 D2:** M1 already has `D1` in `raw/`, and this is a new date for
  it, so this session is day 2. Its subject and procedures files are
  already in `raw/`, so they aren't handed in again.
- **M2 D1:** a new mouse. Its subject and procedures files come with its
  first session. The cohort already exists, so no `cohort.yaml`; a new
  cohort would add one at `_example/2027-01_2P-Testing/cohort.yaml`.

The full walk-through, with more cases, is "Adding data" in `../../README.md`.
