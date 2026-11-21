# Releases, DOIs and publications

Every Raphael version that projects use is a **GitHub Release**: a fixed,
tagged snapshot with notes. With Zenodo connected, each release is archived
and gets its own **DOI**, which papers cite. Projects built on Raphael
(starter_project and its forks) release the same way, so a paper can cite
the exact analysis code and the exact Raphael it ran on.

## Versions

`MAJOR.MINOR.PATCH`, tagged `vMAJOR.MINOR.PATCH`. While Raphael is 0.x:

| Change | Bump | Example |
|---|---|---|
| alters outputs (a threshold, a filter, a default) or breaks imports | middle | 0.2.0 → 0.3.0 |
| fixes and additions that leave existing results unchanged | last | 0.2.0 → 0.2.1 |

Projects pin a release (`uv add "raphael @ git+https://github.com/HDClark94/Raphael" --tag vX.Y.Z`)
and move on purpose, after reading the changelog.

## Releasing Raphael

As you work, add a line under `## [Unreleased]` in
[CHANGELOG.md](../CHANGELOG.md) for each change that matters to a project.
`main` is protected (changes arrive by pull request, once CI passes), so a
release is a small pull request, then a tag on `main`:

```bash
git switch main && git pull
git switch -c release-0.3.0
uv run python .github/scripts/release_tools.py prepare 0.3.0
git diff                                  # versions bumped, changelog dated, uv.lock updated
git commit -am "Release 0.3.0"
git push -u origin release-0.3.0
gh pr create --fill                       # merge it once CI passes
git switch main && git pull               # after the merge
git tag -a v0.3.0 -m "v0.3.0"
git push origin v0.3.0
```

`prepare` writes the version into `pyproject.toml`, `src/raphael/__init__.py`
and `CITATION.cff`, turns `[Unreleased]` into a dated `[0.3.0]` section, and
re-locks. Pushing the tag starts the **Release** workflow
([`.github/workflows/release.yml`](../.github/workflows/release.yml)):

1. **check**: the tag matches every version and a dated changelog section, and is on `main`;
2. **ci**: the full CI, including every downstream project (docs/ci.md);
3. **publish**: builds the package and creates the GitHub Release, with the
   changelog section as its notes and the wheel and sdist attached.

Nothing is published unless all three pass. If one fails, fix the problem
(by pull request), then move the tag to the fixed `main`:

```bash
git push --delete origin v0.3.0 && git tag -d v0.3.0
git tag -a v0.3.0 -m "v0.3.0" && git push origin v0.3.0
```

Never move a tag after its release is published: projects and DOIs point at
it. Release a new version instead.

## DOIs with Zenodo (one-off set-up)

[Zenodo](https://zenodo.org) (run by CERN, free) archives each GitHub
Release of a connected repository and mints a DOI for it.

1. Log in at <https://zenodo.org> with your GitHub account.
2. Open **GitHub** in your Zenodo account menu, and switch **on**
   `HDClark94/Raphael`.
3. Make the next release. Zenodo archives only releases published after
   the switch, so earlier releases have no DOI.

Zenodo reads [CITATION.cff](../CITATION.cff) for the title, authors and
licence. Each release gets a **version DOI**; the record also has a
**concept DOI** that always resolves to the latest version. After the first
DOI:

- add `doi: <concept DOI>` to `CITATION.cff`, and the Zenodo DOI badge to
  the README;
- in papers, cite the **version** DOI of the release you used.

Add yourself to `authors` in `CITATION.cff` when you contribute, so you are
credited on the next DOI record.

## Publishing a paper from a project

The analysis behind a paper lives in a project repository built on
Raphael, so a paper should cite two releases: **the project** (the code that
made the figures) and **Raphael** (the version it ran on).

In the project (starter_project and its forks have the same tooling):

1. Pin Raphael to a release, and commit `pyproject.toml` and `uv.lock`.
   An editable or branch install of Raphael is fine while working, never for
   a release.
2. Make sure every figure is rebuilt by a script from the processed data.
3. Release the project: `prepare`, commit, tag and push, as above. Its
   release workflow refuses to publish unless Raphael is pinned to a
   released tag in `uv.lock`, so the published code always names a
   Raphael version that exists.
4. Connect the project to Zenodo for its own DOI. Zenodo's GitHub
   integration archives public repositories; a private project can be made
   public at publication, or its release archive uploaded to Zenodo by hand.

Anyone can then reproduce the environment exactly:

```bash
git clone https://github.com/<owner>/<project>.git && cd <project>
git checkout v1.0.0
uv sync --frozen                 # the project's uv.lock pins the Raphael release
```

A code-availability statement could read:

> Analysis code is available at https://github.com/<owner>/<project>
> (release v1.0.0, doi:<project DOI>). It uses Raphael v0.3.0
> (doi:<Raphael version DOI>), the labs' shared processing package.
