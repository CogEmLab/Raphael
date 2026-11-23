# Continuous integration

Every push to `main`, every pull request, and every Monday morning,
GitHub Actions runs [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
Results are on the repository's **Actions** tab and in the badge on the README.
It is free, because Raphael is public.

| Job | What it checks |
|---|---|
| `lint` | `ruff check src tests` |
| `test (core)` | the test suite without the `gpu` extra (Kilosort and suite2p tests skip) |
| `test (gpu, ml and tracking extras, on CPU)` | the full suite with Kilosort, suite2p, torch, XGBoost and DeepLabCut installed, on GitHub's CPU runners |
| `test (Windows)` | the suite with the ml extra on Windows, so the labs' Windows PCs keep working |
| `find downstream projects` | reads `.github/downstream/*.yml` (below) |
| `downstream (<project>)` | each project that uses Raphael, with this Raphael swapped in |

## Downstream projects

Raphael is mostly used by being imported from other repositories, so CI
tests every one of them. Each project has a config file in
[`.github/downstream/`](../.github/downstream/), named after it:

```
.github/downstream/
    starter_project.yml
    social_manifolds.yml
```

For each enabled project, the job:

1. checks out the project and installs it exactly as its `uv.lock` pins it
   (and fails if the project does not install Raphael at all);
2. replaces the pinned Raphael with the Raphael being tested (and fails if
   the swap did not take);
3. runs the project's tests.

So a pull request to Raphael shows which projects it would break, before a
release. The Monday run catches projects that changed when Raphael did not.
The run's summary page lists every configured project and whether it was
tested.

### A project's config

```yaml
# .github/downstream/my_project.yml
repo: HDClark94/my-project        # required: owner/name on GitHub
branch: main                      # default: the repository's default branch
private: false                    # true: needs the DOWNSTREAM_TOKEN secret
enabled: true                     # false: listed, but not tested
extras: [dev]                     # uv extras its tests need (CI has no GPU)
test: uv run pytest -q -rs        # the command that runs its tests
contact: HDClark94                # GitHub user to tell when it breaks
```

Only `repo` is required; the rest have the defaults shown. A mistake in a
file (an unknown field, a malformed repo) fails CI, and Raphael's own tests
(`tests/test_downstream_configs.py`) check the files locally too.

### Linking a project

Project owners: tell the Raphael owner (currently Harry Clark, @HDClark94) the
repository URL and whether it's private. The Raphael owner then:

1. Adds `.github/downstream/<project>.yml` in a pull request to Raphael.
2. Checks the project has tests that run with its `test` command on
   generated or simulated data (CI has no lab data and no GPU), and a
   committed `uv.lock` that includes Raphael.
3. If it's private, gives the token read access to it (below).

To pause a project (for example while it moves to a new Raphael), set
`enabled: false` and say why in a comment, as `social_manifolds.yml` does.

**Private projects.** GitHub's default token can only read public
repositories. For a private project, create a fine-grained personal access
token (GitHub → Settings → Developer settings → Fine-grained tokens) with
read-only **Contents** access to the private project repositories. Add it to
Raphael as a repository secret named `DOWNSTREAM_TOKEN` (Raphael →
Settings → Secrets and variables → Actions). One token covers all of them;
when a new private project is linked, give the token access to it too. The
minutes count against Raphael, so this stays free. Pull requests from forks
of Raphael get no secrets, so they skip private projects with a warning.

### When a downstream project fails

- If the change in Raphael is a mistake, fix Raphael.
- If the change is intended (a renamed function, a new default), update the
  project, ideally in the same week, and say so in the release notes so
  others can update theirs.

## Running the same checks locally

```bash
uv sync --frozen --extra dev            # add --extra gpu for the full suite
uv run ruff check src tests
uv run pytest -rs
```
