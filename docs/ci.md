# Continuous integration

Every push to `main`, every pull request, and every Monday morning,
GitHub Actions runs [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
Results are on the repository's **Actions** tab and in the badge on the README.
It is free, because Raphael is public.

| Job | What it checks |
|---|---|
| `lint` | `ruff check src tests` |
| `test (core)` | the test suite without the `gpu` extra (Kilosort and suite2p tests skip) |
| `test (gpu extra, on CPU)` | the full suite with Kilosort, suite2p and torch installed, on GitHub's CPU runners |
| `downstream (<repo>)` | each project that uses Raphael, with this Raphael swapped in (below) |

## Downstream projects

For each linked project, the job:

1. checks out the project and installs it exactly as its `uv.lock` pins it;
2. replaces the pinned Raphael with the Raphael being tested (and fails if
   the swap did not take);
3. runs the project's own tests (`uv run pytest`).

So a pull request to Raphael shows which projects it would break, before a
release. The Monday run catches projects that changed when Raphael did not.

### Linking a project

Add an entry to the `downstream` matrix in `ci.yml`:

```yaml
        include:
          - repo: HDClark94/starter_project
            extras: --extra dev
          - repo: <owner>/<project>          # the new project
            extras: --extra dev              # the extras its tests need
```

The project needs:

- tests that run with `uv run pytest`, on generated or simulated data
  (CI has no lab data and no GPU);
- a committed `uv.lock`.

**Private projects.** GitHub's default token can only read public
repositories. For a private project, create a fine-grained personal access
token (GitHub → Settings → Developer settings → Fine-grained tokens) with
read-only **Contents** access to that repository. Add it to Raphael as a
repository secret named `DOWNSTREAM_TOKEN` (Raphael → Settings → Secrets and
variables → Actions). One token can cover several private projects. The
minutes still count against Raphael, so it stays free.

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
