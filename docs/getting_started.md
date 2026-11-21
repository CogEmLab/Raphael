# Getting started

From a fresh laptop to running the example analysis on your own copy of
[starter_project](https://github.com/HDClark94/starter_project), the
template every new project starts from. It takes about 20 minutes, most of it
downloads.

You will set up:

- **VS Code**, the editor we recommend, with **Claude Code** as an extension
  so you can ask Claude about, and have it change, the code you have open;
- **git** and a **GitHub** account, to keep your project under version control;
- **uv**, which installs the exact Python environment a project needs,
  Raphael included.

The commands are for macOS (and Linux). On Windows, use
[WSL](https://learn.microsoft.com/windows/wsl/install) and follow the Linux
lines.

## 1. Install the tools (once per laptop)

**git.** On a Mac, this installs it along with Apple's command-line tools:

```bash
xcode-select --install
git --version                     # check: prints a version
```

Tell git who you are (use the email of your GitHub account):

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.org"
```

**uv**, then open a new terminal so it is on your `PATH`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version                      # check, in a new terminal
```

You do not need to install Python: uv downloads Python 3.11 for each project.

**GitHub.** Make an account at [github.com](https://github.com) if you do not
have one, and ask a maintainer to add you to the lab's repositories.

**VS Code.** Download it from [code.visualstudio.com](https://code.visualstudio.com)
and move it to Applications. Open it, press `Cmd+Shift+P`, run
**Shell Command: Install 'code' command in PATH**, and open a new terminal.
Then install the extensions:

```bash
code --install-extension anthropic.claude-code     # Claude Code
code --install-extension ms-python.python          # Python
code --install-extension ms-toolsai.jupyter        # notebooks
code --install-extension charliermarsh.ruff        # formatting and linting
```

(starter_project also lists these, so VS Code offers to install any you missed
when you open it.)

**Claude Code.** Click the Claude icon at the top right of an open editor, or
press `Cmd+Esc`, and sign in with your Claude account when it asks.

## 2. Fork starter_project

A fork is your own copy of the repository on GitHub, which you can change
freely and which still knows where it came from.

1. Go to <https://github.com/HDClark94/starter_project>.
2. Click **Fork** (top right).
3. Set **Repository name** to your project's name, e.g. `my-project`, and
   click **Create fork**.

> **Private projects.** A fork of a public repository is public. If your
> project must stay private until publication, instead use GitHub's
> **Import repository** (the `+` menu, top right) with
> `https://github.com/HDClark94/starter_project` as the source, and choose
> **Private**. The steps below are the same.

## 3. Clone it and install

Replace `<you>` with your GitHub username and `my-project` with your repository's name:

```bash
mkdir -p ~/Documents && cd ~/Documents
git clone https://github.com/<you>/my-project.git
cd my-project
git remote add upstream https://github.com/HDClark94/starter_project.git
uv sync
```

`uv sync` creates `.venv/` in the project, with Python 3.11, Raphael (the
version pinned in `pyproject.toml`) and everything else in `uv.lock`. It
installs Raphael like any other package: none of Raphael's code is copied into
your repository.

Check it worked:

```bash
uv run python -c "import raphael; print('raphael', raphael.__version__)"
```

Only if you will spike-sort or run suite2p on this machine (most people do
that on the cluster), add the GPU packages, a ~3 GB download:

```bash
uv sync --extra gpu
```

## 4. Open it in VS Code

```bash
code .
```

- Click **Install** if VS Code offers the recommended extensions.
- VS Code picks up the project's environment (`.venv`) on its own. If a
  Python file shows import errors, press `Cmd+Shift+P`, run
  **Python: Select Interpreter**, and choose `./.venv/bin/python`.
- The terminal inside VS Code (`` Ctrl+` ``) opens in the project folder; use
  it for the `uv run ...` commands below.

## 5. Run the examples

Make the demo data (a simulated recording and a processed session), then run
the analysis:

```bash
uv run python scripts/make_demo_data.py
uv run python scripts/analyse_tuning.py           # writes figures/sub-M1_ses-D1_typ-OF1_tuning.pdf
uv run python scripts/process_raw_ephys.py lfp --demo
```

Or use the notebooks: open `notebooks/02_analyse_processed_data.ipynb`, click
**Select Kernel** (top right) → **Python Environments** → `.venv`, then
**Run All**. `01_process_raw_data.ipynb` runs the processing steps the same way.

| | Processing raw data | Analysing processed data |
|---|---|---|
| Notebook | `notebooks/01_process_raw_data.ipynb` | `notebooks/02_analyse_processed_data.ipynb` |
| Script | `scripts/process_raw_ephys.py` | `scripts/analyse_tuning.py` |
| Uses | Raphael's `lfp` and `kilosort4` job stages | `raphael.population`, `raphael.tuning_scores` |

## 6. Use Claude Code

Open the Claude panel (Claude icon, or `Cmd+Esc`). It sees the repository and
reads `CLAUDE.md`, which tells it that neural processing comes from Raphael
and that project code belongs in `src/`. Things to try:

- *"Explain what `scripts/analyse_tuning.py` does, step by step."*
- *"Add a head-direction tuning-curve plot to the analysis notebook."*
- *"Where in Raphael is spike sorting configured, and what are the defaults?"*
- *"My behaviour is in DeepLabCut CSVs; rewrite `behaviour.py` to load them."*

Claude shows each change before applying it. Read changes before you accept
them, and commit often so any change can be undone.

## 7. Make it yours and push

Follow **Making it yours** in your project's README (rename the package, fill
in `configs/`, replace the arena and behaviour). Then commit and push:

```bash
git add -A
git commit -m "Start my-project from starter_project"
git push
```

Then **tell the Raphael owner** (currently Harry Clark,
[@HDClark94](https://github.com/HDClark94)) about your project: send its
repository URL and say whether it's private. Harry adds it to Raphael's CI
([ci.md](ci.md)), so every change to Raphael is tested against your
project's tests before it's released.

## Later

**Run on the cluster.** Set your MSC login once, then follow
[msc_cluster.md](msc_cluster.md):

```bash
uv run raphael-jobs cluster-user <MUW-ID>
```

**Move to a newer Raphael**, when one is released:

```bash
uv add "raphael @ git+https://github.com/HDClark94/Raphael" --tag v0.2.0
git commit -am "Raphael 0.2.0"
```

**Get improvements to the template** (new examples, fixes):

```bash
git fetch upstream
git merge upstream/main
```

**Work on Raphael itself** at the same time as your project. Clone it beside
the project and point the project at your checkout:

```bash
git clone https://github.com/HDClark94/Raphael.git ~/Documents/Raphael
cd ~/Documents/my-project
uv add --editable ../Raphael
```

Before committing your project, go back to a released Raphael with the
`uv add ... --tag` line above.

## If something goes wrong

| Problem | Fix |
|---|---|
| `uv: command not found` | open a new terminal; or run `source ~/.local/bin/env` |
| `code: command not found` | in VS Code, `Cmd+Shift+P` → **Shell Command: Install 'code' command in PATH** |
| uv says no solution because of `requires-python` | the project must say `requires-python = "==3.11.*"`, as Raphael requires |
| notebook cannot find `raphael` or `starter_project` | the kernel is not `.venv`: **Select Kernel** → `.venv`; run `uv sync` first |
| `ModuleNotFoundError: kilosort` | spike sorting needs `uv sync --extra gpu` |
| Kilosort fails with an MPS / float64 error on a Mac | Kilosort 4.0.16 cannot use Apple GPUs; `process_raw_ephys.py` sorts on the CPU, and real data belongs on the cluster |
