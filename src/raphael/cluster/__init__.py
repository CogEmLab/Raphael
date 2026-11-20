"""Running heavy stages elsewhere: GPU checks, provenance and job bundles.

Everything runs from one uv lockfile (`uv sync --frozen`), so a result made
on a cluster node can be traced to the exact package versions that made it.
See docs/cluster.md.
"""
