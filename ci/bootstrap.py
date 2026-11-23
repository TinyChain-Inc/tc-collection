#!/usr/bin/env python3
"""Populate a fresh sibling layout from repository-owned dependency pins."""

import json
from pathlib import Path
import re
import subprocess


def bootstrap():
    repository = Path(__file__).resolve().parent.parent
    workspace = repository.parent
    dependencies = json.loads((repository / "ci/dependencies.json").read_text())

    # Validate every destination before creating any repository. Existing local
    # checkouts are never reset, patched, or replaced by CI setup.
    for dependency in dependencies:
        destination = (workspace / dependency["path"]).resolve()
        destination.relative_to(workspace)
        if destination == workspace or destination.exists():
            raise ValueError(f"dependency destination must be absent: {destination}")
        if not re.fullmatch(r"[0-9a-f]{40}", dependency["revision"]):
            raise ValueError(f"dependency revision must be a commit SHA: {dependency}")

    for dependency in dependencies:
        destination = workspace / dependency["path"]
        checkout = destination
        print(f"{dependency['path']}: {dependency['revision']}", flush=True)
        subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "remote", "add", "origin", dependency["repository"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "fetch", "--depth=1", "origin", dependency["revision"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
            check=True,
        )


if __name__ == "__main__":
    bootstrap()
