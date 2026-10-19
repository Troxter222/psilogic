#!/usr/bin/env python3
"""Compare inspected package sources and non-research Git status with snapshot."""

import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
ROOT = RESEARCH.parents[1]
snapshot = json.loads((RESEARCH / "source_snapshot.json").read_text())
changed = [
    name
    for name, digest in snapshot["source_sha256"].items()
    if not (ROOT / name).exists()
    or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
]
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
status = subprocess.check_output(
    ["git", "status", "--porcelain", "--untracked-files=normal", "--", ".", ":(exclude)research"],
    cwd=ROOT,
    text=True,
)
result = {
    "status": "pass"
    if not changed
    and revision == snapshot["revision"]
    and status == snapshot["git_status_excluding_research"]
    else "fail",
    "package_files_checked": len(snapshot["source_sha256"]),
    "changed_package_files": changed,
    "revision_unchanged": revision == snapshot["revision"],
    "nonresearch_git_status_unchanged": status == snapshot["git_status_excluding_research"],
    "limitation": "Git status comparison checks paths/status; package files additionally checked byte-for-byte by SHA256.",
}
(HERE / "source_integrity_results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(result["status"] != "pass")
