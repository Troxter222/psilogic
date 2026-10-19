#!/usr/bin/env python3
"""Compile the exact checked source using an existing pinned Mathlib checkout.

This helper does not install tools or fetch dependencies. For a standalone
installation, use the lakefile in this directory and `lake build` instead.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--mathlib", type=Path, required=True)
parser.add_argument("--lake", type=Path, required=True)
args = parser.parse_args()
source = HERE / "PsiLogic.lean"
command = [str(args.lake), "env", "lean", "-DautoImplicit=false", str(source)]
result = subprocess.run(
    command, cwd=args.mathlib, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
)
contains_placeholder_axiom = "sorryAx" in result.stdout
verified = result.returncode == 0 and not contains_placeholder_axiom
version = subprocess.run(
    [str(args.lake.parent / "lean"), "--version"], text=True, stdout=subprocess.PIPE
).stdout.strip()
revision = subprocess.run(
    ["git", "-C", str(args.mathlib), "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE
).stdout.strip()
log = (
    "Command: "
    + " ".join(command)
    + "\nWorking directory: "
    + str(args.mathlib)
    + "\nRuntime: "
    + version
    + "\nMathlib commit: "
    + revision
    + "\nSource SHA256: "
    + hashlib.sha256(source.read_bytes()).hexdigest()
    + "\n\n"
    + result.stdout
    + "\nExit status: "
    + str(result.returncode)
    + "\n"
)
(HERE / "build.log").write_text(log)
status = {
    "status": "kernel_checked" if verified else "build_failed",
    "exit_code": result.returncode,
    "contains_placeholder_axiom": contains_placeholder_axiom,
    "runtime": version,
    "mathlib_commit": revision,
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "theorem_count": sum(line.startswith("theorem ") for line in source.read_text().splitlines()),
    "scope": "Scalar algebra and finite sums under explicit assumptions; not full optimizer or stochastic convergence.",
}
(HERE / "build_status.json").write_text(json.dumps(status, indent=2) + "\n")
print(log)
raise SystemExit(0 if verified else result.returncode or 1)
