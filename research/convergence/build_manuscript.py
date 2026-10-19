#!/usr/bin/env python3
"""Compile the research manuscript without changing the pre-existing paper."""

from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "verification" / "latex"
BUILD.mkdir(parents=True, exist_ok=True)
commands = [
    [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={BUILD}",
        "manuscript.tex",
    ],
    ["bibtex", str((BUILD / "manuscript").relative_to(ROOT))],
    [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={BUILD}",
        "manuscript.tex",
    ],
    [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={BUILD}",
        "manuscript.tex",
    ],
]
records = []
for i, command in enumerate(commands, 1):
    proc = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    logfile = BUILD / f"pass_{i}.log"
    logfile.write_text(proc.stdout)
    records.append(
        {"command": command, "exit_code": proc.returncode, "log": str(logfile.relative_to(ROOT))}
    )
    if proc.returncode:
        (BUILD / "build_results.json").write_text(
            json.dumps({"status": "fail", "commands": records}, indent=2) + "\n"
        )
        print(proc.stdout[-4000:])
        raise SystemExit(proc.returncode)
log = (BUILD / "manuscript.log").read_text()
issues = [
    line
    for line in log.splitlines()
    if "Overfull" in line or "undefined" in line or "Rerun to get" in line
]
shutil.copy2(BUILD / "manuscript.pdf", ROOT / "manuscript.pdf")
result = {
    "status": "pass" if not issues else "needs_layout_review",
    "commands": records,
    "issues": issues,
    "tex_sha256": hashlib.sha256((ROOT / "manuscript.tex").read_bytes()).hexdigest(),
    "pdf_sha256": hashlib.sha256((ROOT / "manuscript.pdf").read_bytes()).hexdigest(),
}
(BUILD / "build_results.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {"status": result["status"], "issues": issues, "pdf": str(ROOT / "manuscript.pdf")},
        indent=2,
    )
)
raise SystemExit(bool(issues))
