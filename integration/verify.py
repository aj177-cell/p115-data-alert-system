"""Prove that the combination is add-only.

python -m integration.verify              # SHA-256 of every Milestone 1 and v3 file
python -m integration.verify --run-tests  # also run the three test suites

Manifests (integration/manifests/):
  milestone1.sha256          every file of the milestone-1 commit (16 files)
  detectors_v3_code.sha256   v3 code, tests and READMEs, as shipped in p115_detectors_v3_final.zip
  detectors_v3_results.sha256  v3 result files; these change legitimately if the v3 scripts are re-run
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = Path(__file__).resolve().parent / "manifests"
CHECKS = (("Milestone 1 files", "milestone1.sha256", True),
          ("detectors v3 code", "detectors_v3_code.sha256", True),
          ("detectors v3 results", "detectors_v3_results.sha256", False))
SUITES = (("Milestone 1 tests (unchanged)", [sys.executable, "-m", "unittest", "discover", "-s", "tests"], ROOT),
          ("detectors v3 tests (unchanged)", [sys.executable, "-m", "pytest", "detectors", "-q", "-p", "no:cacheprovider"],
           ROOT / "p115_detectors"),
          ("integration tests (new)", [sys.executable, "-m", "unittest", "discover", "-s", "integration/tests", "-t", "."], ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(manifest: str, root: Path = ROOT) -> tuple[int, list[str], list[str]]:
    """Return (files listed, changed files, missing files) for one manifest."""
    changed, missing = [], []
    lines = [line.split("  ", 1) for line in (MANIFESTS / manifest).read_text().splitlines() if line.strip()]
    for expected, relative in lines:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
        elif sha256(path) != expected:
            changed.append(relative)
    return len(lines), changed, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-tests", action="store_true", help="also run the three test suites")
    args = parser.parse_args()
    ok = True
    for name, manifest, required in CHECKS:
        total, changed, missing = check(manifest)
        status = "unchanged" if not changed and not missing else f"{len(changed)} changed, {len(missing)} missing"
        print(f"{name:<24} {total - len(changed) - len(missing):>4}/{total} {status}")
        for relative in changed + missing:
            print(f"    {'changed' if relative in changed else 'missing'}: {relative}")
        ok &= not (required and (changed or missing))
    if args.run_tests:
        for name, command, cwd in SUITES:
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
            output = (result.stdout + result.stderr).strip().splitlines()
            summary = next((line for line in reversed(output) if line.startswith(("Ran ", "OK", "FAILED")) or " passed" in line), "")
            ran = next((line for line in reversed(output) if line.startswith("Ran ")), "")
            print(f"{name:<32} {'PASS' if result.returncode == 0 else 'FAIL'}  {ran or summary}")
            ok &= result.returncode == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
