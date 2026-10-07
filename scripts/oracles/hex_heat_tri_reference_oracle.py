#!/usr/bin/env python3
"""Evidence writer for the Reality Anchor Phase 0a reference case ``hex-heat-tri``.

Runs ``run_hex_heat_tri_study`` (spec/REFERENCE_CASE_HEX_HEAT_TRI_v1.md), derives
the ReferenceSpec v1 claim ceiling, and prints (or writes) strict JSON bound to
the source revision. Exit code 0 only if every gate criterion holds; 1
otherwise. The same study runs in ``tests (cpu)`` via
``src/toolkit/test_reference_case_hex_v3.py``; this script produces the
auditable artefact for ``results/``.

    python scripts/oracles/hex_heat_tri_reference_oracle.py --output results/<name>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src" / "toolkit"))

import numpy  # noqa: E402

from reference_case_hex_v3 import hex_heat_tri_reference_spec, run_hex_heat_tri_study  # noqa: E402


def _source_revision() -> dict:
    """Git HEAD and whether the work tree is clean (GITHUB_SHA preferred in CI)."""
    sha = os.environ.get("GITHUB_SHA")
    dirty = None
    try:
        if not sha:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True,
                                 text=True, check=True, stdin=subprocess.DEVNULL).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=_REPO, capture_output=True,
                                text=True, check=True, stdin=subprocess.DEVNULL).stdout
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        sha = sha or None
    return {"git_sha": sha, "worktree_dirty": dirty}


def build_report(evidence_ref: str) -> dict:
    study = run_hex_heat_tri_study()
    spec = hex_heat_tri_reference_spec(study, evidence_ref=evidence_ref)
    study_bytes = json.dumps(study, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "source": _source_revision(),
        "environment": {"python": platform.python_version(), "numpy": numpy.__version__,
                        "platform": platform.platform()},
        "study_sha256": hashlib.sha256(study_bytes).hexdigest(),
        "study": study,
        "reference_spec": spec.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write the JSON report here (default: stdout)")
    args = parser.parse_args(argv)
    evidence_ref = args.output.as_posix() if args.output else "scripts/oracles/hex_heat_tri_reference_oracle.py"
    report = build_report(evidence_ref)
    text = json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    study = report["study"]
    failed = [name for name, ok in study["criteria"].items() if not ok]
    print(
        f"[HEX-HEAT-TRI] p={study['celik_fine']['observed_order']:.4f} "
        f"GCI_fine={study['celik_fine']['gci_fine']:.3e} "
        f"ceiling={report['reference_spec']['claim_ceiling']} | "
        f"{'PASS' if study['passed'] else 'FAIL ' + ','.join(failed)}",
        file=sys.stderr,
    )
    return 0 if study["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
