#!/usr/bin/env python3
"""Evidence writer for the whole Reality Anchor Phase 0 (issue #60, W1-K08).

Runs the three reference studies:
- ``hex-heat-tri`` (spec/REFERENCE_CASE_HEX_HEAT_TRI_v1.md);
- ``hex-heat-torus`` and ``hex-mms-tri`` (spec/REFERENCE_CASES_HEX_PHASE0B_v1.md).

It derives their ReferenceSpec v1 claim ceilings (four specs: the MMS case
yields an oracle spec and a solver spec) and prints or writes strict JSON
bound to the source revision. The exit code is 0 only if every study passes
its gate and every spec reaches the expected ceiling; otherwise it is 1.

The same studies run in ``tests (cpu)`` via
``src/toolkit/test_reference_case_hex_v3.py``. This script produces the
auditable artefact for ``results/``.

    python scripts/oracles/hex_phase0_reference_oracle.py --output results/<name>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src" / "toolkit"))

import numpy  # noqa: E402

from hex_heat_tri_reference_oracle import _source_revision  # noqa: E402  (same directory)
from reference_case_hex_v3 import (  # noqa: E402
    hex_heat_torus_reference_spec,
    hex_heat_tri_reference_spec,
    hex_mms_tri_reference_specs,
    run_hex_heat_torus_study,
    run_hex_heat_tri_study,
    run_hex_mms_tri_study,
)

_EXPECTED = {
    "hex-heat-tri/v1": "verified_numerical_reference",
    "hex-heat-torus/v1": "verified_numerical_reference",
    "hex-mms-tri/oracle": "mathematical_oracle",
    "hex-mms-tri/v1": "verified_numerical_reference",
}


def _digest(value: dict) -> str:
    data = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_report(evidence_ref: str) -> dict:
    tri = run_hex_heat_tri_study()
    torus = run_hex_heat_torus_study()
    mms = run_hex_mms_tri_study()
    mms_oracle, mms_solver = hex_mms_tri_reference_specs(mms, evidence_ref=evidence_ref)
    specs = [
        hex_heat_tri_reference_spec(tri, evidence_ref=evidence_ref),
        hex_heat_torus_reference_spec(torus, evidence_ref=evidence_ref),
        mms_oracle,
        mms_solver,
    ]
    studies = {"hex-heat-tri": tri, "hex-heat-torus": torus, "hex-mms-tri": mms}
    return {
        "source": _source_revision(),
        "environment": {"python": platform.python_version(), "numpy": numpy.__version__,
                        "platform": platform.platform()},
        "study_sha256": {name: _digest(study) for name, study in studies.items()},
        "studies": studies,
        "reference_specs": [spec.to_dict() for spec in specs],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write the JSON report here (default: stdout)")
    args = parser.parse_args(argv)
    evidence_ref = args.output.as_posix() if args.output else "scripts/oracles/hex_phase0_reference_oracle.py"
    report = build_report(evidence_ref)
    text = json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    ok = True
    for name, study in report["studies"].items():
        failed = [key for key, passed in study["criteria"].items() if not passed]
        celik = study["celik_fine"]
        print(f"[PHASE0] {name:15s} p={celik['observed_order']:.4f} GCI_fine={celik['gci_fine']:.3e} | "
              f"{'PASS' if study['passed'] else 'FAIL ' + ','.join(failed)}", file=sys.stderr)
        ok = ok and study["passed"]
    for spec in report["reference_specs"]:
        expected = _EXPECTED[spec["reference_id"]]
        hit = spec["claim_ceiling"] == expected
        print(f"[PHASE0] {spec['reference_id']:19s} ceiling={spec['claim_ceiling']}"
              f"{'' if hit else ' (expected ' + expected + ')'}", file=sys.stderr)
        ok = ok and hit
    print(f"[PHASE0] {'PASS' if ok else 'FAIL'}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
