#!/usr/bin/env python3
"""Render and verify the machine-owned README CI reality-anchor block."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

BEGIN = "<!-- BEGIN GENERATED CI REALITY ANCHOR -->"
END = "<!-- END GENERATED CI REALITY ANCHOR -->"


@dataclass(frozen=True)
class CIEvidence:
    main_sha: str
    workflow_run_id: int
    engine_doctests_total: int
    engine_selftests_total: int
    toolkit_tests: int
    branch_coverage_percent: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CIEvidence":
        evidence = cls(
            main_sha=str(data["main_sha"]),
            workflow_run_id=int(data["workflow_run_id"]),
            engine_doctests_total=int(data["engine_doctests_total"]),
            engine_selftests_total=int(data["engine_selftests_total"]),
            toolkit_tests=int(data["toolkit_tests"]),
            branch_coverage_percent=int(data["branch_coverage_percent"]),
        )
        if re.fullmatch(r"[0-9a-f]{40}", evidence.main_sha) is None:
            raise ValueError("main_sha must be a lowercase 40-hex commit SHA")
        if evidence.workflow_run_id <= 0:
            raise ValueError("workflow_run_id must be positive")
        if min(
            evidence.engine_doctests_total,
            evidence.engine_selftests_total,
            evidence.toolkit_tests,
        ) <= 0:
            raise ValueError("test totals must be positive")
        if not 0 <= evidence.branch_coverage_percent <= 100:
            raise ValueError("branch_coverage_percent must be within [0, 100]")
        return evidence


def render_status_block(evidence: CIEvidence) -> str:
    return "\n".join(
        [
            BEGIN,
            f"> main \`{evidence.main_sha}\` · CI run \`{evidence.workflow_run_id}\`",
            "",
            "| Gate | Verifizierter letzter Main-Lauf |",
            "|---|---|",
            (
                f"| Engine | {evidence.engine_doctests_total}/{evidence.engine_doctests_total} "
                f"Doctests + {evidence.engine_selftests_total}/{evidence.engine_selftests_total} Selftests |"
            ),
            f"| Toolkit | {evidence.toolkit_tests}/{evidence.toolkit_tests} Tests |",
            f"| Branch-Coverage | {evidence.branch_coverage_percent} % |",
            END,
        ]
    )


def extract_status_block(readme: str) -> str:
    if readme.count(BEGIN) != 1 or readme.count(END) != 1:
        raise ValueError("README must contain exactly one generated reality-anchor block")
    start = readme.index(BEGIN)
    stop = readme.index(END, start) + len(END)
    return readme[start:stop]


def check_readme(readme: str, evidence: CIEvidence) -> list[str]:
    actual = extract_status_block(readme)
    expected = render_status_block(evidence)
    return [] if actual == expected else ["README generated CI reality-anchor block is stale"]


def load_evidence(path: Path) -> CIEvidence:
    return CIEvidence.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", nargs=2, metavar=("README", "EVIDENCE"))
    parser.add_argument("--render", metavar="EVIDENCE")
    args = parser.parse_args()

    if args.render:
        print(render_status_block(load_evidence(Path(args.render))))
        return 0
    if args.check:
        readme_path, evidence_path = map(Path, args.check)
        mismatches = check_readme(
            readme_path.read_text(encoding="utf-8"),
            load_evidence(evidence_path),
        )
        if mismatches:
            for item in mismatches:
                print(f"ERROR: {item}")
            return 1
        print("OK: README CI reality anchor matches structured evidence")
        return 0
    parser.error("use --check or --render")


if __name__ == "__main__":
    raise SystemExit(main())
