#!/usr/bin/env python3
from __future__ import annotations

import unittest

from ci_reality_anchor_v1 import CIEvidence, check_readme, render_status_block


class TestCIRealityAnchor(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = CIEvidence.from_mapping(
            {
                "main_sha": "e4a0e5e5f402f4b1df67443e090d6d2276fc8aba",
                "workflow_run_id": 36478545347,
                "engine_doctests_total": 49,
                "engine_selftests_total": 26,
                "toolkit_tests": 299,
                "branch_coverage_percent": 83,
            }
        )

    def test_generated_block_roundtrip(self) -> None:
        block = render_status_block(self.evidence)
        self.assertEqual(check_readme(block, self.evidence), [])
        self.assertIn("299/299", block)
        self.assertIn("83 %", block)

    def test_stale_block_fails(self) -> None:
        stale = render_status_block(self.evidence).replace("299/299", "275/275")
        self.assertNotEqual(check_readme(stale, self.evidence), [])

    def test_duplicate_machine_block_fails_closed(self) -> None:
        block = render_status_block(self.evidence)
        with self.assertRaises(ValueError):
            check_readme(block + "\n" + block, self.evidence)

    def test_invalid_sha_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CIEvidence.from_mapping(
                {
                    "main_sha": "not-a-sha",
                    "workflow_run_id": 1,
                    "engine_doctests_total": 49,
                    "engine_selftests_total": 26,
                    "toolkit_tests": 299,
                    "branch_coverage_percent": 83,
                }
            )


if __name__ == "__main__":
    unittest.main()
