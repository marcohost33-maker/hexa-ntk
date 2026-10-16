"""
test_reference_oracle_source_binding.py
=======================================
The Phase 0 evidence writers (scripts/oracles/hex_heat_tri_reference_oracle.py,
scripts/oracles/hex_phase0_reference_oracle.py) may exit 0 only if the
recorded source revision is bound to the executed bytes: git answered, the
work tree is clean and GITHUB_SHA (when set) equals HEAD (Codex P2 on #61).
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

_ORACLES = Path(__file__).resolve().parents[2] / "scripts" / "oracles"
if str(_ORACLES) not in sys.path:
    sys.path.insert(0, str(_ORACLES))

import hex_heat_tri_reference_oracle as tri_oracle  # noqa: E402
import hex_phase0_reference_oracle as phase0_oracle  # noqa: E402

_HEAD = "a" * 40


def _fake_git(head: str | None = _HEAD, status: str = "", fail: bool = False):
    def run(cmd, **kwargs):
        if fail:
            raise subprocess.CalledProcessError(128, cmd)
        out = head if cmd[1] == "rev-parse" else status
        return subprocess.CompletedProcess(cmd, 0, stdout=(out or "") + "\n", stderr="")

    return run


class TestSourceRevision(unittest.TestCase):
    def _revision(self, github_sha: str | None = None, **git) -> dict:
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_SHA"}
        if github_sha is not None:
            env["GITHUB_SHA"] = github_sha
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(tri_oracle.subprocess, "run", side_effect=_fake_git(**git)):
            return tri_oracle._source_revision()

    def test_clean_verified_checkout_is_bound(self) -> None:
        self.assertEqual(self._revision(), {"git_sha": _HEAD, "github_sha": None, "worktree_dirty": False,
                                            "bound": True})
        self.assertTrue(self._revision(github_sha=_HEAD)["bound"])

    def test_unbound_states(self) -> None:
        cases = {
            "dirty work tree": self._revision(status=" M src/toolkit/hex_fv_solver_hex_v3.py"),
            "untracked file": self._revision(status="?? scratch.py"),
            "git unavailable": self._revision(fail=True),
            "git unavailable, GITHUB_SHA only": self._revision(github_sha=_HEAD, fail=True),
            "stale GITHUB_SHA": self._revision(github_sha="b" * 40),
            "empty HEAD": self._revision(head=""),
        }
        for name, revision in cases.items():
            with self.subTest(case=name):
                self.assertIs(revision["bound"], False)
        self.assertEqual(cases["git unavailable, GITHUB_SHA only"]["git_sha"], _HEAD)  # recorded, not verified
        self.assertIsNone(cases["git unavailable"]["worktree_dirty"])


class TestWritersFailClosedOnUnboundSource(unittest.TestCase):
    """main() of both writers with a passing study: exit 0 iff the source is bound."""

    @staticmethod
    def _source(bound: bool) -> dict:
        return {"git_sha": _HEAD, "github_sha": None, "worktree_dirty": not bound, "bound": bound}

    @staticmethod
    def _passing_study() -> dict:
        return {"criteria": {"C1": True}, "passed": True, "celik_fine": {"observed_order": 2.0, "gci_fine": 1e-6}}

    def _main(self, module, report: dict) -> tuple[int, str]:
        stderr = io.StringIO()
        with mock.patch.object(module, "build_report", return_value=report), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = module.main([])
        return code, stderr.getvalue()

    def test_phase0_writer(self) -> None:
        specs = [{"reference_id": rid, "claim_ceiling": ceiling} for rid, ceiling in phase0_oracle._EXPECTED.items()]
        for bound, expected in ((True, 0), (False, 1)):
            with self.subTest(bound=bound):
                report = {"source": self._source(bound), "studies": {"s": self._passing_study()},
                          "reference_specs": specs}
                code, stderr = self._main(phase0_oracle, report)
                self.assertEqual(code, expected, stderr)
                self.assertEqual("NOT bound" in stderr, not bound)

    def test_heat_tri_writer(self) -> None:
        for bound, expected in ((True, 0), (False, 1)):
            with self.subTest(bound=bound):
                report = {"source": self._source(bound), "study": self._passing_study(),
                          "reference_spec": {"claim_ceiling": "verified_numerical_reference"}}
                code, stderr = self._main(tri_oracle, report)
                self.assertEqual(code, expected, stderr)
                self.assertEqual("NOT bound" in stderr, not bound)

    def test_missing_or_non_bool_bound_field_is_unbound(self) -> None:
        for source in ({"git_sha": _HEAD, "worktree_dirty": False}, dict(self._source(True), bound="yes"),
                       dict(self._source(True), bound=1)):
            with self.subTest(source=source), contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(tri_oracle._report_source_binding(source, "T"))


if __name__ == "__main__":
    unittest.main()
