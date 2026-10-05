"""
test_hex_timestep_contract.py
=============================
Post-merge regression tests for PR #45 (hex L2 stability vs positivity).

Covers the Codex review threads on PR #45:
  * overflow/underflow of h*h must not yield inf/0.0 bounds (fail-closed);
  * returned timestep bounds never exceed the exact analytic boundary
    (rounded toward zero), checked against exact rational arithmetic;
  * the canonical spec carries the positivity contract h^2/(4 D);
  * README source anchors resolve to the code they quote.

The engine is loaded by path (it is not a toolkit module).
"""
from __future__ import annotations

import doctest
import importlib.util
import random
import re
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_ENGINE = _ROOT / "src" / "hexantk_v3.py"
_spec = importlib.util.spec_from_file_location("hexantk_v3_contract", _ENGINE)
assert _spec is not None and _spec.loader is not None
engine = importlib.util.module_from_spec(_spec)
# dataclasses resolve their module via sys.modules during class creation
sys.modules[_spec.name] = engine
_spec.loader.exec_module(engine)


def _exact_l2(h: float, d: float, safety: float = 1.0) -> Fraction:
    return Fraction(safety) * Fraction(h) ** 2 / (3 * Fraction(d))


def _exact_pos(h: float, d: float, safety: float = 1.0) -> Fraction:
    return Fraction(safety) * Fraction(h) ** 2 / (4 * Fraction(d))


class TestUnrepresentableBoundsRejected(unittest.TestCase):
    """Finite, valid inputs whose bound is not a finite positive binary64."""

    def test_cfl_overflow_rejected(self):
        with self.assertRaises(ValueError):
            engine.hex_cfl_max_dt(1e200, 1.0)

    def test_cfl_underflow_rejected(self):
        with self.assertRaises(ValueError):
            engine.hex_cfl_max_dt(1e-200, 1.0)

    def test_positivity_overflow_rejected(self):
        with self.assertRaises(ValueError):
            engine.hex_positivity_max_dt(1e200, 1.0)

    def test_positivity_underflow_rejected(self):
        with self.assertRaises(ValueError):
            engine.hex_positivity_max_dt(1e-200, 1.0)

    def test_stencil_weight_tiny_h_rejected(self):
        # before: ZeroDivisionError (not the documented ValueError contract)
        with self.assertRaises(ValueError):
            engine.hex_laplacian_stencil_weight(1e-200)

    def test_stencil_weight_huge_h_rejected(self):
        # before: silently 0.0
        with self.assertRaises(ValueError):
            engine.hex_laplacian_stencil_weight(1e200)

    def test_extreme_but_representable_is_exact(self):
        # h*h overflows binary64 in naive order, but h^2/D is representable.
        dt = engine.hex_positivity_max_dt(1e200, 1e200)
        self.assertTrue(0.0 < dt < float("inf"))
        self.assertLessEqual(Fraction(dt), _exact_pos(1e200, 1e200))


class TestBoundsRoundTowardZero(unittest.TestCase):
    """Returned dt must never exceed the exact rational bound."""

    CASE = (1.3865996863827346, 0.6157903110245121)  # from the review thread

    def test_review_case_positivity_centre_weight_nonnegative(self):
        h, d = self.CASE
        dt = engine.hex_positivity_max_dt(h, d)
        centre = 1 - 4 * Fraction(d) * Fraction(dt) / Fraction(h) ** 2
        self.assertGreaterEqual(centre, 0)

    def test_review_case_l2_not_above_exact(self):
        h, d = self.CASE
        self.assertLessEqual(Fraction(engine.hex_cfl_max_dt(h, d, 1.0)), _exact_l2(h, d))

    def test_random_scan_never_above_exact_and_within_one_ulp(self):
        rng = random.Random(20260927)
        for _ in range(5000):
            h = rng.uniform(1e-3, 1e3)
            d = rng.uniform(1e-3, 1e3)
            s = rng.choice([1.0, 0.85, rng.uniform(1e-3, 1.0)])
            for fn, exact_fn in ((engine.hex_positivity_max_dt, _exact_pos),
                                 (engine.hex_cfl_max_dt, _exact_l2)):
                got = fn(h, d, s)
                exact = exact_fn(h, d, s)
                self.assertLessEqual(Fraction(got), exact, (fn.__name__, h, d, s))
                # tight: the next representable value above is already > exact
                self.assertGreater(Fraction(float(np.nextafter(got, np.inf))), exact)


class TestExactFromGivenReal(unittest.TestCase):
    """Bounds are computed from the GIVEN numbers.Real, not from float(x)."""

    def _centre(self, h, d, dt) -> Fraction:
        return 1 - 4 * Fraction(d) * Fraction(dt) / Fraction(h) ** 2

    def test_int_above_2_53_centre_weight_nonnegative(self):
        # float(2**53 + 3) == 2**53 + 4: the old path bounded a larger h.
        h, d = 2**53 + 3, 1
        dt = engine.hex_positivity_max_dt(h, d)
        self.assertGreaterEqual(self._centre(h, d, dt), 0)
        self.assertLessEqual(Fraction(dt), _exact_pos(h, d))

    def test_fraction_inputs_centre_weight_nonnegative(self):
        h, d = Fraction(7, 3), Fraction(1, 7)
        dt = engine.hex_positivity_max_dt(h, d)
        self.assertGreaterEqual(self._centre(h, d, dt), 0)
        self.assertLessEqual(Fraction(engine.hex_cfl_max_dt(h, d, 1)), _exact_l2(h, d))

    def test_fraction_scan_never_above_exact_bound_of_given_inputs(self):
        rng = random.Random(20260927)
        for _ in range(3000):
            h = Fraction(rng.randint(1, 10**6), rng.randint(1, 10**6))
            d = Fraction(rng.randint(1, 10**6), rng.randint(1, 10**6))
            self.assertLessEqual(Fraction(engine.hex_positivity_max_dt(h, d)), _exact_pos(h, d))
            self.assertLessEqual(Fraction(engine.hex_cfl_max_dt(h, d, 1)), _exact_l2(h, d))

    def test_safety_just_above_one_rejected(self):
        # rounds to 1.0 in binary64 but is > 1 exactly
        with self.assertRaises(ValueError):
            engine.hex_positivity_max_dt(1.0, 1.0, Fraction(2**60 + 1, 2**60))

    def test_numpy_scalars_are_exact(self):
        h, d = np.float32(0.1), np.int64(3)
        dt = engine.hex_positivity_max_dt(h, d)
        self.assertLessEqual(Fraction(dt), Fraction(*h.as_integer_ratio()) ** 2 / (4 * 3))


class TestExactOperandsBeyondBinary64(unittest.TestCase):
    """Codex #47: operands are validated exactly; only the RESULT must be a
    finite positive binary64 (spec 5.2a)."""

    @staticmethod
    def _call(fn, *args):
        """Result or the raised ValueError, so a rejection fails by assertion."""
        try:
            return fn(*args)
        except ValueError as exc:
            return exc

    def test_huge_operands_that_cancel_give_exact_result(self):
        self.assertEqual(self._call(engine.hex_positivity_max_dt, 10**400, 10**800), 0.25)
        dt = self._call(engine.hex_cfl_max_dt, 10**400, 10**800, 1)
        self.assertIsInstance(dt, float)
        self.assertLessEqual(Fraction(dt), Fraction(1, 3))
        self.assertEqual(dt, engine.hex_cfl_max_dt(1.0, 1.0, 1.0))

    def test_tiny_operands_that_cancel_give_exact_result(self):
        h, d = Fraction(1, 10**400), Fraction(1, 10**800)
        self.assertEqual(self._call(engine.hex_positivity_max_dt, h, d), 0.25)

    def test_unrepresentable_result_still_raises(self):
        with self.assertRaises(ValueError):
            engine.hex_positivity_max_dt(10**400, 1)  # overflow
        with self.assertRaises(ValueError):
            engine.hex_positivity_max_dt(Fraction(1, 10**400), 1)  # underflow to 0
        with self.assertRaises(ValueError):
            engine.hex_laplacian_stencil_weight(Fraction(1, 10**200))  # overflow

    def test_invalid_operands_still_rejected(self):
        for bad in (float("nan"), float("inf"), -1, 0, Fraction(0), True, "1.0",
                    np.float64("nan"), np.float32("inf"), None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                engine.hex_positivity_max_dt(bad, 1.0)
        for bad_safety in (Fraction(10**400 + 1, 10**400), 0, float("nan")):
            with self.assertRaises(ValueError, msg=repr(bad_safety)):
                engine.hex_positivity_max_dt(1.0, 1.0, bad_safety)


class TestDocsContract(unittest.TestCase):
    def test_agents_quoted_hook_anchor_resolves(self):
        agents = (_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        m = re.search(r"Shape-Validierung \(src/hexantk_v3\.py:(\d+)-(\d+)\)", agents)
        self.assertIsNotNone(m)
        lines = _ENGINE.read_text(encoding="utf-8").splitlines()
        a, b = int(m.group(1)), int(m.group(2))
        self.assertIn("This config is a FUTURE-EXTENSION HOOK", lines[a - 1])
        self.assertIn("GAOT-ready.", "\n".join(lines[a - 1:b]))

    def test_readme_toolkit_counts_current(self):
        readme = (_ROOT / "README.md").read_text(encoding="utf-8")
        m = re.search(r"(\d+) Module \+ (\d+) Testmodule", readme)
        self.assertIsNotNone(m)
        toolkit = _ROOT / "src" / "toolkit"
        tests = [p for p in toolkit.glob("test_*.py")]
        modules = [p for p in toolkit.glob("*.py")
                   if not p.name.startswith("test_") and p.name != "__init__.py"]
        self.assertEqual((int(m.group(1)), int(m.group(2))), (len(modules), len(tests)))

    def test_spec_contains_positivity_contract(self):
        spec = (_ROOT / "spec" / "HexaNTK_Review_Gehaertet.md").read_text(encoding="utf-8")
        self.assertIn("h²/(4Dᵥ)", spec)
        self.assertIn("hex_positivity_max_dt", spec)

    def test_readme_quoted_hook_anchor_resolves(self):
        readme = (_ROOT / "README.md").read_text(encoding="utf-8")
        m = re.search(r'GAOT-ready\." — `src/hexantk_v3\.py:(\d+)-(\d+)`', readme)
        self.assertIsNotNone(m)
        lines = _ENGINE.read_text(encoding="utf-8").splitlines()
        a, b = int(m.group(1)), int(m.group(2))
        block = "\n".join(lines[a - 1:b])
        self.assertIn("This config is a FUTURE-EXTENSION HOOK", lines[a - 1])
        self.assertIn("GAOT-ready.", block)

    # Codex review #46 (PRRT_kwDOSwvSC86m2eeU): after merging main the README
    # still said 378 unit tests while 409 ran. Every count the README states
    # about the current tree is pinned to the tree itself; a reworded or
    # removed statement fails here too (each must be found exactly once).

    def _readme_counts(self, pattern: str) -> list:
        readme = (_ROOT / "README.md").read_text(encoding="utf-8")
        found = re.findall(pattern, readme)
        self.assertEqual(len(found), 1, f"README: {pattern!r} found {len(found)}x")
        return [int(x) for x in (found[0] if isinstance(found[0], tuple) else (found[0],))]

    def test_readme_unit_test_count_current(self):
        toolkit = _ROOT / "src" / "toolkit"
        suite = unittest.defaultTestLoader.discover(
            start_dir=str(toolkit), pattern="test_*.py", top_level_dir=str(toolkit))
        n = suite.countTestCases()
        for pattern in (r"\| Toolkit-Tests \|[^\n]*\| (\d+) Tests, ",
                        r"# (\d+) Unit-Tests \(",
                        r"Toolkit-Unit-Tests \((\d+), Stand"):
            with self.subTest(pattern=pattern):
                self.assertEqual(self._readme_counts(pattern), [n])

    def test_readme_provenance_counts_current(self):
        toolkit = [p for p in (_ROOT / "src" / "toolkit").iterdir()
                   if p.is_file() and p.name != "__init__.py"]
        spec = [p for p in (_ROOT / "spec").iterdir() if p.is_file()]
        self.assertEqual(self._readme_counts(r"(\d+) Toolkit- \+ (\d+) Spec-Eintraege"),
                         [len(toolkit), len(spec)])

    def test_readme_doctest_count_current(self):
        n = sum(len(t.examples) for t in doctest.DocTestFinder().find(engine))
        self.assertEqual(self._readme_counts(r"\| (\d+)/\d+ Doctests"), [n])
        self.assertEqual(self._readme_counts(r"# (\d+) Doctests \+"), [n])

    def test_readme_mutant_count_current(self):
        path = _ROOT / "scripts" / "mutation_probe.py"
        spec = importlib.util.spec_from_file_location("mutation_probe_count", path)
        assert spec is not None and spec.loader is not None
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        self.assertEqual(self._readme_counts(r"\| \d+/(\d+) Mutanten getoetet"), [len(probe.MUTANTS)])
        self.assertEqual(self._readme_counts(r"optional: (\d+) Mutanten"), [len(probe.MUTANTS)])

    def test_readme_engine_line_count_current(self):
        readme = (_ROOT / "README.md").read_text(encoding="utf-8")
        m = re.search(r"`src/hexantk_v3\.py` \((\d+) Zeilen;", readme)
        self.assertIsNotNone(m)
        n = len(_ENGINE.read_text(encoding="utf-8").splitlines())
        self.assertEqual(int(m.group(1)), n)


if __name__ == "__main__":
    unittest.main()
