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


class TestDocsContract(unittest.TestCase):
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

    def test_readme_engine_line_count_current(self):
        readme = (_ROOT / "README.md").read_text(encoding="utf-8")
        m = re.search(r"`src/hexantk_v3\.py` \((\d+) Zeilen;", readme)
        self.assertIsNotNone(m)
        n = len(_ENGINE.read_text(encoding="utf-8").splitlines())
        self.assertEqual(int(m.group(1)), n)


if __name__ == "__main__":
    unittest.main()
