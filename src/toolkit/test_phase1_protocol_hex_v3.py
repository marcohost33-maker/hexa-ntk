"""
test_phase1_protocol_hex_v3.py
==============================
Contract tests for the Phase 1 evaluation-protocol instruments
(spec/PHASE1_PROTOCOL_v1.md): closed-form oracles (IQM, log-log
interpolation on a power law, break-even), metamorphic relations (pairing
under scaling, chunk independence, dominated points), the decision truth
table, the run manifest's fail-closed schema, and the cost-accuracy sweep of
the mechanistic baseline against the exact semidiscrete solution (second
order in dt). Synthetic numbers only; no model and no T1 fit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import itertools
import json
import math
import pathlib
import tempfile
import unittest

import numpy as np

import phase1_protocol_hex_v3 as pp
from heat_operator_task_hex_v3 import HeatOperatorTaskConfig, generate_dataset
from hex_fv_solver_hex_v3 import assemble_hex_operator, torus_hex_lattice, triangle_hex_lattice

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _fake_clock(step: float = 0.5):
    """A clock that advances by ``step`` per reading: every timed call lasts exactly ``step``."""
    return itertools.count(0.0, step).__next__


def _semigroup_matrix(lattice, t):
    diag, off = assemble_hex_operator(lattice, 1.0)
    dense = np.diag(np.array(diag, dtype=np.float64))
    for i in range(lattice.num_nodes):
        for s in range(6):
            j = lattice.neighbors[i, s]
            if j >= 0:
                dense[i, j] += off[i, s]
    mu, vec = np.linalg.eigh(dense)
    return (vec * np.exp(t * mu)) @ vec.T


# ------------------------------------------------------------------
# Section 1: timing protocol
# ------------------------------------------------------------------


class TestTimingProtocol(unittest.TestCase):
    def test_defaults_follow_the_preregistration(self):
        p = pp.TimingProtocol()
        self.assertEqual((p.warmup, p.repeats, p.bootstrap_resamples, p.seed, p.confidence),
                         (3, 10, 10_000, 0, 0.95))
        self.assertEqual(p.canonical(), {"warmup": 3, "repeats": 10, "bootstrap_resamples": 10_000,
                                         "seed": 0, "confidence": 0.95})

    def test_validation(self):
        for kwargs, exc in (({"warmup": -1}, ValueError), ({"repeats": 0}, ValueError),
                            ({"bootstrap_resamples": 0}, ValueError), ({"seed": -1}, ValueError),
                            ({"confidence": 1.0}, ValueError), ({"confidence": 0.0}, ValueError),
                            ({"confidence": float("nan")}, ValueError), ({"warmup": True}, TypeError),
                            ({"repeats": 2.0}, TypeError), ({"confidence": "0.95"}, TypeError)):
            with self.subTest(kwargs=kwargs), self.assertRaises(exc):
                pp.TimingProtocol(**kwargs)


class TestMeasureLatency(unittest.TestCase):
    def test_warmup_is_untimed_and_the_median_is_exact(self):
        calls = []
        protocol = pp.TimingProtocol(warmup=2, repeats=5, bootstrap_resamples=100)
        m = pp.measure_latency(lambda: calls.append(1), protocol, clock=_fake_clock(0.5))
        self.assertEqual(len(calls), 7)
        self.assertEqual(m.times_s, (0.5,) * 5)
        self.assertEqual((m.median_s, m.ci_low_s, m.ci_high_s), (0.5, 0.5, 0.5))
        self.assertIs(m.protocol, protocol)
        self.assertEqual(json.loads(json.dumps(m.to_dict()))["median_s"], 0.5)

    def test_variable_durations(self):
        # Durations 1, 3, 2, 10: the median (2.5) differs from the mean (4.0), so a
        # mean-based latency fails here (mutant M09 of the 2026-10-19 probe survived
        # a symmetric 1, 3, 2, 4 design).
        readings = iter([0.0, 1.0, 10.0, 13.0, 20.0, 22.0, 30.0, 40.0])
        protocol = pp.TimingProtocol(warmup=0, repeats=4, bootstrap_resamples=500, seed=3)
        m = pp.measure_latency(lambda: None, protocol, clock=lambda: next(readings))
        self.assertEqual(m.times_s, (1.0, 3.0, 2.0, 10.0))
        self.assertEqual(m.median_s, 2.5)
        self.assertLessEqual(1.0, m.ci_low_s)
        self.assertLessEqual(m.ci_low_s, m.median_s)
        self.assertLessEqual(m.median_s, m.ci_high_s)
        self.assertLessEqual(m.ci_high_s, 10.0)
        again = iter([0.0, 1.0, 10.0, 13.0, 20.0, 22.0, 30.0, 40.0])
        self.assertEqual(pp.measure_latency(lambda: None, protocol, clock=lambda: next(again)), m)

    def test_backwards_clock_and_bad_arguments_fail_closed(self):
        readings = iter([5.0, 4.0])
        with self.assertRaises(ValueError):
            pp.measure_latency(lambda: None, pp.TimingProtocol(warmup=0, repeats=1), clock=lambda: next(readings))
        with self.assertRaises(TypeError):
            pp.measure_latency(42, pp.TimingProtocol())
        with self.assertRaises(TypeError):
            pp.measure_latency(lambda: None, {"warmup": 3})
        with self.assertRaises(TypeError):
            pp.measure_latency(lambda: None, pp.TimingProtocol(), clock="perf_counter")


# ------------------------------------------------------------------
# Section 4: IQM and the paired bootstrap
# ------------------------------------------------------------------


class TestInterquartileMean(unittest.TestCase):
    def test_closed_forms(self):
        self.assertEqual(pp.interquartile_mean([1, 2, 3, 4, 5]), 3.0)
        self.assertEqual(pp.interquartile_mean([1, 2, 3, 4, 100]), 3.0)       # middle three
        self.assertEqual(pp.interquartile_mean([100, 4, 3, 2, 1]), 3.0)       # order invariant
        self.assertEqual(pp.interquartile_mean([1, 2, 3, 4]), 2.5)            # int(0.25 * 4) = 1
        self.assertEqual(pp.interquartile_mean([7.0]), 7.0)
        self.assertEqual(pp.interquartile_mean([1, 2, 3, 4, 100], proportion=0.0), 22.0)
        self.assertEqual(pp.interquartile_mean(np.arange(8.0), proportion=0.3), 3.5)  # drops 2 each end

    def test_validation(self):
        for bad in ([], [[1.0, 2.0]], [1.0, float("nan")], [1.0, float("inf")], [True, False], "123"):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                pp.interquartile_mean(bad)
        for proportion in (0.5, -0.1, float("nan"), "0.25"):
            with self.subTest(proportion=proportion), self.assertRaises((TypeError, ValueError)):
                pp.interquartile_mean([1, 2, 3], proportion=proportion)


class TestPairedBootstrap(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2026)
        self.a = 0.1 + 0.02 * rng.standard_normal((5, 40)) ** 2

    def test_degenerate_design_has_zero_width(self):
        out = pp.paired_bootstrap_iqm(np.full((5, 30), 0.25), n_resamples=300)
        self.assertEqual(set(out), {""})
        self.assertEqual((out[""].point, out[""].low, out[""].high), (0.25, 0.25, 0.25))

    def test_point_estimate_is_the_iqm_of_the_seed_means(self):
        out = pp.paired_bootstrap_iqm({"id": self.a}, n_resamples=200)["id"]
        self.assertEqual(out.point, pp.interquartile_mean(self.a.mean(axis=1)))
        self.assertEqual((out.n_seeds, out.n_samples, out.n_resamples, out.seed), (5, 40, 200, 0))
        self.assertLessEqual(out.low, out.point)
        self.assertLessEqual(out.point, out.high)
        self.assertLess(out.low, out.high)

    def test_pairing_scales_exactly(self):
        # The same seed and sample-id index sets are applied to every matrix: a
        # matrix equal to twice another must yield exactly twice its bounds.
        ids = range(100000, 100040)
        out = pp.paired_bootstrap_iqm({"n16": self.a, "n32": 2.0 * self.a}, sample_ids={"n16": ids, "n32": ids},
                                      n_resamples=400, seed=5)
        self.assertEqual(out["n32"].low, 2.0 * out["n16"].low)
        self.assertEqual(out["n32"].high, 2.0 * out["n16"].high)
        self.assertEqual(out["n32"].point, 2.0 * out["n16"].point)

    def test_pairing_negative_control(self):
        # Without a shared index set the bounds of an unrelated matrix are not a multiple.
        other = 2.0 * self.a[:, ::-1]
        ids = list(range(40))
        out = pp.paired_bootstrap_iqm({"a": self.a, "b": other}, sample_ids={"a": ids, "b": ids},
                                      n_resamples=400, seed=5)
        self.assertNotEqual(out["b"].low, 2.0 * out["a"].low)

    def test_pairing_is_bound_to_equal_sample_ids(self):
        # GPT-6 R2 (2026-10-19): the triangle sets share ids 100000..100499, the torus
        # OOD set uses 200000..200499; equal array shape is no physical pairing.
        triangle, torus = range(100000, 100040), range(200000, 200040)
        pair = {"triangle": self.a, "torus": 2.0 * self.a}
        refusals = [
            {},                                                            # no ids: shape alone
            {"sample_ids": {"triangle": triangle, "torus": torus}},        # disjoint ids
            {"sample_ids": {"triangle": triangle, "torus": list(triangle)[::-1]}},  # same ids, other order
            {"sample_ids": {"triangle": triangle, "torus": [*triangle[:-1], 100040]}},  # one id differs
            {"sample_ids": list(triangle)},                                # one list asserted for both
        ]
        for kwargs in refusals:
            with self.subTest(kwargs=kwargs), self.assertRaises(pp.PairingError):
                pp.paired_bootstrap_iqm(pair, n_resamples=50, **kwargs)
        self.assertTrue(issubclass(pp.PairingError, ValueError))
        # Positive control: the same ids in the same order pair, and the result is the
        # one of the unchecked computation (the check changes no number).
        ok = pp.paired_bootstrap_iqm(pair, sample_ids={"triangle": triangle, "torus": triangle}, n_resamples=50)
        alone = pp.paired_bootstrap_iqm(self.a, sample_ids=triangle, n_resamples=50)[""]
        self.assertEqual(ok["triangle"], alone)
        self.assertEqual(ok["torus"].high, 2.0 * alone.high)
        # Separate bootstraps stay possible for unpaired sets.
        self.assertEqual(pp.paired_bootstrap_iqm({"torus": self.a}, sample_ids={"torus": torus},
                                                 n_resamples=50)["torus"], alone)
        for bad in ({"sample_ids": {"triangle": triangle}},                         # a set without ids
                    {"sample_ids": {"triangle": triangle, "torus": range(200000, 200039)}},  # wrong length
                    {"sample_ids": {"triangle": [0] * 40, "torus": [0] * 40}},      # repeated id
                    {"sample_ids": {"triangle": [-1, *range(39)], "torus": [-1, *range(39)]}},
                    {"sample_ids": {"triangle": "abc", "torus": "abc"}}):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                pp.paired_bootstrap_iqm(pair, n_resamples=50, **bad)

    def test_chunking_does_not_change_the_result_and_seeds_do(self):
        one = pp.paired_bootstrap_iqm(self.a, n_resamples=300, seed=11)[""]
        two = pp.paired_bootstrap_iqm(self.a, n_resamples=300, seed=11, chunk_elements=7)[""]
        self.assertEqual(one, two)
        three = pp.paired_bootstrap_iqm(self.a, n_resamples=300, seed=12)[""]
        self.assertTrue(three.low != one.low or three.high != one.high)

    def test_memory_layout_does_not_change_the_result(self):
        # numpy sums a row pairwise only over contiguous memory: at 3b041db a
        # Fortran-ordered copy of the same values moved the point estimate by an ulp.
        c_order = pp.paired_bootstrap_iqm(_raw_errors(), n_resamples=50)[""]
        for view in (np.asfortranarray(_raw_errors()), _raw_errors()[:, np.arange(500)]):
            with self.subTest(c_contiguous=view.flags["C_CONTIGUOUS"]):
                self.assertFalse(view.flags["C_CONTIGUOUS"])
                self.assertEqual(pp.paired_bootstrap_iqm(view, n_resamples=50)[""], c_order)

    def test_an_outlying_seed_widens_the_interval(self):
        base = pp.paired_bootstrap_iqm(self.a, n_resamples=500, seed=1)[""]
        spoiled = self.a.copy()
        spoiled[0] += 1.0
        wide = pp.paired_bootstrap_iqm(spoiled, n_resamples=500, seed=1)[""]
        self.assertGreater(wide.high - wide.low, base.high - base.low)
        self.assertGreater(wide.high, base.high)

    def test_default_interval_is_the_95_percent_binomial_oracle(self):
        # Closed-form oracle (Equalita mutants D4 alpha = 1 - c and D5 default 0.90
        # survived the module, 2026-10-20). One seed, 16 sample ids, eight zeros and
        # eight ones: a resampled mean is Binomial(16, 1/2) / 16. Its exact 2.5 % and
        # 97.5 % quantiles are 4/16 and 12/16; the 5 % and 95 % quantiles are 5/16
        # and 11/16. The exact CDF is at least 0.0115 away from every one of these
        # levels (about 7 standard errors of a 10 000-resample empirical CDF),
        # so the percentile interval lands on the atoms exactly.
        cdf = np.cumsum([math.comb(16, j) for j in range(17)]) / 2.0 ** 16
        for level, atom in ((0.025, 4), (0.975, 12), (0.05, 5), (0.95, 11)):
            self.assertEqual(int(np.argmax(cdf > level)), atom)
            self.assertGreater(min(cdf[atom] - level, level - cdf[atom - 1]), 0.0115)
        row = np.array([0.0, 1.0] * 8)
        for design in (row[None, :], np.tile(row, (5, 1))):   # 5 identical seeds: seed draws change nothing
            out = pp.paired_bootstrap_iqm(design)[""]          # every argument at its default
            with self.subTest(seeds=design.shape[0]):
                self.assertEqual((out.confidence, out.n_resamples, out.proportion), (0.95, 10_000, 0.25))
                self.assertEqual((out.low, out.point, out.high), (4 / 16, 0.5, 12 / 16))
        ninety = pp.paired_bootstrap_iqm(row[None, :], confidence=0.90)[""]
        self.assertEqual((ninety.low, ninety.high), (5 / 16, 11 / 16))

    def test_the_seed_axis_is_resampled(self):
        # Equalita mutant D10 (seed indices drawn but never applied) survived the module.
        # Every seed row is constant, so the sample-id draws change nothing and the whole
        # interval width comes from the seed draws. Exact law of the IQM of five draws
        # with replacement from {1, 2, 3, 4, 5} (5**5 equally likely tuples, IQM = mean of
        # the middle three): 5 % quantile 5/3, 95 % quantile 13/3, support [1, 5]. The
        # 95 % percentile interval must reach at least as far out as the 90 % quantiles.
        law = sorted(sum(sorted(t)[1:4]) / 3.0 for t in itertools.product(range(1, 6), repeat=5))
        q05, q95 = law[int(0.05 * len(law))], law[int(0.95 * len(law))]
        self.assertEqual((q05, q95), (5 / 3, 13 / 3))
        design = np.repeat(np.arange(1.0, 6.0)[:, None], 16, axis=1)
        out = pp.paired_bootstrap_iqm(design)[""]
        self.assertEqual(out.point, 3.0)
        self.assertGreater(out.high - out.low, 0.0)          # the seed axis contributes variance
        self.assertLessEqual(4 / 3 - 1e-12, out.low)
        self.assertLessEqual(out.low, q05 + 1e-12)
        self.assertLessEqual(q95 - 1e-12, out.high)
        self.assertLessEqual(out.high, 14 / 3 + 1e-12)
        # Negative control: identical seed rows leave only the sample axis.
        flat = pp.paired_bootstrap_iqm(np.full((5, 16), 3.0))[""]
        self.assertEqual(flat.high - flat.low, 0.0)

    def test_the_preregistered_constants_match_the_preregistration(self):
        self.assertEqual((pp.PREREGISTERED_CONFIDENCE, pp.PREREGISTERED_BOOTSTRAP_RESAMPLES,
                          pp.PREREGISTERED_IQM_PROPORTION), (0.95, 10_000, 0.25))
        text = (_ROOT / "spec" / "PREREGISTRATION_PHASE1_v1.md").read_text(encoding="utf-8")
        text = " ".join(text.split())
        self.assertIn("its 95 % interval uses a two-stage paired bootstrap with 10 000 resamples", text)
        self.assertIn("upper 95 % CI bound", text)
        self.assertIn("IQM of the five seed-level mean errors", text)

    def test_validation(self):
        good = np.ones((3, 4))
        for bad in (np.ones(4), np.ones((0, 4)), np.ones((3, 0)), np.full((3, 4), np.nan),
                    {}, {"a": good, "b": np.ones((3, 5))}, {1: good}, np.ones((3, 4), dtype=bool)):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                pp.paired_bootstrap_iqm(bad, n_resamples=10)
        for kwargs in ({"n_resamples": 0}, {"confidence": 1.0}, {"proportion": 0.5},
                       {"seed": -1}, {"chunk_elements": 0}, {"n_resamples": 2.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                pp.paired_bootstrap_iqm(good, **{"n_resamples": 10, **kwargs})


# ------------------------------------------------------------------
# Section 3: frontier and interpolation
# ------------------------------------------------------------------


class TestFrontierAndInterpolation(unittest.TestCase):
    def test_frontier_drops_dominated_points_and_resolves_ties(self):
        np.testing.assert_array_equal(pp.pareto_frontier([1, 2, 3, 4, 5], [4, 3, 3.5, 1, 1.5]), [0, 1, 3])
        np.testing.assert_array_equal(pp.pareto_frontier([2, 1, 1], [1, 3, 2]), [2, 0])   # equal time: lower error
        np.testing.assert_array_equal(pp.pareto_frontier([3, 1, 2], [1, 2, 1]), [1, 2])   # equal error: faster
        np.testing.assert_array_equal(pp.pareto_frontier([1.0], [1.0]), [0])

    def test_frontier_validation(self):
        for times, errors in (([1, 2], [1]), ([0, 1], [1, 2]), ([1, 2], [-1, 1]), ([], []),
                              ([1, float("nan")], [1, 1]), ([1, 2], [float("inf"), 1])):
            with self.subTest(times=times, errors=errors), self.assertRaises((TypeError, ValueError)):
                pp.pareto_frontier(times, errors)

    def test_power_law_is_reproduced_exactly_inside_the_range(self):
        c, p = 3.0, 1.7
        times = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        errors = c * times ** (-p)
        for t in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 11.0, 16.0):
            self.assertAlmostEqual(pp.error_at_time(times, errors, t), c * t ** (-p), delta=1e-12)
        for e in (errors[0], errors[-1], 0.2, 1.0, 2.5):
            t = pp.time_at_error(times, errors, e)
            self.assertAlmostEqual(t, (c / e) ** (1.0 / p), delta=1e-12 * (c / e) ** (1.0 / p))
            self.assertAlmostEqual(pp.error_at_time(times, errors, t), e, delta=1e-12)

    def test_no_extrapolation(self):
        times, errors = [1, 2, 4], [4, 2, 1]
        self.assertIsNone(pp.error_at_time(times, errors, 0.5))
        self.assertIsNone(pp.error_at_time(times, errors, 4.0000001))
        self.assertIsNone(pp.time_at_error(times, errors, 0.5))        # below the floor
        self.assertIsNone(pp.time_at_error(times, errors, 4.5))        # above the coarsest point
        self.assertEqual(pp.error_at_time([2.0], [3.0], 2.0), 3.0)     # one point answers itself only
        self.assertIsNone(pp.error_at_time([2.0], [3.0], 2.5))
        self.assertEqual(pp.time_at_error([2.0], [3.0], 3.0), 2.0)
        self.assertIsNone(pp.time_at_error([2.0], [3.0], 2.0))

    def test_dominated_points_do_not_influence_the_interpolation(self):
        times, errors = [1.0, 2.0, 4.0], [4.0, 2.0, 1.0]
        more_t, more_e = times + [3.0, 2.0], errors + [3.0, 2.5]     # both dominated
        for t in (1.5, 3.0, 3.9):
            self.assertEqual(pp.error_at_time(more_t, more_e, t), pp.error_at_time(times, errors, t))
        for e in (1.2, 2.0, 3.3):
            self.assertEqual(pp.time_at_error(more_t, more_e, e), pp.time_at_error(times, errors, e))
        # The interpolation is log-log linear, not linear: midpoint of (2, 2) and (4, 1).
        self.assertAlmostEqual(pp.error_at_time(times, errors, math.sqrt(8.0)), math.sqrt(2.0), delta=1e-12)
        self.assertNotAlmostEqual(pp.error_at_time(times, errors, 3.0), 1.5, delta=1e-3)
        with self.assertRaises(ValueError):
            pp.error_at_time(times, errors, 0.0)
        with self.assertRaises(TypeError):
            pp.time_at_error(times, errors, "1")


# ------------------------------------------------------------------
# Section 5: decision rules
# ------------------------------------------------------------------


def _summary(high: float) -> pp.BootstrapSummary:
    return pp.BootstrapSummary(point=0.9 * high, low=0.8 * high, high=high, n_seeds=5, n_samples=500,
                               n_resamples=10_000, seed=0, confidence=0.95, proportion=0.25)


class TestDecisionRules(unittest.TestCase):
    TIMES = [1.0, 2.0, 4.0, 8.0]
    ERRORS = [8.0, 4.0, 2.0, 1.0]          # e = 8 / t on the frontier

    def _decide(self, high, latency):
        return pp.decide_test_set(test_set="id", n=32, gino=_summary(high), gino_latency_s=latency,
                                  baseline_times_s=self.TIMES, baseline_errors=self.ERRORS)

    def test_advantage_under_both_rules(self):
        d = self._decide(high=1.5, latency=2.0)
        runtime, accuracy = d.rules
        self.assertEqual(d.verdict, "advantage")
        self.assertTrue(runtime.evaluable and runtime.claim)
        self.assertEqual(runtime.baseline_value, 4.0)
        self.assertTrue(accuracy.evaluable and accuracy.claim)
        self.assertAlmostEqual(accuracy.baseline_value, 8.0 / 1.5, delta=1e-12)
        self.assertEqual((d.test_set, d.n), ("id", 32))
        self.assertEqual(json.loads(json.dumps(d.to_dict()))["verdict"], "advantage")

    def test_h0_when_both_rules_are_evaluable_and_fail(self):
        d = self._decide(high=5.0, latency=2.0)
        runtime, accuracy = d.rules
        self.assertEqual(d.verdict, "H0")
        self.assertTrue(runtime.evaluable)
        self.assertFalse(runtime.claim)
        self.assertTrue(accuracy.evaluable)
        self.assertFalse(accuracy.claim)
        self.assertAlmostEqual(accuracy.baseline_value, 1.6, delta=1e-12)

    def test_no_claim_outside_the_measured_range(self):
        d = self._decide(high=0.5, latency=0.5)
        runtime, accuracy = d.rules
        self.assertEqual(d.verdict, "H0")
        self.assertFalse(runtime.evaluable)
        self.assertIn("outside the time range of the baseline's Pareto frontier", runtime.reason)
        self.assertFalse(accuracy.evaluable)
        self.assertIn("baseline floor", accuracy.reason)
        self.assertIsNone(runtime.baseline_value)
        self.assertIsNone(accuracy.baseline_value)

    def test_above_the_coarsest_error_is_no_claim(self):
        d = self._decide(high=9.0, latency=16.0)
        self.assertEqual(d.verdict, "H0")
        self.assertIn("above the coarsest", d.rules[1].reason)
        self.assertIn("no extrapolation", d.rules[0].reason)

    def test_reasons_name_the_frontier_range_not_the_measured_range(self):
        # E3 (Equalita 2026-10-20): with a dominated slow point the measured times span
        # [1, 4] s but the frontier only [1, 2] s. The rule uses the frontier (no claim at
        # 3 s, conservative); the reason must say so instead of calling [1, 2] "measured".
        d = pp.decide_test_set(test_set="id", n=32, gino=_summary(5.0), gino_latency_s=3.0,
                               baseline_times_s=[1.0, 2.0, 4.0], baseline_errors=[4.0, 2.0, 3.0])
        runtime, accuracy = d.rules
        self.assertFalse(runtime.evaluable)
        self.assertIn("Pareto frontier [1, 2] s (all measured points: [1, 4] s)", runtime.reason)
        self.assertNotIn("measured baseline range", runtime.reason)
        # Errors: frontier [2, 4], measured up to 5 through a dominated point.
        d = pp.decide_test_set(test_set="id", n=32, gino=_summary(4.5), gino_latency_s=1.5,
                               baseline_times_s=[1.0, 2.0, 4.0], baseline_errors=[4.0, 2.0, 5.0])
        self.assertFalse(d.rules[1].evaluable)
        self.assertIn("Pareto frontier 4 (all measured points up to 5)", d.rules[1].reason)
        self.assertEqual(d.verdict, "H0")

    def test_one_rule_evaluable_decides_alone(self):
        d = self._decide(high=3.0, latency=16.0)
        runtime, accuracy = d.rules
        self.assertFalse(runtime.evaluable)
        self.assertTrue(accuracy.evaluable)
        self.assertFalse(accuracy.claim)                   # 16 s is slower than 8/3 s
        self.assertEqual(d.verdict, "H0")
        faster = self._decide(high=3.0, latency=0.5)
        self.assertFalse(faster.rules[0].evaluable)
        self.assertTrue(faster.rules[1].claim)
        self.assertEqual(faster.verdict, "advantage")

    def test_strictness_at_the_boundary(self):
        # Equal is not "below": upper bound 4.0 at the point where the baseline has 4.0.
        self.assertFalse(self._decide(high=4.0, latency=2.0).rules[0].claim)
        # Equal is not "faster": latency 2.0 where the baseline needs 2.0 s for error 4.0.
        self.assertFalse(self._decide(high=4.0, latency=2.0).rules[1].claim)

    def test_the_decision_is_bound_to_the_preregistered_statistic(self):
        # F-E1 (Equalita 2026-10-20): the same data at confidence 0.5 has a lower upper
        # bound, so an unbound rule turned H0 into "advantage". The frontier is placed
        # between the two upper bounds: e_base(4 s) = e lies above the 50 % bound and
        # below the 95 % bound.
        rng = np.random.default_rng(7)
        errors = 0.12 + 0.03 * rng.standard_normal((pp.PREREGISTERED_SEEDS, pp.PREREGISTERED_TEST_SAMPLES)) ** 2
        prereg = pp.paired_bootstrap_iqm(errors)[""]
        loose = pp.paired_bootstrap_iqm(errors, confidence=0.5)[""]
        self.assertLess(loose.high, prereg.high)
        e = 0.5 * (loose.high + prereg.high)
        times, frontier = [1.0, 2.0, 4.0, 8.0], [4 * e, 2 * e, e, e / 2]

        def decide(summary):
            return pp.decide_test_set(test_set="id", n=32, gino=summary, gino_latency_s=4.0,
                                      baseline_times_s=times, baseline_errors=frontier)

        d = decide(prereg)
        self.assertEqual(d.verdict, "H0")
        self.assertTrue(d.rules[0].evaluable and d.rules[1].evaluable)
        self.assertLess(loose.high, d.rules[0].baseline_value)     # unbound, 0.5 would claim
        for summary in (loose, pp.paired_bootstrap_iqm(errors, n_resamples=1)[""],
                        pp.paired_bootstrap_iqm(errors, proportion=0.0, confidence=0.8)[""],
                        pp.paired_bootstrap_iqm(errors, n_resamples=9_999)[""]):
            with self.subTest(confidence=summary.confidence, n_resamples=summary.n_resamples,
                              proportion=summary.proportion), self.assertRaises(pp.DecisionInputError):
                decide(summary)
        # The verdict carries the statistic it rests on.
        as_dict = json.loads(json.dumps(d.to_dict()))
        self.assertIn("gino", as_dict)
        record = as_dict["gino"]
        self.assertEqual((record["confidence"], record["n_resamples"], record["proportion"]),
                         (0.95, 10_000, 0.25))
        self.assertEqual(record["high"], prereg.high)
        self.assertIs(d.gino, prereg)

    def test_malformed_summaries_fail_closed(self):
        good = _summary(1.5)
        cases = [
            {"low": 0.9, "point": 0.5, "high": 0.1},                 # inverted (Equalita probe)
            {"low": 1.4, "point": 1.6, "high": 1.5},                 # point above high
            {"low": 1.5, "point": 1.2, "high": 1.5},                 # point below low
            {"low": float("nan")}, {"point": float("inf")}, {"low": "1"},
            {"confidence": 0.9}, {"confidence": float("nan")}, {"confidence": True},
            {"n_resamples": 10_000.0}, {"n_resamples": True}, {"proportion": 0.2},
            {"n_seeds": 0}, {"n_samples": 0}, {"n_seeds": 5.0}, {"n_seeds": True},
            # Equalita R2-P2-1: subsets with honest labels are another statistic.
            {"n_seeds": 3}, {"n_seeds": 6}, {"n_samples": 20}, {"n_samples": 499},
            # Equalita R2 P3: NEP 50 makes np.float32(0.95) == 0.95 although it is 0.949999988.
            {"confidence": np.float32(0.95)}, {"confidence": np.float16(0.9502)},
            {"confidence": 0.95 + 1e-12}, {"n_resamples": 10_001},
            {"low": -0.1},                                           # rel_l2 is non-negative
        ]
        for change in cases:
            fields = {**good.to_dict(), **change}
            with self.subTest(change=change), self.assertRaises(pp.DecisionInputError):
                pp.decide_test_set(test_set="id", n=32, gino=pp.BootstrapSummary(**fields), gino_latency_s=2.0,
                                   baseline_times_s=self.TIMES, baseline_errors=self.ERRORS)
        self.assertTrue(issubclass(pp.DecisionInputError, ValueError))
        # numpy scalars of the preregistered values are the same statistic.
        numpy_typed = pp.BootstrapSummary(**{**good.to_dict(), "confidence": np.float64(0.95),
                                             "n_resamples": np.int64(10_000), "proportion": np.float64(0.25)})
        self.assertEqual(pp.decide_test_set(test_set="id", n=32, gino=numpy_typed, gino_latency_s=2.0,
                                            baseline_times_s=self.TIMES, baseline_errors=self.ERRORS).verdict,
                         "advantage")

    def test_the_rule_reads_the_validated_bound_once(self):
        # Equalita R2 P3 (TOCTOU): a summary whose ``high`` changes between reads must
        # not pass validation with one value and drive the rule with another.
        class Shifty(pp.BootstrapSummary):
            reads = 0

            def __getattribute__(self, name):
                if name == "high":
                    type(self).reads += 1
                    return 1.5 if type(self).reads == 1 else 0.01
                return super().__getattribute__(name)

        shifty = Shifty(**_summary(1.5).to_dict())
        d = pp.decide_test_set(test_set="id", n=32, gino=shifty, gino_latency_s=2.0,
                               baseline_times_s=self.TIMES, baseline_errors=self.ERRORS)
        self.assertEqual([r.gino_upper_rel_l2 for r in d.rules], [1.5, 1.5])
        honest = pp.decide_test_set(test_set="id", n=32, gino=_summary(1.5), gino_latency_s=2.0,
                                    baseline_times_s=self.TIMES, baseline_errors=self.ERRORS)
        self.assertEqual(d.verdict, honest.verdict)

    def test_validation(self):
        for kwargs in ({"test_set": ""}, {"n": 0}, {"gino": {"high": 1.0}}, {"gino_latency_s": 0.0},
                       {"gino_latency_s": float("nan")}, {"baseline_errors": [8, 4, 2]}):
            full = {"test_set": "id", "n": 32, "gino": _summary(1.5), "gino_latency_s": 2.0,
                    "baseline_times_s": self.TIMES, "baseline_errors": self.ERRORS, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                pp.decide_test_set(**full)


# ------------------------------------------------------------------
# Section 5a: the decision-grade entry from raw errors (HEX-P1-01)
# ------------------------------------------------------------------


_IDS = list(range(100000, 100500))
# spec/PREREGISTRATION_PHASE1_v1.md section 4: the baseline runs with these steps.
_REGISTERED_STEPS = (1, 2, 4, 8, 16, 32, 64)


def _curve_points(times, errors, *, steps=_REGISTERED_STEPS, n=32, domain="triangle_dirichlet", sample_count=500,
                  timed_samples=5, config=None, protocol=None, rtol=1e-12, clock="time.perf_counter"):
    """Baseline points with explicit measurement coordinates, as ``baseline_cost_accuracy_curve`` writes them.

    The coordinates are the ones under test (Equalita R4-F1); the values are given. Hand-built
    points stand for a measured curve here; that a value belongs to its coordinates is not
    checkable from the point (claim boundary, see the test of the relabelled curve). The settings
    default to the preregistered ones and are written as the producer writes them (Equalita
    R5-F2: the former fixture had dt = 1/steps, i.e. T = 1, against the registered T = 0.01).
    """
    config = HeatOperatorTaskConfig(domain=domain if domain in ("triangle_dirichlet", "torus_periodic")
                                    else "triangle_dirichlet") if config is None else config
    protocol = pp.TimingProtocol() if protocol is None else protocol
    return tuple(pp.CostAccuracyPoint(
        n=n, steps=s, dt=float(config.t_final) / s, sample_count=sample_count, timed_samples=timed_samples,
        mean_rel_l2=float(e), median_rel_l2=float(e), mean_rel_h1=float(e), mean_rel_max=float(e),
        median_wall_time_s=float(t), wall_time_ci_s=(float(t), float(t)), cg_iterations_mean=1.0, domain=domain,
        kappa=float(config.kappa), t_final=float(config.t_final), task_config_sha256=config.digest(), rtol=rtol,
        warmup=protocol.warmup, repeats=protocol.repeats, timing_protocol_sha256=protocol.digest(), clock=clock)
        for s, t, e in zip(steps, times, errors))


# Equalita R4 (probes/f1_real_grid.json, real_curve_n64_S50.json, 2026-10-21, torch 2.6.0+cpu,
# numpy 2.5.3): genuine baseline_cost_accuracy_curve output on the same 50 ID inputs (ids
# 100000..100049, dataset_seed 1, timed_samples 5). Errors are deterministic; times are one laptop
# run. Kept as committed values so the test needs no solver run; the coordinates are the measured ones.
_R4_N32_REGISTERED = (  # steps 1, 2, 4, 8, 16, 32, 64 on n = 32
    [0.0032430500141344965, 0.0062607500003650784, 0.009617199946660548, 0.008816150017082691,
     0.010499300085939467, 0.028430950012989342, 0.12459750002017245],
    [0.3596096322403904, 0.030100273908731393, 0.0046839673106014446, 0.0034570043610646513,
     0.004812187650726048, 0.005176910099385276, 0.005269112770350161])
_R4_N32_FOREIGN_STEPS = (1, 8, 16, 24, 32, 48, 64)
_R4_N32_FOREIGN = (  # steps 1, 8, 16, 24, 32, 48, 64 on n = 32
    [0.00421584997093305, 0.022238750010728836, 0.027673550066538155, 0.02444500004639849,
     0.023852800019085407, 0.05212105001555756, 0.11880609998479486],
    [0.3596096322403904, 0.0034570043610646513, 0.004812187650726048, 0.005081700399329767,
     0.005176910099385276, 0.005245172492616448, 0.005269112770350161])
_R4_N64_REGISTERED = (  # steps 1, 2, 4, 8, 16, 32, 64 on n = 64
    [0.00466270005563274, 0.008051299955695868, 0.01315584999974817, 0.07461384998168796,
     0.1163086500018835, 0.05254090001108125, 0.12720480002462864],
    [0.36909290998520305, 0.0338011868274127, 0.0076028534651852685, 0.00117874875644159,
     0.0008577921540806324, 0.001195746789200225, 0.0012868805465495873])


def _raw_errors(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 0.12 + 0.03 * rng.standard_normal((pp.PREREGISTERED_SEEDS, pp.PREREGISTERED_TEST_SAMPLES)) ** 2


class TestPreregisteredDecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.errors = _raw_errors()
        honest = pp.paired_bootstrap_iqm(cls.errors, sample_ids=_IDS)[""]
        # Frontier e(t) = 4 e0 / t with e0 between the forged bound 0.001 and the
        # honest bound: at 4 s the baseline error e0 lies below the honest upper
        # bound (no equal-runtime claim) and the honest bound needs about 2 s (no
        # equal-accuracy claim at 4 s), so the honest verdict is H0, while a forged
        # bound of 0.001 claims at equal runtime.
        cls.e0 = 0.5 * (honest.high + 0.001)
        # One point per registered steps value 1, 2, 4, ..., 64 (preregistration section 4).
        cls.times = [float(s) for s in (1, 2, 4, 8, 16, 32, 64)]
        cls.curve = [4 * cls.e0 / t for t in cls.times]

    def kwargs(self, times=None, curve=None, **change):
        """Decision arguments; the baseline curve gets the decided ``n`` and domain unless given."""
        full = {"test_set": "id", "n": 32, "errors": self.errors, "sample_ids": _IDS,
                "seeds": list(pp.PREREGISTERED_TRAINING_SEEDS),
                "run_manifest": pp.build_run_manifest(_manifest_fields()),
                "gino_latency_times_s": [4.0] * 10}
        full.update(change)
        if "baseline_curve" not in change:
            domain = pp._LATTICE_DOMAIN[pp.PREREGISTERED_TEST_SETS.get(full["test_set"], ("triangle",))[0]]
            full["baseline_curve"] = _curve_points(self.times if times is None else times,
                                                   self.curve if curve is None else curve,
                                                   n=full["n"], domain=domain)
        return full

    def test_a_relabelled_summary_cannot_reach_a_preregistered_decision(self):
        # GPT-6 HEX-P1-01 (2026-10-21): dataclasses.replace(summary, low=0.001, point=0.001,
        # high=0.001) keeps every preregistered label and switches H0 to advantage.
        honest = pp.decide_preregistered(**self.kwargs())
        self.assertEqual(honest.verdict, "H0")
        self.assertTrue(honest.rules[0].evaluable and honest.rules[1].evaluable)
        forged = dataclasses.replace(honest.gino, low=0.001, point=0.001, high=0.001)
        # Positive control of the forger: on the diagnostic helper it does flip the verdict.
        diagnostic = pp.decide_test_set(test_set="id", n=32, gino=forged, gino_latency_s=4.0,
                                        baseline_times_s=self.times, baseline_errors=self.curve)
        self.assertEqual(diagnostic.verdict, "advantage")
        self.assertEqual(diagnostic.to_dict().get("grade"), "diagnostic")
        # (1) The decision-grade entry has no slot for a summary.
        self.assertNotIn("gino", inspect.signature(pp.decide_preregistered).parameters)
        with self.assertRaises(TypeError):
            pp.decide_preregistered(**self.kwargs(), gino=forged)
        # (2) Replacing the summary in the decision object is refused on serialisation.
        tampered = dataclasses.replace(honest, gino=forged, verdict="advantage", rules=diagnostic.rules)
        with self.assertRaises(pp.DecisionProvenanceError):
            tampered.to_dict()
        # (2') The self-digest is drift protection only: a forger who recomputes it gets a
        # serialisable object, which the recomputation from the raw data then refuses.
        resealed = dataclasses.replace(tampered, record_sha256=hashlib.sha256(
            json.dumps(tampered._body(), allow_nan=False, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")).hexdigest())
        self.assertEqual(resealed.to_dict()["verdict"], "advantage")
        with self.assertRaisesRegex(pp.DecisionProvenanceError, "does not follow from the raw data"):
            pp.verify_preregistered_decision(resealed.to_dict(), **self.kwargs())
        # (3) Strongest forger: edits the serialised record and recomputes its self-digest.
        record = json.loads(json.dumps(honest.to_dict()))
        record["gino"].update(low=0.001, point=0.001, high=0.001)
        record["verdict"] = "advantage"
        record["rules"] = [r.to_dict() for r in diagnostic.rules]
        body = {k: v for k, v in record.items() if k != "record_sha256"}
        record["record_sha256"] = hashlib.sha256(
            json.dumps(body, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(pp.DecisionProvenanceError, "does not follow from the raw data"):
            pp.verify_preregistered_decision(record, **self.kwargs())
        # (3') Without recomputing the digest the edit is drift.
        drifted = json.loads(json.dumps(honest.to_dict()))
        drifted["gino"]["high"] = 0.001
        with self.assertRaisesRegex(pp.DecisionProvenanceError, "record_sha256 does not match"):
            pp.verify_preregistered_decision(drifted, **self.kwargs())
        # Positive control: the honest record verifies after a JSON round trip.
        again = pp.verify_preregistered_decision(json.loads(json.dumps(honest.to_dict())), **self.kwargs())
        self.assertEqual(again.to_dict(), honest.to_dict())

    def test_the_summary_is_computed_from_the_raw_table(self):
        d = pp.decide_preregistered(**self.kwargs())
        self.assertEqual(d.gino, pp.paired_bootstrap_iqm(self.errors, sample_ids=_IDS)[""])
        self.assertEqual((d.gino.confidence, d.gino.n_resamples, d.gino.proportion, d.gino.n_seeds,
                          d.gino.n_samples), (0.95, 10_000, 0.25, 5, 500))
        self.assertEqual(d.gino_latency_s, 4.0)
        record = d.to_dict()
        self.assertEqual((record["grade"], record["domain"], record["schema"]),
                         ("preregistered", "triangle", "hexantk_phase1_decision_record_v1"))
        prov = record["provenance"]
        self.assertEqual(prov["run_manifest_sha256"], self.kwargs()["run_manifest"]["manifest_sha256"])
        self.assertEqual(prov["git_sha"], "a" * 40)
        self.assertEqual(prov["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(prov["sample_id_range"], [100000, 100499])
        self.assertTrue(prov["authenticity"].startswith("unattested"))
        canon = np.ascontiguousarray(self.errors.astype("<f8"))
        header = json.dumps({"dtype": "<f8", "order": "C", "shape": [5, 500]}, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
        self.assertEqual(prov["raw_errors_sha256"], hashlib.sha256(header + b"\n" + canon.tobytes()).hexdigest())
        json.dumps(record, allow_nan=False)

    def test_row_and_column_order_is_canonical_and_bound(self):
        base = pp.decide_preregistered(**self.kwargs()).to_dict()
        rng = np.random.default_rng(3)
        rows, cols = rng.permutation(5), rng.permutation(500)
        shuffled = self.kwargs(errors=self.errors[rows][:, cols], seeds=[int(s) for s in rows],
                               sample_ids=[_IDS[c] for c in cols])
        self.assertEqual(pp.decide_preregistered(**shuffled).to_dict(), base)
        # -0.0 and 0.0 are the same measurement and the same bytes.
        zero = self.errors.copy()
        zero[0, 0] = 0.0
        neg = zero.copy()
        neg[0, 0] = -0.0
        self.assertEqual(pp.decide_preregistered(**self.kwargs(errors=zero)).to_dict(),
                         pp.decide_preregistered(**self.kwargs(errors=neg)).to_dict())
        # Moving the columns without their ids is another table: the record no longer verifies.
        moved = self.kwargs(errors=self.errors[:, cols])
        with self.assertRaises(pp.DecisionProvenanceError):
            pp.verify_preregistered_decision(base, **moved)
        # One ulp in one cell, another latency call, another manifest: each breaks verification.
        bumped = self.errors.copy()
        bumped[2, 17] = np.nextafter(bumped[2, 17], 1.0)
        other_manifest = pp.build_run_manifest({**_manifest_fields(), "git_sha": "c" * 40})
        for change in ({"errors": bumped}, {"gino_latency_times_s": [4.0] * 9 + [4.5]},
                       {"run_manifest": other_manifest}, {"curve": [*self.curve[:6], self.curve[6] * 0.99]}):
            with self.subTest(change=sorted(change)), self.assertRaises(pp.DecisionProvenanceError):
                pp.verify_preregistered_decision(base, **self.kwargs(**change))

    def test_the_resolution_axis_stays_paired(self):
        # The per-lattice summary equals the joint, id-paired bootstrap of the N axis.
        coarse = 1.5 * _raw_errors(8)
        joint = pp.paired_bootstrap_iqm({"n16": coarse, "n32": self.errors}, sample_ids={"n16": _IDS, "n32": _IDS})
        d16 = pp.decide_preregistered(**self.kwargs(test_set="resolution_ood", n=16, errors=coarse))
        self.assertEqual(d16.gino, joint["n16"])
        self.assertEqual(pp.decide_preregistered(**self.kwargs()).gino, joint["n32"])

    def test_registered_sets_seeds_ids_and_lattices_only(self):
        torus_ids = list(range(200000, 200500))
        self.assertEqual(pp.decide_preregistered(**self.kwargs(test_set="composite_ood", sample_ids=torus_ids))
                         .to_dict()["domain"], "torus")
        refused = [
            {"test_set": "torus"}, {"test_set": ""}, {"n": 64}, {"test_set": "resolution_ood", "n": 32},
            {"test_set": "composite_ood"},                                     # triangle ids on the torus set
            {"sample_ids": torus_ids},                                         # torus ids on the ID set
            {"sample_ids": [*_IDS[:-1], 100500]}, {"seeds": [0, 1, 2, 3, 5]},
            {"errors": self.errors[:, :499], "sample_ids": _IDS[:499]}, {"errors": self.errors[:3]},
            {"errors": np.vstack([self.errors, self.errors[:1]])},
            {"errors": np.where(np.arange(500) == 3, -1e-3, self.errors)},
        ]
        for change in refused:
            with self.subTest(change=sorted(change)), self.assertRaises(pp.DecisionInputError):
                pp.decide_preregistered(**self.kwargs(**change))
        manifest = pp.build_run_manifest(_manifest_fields())
        edited = {**manifest, "git_sha": "c" * 40}                             # edited after digesting
        for change in ({"run_manifest": edited}, {"run_manifest": {**manifest, "schema": "x"}},
                       {"run_manifest": pp.build_run_manifest({**_manifest_fields(), "seeds": [0, 1, 2, 3]})},
                       {"run_manifest": _manifest_fields()}, {"run_manifest": [1]}):
            with self.subTest(manifest=change), self.assertRaises(pp.DecisionProvenanceError):
                pp.decide_preregistered(**self.kwargs(**change))
        for change in ({"errors": {"id": self.errors}}, {"errors": np.full((5, 500), np.nan)},
                       {"sample_ids": [100000] * 500}, {"sample_ids": "x"}, {"seeds": [0, 0, 1, 2, 3]},
                       # 10 calls each, so the repeats rule (HEXA-TIM-01) cannot mask the check under test
                       {"gino_latency_times_s": []}, {"gino_latency_times_s": [4.0] * 9 + [0.0]},
                       {"gino_latency_times_s": [4.0] * 9 + [float("nan")]}, {"gino_latency_times_s": [[[1.0]]]},
                       {"gino_latency_times_s": [True] * 10}, {"curve": [8, 4, 2]}, {"n": 0}):
            with self.subTest(bad=sorted(change)), self.assertRaises((TypeError, ValueError)):
                pp.decide_preregistered(**self.kwargs(**change))
        for bad in (None, {}, {"record_sha256": 1}, {"record_sha256": "x", "v": float("nan")}):
            with self.subTest(record=bad), self.assertRaises(pp.DecisionProvenanceError):
                pp.verify_preregistered_decision(bad, **self.kwargs())

    def test_latency_without_the_registered_repeats_is_refused(self):
        # GPT-6 HEXA-TIM-01 (2026-10-21): gino_latency_times_s=[0.001], a single timed call,
        # reached a decision-grade verdict although the preregistration measures 10 repeats
        # (TimingProtocol.repeats). 1-D is one sample of `repeats` calls, 2-D is samples x repeats;
        # a flat 1-D of 2 x 10 calls is refused too (its per-sample repeat count is not visible).
        repeats = pp.TimingProtocol().repeats
        self.assertEqual(repeats, 10)
        honest = pp.decide_preregistered(**self.kwargs()).to_dict()
        for bad in ([0.001], [4.0] * (repeats - 1), [4.0] * (repeats + 1), [4.0] * (2 * repeats),
                    np.full((5, repeats - 1), 4.0), np.full((5, repeats + 1), 4.0),
                    np.full((repeats, 1), 4.0)):                               # transposed: 10 samples x 1 call
            with self.subTest(shape=np.shape(bad)):
                with self.assertRaises(pp.DecisionInputError) as caught:
                    pp.decide_preregistered(**self.kwargs(gino_latency_times_s=bad))
                self.assertIs(type(caught.exception), pp.DecisionInputError)
                message = str(caught.exception)
                self.assertIn(f"({repeats},)", message)                       # expected form
                self.assertIn(f"got shape {np.shape(bad)}", message)            # received form
                with self.assertRaises(pp.DecisionInputError):
                    pp.verify_preregistered_decision(honest, **self.kwargs(gino_latency_times_s=bad))

    def test_the_registered_repeats_stay_valid(self):
        # Positive control for HEXA-TIM-01: exactly `repeats` calls, 1-D or samples x repeats,
        # give the same verdict, rules, summary and latency as before the repeats rule.
        repeats = pp.TimingProtocol().repeats
        self.assertEqual(repeats, 10)
        # The class default is the registered 1-D form; the 2-D forms must reproduce its record.
        self.assertEqual(self.kwargs()["gino_latency_times_s"], [4.0] * repeats)

        def decide(latency):
            try:
                return pp.decide_preregistered(**self.kwargs(gino_latency_times_s=latency)).to_dict()
            except pp.DecisionInputError as exc:                              # refusal = test verdict
                self.fail(f"registered latency shape {np.shape(latency)} refused: {exc}")

        honest = decide([4.0] * repeats)
        self.assertEqual(honest["verdict"], "H0")
        for good in (np.full((1, repeats), 4.0), np.full((5, repeats), 4.0), np.full((repeats, repeats), 4.0)):
            with self.subTest(shape=np.shape(good)):
                d = decide(good)
                for key in ("verdict", "rules", "gino", "gino_latency_s"):
                    self.assertEqual(d[key], honest[key], key)
                self.assertEqual(d["provenance"]["gino_latency_times_shape"], list(np.shape(good)))

    def test_a_self_consistent_fabrication_is_out_of_reach(self):
        # Claim boundary: a forger who fabricates the raw table gets a record that is
        # consistent with the fabricated bytes. Only the raw digest differs from the honest
        # one; authenticity needs a signature or a pipeline attestation (not implemented).
        fabricated = 0.001 * (1.0 + 0.01 * np.random.default_rng(9).random((5, 500)))
        fake = pp.decide_preregistered(**self.kwargs(errors=fabricated))
        self.assertEqual(fake.verdict, "advantage")
        self.assertEqual(pp.verify_preregistered_decision(fake.to_dict(), **self.kwargs(errors=fabricated)).verdict,
                         "advantage")
        honest = pp.decide_preregistered(**self.kwargs())
        self.assertNotEqual(fake.provenance["raw_errors_sha256"], honest.provenance["raw_errors_sha256"])
        self.assertTrue(fake.provenance["authenticity"].startswith("unattested"))
        with self.assertRaises(pp.DecisionProvenanceError):
            pp.verify_preregistered_decision(fake.to_dict(), **self.kwargs())

    def _convex_case(self):
        """Equalita R3 F1 setting: a convex 7-point curve and GINO errors whose honest upper bound is 0.0475."""
        steps = np.array(_REGISTERED_STEPS, dtype=np.float64)
        times, curve = list(0.01 * steps), list(0.2 / steps + 0.003)
        raw = 1.0 + 0.3 * np.random.default_rng(7).random((5, 500))
        errors = 0.0475 * raw / pp.paired_bootstrap_iqm(raw, sample_ids=_IDS)[""].high
        latency = [0.05] * 10
        return times, curve, errors, latency

    def test_a_baseline_curve_off_the_registered_steps_grid_is_refused(self):
        # Equalita R3 F1 (2026-10-21): the GENUINE baseline points for steps {1, 64}, {1, 32} or
        # {2, 64} alone reached grade "preregistered" and verdict "advantage": the chord between
        # two points of a convex curve lies above it. Preregistration section 4 runs the baseline
        # with steps 1, 2, 4, 8, 16, 32, 64; any other number of points is another curve.
        times, curve, errors, latency = self._convex_case()
        summary = pp.paired_bootstrap_iqm(errors, sample_ids=_IDS)[""]
        # Positive control of the forger (diagnostic helper, no grid binding): the subset flips H0.
        self.assertEqual(pp.decide_test_set(test_set="id", n=32, gino=summary, gino_latency_s=0.05,
                                            baseline_times_s=times, baseline_errors=curve).verdict, "H0")
        self.assertEqual(pp.decide_test_set(test_set="id", n=32, gino=summary, gino_latency_s=0.05,
                                            baseline_times_s=[times[0], times[6]],
                                            baseline_errors=[curve[0], curve[6]]).verdict, "advantage")
        subsets = ([0, 6], [0, 5], [1, 6], [0, 3, 6], [0], [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
        honest = pp.decide_preregistered(**self.kwargs(errors=errors, gino_latency_times_s=latency,
                                                       times=times, curve=curve))
        self.assertEqual(honest.verdict, "H0")
        for keep in subsets:
            # The subset's points with their own (genuine) steps coordinates.
            points = _curve_points([times[i] for i in keep], [curve[i] for i in keep],
                                   steps=[_REGISTERED_STEPS[i] for i in keep])
            change = {"errors": errors, "gino_latency_times_s": latency, "baseline_curve": points}
            with self.subTest(steps=[_REGISTERED_STEPS[i] for i in keep]):
                with self.assertRaisesRegex(pp.DecisionInputError, "are not the registered steps"):
                    pp.decide_preregistered(**self.kwargs(**change))
                with self.assertRaisesRegex(pp.DecisionInputError, "are not the registered steps"):
                    pp.verify_preregistered_decision(honest.to_dict(), **self.kwargs(**change))
        # The same class one level up: the genuine subset padded to seven points by repetition
        # passes a count check and keeps the chord (measured on the first fix, ef4e797: advantage).
        # Equalita R4-F6: each half of the check is pinned on its own (only times repeated, only
        # errors repeated), since the first three pads repeat both.
        only_times = [times[0]] + [times[6]] * 6
        only_errors = [curve[0]] + [curve[6]] * 6
        pads = [([times[i] for i in pad], [curve[i] for i in pad], str(pad))
                for pad in ([0, 6, 6, 6, 6, 6, 6], [0, 0, 0, 6, 6, 6, 6], [1, 1, 1, 6, 6, 6, 6])]
        pads += [(only_times, curve, "times only"), (times, only_errors, "errors only")]
        for pad_times, pad_errors, label in pads:
            change = {"errors": errors, "gino_latency_times_s": latency, "times": pad_times, "curve": pad_errors}
            with self.subTest(padded=label):
                with self.assertRaisesRegex(pp.DecisionInputError, "must not repeat a time or an error"):
                    pp.decide_preregistered(**self.kwargs(**change))
        # An eighth point (steps 128) is not registered either.
        with self.assertRaisesRegex(pp.DecisionInputError, "are not the registered steps"):
            pp.decide_preregistered(**self.kwargs(baseline_curve=_curve_points(
                [*self.times, 128.0], [*self.curve, self.curve[-1] / 2], steps=(*_REGISTERED_STEPS, 128))))

    def _r4_case(self):
        """Equalita R4-F1 setting, genuine n = 32 curve: GINO latency t(steps 4), upper bound 1.25 e(steps 4)."""
        times, curve = _R4_N32_REGISTERED
        self.assertEqual(len(times), len(curve))
        raw = 1.0 + 0.3 * np.random.default_rng(7).random((5, 500))
        high = 1.25 * curve[2]
        errors = high * raw / pp.paired_bootstrap_iqm(raw, sample_ids=_IDS)[""].high
        return errors, [times[2]] * 10

    def test_a_genuine_curve_from_another_grid_or_lattice_is_refused(self):
        # Equalita R4-F1 (2026-10-21, 646254d): R3 bound only the number of points. The GENUINE
        # n = 32 curve measured on steps (1, 8, 16, 24, 32, 48, 64), and the genuine n = 64 curve
        # submitted for n = 32, both reached "advantage" (registered: H0), and the record named
        # steps [1..64] and n 32. Each point now carries its coordinates and they are checked.
        errors, latency = self._r4_case()
        summary = pp.paired_bootstrap_iqm(errors, sample_ids=_IDS)[""]
        registered = _curve_points(*_R4_N32_REGISTERED)
        foreign_grid = _curve_points(*_R4_N32_FOREIGN, steps=_R4_N32_FOREIGN_STEPS)
        foreign_lattice = _curve_points(*_R4_N64_REGISTERED, n=64)
        # Positive control of the attack (diagnostic helper, values only): both foreign curves flip H0.
        for points, verdict in ((registered, "H0"), (foreign_grid, "advantage"), (foreign_lattice, "advantage")):
            with self.subTest(diagnostic=points[0].n, steps=[p.steps for p in points]):
                self.assertEqual(pp.decide_test_set(
                    test_set="id", n=32, gino=summary, gino_latency_s=latency[0],
                    baseline_times_s=[p.median_wall_time_s for p in points],
                    baseline_errors=[p.mean_rel_l2 for p in points]).verdict, verdict)
        # Positive control of the guard: the registered genuine curve is accepted with H0.
        honest = pp.decide_preregistered(**self.kwargs(errors=errors, gino_latency_times_s=latency,
                                                       baseline_curve=registered))
        self.assertEqual(honest.verdict, "H0")
        prov = honest.to_dict()["provenance"]
        self.assertEqual(prov["baseline_steps"], list(_REGISTERED_STEPS))
        self.assertEqual(prov["baseline_n"], [32])
        for points, reason in ((foreign_grid, r"steps \[1, 8, 16, 24, 32, 48, 64\] are not the registered steps"),
                               (foreign_lattice, "measured on lattice n = 64, the decision is for n = 32")):
            change = {"errors": errors, "gino_latency_times_s": latency, "baseline_curve": points}
            with self.subTest(refused=reason):
                with self.assertRaisesRegex(pp.DecisionInputError, reason):
                    pp.decide_preregistered(**self.kwargs(**change))
                with self.assertRaisesRegex(pp.DecisionInputError, reason):
                    pp.verify_preregistered_decision(honest.to_dict(), **self.kwargs(**change))
        # The same values in another order (Equalita R4 E7: 31 permutations were accepted and
        # recorded with wrong steps) carry their steps in another order and are refused.
        reversed_points = tuple(reversed(registered))
        with self.assertRaisesRegex(pp.DecisionInputError, "are not the registered steps"):
            pp.decide_preregistered(**self.kwargs(errors=errors, gino_latency_times_s=latency,
                                                  baseline_curve=reversed_points))

    def test_point_coordinates_and_settings_are_checked_against_the_preregistration(self):
        # Renamed after Equalita R5-F2 (formerly "every_measurement_coordinate..._is_checked", which
        # overclaimed: dt was not compared with the preregistration). Checked: steps, n, domain,
        # kappa, t_final, dt, task configuration digest, rtol, warm-ups, repeats, timing protocol
        # digest, sample count, timed samples. Not checked (claim boundary, spec section 5a): which
        # 500 inputs, the machine and thread policy.
        # Domain, sample count and timed samples are coordinates too: an n = 32 torus curve has the
        # registered n and steps but other inputs; a 50-input curve is not "the same test inputs".
        cases = (
            ({"domain": "torus_periodic"}, "measured on domain 'torus_periodic'"),
            ({"domain": "triangle"}, "measured on domain 'triangle'"),      # registry name, not the lattice's
            ({"sample_count": 50}, "sample_count 50"),
            ({"sample_count": 501}, "sample_count 501"),
            ({"timed_samples": 0}, "timed_samples 0"),
            ({"timed_samples": 501}, "timed_samples 501"),
            ({"n": 16}, "lattice n = 16"),
            # Equalita R5-F1: the settings, each alone (the fixture writes them as the producer does).
            ({"config": HeatOperatorTaskConfig(kappa=2.0)}, r"solved with kappa = 2\.0;.*:51-55 \[D1\]"),
            ({"config": HeatOperatorTaskConfig(t_final=0.02)}, r"solved with t_final = 0\.02;"),
            ({"config": HeatOperatorTaskConfig(lambda_cutoff=500.0)}, "task configuration digest"),
            ({"config": HeatOperatorTaskConfig(spectral_decay=2.0)}, "task configuration digest"),
            ({"rtol": 1e-6}, r"rtol = 1e-06; the baseline's tolerance is 1e-12"),
            ({"protocol": pp.TimingProtocol(warmup=0)}, r"warmup = 0; .* \(preregistration :145\)"),
            ({"protocol": pp.TimingProtocol(repeats=1)}, r"repeats = 1; .* \(preregistration :145\)"),
            ({"protocol": pp.TimingProtocol(bootstrap_resamples=200)}, "timing protocol digest"),
            ({"protocol": pp.TimingProtocol(seed=1)}, "timing protocol digest"),
            ({"clock": "custom:count.__next__"}, r"timed with clock 'custom:count\.__next__'"),
        )
        for coords, reason in cases:
            with self.subTest(coords=coords):
                points = _curve_points(self.times, self.curve, **coords)
                with self.assertRaisesRegex(pp.DecisionInputError, reason):
                    pp.decide_preregistered(**self.kwargs(baseline_curve=points))
        # One deviating point among seven is enough.
        for field, value, reason in (("n", 64, "lattice n = 64"),
                                     ("domain", "torus_periodic", "domain 'torus_periodic'"),
                                     ("sample_count", 499, "sample_count 499"),
                                     ("kappa", 2.0, "kappa = 2.0"),
                                     # dt inconsistent with t_final / steps (R5 matrix: dt = 0.01 at every
                                     # steps, and the former fixture's dt = 1 / steps)
                                     ("dt", 0.01, r"dt = 0\.01, not t_final / steps = 0\.000625"),
                                     ("dt", 1.0 / 16, r"dt = 0\.0625, not t_final / steps"),
                                     ("warmup", 2, "warmup = 2")):
            with self.subTest(single=field):
                points = list(_curve_points(self.times, self.curve))
                points[4] = dataclasses.replace(points[4], **{field: value})
                with self.assertRaisesRegex(pp.DecisionInputError, reason):
                    pp.decide_preregistered(**self.kwargs(baseline_curve=tuple(points)))
        # The torus set accepts a torus curve (domain from the test set, not a constant).
        torus = pp.decide_preregistered(**self.kwargs(test_set="composite_ood", sample_ids=list(range(200000, 200500))))
        self.assertEqual({p["domain"] for p in torus.provenance["baseline_points"]}, {"torus_periodic"})
        # Types and values of the coordinates fail closed, not by coercion.
        for field, value in (("steps", 1.0), ("steps", True), ("n", 32.0), ("sample_count", "500"),
                             ("mean_rel_l2", float("nan")), ("median_wall_time_s", 0.0),
                             ("median_rel_l2", float("inf")), ("wall_time_ci_s", (1.0,)), ("dt", -1.0),
                             ("kappa", "1.0"), ("t_final", True), ("rtol", float("nan")), ("warmup", 3.0),
                             ("repeats", True), ("task_config_sha256", None), ("timing_protocol_sha256", b"x")):
            with self.subTest(field=field, value=value):
                points = list(_curve_points(self.times, self.curve))
                points[0] = dataclasses.replace(points[0], **{field: value})
                with self.assertRaises(pp.DecisionInputError):
                    pp.decide_preregistered(**self.kwargs(baseline_curve=tuple(points)))

    def test_record_only_fields_are_plausible(self):
        # Equalita R5-F4: fields that do not enter the verdict still stand in the decision-grade
        # record; an inverted interval, a negative error or a negative iteration count was accepted.
        for field, value, reason in (("wall_time_ci_s", (9.0, 1.0), r"\(9\.0, 1\.0\) is not ordered low <= high"),
                                     ("wall_time_ci_s", (-1.0, 1.0), r"wall_time_ci_s must be .* >= 0"),
                                     ("median_rel_l2", -1.0, r"median_rel_l2 must be .* >= 0"),
                                     ("mean_rel_h1", -1e-9, r"mean_rel_h1 must be .* >= 0"),
                                     ("mean_rel_max", -1.0, r"mean_rel_max must be .* >= 0"),
                                     ("cg_iterations_mean", -5.0, r"cg_iterations_mean must be .* >= 0")):
            with self.subTest(field=field, value=value):
                points = list(_curve_points(self.times, self.curve))
                points[3] = dataclasses.replace(points[3], **{field: value})
                with self.assertRaisesRegex(pp.DecisionInputError, reason):
                    pp.decide_preregistered(**self.kwargs(baseline_curve=tuple(points)))
        # Positive control at the bounds: zero errors and iterations, a degenerate and a wide
        # interval (a median outside the interval is not refused: the percentile bootstrap does
        # not guarantee containment, and a genuine curve must pass).
        for field, value in (("median_rel_l2", 0.0), ("mean_rel_h1", 0.0), ("mean_rel_max", 0.0),
                             ("cg_iterations_mean", 0.0), ("wall_time_ci_s", (0.0, 0.0)),
                             ("wall_time_ci_s", (1.0, 1e9)), ("wall_time_ci_s", [2.0, 3.0])):
            with self.subTest(accepted=field, value=value):
                points = list(_curve_points(self.times, self.curve))
                points[3] = dataclasses.replace(points[3], **{field: value})
                record = pp.decide_preregistered(**self.kwargs(baseline_curve=tuple(points))).to_dict()
                expected = list(value) if field == "wall_time_ci_s" else value
                self.assertEqual(record["provenance"]["baseline_points"][3][field], expected)

    def test_each_field_is_read_once_into_the_record(self):
        # Equalita R5-F3: the checks read a coordinate through _point_int and the record read it
        # again with int(); a numbers.Integral whose int() gives 32 first and 64 afterwards was
        # checked as 32 and recorded as n = [64]. Now one plain copy serves checks and record.
        import numbers

        class FlipInt(numbers.Integral):
            def __init__(self, first, then):
                self.values = [first, then]

            def __int__(self):
                return self.values.pop(0) if len(self.values) > 1 else self.values[0]

            __index__ = __int__

            def __eq__(self, other):
                return int(self.values[0]) == other

            def __hash__(self):
                return 0

        for name in ("__abs__", "__add__", "__and__", "__ceil__", "__floor__", "__floordiv__", "__invert__",
                     "__le__", "__lshift__", "__lt__", "__mod__", "__mul__", "__neg__", "__or__", "__pos__",
                     "__pow__", "__radd__", "__rand__", "__rfloordiv__", "__rlshift__", "__rmod__", "__rmul__",
                     "__ror__", "__round__", "__rpow__", "__rrshift__", "__rshift__", "__rtruediv__", "__rxor__",
                     "__truediv__", "__trunc__", "__xor__"):
            setattr(FlipInt, name, lambda self, *a: NotImplemented)
        FlipInt.__abstractmethods__ = frozenset()

        class SameStr(str):
            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

            __hash__ = str.__hash__

        honest = pp.decide_preregistered(**self.kwargs()).to_dict()
        for field, make in (("n", lambda: FlipInt(32, 64)), ("steps", None), ("sample_count", lambda: FlipInt(500, 50)),
                            ("repeats", lambda: FlipInt(10, 1))):
            with self.subTest(field=field):
                points = list(_curve_points(self.times, self.curve))
                if field == "steps":
                    points = [dataclasses.replace(p, steps=FlipInt(p.steps, 3)) for p in points]
                else:
                    points = [dataclasses.replace(p, **{field: make()}) for p in points]
                self.assertEqual(pp.decide_preregistered(**self.kwargs(baseline_curve=tuple(points))).to_dict(),
                                 honest)
        # A str subclass that equals everything is checked and recorded by its characters.
        points = [dataclasses.replace(p, domain=SameStr("torus_periodic")) for p in _curve_points(self.times, self.curve)]
        with self.assertRaisesRegex(pp.DecisionInputError, "measured on domain 'torus_periodic'"):
            pp.decide_preregistered(**self.kwargs(baseline_curve=tuple(points)))
        points = [dataclasses.replace(p, domain=SameStr("triangle_dirichlet"))
                  for p in _curve_points(self.times, self.curve)]
        record = pp.decide_preregistered(**self.kwargs(baseline_curve=tuple(points))).to_dict()
        self.assertEqual(record, honest)
        self.assertIs(type(pp._point_copy(points[0])["domain"]), str)

    def test_genuine_curve_output_passes_steps_lattice_and_domain(self):
        # Fehler 2. Art against the producer itself, not against hand-built points: the coordinates
        # baseline_cost_accuracy_curve writes (lattice spelling of the domain included) must pass
        # the steps, n and domain checks; with 2 instead of 500 inputs the first refusal is the
        # sample count, which the checks reach only after steps, n, domain and the settings
        # (R5-F1) of that point, so the producer's settings with the registered protocol pass.
        protocol = pp.TimingProtocol()
        for test_set, domain, ids in (("id", "triangle_dirichlet", [100000, 100001]),
                                      ("composite_ood", "torus_periodic", [200000, 200001])):
            with self.subTest(test_set=test_set):
                config = HeatOperatorTaskConfig(domain=domain)
                data = generate_dataset(config, n=32, sample_ids=ids, dataset_seed=1)
                points = pp.baseline_cost_accuracy_curve(config, data["u0"], data["uT"], data["lattice"],
                                                         protocol=protocol, timed_samples=1)
                self.assertEqual([(p.n, p.domain, p.steps) for p in points],
                                 [(32, domain, s) for s in _REGISTERED_STEPS])
                change = {"test_set": test_set, "sample_ids": list(range(ids[0], ids[0] + 500)),
                          "baseline_curve": points}
                with self.assertRaisesRegex(pp.DecisionInputError, r"steps=1 has sample_count 2;"):
                    pp.decide_preregistered(**self.kwargs(**change))

    def test_genuine_curve_with_other_settings_is_refused(self):
        # Equalita R5-F1 (probes/r5_real.out @d6cfdec): the genuine producer run with solver kappa = 2
        # or T = 0.02 against the registered truth reached "advantage" (e_base 0.4797 at S = 500),
        # its record indistinguishable from a registered run for kappa. The producer now writes the
        # settings into every point and the gate refuses them before the sample count (here one
        # input, so a point that passes every setting is refused at sample_count 1).
        registered = HeatOperatorTaskConfig()
        data = generate_dataset(registered, n=32, sample_ids=[100000], dataset_seed=1)
        cases = (
            (HeatOperatorTaskConfig(kappa=2.0), {}, r"steps=1 was solved with kappa = 2\.0;"),
            (HeatOperatorTaskConfig(t_final=0.02), {}, r"steps=1 was solved with t_final = 0\.02;"),
            (HeatOperatorTaskConfig(lambda_cutoff=500.0), {}, r"steps=1 carries task configuration digest"),
            (registered, {"rtol": 1e-6}, r"steps=1 was solved with rtol = 1e-06;"),
            (registered, {"protocol": pp.TimingProtocol(warmup=0, repeats=1)}, r"steps=1 was timed with warmup = 0;"),
            (registered, {"protocol": pp.TimingProtocol(bootstrap_resamples=200)},
             r"steps=1 carries timing protocol digest"),
            # Self-check of the fix (same class): the clock is a producer parameter that sets the
            # times; a slow fake clock makes the baseline look slow.
            (registered, {"clock": _fake_clock(10.0)}, r"steps=1 was timed with clock 'custom:"),
            (registered, {}, r"steps=1 has sample_count 1;"),                       # positive control
        )
        for config, extra, reason in cases:
            with self.subTest(config=config.canonical(), extra=sorted(extra)):
                points = pp.baseline_cost_accuracy_curve(config, data["u0"], data["uT"], data["lattice"],
                                                         timed_samples=1, **extra)
                self.assertEqual({p.clock for p in points},
                                 {"time.perf_counter" if "clock" not in extra else pp._clock_name(extra["clock"])})
                self.assertEqual({(p.kappa, p.t_final, p.task_config_sha256, p.rtol) for p in points},
                                 {(config.kappa, config.t_final, config.digest(), extra.get("rtol", 1e-12))})
                protocol = extra.get("protocol", pp.TimingProtocol())
                self.assertEqual({(p.warmup, p.repeats, p.timing_protocol_sha256) for p in points},
                                 {(protocol.warmup, protocol.repeats, protocol.digest())})
                self.assertEqual([p.dt for p in points], [config.t_final / s for s in _REGISTERED_STEPS])
                with self.assertRaisesRegex(pp.DecisionInputError, reason):
                    pp.decide_preregistered(**self.kwargs(baseline_curve=points))

    def test_the_former_float_list_form_is_refused(self):
        # API change 2026-10-21 (Equalita R4-F1): two bare float lists carry no coordinates.
        params = inspect.signature(pp.decide_preregistered).parameters
        self.assertIn("baseline_curve", params)
        self.assertNotIn("baseline_times_s", params)
        self.assertNotIn("baseline_errors", params)
        full = self.kwargs()
        old = {k: v for k, v in full.items() if k != "baseline_curve"}
        with self.assertRaisesRegex(TypeError, "baseline_times_s"):
            pp.decide_preregistered(**old, baseline_times_s=self.times, baseline_errors=self.curve)
        points = full["baseline_curve"]
        subclass = type("PointLike", (pp.CostAccuracyPoint,), {})
        for bad in (self.times, (self.times, self.curve), np.array([self.times, self.curve]),
                    [p.to_dict() for p in points], {"times": self.times}, "points",
                    tuple(subclass(**dataclasses.asdict(p)) for p in points)):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaisesRegex(TypeError, "tuple of CostAccuracyPoint"):
                    pp.decide_preregistered(**{**old, "baseline_curve": bad})
        # A list of points is accepted like the tuple.
        self.assertEqual(pp.decide_preregistered(**{**old, "baseline_curve": list(points)}).to_dict(),
                         pp.decide_preregistered(**full).to_dict())

    def test_the_checked_points_are_the_recorded_points(self):
        # Self-check of the R4-F1 fix (same class one level up): checks and record read the curve
        # in several passes. A sequence that yields the registered points on the first passes and
        # the genuine n = 64 points afterwards had the n = 32 points checked and the n = 64 points
        # decided and recorded. One snapshot serves every check and the record.
        import collections.abc

        errors, latency = self._r4_case()
        registered = _curve_points(*_R4_N32_REGISTERED)
        foreign = _curve_points(*_R4_N64_REGISTERED, n=64)

        class Switching(collections.abc.Sequence):
            def __init__(self, switch_after):
                self.passes, self.switch_after = 0, switch_after

            def __len__(self):
                return len(registered)

            def __getitem__(self, i):
                return (registered if self.passes <= self.switch_after else foreign)[i]

            def __iter__(self):
                self.passes += 1
                return iter(registered if self.passes <= self.switch_after else foreign)

        honest = pp.decide_preregistered(**self.kwargs(errors=errors, gino_latency_times_s=latency,
                                                       baseline_curve=registered)).to_dict()
        self.assertEqual(honest["verdict"], "H0")
        for switch_after in (1, 2, 3):
            with self.subTest(switch_after=switch_after):
                d = pp.decide_preregistered(**self.kwargs(errors=errors, gino_latency_times_s=latency,
                                                          baseline_curve=Switching(switch_after))).to_dict()
                self.assertEqual(d, honest)
                self.assertEqual(d["provenance"]["baseline_n"], [32])

    def test_a_relabelled_curve_is_out_of_reach(self):
        # Claim boundary (as for the raw table): a caller who builds points by hand can put the
        # registered coordinates on values measured elsewhere. The point cannot show where it was
        # measured; only a recomputation of the errors or an attestation of the timing run can.
        errors, latency = self._r4_case()
        relabelled = _curve_points(*_R4_N32_FOREIGN)              # foreign-grid values, registered steps
        d = pp.decide_preregistered(**self.kwargs(errors=errors, gino_latency_times_s=latency,
                                                  baseline_curve=relabelled))
        self.assertEqual(d.verdict, "advantage")
        self.assertTrue(d.provenance["authenticity"].startswith("unattested"))

    def test_the_registered_steps_grid_is_accepted_and_recorded(self):
        # Positive control for F1: the full 7-point curve keeps its verdict, and the record is
        # written from the points: every point with its coordinates, so a reviewer can rerun them
        # (spec/PHASE1_PROTOCOL_v1.md section 5a, claim boundary).
        self.assertEqual(tuple(pp._DEFAULT_STEPS_GRID), _REGISTERED_STEPS)
        times, curve, errors, latency = self._convex_case()
        legit = pp.decide_preregistered(**self.kwargs(errors=errors, gino_latency_times_s=latency,
                                                      times=times, curve=curve))
        self.assertEqual(legit.verdict, "H0")
        for test_set, n in (("id", 32), ("resolution_ood", 16)):
            with self.subTest(test_set=test_set, n=n):
                change = self.kwargs(test_set=test_set, n=n)
                prov = pp.decide_preregistered(**change).to_dict()["provenance"]
                self.assertEqual(prov.get("baseline_steps"), list(_REGISTERED_STEPS))
                self.assertEqual(prov.get("baseline_n"), [n])
                expected = [{**p.to_dict(), "wall_time_ci_s": list(p.wall_time_ci_s)}
                            for p in change["baseline_curve"]]
                self.assertEqual(prov.get("baseline_points"), expected)

    def test_the_run_manifest_content_is_bound_to_the_preregistration(self):
        # Equalita R3 F2 (2026-10-21): form and self-digest were checked, the content not; a
        # manifest with test ids in the training split, a zero preregistration digest, CUDA and
        # float16-amp hardware or neuraloperator 9.9 reached a decision-grade "advantage".
        def manifest(**change):
            return pp.build_run_manifest({**_manifest_fields(), **change})

        hardware = _manifest_fields()["hardware"]
        backend = _manifest_fields()["backend"]
        forged = (
            ("split_ids", {"split_ids": {"train": list(_IDS), "val": [3]}}),         # train = test ids
            ("split_ids", {"split_ids": {"train": [0, 1, 100000], "val": [3]}}),
            ("split_ids", {"split_ids": {"train": [0, 1, 2], "val": [200000]}}),
            ("split_ids", {"split_ids": {"train": [0, 1, 2000], "val": [3]}}),       # outside the pool
            ("datasets", {"datasets": {"train_pool": "c" * 64, "resolution_ood": "d" * 64}}),
            # Equalita R4-F6: every F2 message cites "(preregistration :NN)", so the rule name
            # "preregistration" matched any refusal; the digest rule's own words are pinned.
            ("is not the file_sha256", {"preregistration": {"version": "PREREGISTRATION_PHASE1_v1 DRAFT",
                                                            "sha256": "0" * 64}}),
            ("cuda", {"hardware": {**hardware, "cuda": "12.4"}}),
            ("cuda", {"hardware": {**hardware, "cuda": True}}),
            ("cuda", {"hardware": {**hardware, "cuda": None}}),
            ("dtype", {"hardware": {**hardware, "dtype": "float16-amp"}}),
            # Equalita R4-F3: float32 with AMP or autocast was accepted (preregistration :103,
            # :122, :228); AMP must be recorded, as False.
            ("hardware amp = 'bfloat16'", {"hardware": {**hardware, "amp": "bfloat16"}}),
            ("hardware amp = True", {"hardware": {**hardware, "amp": True}}),
            ("hardware amp = None", {"hardware": {**hardware, "amp": None}}),
            ("hardware amp = 0", {"hardware": {**hardware, "amp": 0}}),
            ("hardware amp = '<missing>'", {"hardware": {k: v for k, v in hardware.items() if k != "amp"}}),
            ("hardware autocast = True", {"hardware": {**hardware, "autocast": True}}),
            ("hardware autocast = 'bf16'", {"hardware": {**hardware, "autocast": "bf16"}}),
            ("neuraloperator", {"backend": {**backend, "neuraloperator": "9.9"}}),
            # R4-F6: the nearest version, which a prefix rule "2." or "2.0" would let through.
            ("neuraloperator = '2.0.1'", {"backend": {**backend, "neuraloperator": "2.0.1"}}),
            ("neuraloperator", {"backend": {k: v for k, v in backend.items() if k != "neuraloperator"}}),
        )
        for rule, change in forged:
            with self.subTest(rule=rule, change=change):
                with self.assertRaisesRegex(pp.DecisionProvenanceError, rule):
                    pp.decide_preregistered(**self.kwargs(run_manifest=manifest(**change)))
        # The dataset digest of the decided set itself must be named.
        torus_ids = list(range(200000, 200500))
        no_torus = manifest(datasets={"train_pool": "c" * 64, "id": "d" * 64})
        with self.assertRaisesRegex(pp.DecisionProvenanceError, "datasets"):
            pp.decide_preregistered(**self.kwargs(test_set="composite_ood", sample_ids=torus_ids,
                                                  run_manifest=no_torus))
        # An unreadable preregistration file fails closed.
        original = getattr(pp, "_PREREGISTRATION_FILE", None)
        try:
            pp._PREREGISTRATION_FILE = _ROOT / "spec" / "does-not-exist.md"
            try:
                pp.decide_preregistered(**self.kwargs())
            except pp.DecisionProvenanceError as exc:
                self.assertIn("preregistration file cannot be read", str(exc))
            except OSError as exc:                                             # refusal type = test verdict
                self.fail(f"an unreadable preregistration file escaped as {type(exc).__name__}, "
                          "not as DecisionProvenanceError")
            else:
                self.fail("an unreadable preregistration file was accepted")
        finally:
            pp._PREREGISTRATION_FILE = original
        # Positive control: the registered manifest (pool ids, the set's digest, the file's digest,
        # CPU without CUDA, float32, neuraloperator 2.0.0) is accepted with the same verdict.
        self.assertEqual(pp.decide_preregistered(**self.kwargs()).verdict, "H0")
        self.assertIs(_manifest_fields()["hardware"]["amp"], False)
        self.assertEqual(pp.decide_preregistered(**self.kwargs(
            run_manifest=manifest(hardware={**hardware, "autocast": False}))).verdict, "H0")
        self.assertEqual(pp.decide_preregistered(**self.kwargs(
            run_manifest=manifest(split_ids={"train": list(range(1800)), "val": list(range(1800, 2000))}))).verdict,
            "H0")

    def test_the_latency_is_the_median_of_the_timed_calls(self):
        # Equalita R3 F8: every latency test used ten equal calls, so median -> mean survived.
        # Spec section 5a: "The latency is their median". One slow call moves the mean, not the median.
        honest = pp.decide_preregistered(**self.kwargs()).to_dict()
        for calls in ([4.0] * 9 + [40.0], [[4.0] * 9 + [40.0], [4.0] * 10, [1.0] * 4 + [4.0] * 6]):
            with self.subTest(calls=calls):
                d = pp.decide_preregistered(**self.kwargs(gino_latency_times_s=calls)).to_dict()
                self.assertEqual(d["gino_latency_s"], 4.0)
                self.assertNotEqual(float(np.mean(calls)), 4.0)               # the mutant would differ
                self.assertEqual(d["rules"], honest["rules"])

    def test_the_baseline_curve_values_are_bound_in_the_record(self):
        # Equalita R3 F9: removing baseline_times_s from the record survived every test. Spec
        # section 5a: the baseline curve values are bound in the record. A changed time far from
        # GINO's latency leaves both rules unchanged, so only the record can notice it.
        base = pp.decide_preregistered(**self.kwargs()).to_dict()
        self.assertEqual(base["provenance"].get("baseline_times_s"), self.times)
        self.assertEqual(base["provenance"].get("baseline_errors"), self.curve)
        moved = self.kwargs(times=[*self.times[:6], 65.0])
        self.assertEqual(pp.decide_preregistered(**moved).to_dict()["rules"], base["rules"])
        with self.assertRaisesRegex(pp.DecisionProvenanceError, "provenance"):
            pp.verify_preregistered_decision(base, **moved)


# ------------------------------------------------------------------
# Section 6: break-even
# ------------------------------------------------------------------


class TestBreakEven(unittest.TestCase):
    def test_closed_form(self):
        self.assertEqual(pp.break_even_queries(100.0, 50.0, 1.0, 2.0), 150)
        self.assertEqual(pp.break_even_queries(10.0, 0.0, 1.0, 4.0), 4)       # ceil(10 / 3)
        self.assertEqual(pp.break_even_queries(0.0, 0.0, 1.0, 2.0), 0)
        self.assertIsNone(pp.break_even_queries(10.0, 0.0, 2.0, 2.0))         # not faster
        self.assertIsNone(pp.break_even_queries(10.0, 0.0, 3.0, 2.0))

    def test_validation(self):
        for args in ((-1.0, 0.0, 1.0, 2.0), (1.0, float("nan"), 1.0, 2.0), (1.0, 1.0, 0.0, 2.0),
                     (1.0, 1.0, 1.0, float("inf")), (True, 1.0, 1.0, 2.0), ("1", 1.0, 1.0, 2.0)):
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                pp.break_even_queries(*args)


# ------------------------------------------------------------------
# Section 7: run manifest
# ------------------------------------------------------------------


def _manifest_fields():
    return {
        "git_sha": "a" * 40,
        # The registered file's digest and one dataset digest per registered set: the
        # decision-grade entry checks both (Equalita R3 F2).
        "preregistration": {"version": "PREREGISTRATION_PHASE1_v1 DRAFT",
                            "sha256": pp.file_sha256(_ROOT / "spec" / "PREREGISTRATION_PHASE1_v1.md")},
        "datasets": {"train_pool": "c" * 64, "id": "d" * 64, "resolution_ood": "d" * 64,
                     "composite_ood": "9" * 64},
        "split_ids": {"train": [0, 1, 2], "val": [3]},
        "backend": {"python": "3.12.10", "torch": "2.6.0", "numpy": "2.5.3", "neuraloperator": "2.0.0"},
        "config_digest": "e" * 64,
        "seeds": [0, 1, 2, 3, 4],
        "hardware": {"machine": "test-box", "threads": 4, "dtype": "float32", "cuda": False, "amp": False},
        "reference_spec_ids": ["hex-heat-tri", "hex-heat-torus"],
        "metric_spec_sha256": "f" * 64,
        "result_digests": {},
    }


class TestRunManifest(unittest.TestCase):
    def test_digest_binds_the_canonical_body(self):
        manifest = pp.build_run_manifest(_manifest_fields())
        self.assertEqual((manifest["schema"], manifest["protocol"]),
                         ("hexantk_phase1_run_manifest_v1", pp.PROTOCOL_ID))
        body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        payload = json.dumps(body, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(manifest["manifest_sha256"], hashlib.sha256(payload).hexdigest())
        reordered = dict(reversed(list(_manifest_fields().items())))
        self.assertEqual(pp.build_run_manifest(reordered)["manifest_sha256"], manifest["manifest_sha256"])
        fields = _manifest_fields()
        fields["extra"] = {"note": "x", "amp": False, "value": np.float64(1.5), "count": np.int64(3)}
        with_extra = pp.build_run_manifest(fields)
        self.assertEqual(with_extra["extra"], {"note": "x", "amp": False, "value": 1.5, "count": 3})
        self.assertNotEqual(with_extra["manifest_sha256"], manifest["manifest_sha256"])
        json.dumps(with_extra, allow_nan=False)

    def test_every_required_key_is_required(self):
        for key in pp._REQUIRED_MANIFEST_KEYS:
            fields = _manifest_fields()
            del fields[key]
            with self.subTest(missing=key), self.assertRaises(pp.RunManifestError):
                pp.build_run_manifest(fields)

    def test_fail_closed_on_malformed_fields(self):
        cases = [
            ("git_sha", "A" * 40), ("git_sha", "a" * 39), ("config_digest", "e" * 63),
            ("metric_spec_sha256", 12), ("datasets", {}), ("datasets", {"x": "c" * 63}),
            ("preregistration", {"version": "v1"}), ("preregistration", {"version": "", "sha256": "b" * 64}),
            ("split_ids", {"train": [0, 1], "val": [1]}), ("split_ids", {"train": [0, 0], "val": [1]}),
            ("split_ids", {"train": [0], "val": []}), ("split_ids", {"train": [0], "val": [True]}),
            ("split_ids", {"train": [0], "val": [1], "test": [2]}), ("backend", {"torch": ""}),
            ("backend", {}), ("seeds", [0, 0]), ("seeds", [-1]), ("seeds", []), ("seeds", [0.5]),
            ("hardware", {"machine": "m", "threads": 1, "dtype": "float32"}),
            ("hardware", {"machine": "m", "threads": float("nan"), "dtype": "float32", "cuda": False}),
            ("reference_spec_ids", []), ("reference_spec_ids", "hex-heat-tri"), ("reference_spec_ids", [""]),
            ("result_digests", {"a": 1}), ("extra", [1, 2]), ("extra", {"x": object()}),
            ("extra", {"x": float("inf")}),
        ]
        for key, value in cases:
            fields = _manifest_fields()
            fields[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(pp.RunManifestError):
                pp.build_run_manifest(fields)
        with self.assertRaises(pp.RunManifestError):
            pp.build_run_manifest({**_manifest_fields(), "unknown": 1})
        with self.assertRaises(pp.RunManifestError):
            pp.build_run_manifest([("git_sha", "a" * 40)])

    def test_file_sha256_is_lf_normalised(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf, crlf = pathlib.Path(tmp, "lf.md"), pathlib.Path(tmp, "crlf.md")
            lf.write_bytes(b"# a\nb\n")
            crlf.write_bytes(b"# a\r\nb\r\n")
            self.assertEqual(pp.file_sha256(lf), hashlib.sha256(b"# a\nb\n").hexdigest())
            self.assertEqual(pp.file_sha256(crlf), pp.file_sha256(lf))
        prereg = pp.file_sha256(_ROOT / "spec" / "PREREGISTRATION_PHASE1_v1.md")
        self.assertRegex(prereg, r"^[0-9a-f]{64}$")

    def test_source_revision(self):
        source = pp.source_revision()
        self.assertRegex(source["git_sha"], r"^[0-9a-f]{40}$")
        self.assertIsInstance(source["worktree_dirty"], bool)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(pp.RunManifestError):
                pp.source_revision(pathlib.Path(tmp))


# ------------------------------------------------------------------
# Section 2: cost-accuracy curve of the mechanistic baseline
# ------------------------------------------------------------------


class TestCostAccuracyCurve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = HeatOperatorTaskConfig()
        cls.lattice = triangle_hex_lattice(12)
        data = generate_dataset(cls.config, n=12, sample_ids=[0, 1, 2], dataset_seed=1)
        cls.u0 = data["u0"]
        cls.truth = data["uT"]
        cls.semidiscrete = cls.u0 @ _semigroup_matrix(cls.lattice, cls.config.t_final).T
        cls.protocol = pp.TimingProtocol(warmup=1, repeats=3, bootstrap_resamples=200)

    def test_second_order_in_dt_against_the_semidiscrete_solution(self):
        points = pp.baseline_cost_accuracy_curve(self.config, self.u0, self.semidiscrete, self.lattice,
                                                 steps_grid=(2, 4, 8), protocol=self.protocol,
                                                 clock=_fake_clock())
        errors = [p.mean_rel_l2 for p in points]
        for coarse, fine in zip(errors, errors[1:]):
            self.assertGreater(coarse / fine, 3.5)
            self.assertLess(coarse / fine, 4.5)
        iterations = [p.cg_iterations_mean for p in points]
        self.assertEqual(iterations, sorted(iterations))
        self.assertLess(iterations[0], iterations[-1])
        for p, steps in zip(points, (2, 4, 8)):
            self.assertEqual((p.n, p.steps, p.sample_count, p.timed_samples), (12, steps, 3, 3))
            self.assertAlmostEqual(p.dt, self.config.t_final / steps)
            self.assertEqual((p.median_wall_time_s, p.wall_time_ci_s), (0.5, (0.5, 0.5)))
            self.assertGreater(p.mean_rel_h1, 0.0)
            self.assertGreater(p.mean_rel_max, 0.0)
            self.assertGreater(p.median_rel_l2, 0.0)
            json.dumps(p.to_dict(), allow_nan=False)

    def test_timed_samples_bounds_the_clock_calls(self):
        readings = []

        def clock():
            # Cubic readings: the k-th timed call lasts (2k)^3 - (2k-1)^3, so the
            # per-point durations are skewed and the median differs from the mean.
            readings.append(len(readings))
            return float(len(readings)) ** 3

        points = pp.baseline_cost_accuracy_curve(self.config, self.u0, self.truth, self.lattice,
                                                 steps_grid=(1, 2), protocol=self.protocol,
                                                 timed_samples=1, clock=clock)
        self.assertEqual(len(readings), 2 * 1 * self.protocol.repeats * 2)   # steps x samples x repeats x 2
        self.assertEqual([p.timed_samples for p in points], [1, 1])
        # steps=1: durations 7, 37, 91 (median 37, mean 45); steps=2: 169, 271, 397 (median 271, mean 279)
        self.assertEqual([p.median_wall_time_s for p in points], [37.0, 271.0])
        self.assertEqual([p.wall_time_ci_s for p in points], [(37.0, 37.0), (271.0, 271.0)])

    def test_continuous_truth_floors_and_the_frontier_keeps_the_best_point(self):
        points = pp.baseline_cost_accuracy_curve(self.config, self.u0, self.truth, self.lattice,
                                                 steps_grid=(1, 2, 4, 8), protocol=self.protocol,
                                                 clock=_fake_clock())
        errors = np.array([p.mean_rel_l2 for p in points])
        # Spatial error of N = 12 dominates: the curve is not monotone in steps.
        self.assertFalse(np.all(np.diff(errors) <= 0))
        times = np.array([p.median_wall_time_s for p in points])      # all 0.5 under the fake clock
        np.testing.assert_array_equal(pp.pareto_frontier(times, errors), [int(np.argmin(errors))])
        # With distinct times the dominated points fall away and the frontier is short.
        frontier = pp.pareto_frontier(np.array([1.0, 2.0, 3.0, 4.0]), errors)
        self.assertLess(len(frontier), len(points))
        self.assertEqual(frontier[0], 0)

    def test_validation(self):
        kwargs = dict(steps_grid=(1,), protocol=self.protocol, clock=_fake_clock())
        with self.assertRaises(ValueError):
            pp.baseline_cost_accuracy_curve(self.config, self.u0, self.truth, torus_hex_lattice(8), **kwargs)
        with self.assertRaises(TypeError):
            pp.baseline_cost_accuracy_curve({"domain": "triangle_dirichlet"}, self.u0, self.truth,
                                            self.lattice, **kwargs)
        with self.assertRaises(ValueError):
            pp.baseline_cost_accuracy_curve(self.config, self.u0, self.truth[:2], self.lattice, **kwargs)
        with self.assertRaises(ValueError):
            pp.baseline_cost_accuracy_curve(self.config, self.u0, self.truth, self.lattice,
                                            **{**kwargs, "steps_grid": (1, 1)})
        with self.assertRaises(ValueError):
            pp.baseline_cost_accuracy_curve(self.config, self.u0, self.truth, self.lattice,
                                            **{**kwargs, "steps_grid": (0,)})
        with self.assertRaises(TypeError):
            pp.baseline_cost_accuracy_curve(self.config, self.u0, self.truth, self.lattice,
                                            **{**kwargs, "steps_grid": 4})
        with self.assertRaises(ValueError):
            pp.baseline_cost_accuracy_curve(self.config, self.u0, self.truth, self.lattice,
                                            timed_samples=4, **kwargs)
        with self.assertRaises(ValueError):
            pp.baseline_cost_accuracy_curve(self.config, self.u0, self.truth, self.lattice,
                                            rtol=0.0, **kwargs)
        bad = self.u0.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            pp.baseline_cost_accuracy_curve(self.config, bad, self.truth, self.lattice, **kwargs)
        with self.assertRaises(TypeError):
            pp.baseline_cost_accuracy_curve(self.config, self.u0, self.truth, self.lattice,
                                            **{**kwargs, "protocol": None})


# ------------------------------------------------------------------
# Section 11: D4 float-order stability of a radius graph (HEX-P2-02)
# ------------------------------------------------------------------


_U32 = pp.FLOAT32_UNIT_ROUNDOFF
# Geometry of the radius-tie survey: latent grid [0, 1] x [0, sqrt(3)/2] (corner
# norm sqrt(7/4)), nodes inside the same box (max node norm 1.2934 on the torus).
_SURVEY_NORM = math.sqrt(1.75)
# Coordinates as computed in float32: nodes rounded from float64 (<= u |x|), the
# latent axes by torch.linspace (measured <= 0.98 u per component, budgeted 3 u).
_SURVEY_INPUT_ERROR = 4.0 * _U32 * _SURVEY_NORM


def _emulated_cdist32(x64: np.ndarray, y64: np.ndarray) -> np.ndarray:
    """float32 matmul-path distances under every order of the four-term sum: (orders, pairs)."""
    f = np.float32
    x, y = x64.astype(f), y64.astype(f)
    nx = (x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1]).astype(f)
    ny = (y[:, 0] * y[:, 0] + y[:, 1] * y[:, 1]).astype(f)
    terms = [(f(-2.0) * x[:, 0]) * y[:, 0], (f(-2.0) * x[:, 1]) * y[:, 1], nx * f(1.0), f(1.0) * ny]
    sums = []
    for order in itertools.permutations(range(4)):
        acc = terms[order[0]]
        for k in order[1:]:
            acc = (acc + terms[k]).astype(f)                      # sequential, every order
        sums.append(acc)
        sums.append(((terms[order[0]] + terms[order[1]]).astype(f)
                     + (terms[order[2]] + terms[order[3]]).astype(f)).astype(f))   # pairwise
    return np.sqrt(np.maximum(np.stack(sums), f(1e-30))).astype(f)


class TestRadiusFloatStability(unittest.TestCase):
    def test_closed_forms(self):
        r = 0.046875
        self.assertLessEqual(pp.cdist_rounding_margin(r, max_point_norm=1.0, unit_roundoff=0.0), 1e-15)
        self.assertAlmostEqual(pp.cdist_rounding_margin(r, max_point_norm=1.0, unit_roundoff=0.0,
                                                        input_abs_error=1e-6), 1e-6 / r, delta=1e-15)
        small = pp.cdist_rounding_margin(r, max_point_norm=0.5)
        large = pp.cdist_rounding_margin(r, max_point_norm=1.5)
        self.assertLess(small, large)                         # the matmul path loses (2R / r)^2
        self.assertLess(pp.cdist_rounding_margin(r, max_point_norm=1.5, unit_roundoff=pp.FLOAT64_UNIT_ROUNDOFF),
                        1e-11)
        # Leading order: m ~ eps / (2 r^2) with eps = gamma_6 (2R)^2.
        eps = 6 * _U32 * (2 * 1.5) ** 2
        self.assertAlmostEqual(large / (eps / (2 * r * r)), 1.0, delta=0.01)

    def test_the_bound_holds_for_every_emulated_summation_order(self):
        # Independent oracle: emulate the float32 matmul path under all 24 sequential and
        # 24 pairwise orders; no pair with a margin above the bound may flip, and the
        # emulation must see flips below it (positive control: it is not blind).
        rng = np.random.default_rng(2026)
        r = 0.046875
        x = rng.uniform([0.0, 0.0], [1.0 - 2.1 * r, math.sqrt(3) / 2 - 2.1 * r], size=(40_000, 2)) + 1.05 * r
        theta = rng.uniform(0.0, 2 * math.pi, size=40_000)
        t = rng.uniform(-6e-4, 6e-4, size=40_000)
        y = x + (r * (1.0 + t))[:, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        d = np.sqrt(np.sum((x - y) ** 2, axis=1))
        margin = np.abs(d - r) / r
        bound = pp.cdist_rounding_margin(r, max_point_norm=_SURVEY_NORM, input_abs_error=2 * _U32 * _SURVEY_NORM)
        d32 = _emulated_cdist32(x, y).astype(np.float64)
        flips = (d32 <= np.float32(r)) != (d <= r)[None, :]
        self.assertTrue(flips.any())                                       # positive control
        self.assertFalse(flips[:, margin > bound].any())
        near = np.abs(d - r) <= 0.1 * r
        self.assertLess(float(np.max(np.abs(d32[:, near] - d[near]))) / r, bound)
        # Negative control: a bound that ignores the matmul cancellation (direct-difference
        # accuracy, a few u) is violated by the emulation.
        naive = 4 * _U32
        self.assertTrue(flips[:, margin > naive].any())

    def test_the_bound_covers_the_committed_float32_survey(self):
        # Second oracle: torch 2.6.0 float32 vs float64 graphs measured by the survey (Windows,
        # source 69740d1). The measured near-radius error never exceeds the bound, and no
        # flipping candidate has a margin above it.
        survey = json.loads((_ROOT / "results" / "2026-10-20_gino_radius_tie_survey_float32-graphs.json")
                            .read_text(encoding="utf-8"))["survey"]
        tie_free = [c for c in survey["candidates"] if c["tie_free_on_all"]]
        self.assertEqual(len(tie_free), 12)
        above = []
        for c in survey["candidates"]:
            bound = pp.cdist_rounding_margin(c["radius"], max_point_norm=_SURVEY_NORM,
                                             input_abs_error=_SURVEY_INPUT_ERROR)
            with self.subTest(latent=c["latent_resolution"], factor=c["radius_factor"]):
                self.assertLess(c["float32_error_near_radius"], bound)
                if c["float32_flips_total"] > 0:
                    self.assertLessEqual(c["min_relative_margin"], bound)
            if c["tie_free_on_all"] and c["min_relative_margin"] > bound:
                above.append((c["latent_resolution"], c["radius_factor"]))
        # Finding for the owner (spec/GINO_ADAPTER_PHASE1_v2.md, amendment 2026-10-21): no
        # tie-free candidate clears the order-independent bound; the largest ratio is (16, 1.5).
        self.assertEqual(above, [])
        ratios = {(c["latent_resolution"], c["radius_factor"]):
                  c["min_relative_margin"] / pp.cdist_rounding_margin(
                      c["radius"], max_point_norm=_SURVEY_NORM, input_abs_error=_SURVEY_INPUT_ERROR)
                  for c in tie_free}
        self.assertEqual(max(ratios, key=ratios.get), (16, 1.5))

    def test_stability_verdicts(self):
        self.assertEqual(pp.radius_graph_stability(2e-4, budget_relative=1e-4).status, "stable")
        equal = pp.radius_graph_stability(1e-4, budget_relative=1e-4)
        self.assertEqual(equal.status, "refused")                              # equality is not stable
        frozen = pp.radius_graph_stability(5e-5, budget_relative=1e-4, frozen_order_flips=0)
        self.assertEqual(frozen.status, "frozen_order_required")
        self.assertIn("pin", frozen.reason)
        self.assertEqual(pp.radius_graph_stability(5e-5, budget_relative=1e-4, frozen_order_flips=3).status,
                         "refused")
        # Equalita R3 F4: flips measured on the frozen order contradict a budget the margin
        # clears; that was "stable" (fail-open). A measurement against the budget is refused.
        contradicted = pp.radius_graph_stability(2e-4, budget_relative=1e-4, frozen_order_flips=3)
        self.assertEqual(contradicted.status, "refused")
        self.assertIn("contradict", contradicted.reason)
        # Equalita R4-F6: the boundary itself, a single measured flip, is a contradiction too
        # (a mutant "flips >= 2" kept every test above green).
        single = pp.radius_graph_stability(2e-4, budget_relative=1e-4, frozen_order_flips=1)
        self.assertEqual(single.status, "refused")
        self.assertIn("1 measured flips", single.reason)
        self.assertEqual(pp.radius_graph_stability(2e-4, budget_relative=1e-4, frozen_order_flips=0).status,
                         "stable")
        verdict = pp.radius_graph_stability(3e-4, budget_relative=1e-4)
        self.assertEqual(verdict.to_dict()["margin_over_budget"], 3e-4 / 1e-4)
        json.dumps(verdict.to_dict(), allow_nan=False)
        for args, kwargs in (((float("nan"),), {"budget_relative": 1e-4}),
                             ((2e-4,), {"budget_relative": float("nan")}),
                             ((float("inf"),), {"budget_relative": 1e-4}),
                             ((2e-4,), {"budget_relative": 0.0}), ((-1e-5,), {"budget_relative": 1e-4}),
                             (("1",), {"budget_relative": 1e-4}),
                             ((2e-4,), {"budget_relative": 1e-4, "frozen_order_flips": -1}),
                             ((2e-4,), {"budget_relative": 1e-4, "frozen_order_flips": True}),
                             ((2e-4,), {"budget_relative": 1e-4, "frozen_order_flips": 0.0})):
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                pp.radius_graph_stability(*args, **kwargs)
        for kwargs in ({"radius": 0.0}, {"radius": float("nan")}, {"max_point_norm": -1.0},
                       {"unit_roundoff": float("nan")}, {"input_abs_error": -1e-9}, {"unit_roundoff": 0.2},
                       {"radius": 1e-4}):          # rounding swamps r^2: no margin can help
            full = {"radius": 0.05, "max_point_norm": 1.0, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                pp.cdist_rounding_margin(full.pop("radius"), **full)


if __name__ == "__main__":
    unittest.main()
