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

import hashlib
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
        out = pp.paired_bootstrap_iqm({"n16": self.a, "n32": 2.0 * self.a}, n_resamples=400, seed=5)
        self.assertEqual(out["n32"].low, 2.0 * out["n16"].low)
        self.assertEqual(out["n32"].high, 2.0 * out["n16"].high)
        self.assertEqual(out["n32"].point, 2.0 * out["n16"].point)

    def test_pairing_negative_control(self):
        # Without a shared index set the bounds of an unrelated matrix are not a multiple.
        other = 2.0 * self.a[:, ::-1]
        out = pp.paired_bootstrap_iqm({"a": self.a, "b": other}, n_resamples=400, seed=5)
        self.assertNotEqual(out["b"].low, 2.0 * out["a"].low)

    def test_chunking_does_not_change_the_result_and_seeds_do(self):
        one = pp.paired_bootstrap_iqm(self.a, n_resamples=300, seed=11)[""]
        two = pp.paired_bootstrap_iqm(self.a, n_resamples=300, seed=11, chunk_elements=7)[""]
        self.assertEqual(one, two)
        three = pp.paired_bootstrap_iqm(self.a, n_resamples=300, seed=12)[""]
        self.assertTrue(three.low != one.low or three.high != one.high)

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
        self.assertIn("outside the measured baseline range", runtime.reason)
        self.assertFalse(accuracy.evaluable)
        self.assertIn("baseline floor", accuracy.reason)
        self.assertIsNone(runtime.baseline_value)
        self.assertIsNone(accuracy.baseline_value)

    def test_above_the_coarsest_error_is_no_claim(self):
        d = self._decide(high=9.0, latency=16.0)
        self.assertEqual(d.verdict, "H0")
        self.assertIn("above the coarsest", d.rules[1].reason)
        self.assertIn("no extrapolation", d.rules[0].reason)

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
        errors = 0.12 + 0.03 * rng.standard_normal((5, 40)) ** 2
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
            {"n_seeds": 0}, {"n_samples": 0}, {"n_seeds": 5.0},
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

    def test_validation(self):
        for kwargs in ({"test_set": ""}, {"n": 0}, {"gino": {"high": 1.0}}, {"gino_latency_s": 0.0},
                       {"gino_latency_s": float("nan")}, {"baseline_errors": [8, 4, 2]}):
            full = {"test_set": "id", "n": 32, "gino": _summary(1.5), "gino_latency_s": 2.0,
                    "baseline_times_s": self.TIMES, "baseline_errors": self.ERRORS, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                pp.decide_test_set(**full)


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
        "preregistration": {"version": "PREREGISTRATION_PHASE1_v1 DRAFT", "sha256": "b" * 64},
        "datasets": {"train_pool": "c" * 64, "id_test": "d" * 64},
        "split_ids": {"train": [0, 1, 2], "val": [3]},
        "backend": {"python": "3.12.10", "torch": "2.6.0", "numpy": "2.5.3", "neuraloperator": "2.0.0"},
        "config_digest": "e" * 64,
        "seeds": [0, 1, 2, 3, 4],
        "hardware": {"machine": "test-box", "threads": 4, "dtype": "float32", "cuda": False},
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


if __name__ == "__main__":
    unittest.main()
