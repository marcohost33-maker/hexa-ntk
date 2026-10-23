"""Tests for ntk_trajectory_hex_v3 (NTK_TRAJECTORY_v1): closed-form kernel trajectories,
metamorphic invariances (output scale, basis rotation), self-computed digests and every
fail-closed guard of the declared checkpoint sequence."""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import unittest
from unittest import mock

import numpy as np
import torch

import ntk_trajectory_hex_v3 as nt

M, K = 6, 3
RNG = np.random.default_rng(11)
WEIGHTS = RNG.uniform(0.5, 2.0, M)
_raw = RNG.standard_normal((M, K))
_q, _ = np.linalg.qr(np.sqrt(WEIGHTS)[:, None] * _raw)
MODES = _q / np.sqrt(WEIGHTS)[:, None]  # phi^T W phi = I
PROJ = (WEIGHTS[:, None] * MODES).T     # (K, M): P = PROJ @ y
U = RNG.uniform(-1.0, 1.0, M)           # generic input field


def _f64(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


def apply(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # param_ntk runs a float64 and a float32 copy: the input follows the model's dtype.
    return model(x.to(next(model.parameters()).dtype))


class Linear(torch.nn.Module):
    """y = A theta + x: the Jacobian A does not depend on theta, the kernel is constant."""

    def __init__(self, a: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("a", _f64(a))
        self.theta = torch.nn.Parameter(_f64(np.zeros(a.shape[1])))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a @ self.theta + x


class Product(torch.nn.Module):
    """y = a b x: Theta = (a^2 + b^2) c c^T with c = P x -- the kernel grows, its direction stays."""

    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(_f64(1.0))
        self.b = torch.nn.Parameter(_f64(1.0))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * self.b * x


class SquarePlusMode(torch.nn.Module):
    """y = a^2 x + b phi_2 with input x = phi_1: Theta = diag(4 a^2, 1, 0) in the mode basis."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("v", _f64(MODES[:, 1]), persistent=False)  # not part of a checkpoint
        self.a = torch.nn.Parameter(_f64(1.0))
        self.b = torch.nn.Parameter(_f64(0.0))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a ** 2 * x + self.b * self.v.to(x.dtype)


class Ignores(torch.nn.Module):
    """Output independent of the parameter: the zero kernel at every checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(_f64(0.0))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


class SquareScale(torch.nn.Module):
    """y = theta^2 x: Theta = 4 theta^2 c c^T, zero at theta = 0."""

    def __init__(self) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(_f64(0.0))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.theta ** 2 * x


class Mutating(Product):
    """Writes into its input but does not read it: only the input digest can notice."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.add_(1.0)  # changes the caller's fixed input in place (float64 path: .to is a no-op)
        return self.a * self.b * torch.ones_like(x)


class WithMetadata(Product):
    """Mimics neuraloperator 2.0.0 BaseModel: state_dict() adds the constructor kwargs as
    "_metadata" (with a function and a class), load_state_dict() pops it from the given dict."""

    META = {"width": 3, "act": torch.tanh, "conv": torch.nn.Linear, "shape": (2, 3)}

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["_metadata"] = self.META
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def _states(**values) -> list:
    """{'a': [1, 2], 'b': [3, 4]} -> [(0, {'a': 1, 'b': 3}), (10, {'a': 2, 'b': 4})] with steps 0, 10, 20 ..."""
    names = list(values)
    n = len(values[names[0]])
    return [(10 * i, {k: _f64(values[k][i]) for k in names}) for i in range(n)]


def _run(model, checkpoints, x=U, *, modes=MODES, weights=WEIGHTS, **kw):
    if "declared_steps" not in kw:
        kw["declared_steps"] = [s for s, _ in checkpoints]
    kw.setdefault("run_id", "test-run")
    return nt.ntk_trajectory(model, kw.pop("apply", apply), _f64(x), modes, weights, checkpoints, **kw)


class TestClosedForms(unittest.TestCase):
    def test_linear_model_kernel_is_constant_and_analytic(self) -> None:
        a = RNG.standard_normal((M, 4))
        ckpts = [(s, {"a": _f64(a), "theta": _f64(RNG.standard_normal(4))}) for s in (0, 5, 50)]
        traj = _run(Linear(a), ckpts)
        expected = PROJ @ a @ a.T @ PROJ.T
        for e in traj.entries:
            np.testing.assert_allclose(e.kernel, expected, rtol=1e-12, atol=1e-12)
            self.assertEqual(e.relative_drift, 0.0)
            self.assertEqual(e.distance_from_initial, 0.0)
        self.assertEqual([e.velocity for e in traj.entries], [None, 0.0, 0.0])
        self.assertEqual(traj.n_real_params, 4)
        self.assertEqual(traj.n_modes, K)

    def test_product_model_drifts_in_magnitude_not_direction(self) -> None:
        a, b = [1.0, 1.5, 2.0], [0.5, 0.5, 1.0]
        traj = _run(Product(), _states(a=a, b=b), target_coefficients=PROJ @ U)
        c = PROJ @ U
        s = [ai ** 2 + bi ** 2 for ai, bi in zip(a, b)]
        for e, si in zip(traj.entries, s):
            np.testing.assert_allclose(e.kernel, si * np.outer(c, c), rtol=1e-12, atol=1e-14)
            self.assertAlmostEqual(e.relative_drift, abs(si / s[0] - 1.0), delta=1e-12)
            self.assertLess(e.distance_from_initial, 1e-14)
            self.assertAlmostEqual(e.effective_rank, 1.0, delta=1e-12)
            self.assertAlmostEqual(e.target_alignment, 1.0, delta=1e-12)
            expected_offdiag = 1.0 - np.sum(c ** 4) / np.sum(c ** 2) ** 2
            self.assertAlmostEqual(e.offdiag_energy_fraction, expected_offdiag, delta=1e-12)
        self.assertGreater(traj.entries[-1].relative_drift, 1.0)  # 1.25 -> 5: drift 3

    def test_square_model_changes_direction_in_closed_form(self) -> None:
        a = [1.0, 0.5, 2.0]
        traj = _run(SquarePlusMode(), _states(a=a, b=[0.0, 0.3, -0.1]), x=MODES[:, 0],
                    target_coefficients=[1.0, 0.0, 0.0])

        def dist(x0: float, x1: float) -> float:
            p, q = 4 * x0 ** 2, 4 * x1 ** 2  # Theta = diag(p, 1, 0)
            return 1.0 - (p * q + 1.0) / math.sqrt((p * p + 1.0) * (q * q + 1.0))

        for i, (e, ai) in enumerate(zip(traj.entries, a)):
            p = 4 * ai ** 2
            np.testing.assert_allclose(e.kernel, np.diag([p, 1.0, 0.0]), rtol=1e-12, atol=1e-13)
            self.assertAlmostEqual(e.distance_from_initial, dist(a[0], ai), delta=1e-12)
            self.assertAlmostEqual(e.effective_rank, (p + 1.0) ** 2 / (p * p + 1.0), delta=1e-12)
            self.assertLess(e.offdiag_energy_fraction, 1e-24)
            self.assertAlmostEqual(e.target_alignment, p / math.sqrt(p * p + 1.0), delta=1e-12)
            self.assertAlmostEqual(e.relative_drift, abs(p - 4.0) / math.sqrt(17.0), delta=1e-12)
            if i:
                self.assertAlmostEqual(e.distance_from_previous, dist(a[i - 1], ai), delta=1e-12)
                self.assertAlmostEqual(e.velocity, dist(a[i - 1], ai) / 10.0, delta=1e-13)
        self.assertGreater(traj.entries[1].distance_from_initial, 0.05)

    def test_zero_kernels_give_null_relative_quantities(self) -> None:
        traj = _run(Ignores(), _states(theta=[0.0, 1.0]), target_coefficients=[1.0, 2.0, 3.0])
        for e in traj.entries:
            np.testing.assert_array_equal(e.kernel, np.zeros((K, K)))
            for value in (e.effective_rank, e.offdiag_energy_fraction, e.relative_drift,
                          e.distance_from_initial, e.target_alignment):
                self.assertIsNone(value)
        self.assertIsNone(traj.entries[1].velocity)
        record = traj.to_dict()
        json.dumps(record, allow_nan=False)
        self.assertIsNone(record["entries"][1]["relative_drift"])

    def test_zero_anchor_then_growing_kernel(self) -> None:
        traj = _run(SquareScale(), _states(theta=[0.0, 1.0, 2.0]))
        self.assertTrue(all(e.relative_drift is None for e in traj.entries))  # Theta_0 = 0
        self.assertIsNone(traj.entries[1].distance_from_previous)             # previous is zero
        self.assertLess(traj.entries[2].distance_from_previous, 1e-14)        # same direction
        self.assertAlmostEqual(traj.entries[2].trace / traj.entries[1].trace, 4.0, delta=1e-12)


class TestInvariances(unittest.TestCase):
    CKPTS = staticmethod(lambda: _states(a=[1.0, 0.5, 2.0], b=[0.0, 0.3, -0.1]))

    def test_output_scale_leaves_every_relative_quantity_unchanged(self) -> None:
        base = _run(SquarePlusMode(), self.CKPTS(), x=MODES[:, 0], target_coefficients=[1.0, 2.0, 0.5])
        scaled = _run(SquarePlusMode(), self.CKPTS(), x=MODES[:, 0], target_coefficients=[2.0, 4.0, 1.0],
                      apply=lambda m, x: 3.0 * apply(m, x))
        for e0, e1 in zip(base.entries, scaled.entries):
            self.assertAlmostEqual(e1.trace, 9.0 * e0.trace, delta=1e-12 * e1.trace)
            for name in ("effective_rank", "offdiag_energy_fraction", "relative_drift",
                         "distance_from_initial", "target_alignment"):
                self.assertAlmostEqual(getattr(e1, name), getattr(e0, name), delta=1e-12, msg=name)
        self.assertEqual(base.inputs_sha256, scaled.inputs_sha256)
        self.assertNotEqual(base.target_sha256, scaled.target_sha256)

    def test_orthogonal_basis_change(self) -> None:
        # Q R stays W-orthonormal; Theta -> R^T Theta R. Distance, drift, rank and spectrum are
        # basis-invariant; mode coupling is a property of the basis and must change.
        r, _ = np.linalg.qr(RNG.standard_normal((K, K)))
        base = _run(SquarePlusMode(), self.CKPTS(), x=MODES[:, 0])
        turned = _run(SquarePlusMode(), self.CKPTS(), x=MODES[:, 0], modes=MODES @ r)
        for e0, e1 in zip(base.entries, turned.entries):
            np.testing.assert_allclose(e1.kernel, r.T @ e0.kernel @ r, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(e1.eigenvalues, e0.eigenvalues, rtol=1e-11, atol=1e-12)
            for name in ("effective_rank", "relative_drift", "distance_from_initial"):
                self.assertAlmostEqual(getattr(e1, name), getattr(e0, name), delta=1e-11, msg=name)
            self.assertGreater(e1.offdiag_energy_fraction, 1e-3)
        self.assertNotEqual(base.basis_sha256, turned.basis_sha256)


class TestLazyRegime(unittest.TestCase):
    """Theory oracle: in NTK parameterisation f = v . tanh(W x) / sqrt(n) the kernel moves
    O(1/sqrt(n)) during training (Jacot et al. 2018; Lee et al. 2019). Same task, same
    steps and learning rate: the wide network must show far less drift and distance.
    Measured 2026-10-22 (float64): n = 16 drift 0.110, distance 5.9e-3; n = 1024 drift
    0.0103, distance 3.2e-5 -- the assertions keep a factor >= 2.5 margin."""

    XS = np.linspace(-1.0, 1.0, 24)
    W = np.full(24, 2.0 / 24)

    class Net(torch.nn.Module):
        def __init__(self, n: int) -> None:
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.n = n
            self.w = torch.nn.Parameter(torch.randn(n, 2, generator=g, dtype=torch.float64))
            self.v = torch.nn.Parameter(torch.randn(n, generator=g, dtype=torch.float64))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.v @ torch.tanh(self.w[:, :1] * x[None, :] + self.w[:, 1:]) / math.sqrt(self.n)

    def _final(self, n: int):
        raw = np.stack([np.ones(24), self.XS, self.XS ** 2], 1)
        q, _ = np.linalg.qr(np.sqrt(self.W)[:, None] * raw)
        modes = q / np.sqrt(self.W)[:, None]
        net, x = self.Net(n), _f64(self.XS)
        target = torch.sin(2.5 * x)
        opt = torch.optim.SGD(net.parameters(), lr=1.0)
        ckpts = [(0, {k: v.clone() for k, v in net.state_dict().items()})]
        for t in range(1, 201):
            opt.zero_grad()
            (0.5 * ((net(x) - target) ** 2).mean()).backward()
            opt.step()
            if t in (50, 200):
                ckpts.append((t, {k: v.clone() for k, v in net.state_dict().items()}))
        net.eval()
        traj = nt.ntk_trajectory(net, apply, x, modes, self.W, ckpts, declared_steps=[0, 50, 200],
                                 run_id=f"lazy-{n}", float32_crosscheck=False)
        return traj.entries[-1]

    def test_wide_network_kernel_stays_put(self) -> None:
        narrow, wide = self._final(16), self._final(1024)
        self.assertGreater(narrow.relative_drift, 0.05)
        self.assertLess(wide.relative_drift, narrow.relative_drift / 4.0)
        self.assertLess(wide.distance_from_initial, narrow.distance_from_initial / 10.0)


class TestBinding(unittest.TestCase):
    def test_record_digests_are_computed_from_the_measured_bytes(self) -> None:
        ckpts = _states(a=[1.0, 2.0], b=[0.5, 0.5])
        traj = _run(Product(), ckpts)
        for (_, state), e in zip(ckpts, traj.entries):
            self.assertEqual(e.state_sha256, nt.state_dict_sha256(state))
        self.assertEqual(traj.inputs_sha256, nt.tensor_sha256(_f64(U)))
        record = traj.to_dict()
        self.assertEqual(record["authenticity"], "unattested")
        self.assertEqual(record["record_sha256"], nt.trajectory_record_sha256(record))
        json.loads(json.dumps(record, allow_nan=False))
        tampered = dict(record, run_id="other")
        self.assertNotEqual(tampered["record_sha256"], nt.trajectory_record_sha256(tampered))

    def test_state_digest_is_order_free_and_sees_every_byte(self) -> None:
        a = {"x": _f64([1.0, 2.0]), "y": _f64(3.0)}
        b = {"y": _f64(3.0), "x": _f64([1.0, 2.0])}
        self.assertEqual(nt.state_dict_sha256(a), nt.state_dict_sha256(b))
        flipped = {"x": _f64([1.0, np.nextafter(2.0, 3.0)]), "y": _f64(3.0)}
        self.assertNotEqual(nt.state_dict_sha256(a), nt.state_dict_sha256(flipped))
        # Same bytes under other key names, another key split, dtype or shape are other states.
        self.assertNotEqual(nt.state_dict_sha256({"x": _f64(1.0), "y": _f64(2.0)}),
                            nt.state_dict_sha256({"u": _f64(1.0), "w": _f64(2.0)}))
        self.assertNotEqual(nt.state_dict_sha256({"ab": _f64(1.0), "c": _f64(2.0)}),
                            nt.state_dict_sha256({"a": _f64(1.0), "bc": _f64(2.0)}))
        self.assertNotEqual(nt.tensor_sha256(_f64([1.0, 2.0])), nt.tensor_sha256(_f64([[1.0, 2.0]])))
        self.assertNotEqual(nt.tensor_sha256(torch.zeros(2, dtype=torch.float64)),
                            nt.tensor_sha256(torch.zeros(4, dtype=torch.float32)))
        for unbindable in ({"x": lambda: 0}, {"x": object()}, {"x": math.nan}, {1: _f64(1.0)},
                           {"x": {"act": torch.nn.Module().forward}}):
            with self.subTest(state=repr(unbindable)[:40]):
                with self.assertRaises(nt.NTKTrajectoryError):
                    nt.state_dict_sha256(unbindable)
        with self.assertRaises(nt.NTKTrajectoryError):
            nt.state_dict_sha256([("x", _f64(1.0))])
        with self.assertRaises(TypeError):
            nt.tensor_sha256(np.zeros(2))

    def test_complex_and_bool_tensors_are_hashed(self) -> None:
        z = torch.tensor([1 + 2j], dtype=torch.complex64)
        self.assertNotEqual(nt.tensor_sha256(z), nt.tensor_sha256(torch.tensor([1 - 2j], dtype=torch.complex64)))
        self.assertNotEqual(nt.tensor_sha256(torch.tensor([True])), nt.tensor_sha256(torch.tensor([False])))

    def test_caller_model_and_states_are_unchanged(self) -> None:
        model = Product()
        ckpts = _states(a=[1.0, 2.0], b=[0.5, 0.25])
        before = [nt.state_dict_sha256(s) for _, s in ckpts]
        _run(model, ckpts)
        self.assertEqual(float(model.a), 1.0)
        self.assertEqual(model.a.dtype, torch.float64)
        self.assertFalse(model.training)
        self.assertEqual([nt.state_dict_sha256(s) for _, s in ckpts], before)

    def test_kernel_distance_contract(self) -> None:
        a = np.diag([1.0, 2.0, 0.0])
        self.assertEqual(nt.kernel_distance(a, a.copy()), 0.0)  # equal kernels: exactly 0
        self.assertLess(nt.kernel_distance(a, 5.0 * a), 1e-15)
        b = np.diag([0.0, 0.0, 1.0])
        self.assertEqual(nt.kernel_distance(a, b), 1.0)  # orthogonal PSD kernels
        self.assertIsNone(nt.kernel_distance(a, np.zeros((3, 3))))
        c = np.diag([2.0, 1.0, 1.0])
        self.assertEqual(nt.kernel_distance(a, c), nt.kernel_distance(c, a))
        tiny = 1e-200 * a  # Frobenius norm would underflow without scaling
        self.assertEqual(nt.kernel_distance(tiny, a), 0.0)
        with self.assertRaises(ValueError):
            nt.kernel_distance(a, np.eye(2))
        with self.assertRaises(ValueError):
            nt.kernel_distance(a, np.full((3, 3), np.nan))
        with self.assertRaises(ValueError):
            nt.kernel_distance(np.ones(3), np.ones(3))

    def test_infinite_float32_deviation_is_written_as_a_string(self) -> None:
        self.assertEqual(nt._json_deviation(math.inf), "inf")
        self.assertIsNone(nt._json_deviation(None))
        self.assertEqual(nt._json_deviation(0.25), 0.25)
        with self.assertRaises(TypeError):
            nt.trajectory_record_sha256([1, 2])


class TestConstructorMetadata(unittest.TestCase):
    def _ckpts(self, meta):
        return [(s, dict(st, _metadata=meta)) for s, st in _states(a=[1.0, 2.0], b=[0.5, 0.5])]

    def test_metadata_is_bound_and_the_callers_dict_is_kept(self) -> None:
        ckpts = self._ckpts(dict(WithMetadata.META))
        traj = _run(WithMetadata(), ckpts)
        for (_, state), e in zip(ckpts, traj.entries):
            self.assertIn("_metadata", state)  # the loader popped from a copy
            self.assertEqual(e.state_sha256, nt.state_dict_sha256(state))
        other = dict(WithMetadata.META, act=torch.relu)
        self.assertNotEqual(nt.state_dict_sha256(ckpts[0][1]),
                            nt.state_dict_sha256(dict(ckpts[0][1], _metadata=other)))

    def test_metadata_of_another_architecture_is_refused(self) -> None:
        for meta in (dict(WithMetadata.META, width=4), dict(WithMetadata.META, act=torch.relu),
                     dict(WithMetadata.META, shape=(3, 2)), _f64(1.0)):
            with self.subTest(meta=repr(meta)[:40]):
                with self.assertRaises(nt.NTKTrajectoryError) as ctx:
                    _run(WithMetadata(), self._ckpts(meta))
                self.assertIn("another architecture", str(ctx.exception))

    def test_unbindable_metadata_is_refused(self) -> None:
        class Lambda(WithMetadata):
            META = {"act": lambda x: x}

        with self.assertRaises(nt.NTKTrajectoryError) as ctx:
            _run(Lambda(), self._ckpts(Lambda.META))
        self.assertIn("cannot be bound", str(ctx.exception))


def _sgd_step(lr: float) -> tuple[Product, dict, dict]:
    """One genuine torch.optim.SGD step on Product with the trajectory's loss input."""
    model = Product()
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)
    apply(model, _f64(U)).square().sum().backward()
    self_check = [p.grad is not None and float(p.grad.abs()) > 0.0 for p in model.parameters()]
    assert all(self_check), "the step must see nonzero gradients, else lr = 0 proves nothing"
    optimizer.step()
    return model, before, copy.deepcopy(model.state_dict())


def _reseal(record: dict) -> dict:
    """Recompute record_sha256 after an edit: the digest is no signature, anyone can do this."""
    record = copy.deepcopy(record)
    record["record_sha256"] = nt.trajectory_record_sha256(record)
    return record


class TestRepeatedStates(unittest.TestCase):
    """HNT-R25-01 (cross-family GPT-6 R25, P2): identical checkpoint bytes are measured and
    marked, not refused. Equal bytes prove neither a mislabelled copy nor real stationarity;
    a copy under a wrong step number is a provenance defect, not a hash defect.

    Replaces TestFailClosed.test_identical_checkpoint_bytes_are_refused (same input
    a = [1, 2, 1]), which pinned the refusal this finding retracted; that case is now
    test_return_to_an_earlier_state_is_measured_and_marked."""

    def test_real_sgd_step_with_zero_lr_is_a_measured_stationary_checkpoint(self) -> None:
        # Positive control: a genuine optimizer step with lr = 0 leaves the bytes unchanged.
        model, before, after = _sgd_step(0.0)
        self.assertEqual(nt.state_dict_sha256(before), nt.state_dict_sha256(after))
        traj = _run(model.eval(), [(0, before), (1, after)])
        first, second = traj.entries
        self.assertFalse(first.state_unchanged_since_previous)
        self.assertIsNone(first.repeated_state_digest_of)
        self.assertIsNone(first.stationary_drift)
        self.assertTrue(second.state_unchanged_since_previous)
        self.assertEqual(second.repeated_state_digest_of, 0)
        self.assertEqual(second.stationary_drift, "measured_zero")
        # Measured again under the FD oracle, not copied from the first entry.
        self.assertLessEqual(second.fd_relative_error, nt.pn.FD_REL_TOL)
        self.assertEqual(second.kernel_sha256, first.kernel_sha256)
        self.assertGreater(second.trace, 0.0)
        self.assertEqual(second.relative_drift, 0.0)
        self.assertEqual(second.distance_from_initial, 0.0)
        self.assertEqual(second.distance_from_previous, 0.0)
        self.assertEqual(second.velocity, 0.0)
        record = traj.to_dict()
        self.assertEqual(record["authenticity"], "unattested")
        self.assertIs(record["entries"][1]["state_unchanged_since_previous"], True)
        self.assertEqual(record["entries"][1]["repeated_state_digest_of"], 0)
        self.assertEqual(record["entries"][1]["stationary_drift"], "measured_zero")
        nt.check_trajectory_record(record)
        nt.check_trajectory_record(json.loads(json.dumps(record, allow_nan=False)))

    def test_identical_bytes_are_not_refused(self) -> None:
        # The retracted contract as an explicit assertion (a refusal is a test failure here,
        # not an error), for the A-A, A-B-A and A-A-A sequences.
        for a in ([1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]):
            with self.subTest(a=a):
                try:
                    traj = _run(Product(), _states(a=a, b=[0.5] * len(a)))
                except nt.NTKTrajectoryError as exc:
                    self.fail(f"identical checkpoint bytes were refused: {exc}")
                self.assertEqual(len(traj.entries), len(a))

    def test_real_sgd_step_with_positive_lr_drifts_and_is_not_marked(self) -> None:
        # Control: the same step with lr > 0 changes the bytes and moves the kernel by the
        # closed form drift |(a1^2 + b1^2) / (a0^2 + b0^2) - 1| (Product, Theta = (a^2+b^2) c c^T).
        model, before, after = _sgd_step(0.05)
        self.assertNotEqual(nt.state_dict_sha256(before), nt.state_dict_sha256(after))
        second = _run(model.eval(), [(0, before), (1, after)]).entries[1]
        self.assertFalse(second.state_unchanged_since_previous)
        self.assertIsNone(second.repeated_state_digest_of)
        self.assertIsNone(second.stationary_drift)
        s0 = float(before["a"]) ** 2 + float(before["b"]) ** 2
        s1 = float(after["a"]) ** 2 + float(after["b"]) ** 2
        self.assertGreater(second.relative_drift, 0.0)
        self.assertAlmostEqual(second.relative_drift, abs(s1 / s0 - 1.0), places=12)

    def test_cross_family_proposal_linear_zero_lr(self) -> None:
        # The R25 proposal (proposed_repo_tests/test_hexa_ntk_r84_stationary_checkpoint.py),
        # run in the repo for the first time and extended by the markers.
        model = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64).eval()
        before = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        optimizer.zero_grad(set_to_none=True)
        x = torch.tensor([[0.4, -0.2]], dtype=torch.float64)
        model(x).sum().backward()
        optimizer.step()
        after = copy.deepcopy(model.state_dict())
        self.assertTrue(all(torch.equal(before[k], after[k]) for k in before))
        traj = nt.ntk_trajectory(
            model, lambda m, inp: m(inp).reshape(-1), x, np.eye(2, dtype=np.float64), np.ones(2, dtype=np.float64),
            [(0, before), (1, after)], declared_steps=(0, 1), run_id="stationary-positive-control",
            model_dtype="float64", float32_crosscheck=False,
        )
        self.assertEqual(traj.entries[1].relative_drift, 0.0)
        self.assertEqual(traj.entries[1].distance_from_initial, 0.0)
        self.assertEqual(traj.entries[1].stationary_drift, "measured_zero")
        self.assertEqual(traj.to_dict()["authenticity"], "unattested")

    def test_markers_name_the_previous_and_the_earliest_repeat(self) -> None:
        # A, B, B: step 20 repeats its predecessor (not the first checkpoint).
        traj = _run(Product(), _states(a=[1.0, 2.0, 2.0], b=[0.5, 0.5, 0.5]))
        self.assertEqual([e.state_unchanged_since_previous for e in traj.entries], [False, False, True])
        self.assertEqual([e.repeated_state_digest_of for e in traj.entries], [None, None, 10])
        self.assertEqual([e.stationary_drift for e in traj.entries], [None, None, "measured_zero"])
        self.assertGreater(traj.entries[2].relative_drift, 0.0)  # still drifted from step 0
        # A, A, A: every repeat names the earliest step, every one is a measured zero.
        traj = _run(Product(), _states(a=[1.0, 1.0, 1.0], b=[0.5, 0.5, 0.5]))
        self.assertEqual([e.repeated_state_digest_of for e in traj.entries], [None, 0, 0])
        self.assertEqual([e.stationary_drift for e in traj.entries], [None, "measured_zero", "measured_zero"])
        nt.check_trajectory_record(traj.to_dict())

    def test_return_to_an_earlier_state_is_measured_and_marked(self) -> None:
        # A, B, A (the input of the retracted refusal test): measured, not unchanged since
        # the previous checkpoint, but a repeat of step 0 with the identical kernel.
        traj = _run(Product(), _states(a=[1.0, 2.0, 1.0], b=[0.5, 0.5, 0.5]))
        last = traj.entries[2]
        self.assertFalse(last.state_unchanged_since_previous)
        self.assertEqual(last.repeated_state_digest_of, 0)
        self.assertIsNone(last.stationary_drift)
        self.assertEqual(last.kernel_sha256, traj.entries[0].kernel_sha256)
        self.assertEqual(last.relative_drift, 0.0)
        self.assertGreater(traj.entries[1].relative_drift, 0.0)
        nt.check_trajectory_record(traj.to_dict())

    def test_zero_kernel_repeat_is_a_measured_zero_with_undefined_distance(self) -> None:
        traj = _run(Ignores(), _states(theta=[0.0, 0.0]))
        second = traj.entries[1]
        self.assertEqual(second.stationary_drift, "measured_zero")
        self.assertIsNone(second.distance_from_previous)  # S undefined for the zero kernel
        nt.check_trajectory_record(traj.to_dict())

    def test_irreproducible_remeasurement_is_not_a_measured_zero(self) -> None:
        # Identical bytes, but the backend returns a kernel differing in the last bit.
        real = nt.pn.param_ntk
        calls = []

        def jittered(*args, **kwargs):
            result = real(*args, **kwargs)
            calls.append(result)
            if len(calls) == 2:
                return dataclasses.replace(result, kernel=result.kernel * (1.0 + 2.0 ** -50))
            return result

        with mock.patch.object(nt.pn, "param_ntk", side_effect=jittered):
            traj = _run(Product(), _states(a=[1.0, 1.0], b=[0.5, 0.5]))
        second = traj.entries[1]
        self.assertTrue(second.state_unchanged_since_previous)
        self.assertEqual(second.stationary_drift, "not_measured")
        self.assertNotEqual(second.kernel_sha256, traj.entries[0].kernel_sha256)
        self.assertGreater(second.relative_drift, 0.0)  # the measured number is still reported
        nt.check_trajectory_record(traj.to_dict())

    def test_a_failing_oracle_at_a_repeated_state_yields_no_record(self) -> None:
        # The drift of a repeat is only a measured zero if PARAM_NTK_v1 ran correctly there:
        # an oracle failure at the repeat aborts the trajectory instead of reusing the kernel.
        real = nt.pn.param_ntk
        calls = []

        def fails_second(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise nt.pn.ParamNTKError("autograd and finite differences disagree (injected)")
            return real(*args, **kwargs)

        with mock.patch.object(nt.pn, "param_ntk", side_effect=fails_second):
            with self.assertRaises(nt.pn.ParamNTKError):
                _run(Product(), _states(a=[1.0, 1.0], b=[0.5, 0.5]))
        self.assertEqual(len(calls), 2)  # the repeat was measured, not skipped

    def test_step_guards_still_hold_for_identical_states(self) -> None:
        state = {"a": _f64(1.0), "b": _f64(0.5)}
        for ckpts, declared in (([(0, state), (0, dict(state))], [0, 0]),          # repeated step number
                                ([(10, state), (0, dict(state))], [10, 0]),         # decreasing
                                ([(0, state), (10, dict(state))], [0, 20])):        # not the declared steps
            with self.subTest(steps=[s for s, _ in ckpts], declared=declared):
                with self.assertRaises(nt.NTKTrajectoryError):
                    _run(Product(), ckpts, declared_steps=declared)


class TestRecordCheck(unittest.TestCase):
    """check_trajectory_record: the markers must follow from the record's own digests."""

    def setUp(self) -> None:
        self.record = _run(Product(), _states(a=[1.0, 1.0, 2.0], b=[0.5, 0.5, 0.5])).to_dict()

    def _refused(self, record, fragment: str) -> None:
        with self.assertRaises(nt.NTKTrajectoryError) as ctx:
            nt.check_trajectory_record(record)
        self.assertIn(fragment, str(ctx.exception))

    def test_a_consistent_record_passes(self) -> None:
        self.assertIsNone(nt.check_trajectory_record(self.record))
        self.assertEqual(self.record["entries"][1]["stationary_drift"], "measured_zero")

    def test_unmarked_repeat_is_refused_even_when_resealed(self) -> None:
        for key, value in (("state_unchanged_since_previous", False), ("repeated_state_digest_of", None),
                           ("stationary_drift", None), ("stationary_drift", "not_measured"),
                           ("repeated_state_digest_of", 10), ("state_unchanged_since_previous", 1),
                           ("repeated_state_digest_of", False)):
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(self.record)
                bad["entries"][1][key] = value
                self._refused(_reseal(bad), key)
        bad = copy.deepcopy(self.record)
        del bad["entries"][1]["stationary_drift"]
        self._refused(_reseal(bad), "missing")

    def test_a_marker_on_a_changed_state_is_refused(self) -> None:
        for key, value in (("state_unchanged_since_previous", True), ("repeated_state_digest_of", 0),
                           ("stationary_drift", "measured_zero")):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.record)
                bad["entries"][2][key] = value
                self._refused(_reseal(bad), key)

    def test_measured_zero_needs_equal_kernels_and_zero_distance(self) -> None:
        bad = copy.deepcopy(self.record)
        bad["entries"][1]["kernel_sha256"] = "0" * 64
        self._refused(_reseal(bad), "stationary_drift")
        bad = copy.deepcopy(self.record)
        bad["entries"][1]["distance_from_previous"] = 1e-3
        self._refused(_reseal(bad), "distance_from_previous")

    def test_envelope_is_checked(self) -> None:
        cases = {
            "record_sha256": dict(self.record, run_id="other"),             # edited, not resealed
            "authenticity": _reseal(dict(self.record, authenticity="attested")),
            "policy": _reseal(dict(self.record, policy="NTK_TRAJECTORY_v0")),
            "strictly increasing": _reseal(dict(self.record, declared_steps=[0, 0, 20])),
            "declared steps": _reseal(dict(self.record, declared_steps=[0, 10, 30])),
            "entries must be": _reseal(dict(self.record, entries="none")),
            "canonical JSON": dict(self.record, trace=math.nan),
        }
        for fragment, record in cases.items():
            with self.subTest(fragment=fragment):
                self._refused(record, fragment)
        self._refused([self.record], "mapping")
        bad = copy.deepcopy(self.record)
        bad["entries"][0]["state_sha256"] = None
        self._refused(_reseal(bad), "must be strings")
        bad = copy.deepcopy(self.record)
        bad["entries"][1]["step"] = True
        bad["declared_steps"] = [0, 1, 20]
        self._refused(_reseal(bad), "declared steps")


class TestFailClosed(unittest.TestCase):
    def _ok(self):
        return _states(a=[1.0, 2.0, 3.0], b=[0.5, 0.5, 0.5])

    def _refused(self, *args, error=nt.NTKTrajectoryError, **kw):
        with self.assertRaises(error) as ctx:
            _run(*args, **kw)
        return str(ctx.exception)

    def test_steps_must_equal_the_declared_steps(self) -> None:
        ckpts = self._ok()
        for declared in ([0, 10], [0, 10, 20, 30], [0, 20, 10], [0, 10, 21]):
            with self.subTest(declared=declared):
                msg = self._refused(Product(), ckpts, declared_steps=declared)
                self.assertTrue("declared" in msg or "increasing" in msg, msg)
        self._refused(Product(), [ckpts[0], ckpts[2], ckpts[1]], declared_steps=[0, 20, 10])

    def test_declared_steps_contract(self) -> None:
        ckpts = self._ok()
        for declared in ([], [0, 0, 10], [True, 10, 20], [-1, 10, 20], [0.0, 10, 20], "abc", None):
            with self.subTest(declared=declared):
                self._refused(Product(), ckpts, declared_steps=declared)
        with mock.patch.object(nt, "MAX_CHECKPOINTS", 2):
            self.assertIn("at most 2", self._refused(Product(), ckpts))

    def test_numpy_integer_steps_are_accepted(self) -> None:
        ckpts = [(np.int64(s), st) for s, st in self._ok()]
        traj = _run(Product(), ckpts, declared_steps=[np.int32(0), 10, 20])
        self.assertEqual(traj.declared_steps, (0, 10, 20))
        self.assertIs(type(traj.entries[1].step), int)

    def test_state_must_match_the_model_exactly(self) -> None:
        good = self._ok()
        cases = {
            "missing": {"a": _f64(1.0)},
            "extra": {"a": _f64(1.0), "b": _f64(2.0), "c": _f64(0.0)},
            "dtype": {"a": torch.tensor(1.0, dtype=torch.float32), "b": _f64(2.0)},
            "shape": {"a": _f64([1.0]), "b": _f64(2.0)},
            "nonfinite": {"a": _f64(math.nan), "b": _f64(2.0)},
            "not a tensor": {"a": 1.0, "b": _f64(2.0)},
        }
        for name, state in cases.items():
            with self.subTest(case=name):
                self._refused(Product(), [good[0], (10, state)])
        self._refused(Product(), [good[0], (10, [("a", _f64(1.0))])])

    def test_checkpoints_must_be_step_state_pairs(self) -> None:
        for ckpts in ({0: {}}, "ckpt", [(0, {"a": _f64(1.0), "b": _f64(1.0)}, "extra")], [[0, {}]]):
            with self.subTest(ckpts=type(ckpts).__name__):
                self._refused(Product(), ckpts, declared_steps=[0])

    def test_model_inputs_and_run_id(self) -> None:
        ckpts = self._ok()
        training = Product().train()
        self.assertIn("training mode", self._refused(training, ckpts))
        self._refused(Product(), ckpts, x=np.full(M, np.inf))
        self._refused(Product(), ckpts, run_id="  ")
        self._refused(Product(), ckpts, run_id=None)
        with self.assertRaises(TypeError):
            nt.ntk_trajectory(Product(), apply, U, MODES, WEIGHTS, ckpts, declared_steps=[0, 10, 20], run_id="r")
        with self.assertRaises(TypeError):
            nt.ntk_trajectory(object(), apply, _f64(U), MODES, WEIGHTS, ckpts, declared_steps=[0, 10, 20],
                              run_id="r")
        with self.assertRaises(TypeError):
            nt.ntk_trajectory(Product(), None, _f64(U), MODES, WEIGHTS, ckpts, declared_steps=[0, 10, 20],
                              run_id="r")

    def test_forward_that_mutates_the_input_is_refused(self) -> None:
        self.assertIn("mutated", self._refused(Mutating(), self._ok()))

    def test_target_and_basis_contract(self) -> None:
        ckpts = self._ok()
        for target in ([1.0, 2.0], [0.0, 0.0, 0.0], [1.0, math.nan, 0.0]):
            with self.subTest(target=target):
                self._refused(Product(), ckpts, target_coefficients=target, error=ValueError)
        self._refused(Product(), ckpts, modes=2.0 * MODES, error=ValueError)  # not W-orthonormal
        with mock.patch.object(nt, "MAX_MODES", 2):
            self.assertIn("at most 2 modes", self._refused(Product(), ckpts))

    def test_detaching_forward_is_refused_by_param_ntk(self) -> None:
        # The trajectory inherits PARAM_NTK_v1's autograd-vs-FD guard at every checkpoint.
        class Detached(Product):
            def forward(self, x):
                return self.a.detach() * self.b.detach() * x + 0.0 * (self.a + self.b)

        with self.assertRaises(nt.pn.ParamNTKError):
            _run(Detached(), self._ok())


class TestPreflightBinding(unittest.TestCase):
    """HNT-R85-01 (cross-family GPT-6 R85/R86/R87): every digest is taken in a preflight
    loop, every state is loaded in a later one. What is loaded must be the bytes the digest
    names, also when the caller's checkpoint objects change in between."""

    def _future_mutating_apply(self, future: dict, mutate: bool):
        calls = []

        def hooked(model, x):
            calls.append(1)
            if mutate and len(calls) == 1:
                future["a"].fill_(3.0)  # during step 0, after every digest exists
            return apply(model, x)
        return hooked

    def test_checkpoint_changed_after_preflight_is_refused(self) -> None:
        ckpts = _states(a=[1.0, 2.0], b=[0.5, 0.5])
        preflight = nt.state_dict_sha256(ckpts[1][1])
        with self.assertRaises(nt.NTKTrajectoryError) as ctx:
            _run(Product(), ckpts, apply=self._future_mutating_apply(ckpts[1][1], True))
        self.assertIn("changed since the preflight digest", str(ctx.exception))
        # The poisoning really happened: the caller's later checkpoint has other bytes now.
        self.assertNotEqual(nt.state_dict_sha256(ckpts[1][1]), preflight)

    def test_same_hook_without_mutation_is_measured(self) -> None:
        # Positive control: the hook itself is harmless; only the mutation is refused.
        ckpts = _states(a=[1.0, 2.0], b=[0.5, 0.5])
        traj = _run(Product(), ckpts, apply=self._future_mutating_apply(ckpts[1][1], False))
        reference = _run(Product(), _states(a=[1.0, 2.0], b=[0.5, 0.5]))
        for (_, state), e, r in zip(ckpts, traj.entries, reference.entries):
            self.assertEqual(e.state_sha256, nt.state_dict_sha256(state))
            self.assertEqual(e.kernel_sha256, r.kernel_sha256)

    def test_mapping_whose_reads_differ_is_refused(self) -> None:
        # A Mapping that serves other bytes once read often enough: what was digested is
        # not what would be loaded. The state is read exactly once for the load.
        from collections.abc import Mapping

        class Drifting(Mapping):
            def __init__(self, early, late, after):
                self._early, self._late, self._after, self._reads = early, late, after, 0

            def __getitem__(self, key):
                self._reads += 1
                return (self._early if self._reads <= self._after else self._late)[key]

            def __iter__(self):
                return iter(self._early)

            def __len__(self):
                return len(self._early)

        early = {"a": _f64(2.0), "b": _f64(0.5)}
        late = {"a": _f64(5.0), "b": _f64(0.5)}
        # Two keys: _check_state reads 2, the preflight digest reads 2 more.
        drifting = Drifting(early, late, after=4)
        with self.assertRaises(nt.NTKTrajectoryError) as ctx:
            _run(Product(), [(0, {"a": _f64(1.0), "b": _f64(0.5)}), (10, drifting)])
        self.assertIn("changed since the preflight digest", str(ctx.exception))
        # Positive control: a Mapping with stable reads is measured.
        steady = Drifting(early, early, after=4)
        traj = _run(Product(), [(0, {"a": _f64(1.0), "b": _f64(0.5)}), (10, steady)])
        self.assertEqual(traj.entries[1].state_sha256, nt.state_dict_sha256(early))


def _v1_metadata_json(value) -> bytes:
    """Independent oracle: the NTK_TRAJECTORY_v1 metadata encoding as first specified
    (canonical JSON, callables as {"__qualified_name__": module.qualname})."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=lambda o: {"__qualified_name__": f"{o.__module__}.{o.__qualname__}"}).encode("utf-8")


class TestMetadataAliasing(unittest.TestCase):
    """HNT-R86-01 (cross-family GPT-6 R86/R87): JSON erased the type of what it encoded.
    A data dict {"__qualified_name__": name} and a str key "1" produced the bytes of the
    callable and of the int key 1, so the digest and the architecture check bound a
    different subject. Such entries are now refused; every other entry keeps its bytes."""

    def test_a_data_dict_cannot_pose_as_a_callable(self) -> None:
        name = f"{torch.tanh.__module__}.{torch.tanh.__qualname__}"
        genuine = {"w": _f64(1.0), "_metadata": {"act": torch.tanh}}
        for alias in ({"act": {"__qualified_name__": name}}, {"act": [{"__qualified_name__": name}]}):
            with self.subTest(alias=alias):
                with self.assertRaises(nt.NTKTrajectoryError) as ctx:
                    nt.state_dict_sha256({"w": _f64(1.0), "_metadata": alias})
                self.assertIn("reserved", str(ctx.exception))
        nt.state_dict_sha256(genuine)  # the callable itself still binds

    def test_the_alias_is_refused_as_a_checkpoint(self) -> None:
        name = f"{torch.tanh.__module__}.{torch.tanh.__qualname__}"
        meta = dict(WithMetadata.META, act={"__qualified_name__": name})
        ckpts = [(s, dict(st, _metadata=meta)) for s, st in _states(a=[1.0, 2.0], b=[0.5, 0.5])]
        with self.assertRaises(nt.NTKTrajectoryError):
            _run(WithMetadata(), ckpts)

    def test_non_string_keys_are_refused(self) -> None:
        for meta in ({1: "x"}, {True: "x"}, {None: "x"}, {"outer": {2.5: "x"}}):
            with self.subTest(meta=meta):
                with self.assertRaises(nt.NTKTrajectoryError):
                    nt.state_dict_sha256({"_metadata": meta})

    def test_deep_or_cyclic_metadata_is_a_contract_error(self) -> None:
        deep: list = []
        node = deep
        for _ in range(5000):
            node.append([])
            node = node[0]
        cyclic: dict = {"width": 3}
        cyclic["self"] = cyclic
        for meta in (deep, cyclic):
            with self.subTest(kind=type(meta).__name__):
                with self.assertRaises(nt.NTKTrajectoryError):
                    nt.state_dict_sha256({"_metadata": meta})

    def test_bytes_of_valid_metadata_are_unchanged(self) -> None:
        # Positive control: digests already written under v1 stay valid (no migration).
        cases = (WithMetadata.META,
                 {"n_modes": (16, 16), "fno": {"act": torch.tanh, "skip": "linear", "rank": 1.0},
                  "flags": [True, False, None], "eps": 1e-12, "name": "gino", "seed": -3},
                 {}, [], 7, "plain", None, 0.5)
        for meta in cases:
            with self.subTest(meta=repr(meta)[:40]):
                self.assertEqual(nt._metadata_bytes("_metadata", meta), _v1_metadata_json(meta))


def _v1_drift(k: np.ndarray, k0: np.ndarray) -> float:
    scale = float(np.abs(k0).max())
    return float(np.linalg.norm((k - k0) / scale)) / float(np.linalg.norm(k0 / scale))


class TestDriftRange(unittest.TestCase):
    """HNT-R86-02 (cross-family GPT-6 R86/R87): finite PSD kernels made the v1 formula
    return inf (subtract, then divide by a tiny anchor scale, then square in the norm),
    which the JSON record then could not hold. A representable drift is now computed, an
    unrepresentable one is a contract error; the value is unchanged wherever v1 was finite."""

    def test_unrepresentable_drift_is_a_contract_error(self) -> None:
        with self.assertRaises(nt.NTKTrajectoryError) as ctx:
            nt._relative_drift(np.diag([1e100, 0.0]), np.diag([1e-300, 0.0]))
        self.assertIn("binary64", str(ctx.exception))

    def test_representable_huge_drift_is_computed(self) -> None:
        # True drift (1e-100 - 1e-300) / 1e-300 = 1e200 - 1: representable, v1 gave inf.
        drift = nt._relative_drift(np.diag([1e-100, 0.0]), np.diag([1e-300, 0.0]))
        self.assertTrue(math.isfinite(drift))
        self.assertLess(abs(drift / 1e200 - 1.0), 1e-15)
        # Huge entries of opposite sign: k - k0 itself overflows; ||[[0,-2],[-2,0]]|| / ||ones|| = sqrt(2).
        k = np.array([[1e308, -1e308], [-1e308, 1e308]])
        k0 = np.array([[1e308, 1e308], [1e308, 1e308]])
        self.assertLess(abs(nt._relative_drift(k, k0) - math.sqrt(2.0)), 4e-16)

    def test_drift_where_v1_was_finite_is_bit_identical(self) -> None:
        rng = np.random.default_rng(86)
        for n in (1, 2, 3, 8):
            for log_scale in (-150, -20, 0, 20, 150):
                for _ in range(10):
                    a, b = rng.standard_normal((2, n, n))
                    k0, k = (a @ a.T) * 10.0 ** log_scale, (b @ b.T) * 10.0 ** (log_scale + rng.integers(-3, 4))
                    with self.subTest(n=n, log_scale=log_scale):
                        self.assertEqual(nt._relative_drift(k, k0), _v1_drift(k, k0))
        self.assertIsNone(nt._relative_drift(np.eye(2), np.zeros((2, 2))))

    def test_intermediate_overflow_of_the_factored_ratio(self) -> None:
        # Codex #83 (8761e55): m / scale = 2 / 1e-308 overflows, the product with the norm
        # factor 0.5 does not. True drift ||k - k0|| / ||k0|| = 4 / 4e-308 = 1e308.
        k0, k = 1e-308 * np.eye(16), np.diag([4.0] + [0.0] * 15)
        drift = nt._relative_drift(k, k0)
        self.assertLess(abs(drift / 1e308 - 1.0), 1e-12)


class TestCodexRound8761e55(unittest.TestCase):
    """Codex review of #83 at 8761e55: the basis and target arrays, the checked snapshot
    and the off-diagonal energy."""

    def test_basis_changed_by_apply_is_not_used(self) -> None:
        modes = np.array(MODES, dtype=np.float64)  # float64: np.asarray would alias it
        calls = []

        def swapping(model, x):
            calls.append(1)
            if len(calls) == 1:
                modes[:, [0, 1]] = modes[:, [1, 0]]  # still W-orthonormal, other basis
            return apply(model, x)

        same = _states(a=[1.0, 1.0], b=[0.5, 0.5])
        traj = _run(Product(), same, modes=modes, apply=swapping)
        reference = _run(Product(), _states(a=[1.0, 1.0], b=[0.5, 0.5]))
        self.assertFalse(np.array_equal(modes, MODES))  # the poisoning happened
        self.assertEqual(traj.basis_sha256, reference.basis_sha256)
        self.assertEqual([e.kernel_sha256 for e in traj.entries], [e.kernel_sha256 for e in reference.entries])
        self.assertEqual(traj.entries[1].distance_from_initial, 0.0)

    def test_target_changed_by_apply_is_not_bound(self) -> None:
        target = np.array([1.0, 0.5, 0.25])
        calls = []

        def mutating(model, x):
            calls.append(1)
            if len(calls) == 1:
                target[:] = [0.0, 0.0, 9.0]
            return apply(model, x)

        traj = _run(Product(), _states(a=[1.0, 2.0], b=[0.5, 0.5]), target_coefficients=target, apply=mutating)
        reference = _run(Product(), _states(a=[1.0, 2.0], b=[0.5, 0.5]),
                         target_coefficients=np.array([1.0, 0.5, 0.25]))
        self.assertEqual(target.tolist(), [0.0, 0.0, 9.0])
        self.assertEqual(traj.target_sha256, reference.target_sha256)
        self.assertEqual([e.target_alignment for e in traj.entries], [e.target_alignment for e in reference.entries])

    def test_mapping_that_changes_before_the_preflight_digest_is_checked_again(self) -> None:
        from collections.abc import Mapping

        class Switching(Mapping):
            """Valid values for the first `after` reads (the preflight check), then others."""

            def __init__(self, early, late, after):
                self._early, self._late, self._after, self._reads = early, late, after, 0

            def __getitem__(self, key):
                self._reads += 1
                return (self._early if self._reads <= self._after else self._late)[key]

            def __iter__(self):
                return iter(self._early)

            def __len__(self):
                return len(self._early)

        early = {"a": _f64(2.0), "b": _f64(0.5)}
        late = {"a": torch.tensor(2.0, dtype=torch.float32), "b": _f64(0.5)}  # load_state_dict would cast
        first = (0, {"a": _f64(1.0), "b": _f64(0.5)})
        with self.assertRaises(nt.NTKTrajectoryError):
            _run(Product(), [first, (10, Switching(early, late, after=2))])
        # Positive control: the same switch AFTER the preflight digest is caught as before,
        # and a stable Mapping is measured.
        with self.assertRaises(nt.NTKTrajectoryError):
            _run(Product(), [first, (10, Switching(early, late, after=4))])
        traj = _run(Product(), [first, (10, Switching(early, early, after=2))])
        self.assertEqual(traj.entries[1].state_sha256, nt.state_dict_sha256(early))

    def test_weak_mode_coupling_is_not_cancelled(self) -> None:
        k = np.array([[1.0, 1e-9], [1e-9, 1.0]])
        offdiag = nt._shape_metrics(k, None)[1]
        self.assertLess(abs(offdiag / (2e-18 / (2.0 + 2e-18)) - 1.0), 1e-12)
        self.assertEqual(nt._shape_metrics(np.eye(3), None)[1], 0.0)


if __name__ == "__main__":
    unittest.main()
