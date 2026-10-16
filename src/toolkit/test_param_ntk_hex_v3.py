"""Tests for param_ntk_hex_v3 (PARAM_NTK_v1): analytic kernels, real/complex
coordinates, PSD, float32 cross-check and the two fail-closed guards."""
from __future__ import annotations

import unittest

import numpy as np
import torch

import param_ntk_hex_v3 as pn

M, K = 6, 3
RNG = np.random.default_rng(7)
WEIGHTS = RNG.uniform(0.5, 2.0, M)
# W-orthonormal modes: phi^T W phi = I
_raw = RNG.standard_normal((M, K))
_q, _ = np.linalg.qr(np.sqrt(WEIGHTS)[:, None] * _raw)
MODES = _q / np.sqrt(WEIGHTS)[:, None]
PROJ = (WEIGHTS[:, None] * MODES).T  # (K, M)


class Linear(torch.nn.Module):
    """y = A theta: the parameter NTK is exactly A A^T (projected: P A A^T P^T)."""

    def __init__(self, a: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("a", torch.tensor(a))
        self.theta = torch.nn.Parameter(torch.tensor(RNG.standard_normal(a.shape[1])))
        self.eval()  # param_ntk measures eval-mode models only

    def forward(self) -> torch.Tensor:
        return self.a @ self.theta


class ComplexScale(torch.nn.Module):
    """y = Re(c) u + Im(c) u^2 with one complex parameter c: J = [u, u^2] in real coordinates."""

    def __init__(self, u: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("u", torch.tensor(u))
        self.c = torch.nn.Parameter(torch.tensor(0.3 - 0.7j, dtype=torch.complex64))
        self.eval()

    def forward(self) -> torch.Tensor:
        return self.c.real * self.u + self.c.imag * self.u**2


class Detaching(Linear):
    """Uses its parameter through .detach(): autograd sees no dependence, the output does."""

    def forward(self) -> torch.Tensor:
        return self.a @ self.theta.detach() + 0.0 * self.theta.sum()


def _run(model, **kw):
    return pn.param_ntk(model, lambda m: m(), MODES, WEIGHTS, **kw)


class TestParamNTK(unittest.TestCase):
    def test_linear_model_matches_the_analytic_kernel(self) -> None:
        a = RNG.standard_normal((M, 4))
        result = _run(Linear(a))
        expected = PROJ @ a @ a.T @ PROJ.T
        np.testing.assert_allclose(result.kernel, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(result.n_real_params, 4)
        self.assertLessEqual(result.fd_relative_error, pn.FD_REL_TOL)

    def test_complex_parameter_is_resolved_into_real_coordinates(self) -> None:
        u = RNG.uniform(-1.0, 1.0, M)
        result = _run(ComplexScale(u))
        jac = np.stack([u, u**2], axis=1)  # d y / d (Re c, Im c)
        np.testing.assert_allclose(result.kernel, PROJ @ jac @ jac.T @ PROJ.T, rtol=1e-10, atol=1e-12)
        self.assertEqual(result.n_real_params, 2)

    def test_kernel_is_symmetric_psd_and_float32_close(self) -> None:
        result = _run(Linear(RNG.standard_normal((M, 5))))
        np.testing.assert_array_equal(result.kernel, result.kernel.T)
        self.assertGreaterEqual(result.eigenvalues[0], -1e-12 * result.eigenvalues[-1])
        self.assertIsNotNone(result.float32_relative_deviation)
        self.assertLess(result.float32_relative_deviation, 1e-5)
        self.assertGreater(result.float32_relative_deviation, 0.0)  # it really measured float32

    def test_measurement_never_changes_the_callers_model(self) -> None:
        model = Linear(RNG.standard_normal((M, 3))).float()
        before = model.theta.detach().clone()
        _run(model)
        self.assertEqual(model.theta.dtype, torch.float32)
        self.assertTrue(torch.equal(model.theta.detach(), before))

    def test_partial_parameter_list_is_refused(self) -> None:
        class Two(Linear):
            def __init__(self, a):
                super().__init__(a)
                self.bias = torch.nn.Parameter(torch.zeros(M, dtype=torch.float64))

            def forward(self):
                return super().forward() + self.bias

        model = Two(RNG.standard_normal((M, 3)))
        _run(model, params=list(model.parameters()))  # positive control: the full list
        with self.assertRaisesRegex(pn.ParamNTKError, "sub-model"):
            _run(model, params=[model.theta])

    def test_forward_outside_autograd_is_refused(self) -> None:
        with self.assertRaisesRegex(pn.ParamNTKError, "finite differences"):
            _run(Detaching(RNG.standard_normal((M, 3))))

    def test_float32_mode_assembles_in_float64_and_still_refuses_detach(self) -> None:
        # Backends without float64 (neuraloperator 2.0.0 GINO): float32 model, float64 assembly.
        a = RNG.standard_normal((M, 4))
        result = _run(Linear(a), model_dtype="float32")
        expected = PROJ @ a @ a.T @ PROJ.T
        np.testing.assert_allclose(result.kernel, expected, rtol=1e-5, atol=1e-6)
        self.assertLessEqual(result.fd_relative_error, pn.FD_REL_TOL_FLOAT32)
        with self.assertRaisesRegex(pn.ParamNTKError, "finite differences"):
            _run(Detaching(a), model_dtype="float32")
        with self.assertRaises(ValueError):
            _run(Linear(a), model_dtype="float16")

    def test_fd_check_validates_the_kernel_rows_mode_by_mode(self) -> None:
        """A defect in the rows that build Theta (here: two modes swapped) must be
        refused; the earlier scalar check used its own autograd pass and could not
        see it (PR #71, 007acc2 v6 finding "all K modes per direction")."""
        model = Linear(RNG.standard_normal((M, 4)))
        original = pn._gram_rows

        def swapped(m, f, p):
            rows, params = original(m, f, p)
            return rows[[1, 0, *range(2, rows.shape[0])]], params

        pn._gram_rows = swapped
        try:
            with self.assertRaisesRegex(pn.ParamNTKError, "disagree"):
                _run(model, float32_crosscheck=False)
        finally:
            pn._gram_rows = original

    def test_training_mode_is_refused(self) -> None:
        # Codex P1 on #71: dropout/batch-norm in training mode would make the kernel random.
        model = Linear(RNG.standard_normal((M, 3)))
        _run(model)  # positive control: eval mode
        model.train()
        with self.assertRaisesRegex(pn.ParamNTKError, "training mode"):
            _run(model)
        model.eval()
        model.add_module("drop", torch.nn.Dropout(0.5))  # a submodule left in training mode
        model.drop.train()
        with self.assertRaisesRegex(pn.ParamNTKError, "training mode"):
            _run(model)

    def test_modes_must_be_w_orthonormal(self) -> None:
        # Codex P1 on #71: raw (unnormalized) modes rescale rows and eigenvalues silently.
        model = Linear(RNG.standard_normal((M, 3)))
        with self.assertRaisesRegex(ValueError, "W-orthonormal"):
            pn.param_ntk(model, lambda m: m(), 2.0 * MODES, WEIGHTS)
        with self.assertRaisesRegex(ValueError, "W-orthonormal"):
            pn.param_ntk(model, lambda m: m(), MODES, 2.0 * WEIGHTS)

    def test_shape_and_basis_contract(self) -> None:
        with self.assertRaises(pn.ParamNTKError):
            pn.param_ntk(Linear(RNG.standard_normal((M, 2))), lambda m: m()[:-1], MODES, WEIGHTS)
        with self.assertRaises(ValueError):
            pn.param_ntk(Linear(RNG.standard_normal((M, 2))), lambda m: m(), MODES, -WEIGHTS)


if __name__ == "__main__":
    unittest.main()
