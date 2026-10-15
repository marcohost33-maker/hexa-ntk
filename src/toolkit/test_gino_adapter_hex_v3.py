"""Contract and optional-backend smoke tests for gino_adapter_hex_v3."""
from __future__ import annotations

import importlib.util
import importlib.metadata
import hashlib
import json
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gino_adapter_hex_v3 import (
    GINO_BACKEND_PIN,
    GinoPhase1Config,
    build_gino,
    predict_gino,
    prepare_gino_batch,
)
from hex_fv_solver_hex_v3 import HexLattice, torus_hex_lattice, triangle_hex_lattice


class TestGinoPhase1Adapter(unittest.TestCase):
    def test_config_validation_and_digest(self) -> None:
        cfg = GinoPhase1Config()
        self.assertEqual(cfg.canonical()["backend"], GINO_BACKEND_PIN)
        expected = {
            "backend": "neuraloperator==2.0.0",
            "resource_policy": "GINO_ADAPTER_PHASE1_v2",
            "geometry_semantics": "shared-euclidean-v1",
            "latent_resolution": 32,
            "n_modes": 12,
            "hidden_channels": 64,
            "n_layers": 4,
            "radius_factor": 2.5,
        }
        self.assertEqual(cfg.canonical(), expected)
        digest = hashlib.sha256(
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(cfg.digest(), digest)
        for field, other in (
            ("latent_resolution", 64), ("n_modes", 11),
            ("hidden_channels", 32), ("n_layers", 3),
            ("radius_factor", 3.0),
        ):
            modified = GinoPhase1Config(**{field: other})
            self.assertNotEqual(modified.digest(), digest, field)
        for wrong_type in (True, "2.5", complex(2.5, 0.0)):
            with self.subTest(radius=wrong_type), self.assertRaises(TypeError):
                GinoPhase1Config(radius_factor=wrong_type)
        with self.assertRaises(ValueError):
            GinoPhase1Config(radius_factor=10 ** 500)
        for kwargs in (
            {"latent_resolution": 7},
            {"n_modes": 16},
            {"hidden_channels": 0},
            {"n_layers": 0},
            {"radius_factor": 0.0},
            {"radius_factor": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GinoPhase1Config(**kwargs)

    def test_reject_excessive_architecture_before_allocating_tensor(self) -> None:
        # Prevent hostile latent-grid/model hyperparameters from exhausting RAM.
        for kwargs in (
            {"latent_resolution": 10**10},
            {"hidden_channels": 10**10},
            {"n_layers": 10**10},
            {"radius_factor": 10**100},
        ):
            with self.subTest(kwargs=kwargs), patch(
                "torch.linspace", side_effect=AssertionError("allocated latent grid")
            ), self.assertRaisesRegex(ValueError, "resource limit"):
                GinoPhase1Config(**kwargs)

    def test_joint_dense_spectral_budget_rejects_giant_parameters(self) -> None:
        # Every individual ceiling passes, but their combination would
        # allocate >60 GiB of dense complex spectral weights alone.
        kwargs = dict(latent_resolution=128, n_modes=63, hidden_channels=512, n_layers=16)
        with self.assertRaisesRegex(ValueError, "combined spectral weight resource limit"):
            GinoPhase1Config(**kwargs)
        # The check runs *before* any backend module construction.
        with patch("gino_adapter_hex_v3.importlib.metadata.version", side_effect=AssertionError("imported")):
            with self.assertRaisesRegex(ValueError, "combined spectral weight resource limit"):
                GinoPhase1Config(**kwargs)

    def test_dense_spectral_budget_boundary_and_digest_policy(self) -> None:
        # Base/preregistered geometry and architecture remain admissible.
        self.assertEqual(GinoPhase1Config().canonical()["resource_policy"], "GINO_ADAPTER_PHASE1_v2")
        GinoPhase1Config(latent_resolution=128, n_modes=12, hidden_channels=128, n_layers=8)
        # Exercise a 256 MiB crossing without OOM, including odd-count rounding.
        with self.assertRaisesRegex(ValueError, "combined spectral weight resource limit"):
            GinoPhase1Config(latent_resolution=128, n_modes=63, hidden_channels=128, n_layers=16)
        # Non-factorized complex64 estimate is a *lower* bound on total
        # train-time GPU memory. Prove the default lies inside the limit.
        m, c, layers = 12, 64, 4
        bytes_est = layers * c**2 * m * (m // 2 + 1) * 8
        self.assertLess(bytes_est, 256 * 1024 * 1024)

    def test_geometry_and_batch_contract(self) -> None:
        lattice = triangle_hex_lattice(8)
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3)
        batch = prepare_gino_batch(lattice, np.ones((2, lattice.num_nodes)), cfg)
        self.assertEqual(tuple(batch["input_geom"].shape), (1, lattice.num_nodes, 2))
        self.assertEqual(tuple(batch["latent_queries"].shape), (1, 8, 8, 2))
        self.assertEqual(tuple(batch["output_queries"].shape), (1, lattice.num_nodes, 2))
        self.assertEqual(tuple(batch["x"].shape), (2, lattice.num_nodes, 1))
        self.assertTrue(torch.isfinite(batch["latent_queries"]).all())
        grid = batch["latent_queries"][0]
        self.assertEqual(tuple(grid[0, 0].tolist()), (0.0, 0.0))
        self.assertAlmostEqual(float(grid[-1, -1, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(grid[-1, -1, 1]), np.sqrt(3) / 2, places=6)
        self.assertAlmostEqual(float(grid[1, 0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(grid[0, 1, 0]), 0.0, places=6)
        self.assertGreater(float(grid[1, 0, 0]), 0.0)
        self.assertGreater(float(grid[0, 1, 1]), 0.0)

    def test_fail_closed_inputs(self) -> None:
        lattice = triangle_hex_lattice(8)
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3)
        with self.assertRaises(ValueError):
            prepare_gino_batch(lattice, np.zeros(lattice.num_nodes + 1), cfg)
        bad = np.zeros(lattice.num_nodes)
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            prepare_gino_batch(lattice, bad, cfg)
        with self.assertRaises(TypeError):
            prepare_gino_batch(lattice, np.zeros(lattice.num_nodes), object())

    def test_factory_and_predict_contract_with_fake_backend(self) -> None:
        lattice = triangle_hex_lattice(32)
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3, hidden_channels=8, n_layers=1)

        class FakeGINO:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __call__(self, **kwargs):
                return kwargs["x"]

        root = types.ModuleType("neuralop")
        models = types.ModuleType("neuralop.models")
        models.GINO = FakeGINO
        root.models = models
        with patch.dict(sys.modules, {"neuralop": root, "neuralop.models": models}), patch(
            "gino_adapter_hex_v3.importlib.metadata.version", return_value="2.0.0"
        ):
            model = build_gino(cfg, lattice)

        self.assertEqual(model.kwargs["gno_coord_dim"], 2)
        self.assertFalse(model.kwargs["gno_use_open3d"])
        self.assertFalse(model.kwargs["gno_use_torch_scatter"])
        self.assertAlmostEqual(model.kwargs["in_gno_radius"], 2.5 * lattice.h)
        out = predict_gino(model, lattice, np.ones((2, lattice.num_nodes)), cfg)
        self.assertEqual(tuple(out.shape), (2, lattice.num_nodes))
        # Codex P2 on #61: a model built with config A must not run under config B
        # (B's latent grid would be used while A's modes/channels/radius stay baked in).
        other = GinoPhase1Config(latent_resolution=16, n_modes=3, hidden_channels=8, n_layers=1)
        with self.assertRaisesRegex(ValueError, "construction config"):
            predict_gino(model, lattice, np.ones((2, lattice.num_nodes)), other)

    def test_reject_wrong_training_resolution_and_backend_version(self) -> None:
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3)
        for n in (8, 16, 64):
            with self.subTest(n=n), self.assertRaisesRegex(ValueError, "N=32"):
                build_gino(cfg, triangle_hex_lattice(n))
        with self.assertRaisesRegex(ValueError, "N=32"):
            build_gino(cfg, torus_hex_lattice(32))
        with patch("gino_adapter_hex_v3.importlib.metadata.version", return_value="2.0.1"):
            with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                build_gino(cfg, triangle_hex_lattice(32))
        with patch(
            "gino_adapter_hex_v3.importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            with self.assertRaisesRegex(RuntimeError, "optional backend missing"):
                build_gino(cfg, triangle_hex_lattice(32))

    def test_reject_fabricated_topology(self) -> None:
        lat = triangle_hex_lattice(8)
        neighbors = lat.neighbors.copy()
        neighbors[0, 0] = 0
        forged = HexLattice(lat.n, lat.h, lat.domain, lat.coords, neighbors)
        with self.assertRaisesRegex(ValueError, "canonical"):
            prepare_gino_batch(forged, np.ones(lat.num_nodes), GinoPhase1Config())

    def test_model_dtype_contract_and_gradient_propagation(self) -> None:
        lattice = triangle_hex_lattice(8)
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3)

        class CheckedDouble(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)

            def forward(self, **batch):
                self.assert_dtype(batch["x"], batch["input_geom"], batch["latent_queries"])
                return self.linear(batch["x"])

            @staticmethod
            def assert_dtype(*args):
                if any(t.dtype != torch.float64 for t in args):
                    raise AssertionError("adapter failed to match model dtype")

        model = CheckedDouble()
        u0 = torch.ones((1, lattice.num_nodes), dtype=torch.float64, requires_grad=True)
        out = predict_gino(model, lattice, u0, cfg)
        self.assertEqual(out.dtype, torch.float64)
        out.sum().backward()
        self.assertIsNotNone(model.linear.weight.grad)
        self.assertTrue(bool(torch.isfinite(model.linear.weight.grad).all()))
        self.assertIsNotNone(u0.grad)
        self.assertGreater(float(torch.sum(torch.abs(u0.grad))), 0.0)

    def test_conversion_underflow_bool_complex_and_caller_alias(self) -> None:
        lattice = triangle_hex_lattice(8)
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3)
        with self.assertRaisesRegex(ValueError, "underflows"):
            prepare_gino_batch(lattice, np.full(lattice.num_nodes, 1e-60), cfg)
        for bad in (
            np.ones(lattice.num_nodes, dtype=bool),
            np.ones(lattice.num_nodes, dtype=np.complex128),
        ):
            with self.subTest(dtype=str(bad.dtype)), self.assertRaises(TypeError):
                prepare_gino_batch(lattice, bad, cfg)
        for bad in (torch.float16, torch.int32, torch.complex64):
            with self.subTest(dtype=str(bad)), self.assertRaises(TypeError):
                prepare_gino_batch(lattice, np.ones(lattice.num_nodes), cfg, dtype=bad)
        supplied = torch.ones((1, lattice.num_nodes))
        batch = prepare_gino_batch(lattice, supplied, cfg)
        supplied[0, 0] = 42
        self.assertEqual(float(batch["x"][0, 0, 0]), 1.0)

    def test_model_mixed_floating_types_fail_closed(self) -> None:
        lattice = triangle_hex_lattice(8)
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3)

        class MixedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1, dtype=torch.float32)
                self.register_buffer("other", torch.ones(1, dtype=torch.float64))

            def forward(self, **batch):
                raise AssertionError("mixed dtype should fail before model invocation")

        with self.assertRaisesRegex(ValueError, "inconsistent floating dtypes"):
            predict_gino(MixedModel(), lattice, np.ones(lattice.num_nodes), cfg)

    def test_reject_oversized_n_without_allocating_lattice(self) -> None:
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3)
        for domain in ("triangle_dirichlet", "torus_periodic"):
            forged = HexLattice(10**10, 1e-10, domain, np.zeros((1, 2)), np.zeros((1, 6), dtype=np.int64))
            with self.subTest(domain=domain), patch(
                "gino_adapter_hex_v3.triangle_hex_lattice",
                side_effect=AssertionError("must not allocate"),
            ), patch(
                "gino_adapter_hex_v3.torus_hex_lattice",
                side_effect=AssertionError("must not allocate"),
            ):
                with self.assertRaisesRegex(ValueError, "resource limit"):
                    prepare_gino_batch(forged, np.zeros(1), cfg)

    def test_reject_shape_dtype_and_spacing_before_factory(self) -> None:
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3)
        base = triangle_hex_lattice(8)
        for altered in (
            HexLattice(32, 1 / 32, "triangle_dirichlet", base.coords, base.neighbors),
            HexLattice(8, base.h, base.domain, base.coords.astype(np.float32), base.neighbors),
            HexLattice(8, base.h, base.domain, base.coords, base.neighbors.astype(np.int32)),
            HexLattice(8, base.h * 1.01, base.domain, base.coords, base.neighbors),
        ):
            with self.subTest(n=altered.n, shape=altered.coords.shape), patch(
                "gino_adapter_hex_v3.triangle_hex_lattice",
                side_effect=AssertionError("must not allocate"),
            ):
                with self.assertRaisesRegex(ValueError, "canonical shapes"):
                    prepare_gino_batch(altered, np.zeros(base.num_nodes), cfg)

    @unittest.skipUnless(importlib.util.find_spec("neuralop"), "optional neuraloperator backend not installed")
    def test_actual_backend_backward_optimizer_smoke(self) -> None:
        """Synthetic optimizer step ONLY; not Phase-1 task data or preregistered training."""
        lattice = triangle_hex_lattice(32)
        cfg = GinoPhase1Config(
            latent_resolution=8, n_modes=3,
            # 2.5 = Phase-1 default; 5.0 has 8 radius ties here and is refused since 2026-10-15.
            hidden_channels=8, n_layers=1, radius_factor=2.5,
        )
        with torch.random.fork_rng():
            torch.manual_seed(317)
            model = build_gino(cfg, lattice)
            model.train()
            data = torch.sin(torch.linspace(0.1, 2.4, lattice.num_nodes)).unsqueeze(0)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer.zero_grad(set_to_none=True)
            before = [p.detach().clone() for p in model.parameters()]
            output = predict_gino(model, lattice, data, cfg)
            self.assertEqual(tuple(output.shape), (1, lattice.num_nodes))
            loss = output.square().mean()
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients, "expected at least one trainable gradient")
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
            self.assertTrue(any(bool(torch.any(g != 0)) for g in gradients))
            optimizer.step()
            self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))
            self.assertTrue(any(not torch.equal(old, p) for old, p in zip(before, model.parameters())))

    @unittest.skipUnless(importlib.util.find_spec("neuralop"), "optional neuraloperator backend not installed")
    def test_actual_backend_forward_smoke(self) -> None:
        lattice = triangle_hex_lattice(32)
        cfg = GinoPhase1Config(
            latent_resolution=8,
            n_modes=3,
            hidden_channels=8,
            n_layers=1,
            radius_factor=2.5,  # Phase-1 default; 5.0 has 8 radius ties (refused since 2026-10-15)
        )
        model = build_gino(cfg, lattice)
        model.eval()
        with torch.no_grad():
            out = predict_gino(model, lattice, torch.ones((2, lattice.num_nodes)), cfg)
        self.assertEqual(tuple(out.shape), (2, lattice.num_nodes))
        self.assertTrue(torch.isfinite(out).all())

    def test_complex_weight_dtype_must_match_real_dtype(self) -> None:
        # model.double() leaves complex64 spectral weights next to float64 tensors.
        lattice = triangle_hex_lattice(8)
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3)

        class Mixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
                self.spectral = torch.nn.Parameter(torch.ones(2, dtype=torch.complex64))

            def forward(self, **batch):
                return self.linear(batch["x"])

        with self.assertRaisesRegex(ValueError, "complex spectral weights"):
            predict_gino(Mixed(), lattice, torch.ones((1, lattice.num_nodes), dtype=torch.float64), cfg)

    @unittest.skipUnless(importlib.util.find_spec("neuralop"), "optional neuraloperator backend not installed")
    def test_actual_backend_float64_is_refused_by_the_adapter(self) -> None:
        # neuraloperator 2.0.0 allocates segment_csr and spectral buffers in float32/complex64:
        # a float64 GINO fails deep in the backend, so the adapter must refuse it first.
        lattice = triangle_hex_lattice(32)
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3, hidden_channels=8, n_layers=1)
        model = build_gino(cfg, lattice).double()
        for p in model.parameters():
            if p.is_complex():
                p.data = p.data.to(torch.complex128)
        with self.assertRaises(Exception) as caught:
            predict_gino(model, lattice, torch.ones((1, lattice.num_nodes), dtype=torch.float64), cfg)
        got = type(caught.exception).__name__
        self.assertIsInstance(caught.exception, ValueError, f"expected ValueError, got {got}")
        self.assertIn("float64", str(caught.exception))


def _neuralop_like_core(dtype: torch.dtype) -> torch.nn.Module:
    """Test double for a neuraloperator submodule: its class lives in ``neuralop.*``."""

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1, bias=False, dtype=dtype)

        def forward(self, **batch):
            return self.linear(batch["x"])

    Core.__module__ = "neuralop.models.gino"
    return Core()


class _Wrapper(torch.nn.Module):
    """A caller-side wrapper module outside ``neuralop.*`` around a backend module."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(self, **batch):
        return self.core(**batch)


class TestGinoHardening20261015(unittest.TestCase):
    """Negative-first hardening of 2026-10-15 (B3a wrapper bypass, B4 radius ties)."""

    def test_float64_wrapper_around_neuralop_submodule_is_refused(self) -> None:
        # B3a: the float64 refusal looked at type(model) only; a wrapper outside
        # neuralop.* around a float64 neuraloperator submodule slipped through.
        lattice = triangle_hex_lattice(8)
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3)
        wrapped = _Wrapper(_neuralop_like_core(torch.float64))
        self.assertFalse(type(wrapped).__module__.startswith("neuralop."))
        u0 = torch.ones((1, lattice.num_nodes), dtype=torch.float64)
        with self.assertRaises(Exception) as caught:
            predict_gino(wrapped, lattice, u0, cfg)
        got = type(caught.exception).__name__
        self.assertIsInstance(caught.exception, ValueError, f"expected ValueError, got {got}")
        self.assertIn("float64", str(caught.exception))

    def test_float32_wrapper_around_neuralop_submodule_runs(self) -> None:
        # Positive control for the B3a test: same wrapper, float32, must run.
        lattice = triangle_hex_lattice(8)
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3)
        out = predict_gino(_Wrapper(_neuralop_like_core(torch.float32)), lattice,
                           torch.ones((1, lattice.num_nodes)), cfg)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (1, lattice.num_nodes))

    @unittest.skipUnless(importlib.util.find_spec("neuralop"), "optional neuraloperator backend not installed")
    def test_actual_backend_float64_wrapper_is_refused(self) -> None:
        lattice = triangle_hex_lattice(32)
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3, hidden_channels=8, n_layers=1)
        model = _Wrapper(build_gino(cfg, lattice)).double()
        for p in model.parameters():
            if p.is_complex():
                p.data = p.data.to(torch.complex128)
        with self.assertRaises(Exception) as caught:
            predict_gino(model, lattice, torch.ones((1, lattice.num_nodes), dtype=torch.float64), cfg)
        got = type(caught.exception).__name__
        self.assertIsInstance(caught.exception, ValueError, f"expected ValueError, got {got}")
        self.assertIn("float64", str(caught.exception))

    def test_radius_ties_are_refused_when_preparing_a_batch(self) -> None:
        # B4: at radius_factor=5.0 (latent 8x8, N=32) eight node-latent distances
        # lie on the GNO radius (spec/GINO_ADAPTER_PHASE1_v2.md); float32 and
        # float64 then build different neighbour graphs. Measured 2026-10-15.
        lattice = triangle_hex_lattice(32)
        tied = GinoPhase1Config(latent_resolution=8, n_modes=3, radius_factor=5.0)
        with self.assertRaises(Exception) as caught:
            prepare_gino_batch(lattice, torch.ones(lattice.num_nodes), tied)
        got = type(caught.exception).__name__
        self.assertIsInstance(caught.exception, ValueError, f"expected ValueError, got {got}")
        self.assertRegex(str(caught.exception), r"\b8\b.*radius")

    def test_radius_ties_are_refused_when_building(self) -> None:
        lattice = triangle_hex_lattice(32)
        tied = GinoPhase1Config(latent_resolution=8, n_modes=3, radius_factor=5.0)
        with self.assertRaises(Exception) as caught:
            build_gino(tied, lattice)
        got = type(caught.exception).__name__
        self.assertIsInstance(caught.exception, ValueError, f"expected ValueError, got {got}")
        self.assertIn("radius", str(caught.exception))

    def test_default_radius_without_ties_runs(self) -> None:
        # Positive control: Phase-1 default 2.5 h, same lattice and latent grid: 0 ties.
        lattice = triangle_hex_lattice(32)
        cfg = GinoPhase1Config(latent_resolution=8, n_modes=3)
        self.assertEqual(cfg.radius_factor, 2.5)
        batch = prepare_gino_batch(lattice, torch.ones(lattice.num_nodes), cfg)
        self.assertEqual(tuple(batch["x"].shape), (1, lattice.num_nodes, 1))
        default = prepare_gino_batch(lattice, torch.ones(lattice.num_nodes), GinoPhase1Config())
        self.assertEqual(tuple(default["latent_queries"].shape), (1, 32, 32, 2))


if __name__ == "__main__":
    unittest.main()
