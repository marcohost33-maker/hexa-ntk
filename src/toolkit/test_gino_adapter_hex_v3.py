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
            hidden_channels=8, n_layers=1, radius_factor=5.0,
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
            radius_factor=5.0,
        )
        model = build_gino(cfg, lattice)
        model.eval()
        with torch.no_grad():
            out = predict_gino(model, lattice, torch.ones((2, lattice.num_nodes)), cfg)
        self.assertEqual(tuple(out.shape), (2, lattice.num_nodes))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
