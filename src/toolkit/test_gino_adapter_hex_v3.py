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
