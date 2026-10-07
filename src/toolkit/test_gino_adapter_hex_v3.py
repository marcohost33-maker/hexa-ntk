"""Contract and optional-backend smoke tests for gino_adapter_hex_v3."""
from __future__ import annotations

import importlib.util
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
from hex_fv_solver_hex_v3 import triangle_hex_lattice


class TestGinoPhase1Adapter(unittest.TestCase):
    def test_config_validation_and_digest(self) -> None:
        cfg = GinoPhase1Config()
        self.assertEqual(cfg.canonical()["backend"], GINO_BACKEND_PIN)
        self.assertEqual(len(cfg.digest()), 64)
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
        lattice = triangle_hex_lattice(8)
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
        with patch.dict(sys.modules, {"neuralop": root, "neuralop.models": models}):
            model = build_gino(cfg, lattice)

        self.assertEqual(model.kwargs["gno_coord_dim"], 2)
        self.assertFalse(model.kwargs["gno_use_open3d"])
        self.assertFalse(model.kwargs["gno_use_torch_scatter"])
        self.assertAlmostEqual(model.kwargs["in_gno_radius"], 2.5 * lattice.h)
        out = predict_gino(model, lattice, np.ones((2, lattice.num_nodes)), cfg)
        self.assertEqual(tuple(out.shape), (2, lattice.num_nodes))

    @unittest.skipUnless(importlib.util.find_spec("neuralop"), "optional neuraloperator backend not installed")
    def test_actual_backend_forward_smoke(self) -> None:
        lattice = triangle_hex_lattice(8)
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
