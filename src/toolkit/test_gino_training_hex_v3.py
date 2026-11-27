"""Tests for gino_training_hex_v3 (spec/GINO_TRAINING_PHASE1_v1.md).

Synthetic tensors only: standard-normal fields on lattice coordinates. No task
data is generated, loaded or fit (spec/PREREGISTRATION_PHASE1_v1.md, lines 14-15).

Two groups. The contract tests run without the optional backend: a stand-in
with the backend's call signature and its non-tensor ``"_metadata"`` state
entry replaces ``build_gino``. ``TestRealBackend`` runs the pinned
neuraloperator 2.0.0 GINO and is skipped where the backend is not installed
(the ``gate`` job installs it).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import logging
import math
import os
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

import gino_training_hex_v3 as gt
from ddp_hex_v3 import (
    DATA_PARALLEL_REFUSAL_ATTR,
    DataParallelRefusedError,
    DistributedContractError,
    refuse_data_parallel,
)
from gino_adapter_hex_v3 import (
    CONFIG_DIGEST_ATTR,
    GINO_DATA_PARALLEL_REFUSAL,
    GinoPhase1Config,
    build_gino,
    predict_gino,
)
from gino_training_hex_v3 import (
    CONFIG_DIGEST_KEY,
    GinoRunnerModule,
    Phase1TrainingRecipe,
    RelativeL2Loss,
    RunnerPairDataset,
    build_phase1_runner,
    mean_rel_l2,
)
from heat_operator_task_hex_v3 import relative_l2
from hex_fv_solver_hex_v3 import torus_hex_lattice, triangle_hex_lattice
from runner_hex_v3 import HexTrainingRunner, RunnerConfig

_HAS_BACKEND = importlib.util.find_spec("neuralop") is not None
_LATTICE = triangle_hex_lattice(32)
_NODES = _LATTICE.num_nodes
# latent 16 is tie-free on every preregistered lattice for these radius factors
# (results/2026-10-30_gino_radius_tie_survey_float32-graphs_linux.json).
_CFG = GinoPhase1Config(latent_resolution=16, n_modes=4, hidden_channels=8, n_layers=2, radius_factor=2.5)
_OTHER_RADIUS = dataclasses.replace(_CFG, radius_factor=2.25)


def setUpModule() -> None:
    logging.disable(logging.CRITICAL)


def tearDownModule() -> None:
    logging.disable(logging.NOTSET)


class _FakeGino(nn.Module):
    """The backend's call signature and its "_metadata" state entry, without the backend."""

    def __init__(self) -> None:
        super().__init__()
        self.mix = nn.Linear(1, 1)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, input_geom, latent_queries, output_queries, x):
        if output_queries is not input_geom or latent_queries.ndim != 4:
            raise AssertionError("geometry was not passed as the adapter prepares it")
        return torch.tanh(self.mix(x)) * self.scale + 0.01 * input_geom[..., :1]

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if state.get("_metadata") is None:  # as neuraloperator: an existing entry is left alone
            state["_metadata"] = {"non_linearity": torch.nn.functional.gelu, "_version": "2.0.0"}
        return state


def _fake_build(config, lattice):
    if lattice.domain != "triangle_dirichlet" or lattice.n != 32:
        raise AssertionError("the radius must be fixed from the N = 32 triangle lattice")
    model = _FakeGino()
    setattr(model, CONFIG_DIGEST_ATTR, config.digest())
    return model


def _fake_backend():
    return patch.object(gt, "build_gino", side_effect=_fake_build)


class _Source:
    """Synthetic {"trajectory_id", "u0", "uT"} samples, float64 like the task dataset."""

    def __init__(self, count: int, seed: int, nodes: int = _NODES) -> None:
        rng = np.random.default_rng(seed)
        self.u0 = rng.standard_normal((count, nodes))
        self.u_t = rng.standard_normal((count, nodes))

    def __len__(self) -> int:
        return len(self.u0)

    def __getitem__(self, index: int):
        return {"trajectory_id": 100 + index, "u0": self.u0[index], "uT": self.u_t[index]}


def _pairs(count: int, seed: int) -> RunnerPairDataset:
    return RunnerPairDataset(_Source(count, seed), _NODES)


def _recipe(**changes) -> Phase1TrainingRecipe:
    values = dict(epochs=4, learning_rate=1e-2, batch_size=4, seed=3)
    values.update(changes)
    return Phase1TrainingRecipe(**values)


def _tensors_equal(a, b) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


class _Stop(Exception):
    pass


def _interrupted_equals_uninterrupted(case: unittest.TestCase, *, use_ema: bool, mid_epoch: bool) -> None:
    """A 4-epoch run stopped and resumed must end with the bits of the uninterrupted run."""
    recipe = _recipe(use_ema=use_ema)
    steps = 1 if mid_epoch else 0
    with tempfile.TemporaryDirectory() as full_dir, tempfile.TemporaryDirectory() as cut_dir:
        full_runner, full_model = build_phase1_runner(recipe, _CFG, _pairs(8, 1), _pairs(6, 2), output_dir=full_dir,
                                                      save_every_n_steps=steps)
        full_runner.run()

        cut_runner, _ = build_phase1_runner(recipe, _CFG, _pairs(8, 1), _pairs(6, 2), output_dir=cut_dir,
                                            save_every_n_steps=steps)
        if mid_epoch:
            # 2 steps per epoch: the 4th step is the second step of the second epoch.
            one_step, calls = cut_runner._train_one_step, [0]

            def stopping(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 4:
                    raise _Stop()
                return one_step(*args, **kwargs)

            cut_runner._train_one_step = stopping
            with case.assertRaises(_Stop):
                cut_runner.run()
        else:
            cut_runner.config.num_epochs = 2
            cut_runner.run()

        checkpoint = os.path.join(cut_dir, "checkpoints", "resume_last.pt")
        resumed_runner, resumed_model = build_phase1_runner(
            recipe, _CFG, _pairs(8, 1), _pairs(6, 2), output_dir=cut_dir, resume_from=checkpoint,
            resume_trusted=True, save_every_n_steps=steps,
        )
        summary = resumed_runner.run()
        case.assertIsNotNone(summary["resume_report"])
        case.assertTrue(_tensors_equal(full_model.state_dict(), resumed_model.state_dict()))
        if use_ema:
            case.assertTrue(_tensors_equal(full_runner.ema_model.state_dict(), resumed_runner.ema_model.state_dict()))
        # Positive control: the run did move the weights.
        fresh = build_phase1_runner(recipe, _CFG, _pairs(8, 1), _pairs(6, 2), output_dir=full_dir)[1]
        case.assertFalse(_tensors_equal(fresh.state_dict(), full_model.state_dict()))


class TestGinoRunnerModule(unittest.TestCase):
    def setUp(self) -> None:
        backend = _fake_backend()
        backend.start()
        self.addCleanup(backend.stop)

    def test_backend_state_holds_a_function_and_the_module_state_does_not(self) -> None:
        module = GinoRunnerModule(_CFG, _LATTICE)
        # Premise: the stand-in writes what neuraloperator 2.0.0 writes.
        self.assertIs(module.gino.state_dict()["_metadata"]["non_linearity"], torch.nn.functional.gelu)
        state = module.state_dict()
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in state.values()))
        self.assertNotIn("_metadata", state)
        self.assertEqual(bytes(state[CONFIG_DIGEST_KEY].tolist()).hex(), _CFG.digest())
        self.assertEqual(sorted(state), sorted(["gino.mix.weight", "gino.mix.bias", "gino.scale", CONFIG_DIGEST_KEY]))

    def test_state_is_loadable_with_weights_only(self) -> None:
        module = GinoRunnerModule(_CFG, _LATTICE)
        with tempfile.TemporaryDirectory() as tmp:
            module_path, backend_path = os.path.join(tmp, "module.pt"), os.path.join(tmp, "backend.pt")
            torch.save(module.state_dict(), module_path)
            torch.save(module.gino.state_dict(), backend_path)
            loaded = torch.load(module_path, weights_only=True)
            # Positive control: the backend's own state is what the safe loader refuses.
            with self.assertRaises(Exception) as refused:
                torch.load(backend_path, weights_only=True)
        self.assertIn("gelu", str(refused.exception))
        self.assertTrue(_tensors_equal(loaded, module.state_dict()))

    def test_state_inside_a_wrapper_keeps_the_prefix_and_drops_the_metadata(self) -> None:
        module = GinoRunnerModule(_CFG, _LATTICE)
        averaged = torch.optim.swa_utils.AveragedModel(module)
        state = averaged.state_dict()
        self.assertNotIn("_metadata", state)
        self.assertIn("module." + CONFIG_DIGEST_KEY, state)
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in state.values()))
        averaged.load_state_dict(state)
        other = torch.optim.swa_utils.AveragedModel(GinoRunnerModule(_OTHER_RADIUS, _LATTICE))
        with self.assertRaisesRegex(RuntimeError, "another GinoPhase1Config"):
            other.load_state_dict(state)

    def test_metadata_of_another_module_in_the_destination_is_kept(self) -> None:
        module = GinoRunnerModule(_CFG, _LATTICE)
        destination = {"_metadata": {"owner": "a sibling module"}}
        state = module.state_dict(destination=destination, prefix="child.")
        self.assertEqual(state["_metadata"], {"owner": "a sibling module"})
        self.assertIn("child." + CONFIG_DIGEST_KEY, state)
        with self.assertRaisesRegex(TypeError, "keyword arguments only"):
            module.state_dict({})

    def test_other_non_tensor_backend_state_is_refused(self) -> None:
        module = GinoRunnerModule(_CFG, _LATTICE)
        plain = _FakeGino.state_dict

        def with_extra(self, destination=None, prefix="", keep_vars=False):
            state = plain(self, destination=destination, prefix=prefix, keep_vars=keep_vars)
            state[prefix + "activation"] = torch.nn.functional.gelu
            return state

        with patch.object(_FakeGino, "state_dict", with_extra):
            with self.assertRaisesRegex(TypeError, "non-tensor state entries"):
                module.state_dict()

    def test_load_checks_the_config_binding(self) -> None:
        module = GinoRunnerModule(_CFG, _LATTICE)
        state = module.state_dict()
        with torch.no_grad():
            module.gino.scale.add_(1.0)
        module.load_state_dict(state)
        self.assertTrue(_tensors_equal(module.state_dict(), state))

        # The radius changes no tensor shape: without the binding these weights load.
        other = GinoRunnerModule(_OTHER_RADIUS, _LATTICE)
        self.assertEqual({k: v.shape for k, v in other.state_dict().items()}, {k: v.shape for k, v in state.items()})
        with self.assertRaisesRegex(RuntimeError, "another GinoPhase1Config"):
            other.load_state_dict(state)
        with self.assertRaisesRegex(RuntimeError, "another GinoPhase1Config"):
            other.load_state_dict(state, strict=False)

        for forged in (state[CONFIG_DIGEST_KEY].to(torch.int64), state[CONFIG_DIGEST_KEY][:16], "digest"):
            with self.subTest(forged=type(forged).__name__):
                with self.assertRaisesRegex(RuntimeError, "another GinoPhase1Config"):
                    module.load_state_dict(dict(state, **{CONFIG_DIGEST_KEY: forged}))

        unbound = {k: v for k, v in state.items() if k != CONFIG_DIGEST_KEY}
        with self.assertRaisesRegex(RuntimeError, "Missing key.*" + CONFIG_DIGEST_KEY):
            module.load_state_dict(unbound)
        result = module.load_state_dict(unbound, strict=False)
        self.assertEqual(result.missing_keys, [CONFIG_DIGEST_KEY])
        self.assertEqual(result.unexpected_keys, [])

    def test_weights_move_between_lattices_and_the_geometry_is_not_saved(self) -> None:
        trained = GinoRunnerModule(_CFG, _LATTICE)
        for lattice in (triangle_hex_lattice(16), torus_hex_lattice(32)):
            with self.subTest(domain=lattice.domain, n=lattice.n):
                other = GinoRunnerModule(_CFG, lattice)
                other.load_state_dict(trained.state_dict())
                self.assertEqual(tuple(other.input_geom.shape), (1, lattice.num_nodes, 2))
                field = torch.randn(2, lattice.num_nodes, generator=torch.Generator().manual_seed(0))
                self.assertEqual(tuple(other(field).shape), (2, lattice.num_nodes))
        self.assertFalse(any("geom" in key or "queries" in key for key in trained.state_dict()))
        self.assertEqual({name for name, _ in trained.named_buffers()}, {"input_geom", "latent_queries"})

    def test_forward_is_predict_gino_and_refuses_other_inputs(self) -> None:
        module = GinoRunnerModule(_CFG, _LATTICE)
        field = torch.randn(3, _NODES, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            self.assertTrue(torch.equal(module(field), predict_gino(module.gino, _LATTICE, field, _CFG)))
        for bad in (field[0], field.unsqueeze(-1), field[:, :-1], field.numpy()):
            with self.assertRaisesRegex(ValueError, "shaped"):
                module(bad)
        with self.assertRaisesRegex(TypeError, "must be torch.float32"):
            module(field.double())
        with self.assertRaisesRegex(TypeError, "GinoPhase1Config"):
            GinoRunnerModule({"latent_resolution": 16}, _LATTICE)

    def test_backend_output_shape_is_checked(self) -> None:
        module = GinoRunnerModule(_CFG, _LATTICE)
        with patch.object(_FakeGino, "forward", lambda self, **kwargs: kwargs["x"][:, :-1]):
            with self.assertRaisesRegex(ValueError, "unexpected output shape"):
                module(torch.zeros(1, _NODES))

    def test_predict_runs_in_eval_mode_and_restores_the_mode(self) -> None:
        module = GinoRunnerModule(_CFG, _LATTICE)
        seen = []
        forward = _FakeGino.forward

        def recording(self, **kwargs):
            seen.append((self.training, torch.is_grad_enabled()))
            return forward(self, **kwargs)

        coarse = triangle_hex_lattice(16)
        with patch.object(_FakeGino, "forward", recording):
            module.train()
            out = module.predict(coarse, np.ones(coarse.num_nodes))
            self.assertTrue(module.gino.training)
            module.eval()
            module.predict(coarse, np.ones(coarse.num_nodes))
            self.assertFalse(module.gino.training)
        self.assertEqual(seen, [(False, False), (False, False)])
        self.assertEqual(tuple(out.shape), (1, coarse.num_nodes))
        self.assertFalse(out.requires_grad)


class TestRelativeL2Loss(unittest.TestCase):
    def test_equals_the_task_metric(self) -> None:
        generator = torch.Generator().manual_seed(5)
        pred = torch.randn(7, _NODES, generator=generator, dtype=torch.float64)
        target = torch.randn(7, _NODES, generator=generator, dtype=torch.float64)
        expected = relative_l2(pred.numpy(), target.numpy(), _LATTICE)
        self.assertAlmostEqual(float(RelativeL2Loss()(pred, target)), float(expected.mean()), delta=1e-15)
        # Mean of ratios, not the pooled ratio: one large sample must not dominate.
        scaled_pred, scaled_target = pred.clone(), target.clone()
        scaled_pred[0] *= 1e3
        scaled_target[0] *= 1e3
        self.assertAlmostEqual(float(RelativeL2Loss()(scaled_pred, scaled_target)), float(expected.mean()), delta=1e-12)

    def test_gradient_and_refusals(self) -> None:
        pred = torch.ones(2, 5, requires_grad=True)
        target = torch.full((2, 5), 2.0)
        loss = RelativeL2Loss()(pred, target)
        self.assertAlmostEqual(float(loss), 0.5, places=6)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(pred.grad).all()) and bool((pred.grad != 0).all()))
        zero_target = target.clone()
        zero_target[1] = 0.0
        with self.assertRaisesRegex(ValueError, "undefined for a zero target"):
            RelativeL2Loss()(pred, zero_target)
        bad_shapes = ((pred[0], target[0]), (pred, target[:, :4]), (pred.unsqueeze(-1), target.unsqueeze(-1)))
        for bad_pred, bad_target in bad_shapes:
            with self.assertRaisesRegex(ValueError, r"\[batch, nodes\]"):
                RelativeL2Loss()(bad_pred, bad_target)


class TestRunnerPairDataset(unittest.TestCase):
    def test_items_are_float32_runner_batches(self) -> None:
        source = _Source(5, 9)
        data = RunnerPairDataset(source, _NODES)
        self.assertEqual(len(data), 5)
        item = data[3]
        self.assertEqual(sorted(item), ["input", "sample_id", "target"])
        self.assertEqual(item["sample_id"], 103)
        for key, raw in (("input", source.u0[3]), ("target", source.u_t[3])):
            self.assertEqual(item[key].dtype, torch.float32)
            self.assertEqual(tuple(item[key].shape), (_NODES,))
            np.testing.assert_array_equal(item[key].numpy(), raw.astype(np.float32))
        subset = RunnerPairDataset(Subset(source, [4, 0]), _NODES)
        self.assertEqual([subset[i]["sample_id"] for i in range(len(subset))], [104, 100])
        self.assertNotIn("sample_id", RunnerPairDataset([{"u0": source.u0[0], "uT": source.u_t[0]}], _NODES)[0])

    def test_lossy_or_malformed_fields_are_refused(self) -> None:
        good = np.ones(4)

        def item(u0=good, u_t=good):
            return RunnerPairDataset([{"u0": u0, "uT": u_t}], 4)[0]

        self.assertEqual(item(u0=np.arange(4))["input"].tolist(), [0.0, 1.0, 2.0, 3.0])
        cases = {
            "must be finite": np.array([1.0, np.nan, 1.0, 1.0]),
            "overflows float32": np.array([1.0, 1e39, 1.0, 1.0]),
            "conversion to float32 underflows a nonzero entry": np.array([1.0, 1e-50, 1.0, 1.0]),
            "must be a real field shaped": np.ones(3),
        }
        for message, field in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, "u0 " + message):
                    item(u0=field)
                with self.assertRaisesRegex(ValueError, "uT " + message):
                    item(u_t=field)
        for field in (np.ones(4, dtype=complex), np.ones(4, dtype=bool), np.array(["a"] * 4), np.ones((1, 4))):
            with self.assertRaisesRegex(ValueError, "real field shaped"):
                item(u0=field)
        for nodes in (0, -1, True, 4.0):
            with self.assertRaisesRegex(ValueError, "positive int"):
                RunnerPairDataset([], nodes)
        with self.assertRaisesRegex(TypeError, "map-style"):
            RunnerPairDataset(iter([]), 4)


class TestPhase1TrainingRecipe(unittest.TestCase):
    def test_runner_defaults_are_not_the_recipe(self) -> None:
        """Why the recipe exists: these runner defaults are named nowhere in the preregistration."""
        defaults = RunnerConfig()
        self.assertEqual(defaults.grad_clip_max_norm, 1.0)
        self.assertTrue(defaults.use_ema)
        self.assertEqual(defaults.best_metric_name, "val_nrmse")

    def test_digest_binds_every_setting(self) -> None:
        base = _recipe()
        self.assertEqual(base.digest(), _recipe().digest())
        changes = dict(epochs=5, learning_rate=2e-2, batch_size=8, seed=4, weight_decay=1e-4,
                       adam_betas=(0.8, 0.999), adam_eps=1e-7, cosine_eta_min=1e-6, grad_clip_max_norm=1.0,
                       use_ema=True, ema_decay=0.99, warmup_steps=1, shuffle=False)
        self.assertEqual(set(changes), {f.name for f in dataclasses.fields(Phase1TrainingRecipe)})
        digests = {base.digest()}
        for name, value in changes.items():
            digests.add(dataclasses.replace(base, **{name: value}).digest())
        self.assertEqual(len(digests), len(changes) + 1)
        canonical = base.canonical()
        self.assertEqual(canonical["grad_clip_max_norm"], "inf")
        self.assertIs(canonical["amp"], False)
        self.assertEqual(canonical["learning_rate"], (1e-2).hex())
        self.assertIn("last epoch", canonical["registered_weights"])

    def test_invalid_settings_are_refused(self) -> None:
        bad = [
            ("epochs", 0), ("epochs", True), ("epochs", 2.0), ("batch_size", 0), ("seed", -1), ("warmup_steps", -1),
            ("learning_rate", 0.0), ("learning_rate", math.inf), ("learning_rate", "1e-3"), ("learning_rate", True),
            ("weight_decay", -1e-4), ("weight_decay", math.nan), ("cosine_eta_min", -1.0),
            ("adam_eps", 0.0), ("adam_betas", [0.9, 0.999]), ("adam_betas", (0.9,)), ("adam_betas", (0.9, 1.0)),
            ("grad_clip_max_norm", 0.0), ("grad_clip_max_norm", math.nan), ("ema_decay", 1.5),
            ("use_ema", 1), ("shuffle", "yes"),
        ]
        for name, value in bad:
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    _recipe(**{name: value})


class TestBuildPhase1Runner(unittest.TestCase):
    def setUp(self) -> None:
        backend = _fake_backend()
        backend.start()
        self.addCleanup(backend.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def _build(self, recipe=None, **kwargs):
        return build_phase1_runner(recipe or _recipe(), _CFG, _pairs(8, 1), _pairs(6, 2),
                                   output_dir=kwargs.pop("output_dir", self.tmp), **kwargs)

    def test_runner_is_built_from_the_recipe_alone(self) -> None:
        recipe = _recipe(weight_decay=1e-3, adam_betas=(0.8, 0.99), adam_eps=1e-7, cosine_eta_min=1e-5)
        runner, model = self._build(recipe)
        self.assertIs(runner.model, model)
        self.assertIsNone(runner.adapter)
        self.assertIsInstance(runner.loss_fn, RelativeL2Loss)
        config = runner.config
        self.assertEqual((config.num_epochs, config.grad_clip_max_norm, config.use_ema, config.use_amp,
                          config.warmup_steps, config.best_metric_name),
                         (4, math.inf, False, False, 0, "val_loss"))
        self.assertIsNone(runner.ema_model)
        self.assertIs(type(runner.optimizer), torch.optim.Adam)
        group = runner.optimizer.param_groups[0]
        self.assertEqual((group["lr"], group["betas"], group["eps"], group["weight_decay"]),
                         (1e-2, (0.8, 0.99), 1e-7, 1e-3))
        self.assertIs(type(runner.scheduler), torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual((runner.scheduler.T_max, runner.scheduler.eta_min), (4, 1e-5))
        self.assertEqual((runner.train_loader.batch_size, runner.train_loader.drop_last), (4, False))
        self.assertIsInstance(runner.train_loader.sampler, torch.utils.data.RandomSampler)
        self.assertIsInstance(runner.val_loader.sampler, torch.utils.data.SequentialSampler)
        ema_runner, _ = self._build(_recipe(use_ema=True, ema_decay=0.5, grad_clip_max_norm=2.0, warmup_steps=3,
                                            shuffle=False))
        self.assertEqual((ema_runner.config.use_ema, ema_runner.config.ema_decay,
                          ema_runner.config.grad_clip_max_norm, ema_runner.config.warmup_steps),
                         (True, 0.5, 2.0, 3))
        self.assertIsNotNone(ema_runner.ema_model)
        self.assertIsInstance(ema_runner.train_loader.sampler, torch.utils.data.SequentialSampler)

    def test_learning_rate_follows_the_cosine_per_epoch(self) -> None:
        runner, _ = self._build()
        summary = runner.run()
        rates = [report["lr"] for report in summary["history"]["train"]]
        expected = [1e-2 * 0.5 * (1.0 + math.cos(math.pi * (epoch + 1) / 4)) for epoch in range(4)]
        np.testing.assert_allclose(rates, expected, rtol=1e-12, atol=1e-18)
        self.assertEqual(summary["global_step"], 8)

    def test_the_seed_fixes_the_run(self) -> None:
        def trained(seed: int):
            with tempfile.TemporaryDirectory() as tmp:
                runner, model = self._build(_recipe(seed=seed), output_dir=tmp)
                initial = {k: v.clone() for k, v in model.state_dict().items()}
                runner.run()
                return initial, model.state_dict()

        first_initial, first = trained(3)
        second_initial, second = trained(3)
        other_initial, other = trained(4)
        self.assertTrue(_tensors_equal(first_initial, second_initial))
        self.assertTrue(_tensors_equal(first, second))
        self.assertFalse(_tensors_equal(first_initial, first))
        self.assertFalse(_tensors_equal(first_initial, other_initial))
        self.assertFalse(_tensors_equal(first, other))

    def test_inputs_are_checked(self) -> None:
        with self.assertRaisesRegex(TypeError, "Phase1TrainingRecipe"):
            build_phase1_runner({"epochs": 4}, _CFG, _pairs(8, 1), _pairs(6, 2), output_dir=self.tmp)
        coarse = RunnerPairDataset(_Source(4, 1, nodes=triangle_hex_lattice(16).num_nodes),
                                   triangle_hex_lattice(16).num_nodes)
        for train, val in ((coarse, _pairs(6, 2)), (_pairs(8, 1), coarse), (_Source(8, 1), _pairs(6, 2))):
            with self.assertRaisesRegex(ValueError, "N = 32 triangle lattice"):
                build_phase1_runner(_recipe(), _CFG, train, val, output_dir=self.tmp)
        with self.assertRaisesRegex(ValueError, "val_data is empty"):
            build_phase1_runner(_recipe(), _CFG, _pairs(8, 1), RunnerPairDataset([], _NODES), output_dir=self.tmp)

    def test_checkpoints_load_with_weights_only(self) -> None:
        for use_ema in (False, True):
            with self.subTest(use_ema=use_ema), tempfile.TemporaryDirectory() as tmp:
                runner, model = self._build(_recipe(use_ema=use_ema), output_dir=tmp)
                runner.run()
                for name in ("best.pt", "last.pt"):
                    payload = torch.load(os.path.join(tmp, "checkpoints", name), weights_only=True)
                    self.assertIsInstance(payload, dict)
                reloaded = GinoRunnerModule(_CFG, _LATTICE)
                last = torch.load(os.path.join(tmp, "checkpoints", "last.pt"), weights_only=True)
                states = [value for value in last.values()
                          if isinstance(value, dict) and CONFIG_DIGEST_KEY in value]
                self.assertEqual(len(states), 1)
                reloaded.load_state_dict(states[0])
                self.assertTrue(_tensors_equal(reloaded.state_dict(), model.state_dict()))

    def test_resume_at_an_epoch_boundary_is_bitwise(self) -> None:
        for use_ema in (False, True):
            with self.subTest(use_ema=use_ema):
                _interrupted_equals_uninterrupted(self, use_ema=use_ema, mid_epoch=False)

    def test_resume_inside_an_epoch_is_bitwise(self) -> None:
        for use_ema in (False, True):
            with self.subTest(use_ema=use_ema):
                _interrupted_equals_uninterrupted(self, use_ema=use_ema, mid_epoch=True)


class _PlainOperator(nn.Module):
    """A non-GINO model with the runner's call signature: the negative control."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.scale


def _runner_on(model: nn.Module, out: str, *, distributed: bool) -> HexTrainingRunner:
    return HexTrainingRunner(
        RunnerConfig(output_dir=out, num_epochs=1, distributed=distributed),
        model, torch.optim.SGD(model.parameters(), lr=1e-2),
        DataLoader(_pairs(4, 1), batch_size=2), DataLoader(_pairs(2, 2), batch_size=2),
        loss_fn=RelativeL2Loss(),
    )


class TestNeverDataParallel(unittest.TestCase):
    """Owner decision 2026-11-27 (spec section 8a): GINO is never trained data-parallel.

    Through the public entry, HexTrainingRunner: a GinoRunnerModule with
    ``distributed=True`` (no process group, or already inside DDP in a gloo group)
    and a DDP-wrapped one with ``distributed=False`` raise DataParallelRefusedError;
    a non-GINO model and the single-process GINO run as before.
    """

    def setUp(self) -> None:
        backend = _fake_backend()  # the stand-in is not marked: the class itself must refuse
        backend.start()
        self.addCleanup(backend.stop)

    def _assert_refusal(self, caught: BaseException) -> None:
        self.assertIsInstance(caught, DataParallelRefusedError)
        self.assertIsInstance(caught, DistributedContractError)
        text = str(caught)
        self.assertTrue(text.startswith("HEXANTK-DDP-REFUSED: "), text)
        self.assertIn("spec/GINO_TRAINING_PHASE1_v1.md section 8a", text)
        self.assertIn("never data-parallel", text)

    def test_the_module_class_carries_the_refusal(self) -> None:
        self.assertEqual(getattr(GinoRunnerModule, DATA_PARALLEL_REFUSAL_ATTR), GINO_DATA_PARALLEL_REFUSAL)
        self.assertIsNone(getattr(_FakeGino(), DATA_PARALLEL_REFUSAL_ATTR, None))  # the stand-in is unmarked

    def test_distributed_runner_refuses_gino_without_a_process_group(self) -> None:
        if dist.is_initialized():
            self.skipTest("a process group is already initialized in this process")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Exception) as caught:
                _runner_on(GinoRunnerModule(_CFG, _LATTICE), tmp, distributed=True)
            self._assert_refusal(caught.exception)
            # negative control: a non-GINO model fails for the missing group, not with the refusal
            with self.assertRaises(DistributedContractError) as plain:
                _runner_on(_PlainOperator(), tmp, distributed=True)
        self.assertNotIsInstance(plain.exception, DataParallelRefusedError)
        self.assertIn("not initialized", str(plain.exception))

    def test_runner_refuses_gino_inside_ddp_in_a_gloo_group(self) -> None:
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo backend not available")
        if dist.is_initialized():
            self.skipTest("a process group is already initialized in this process")
        with tempfile.TemporaryDirectory() as tmp:
            dist.init_process_group("gloo", init_method=Path(tmp, "rendezvous").as_uri(), rank=0, world_size=1,
                                    timeout=timedelta(seconds=60))
            try:
                # the stand-in has no complex weight, so DDP constructs: only the refusal can stop it
                wrapped = DistributedDataParallel(GinoRunnerModule(_CFG, _LATTICE))
                for distributed in (True, False):
                    with self.subTest(distributed=distributed):
                        with self.assertRaises(Exception) as caught:
                            _runner_on(wrapped, os.path.join(tmp, f"gino-{distributed}"), distributed=distributed)
                        self._assert_refusal(caught.exception)
                # negative control: the same path with a non-GINO model constructs
                runner = _runner_on(DistributedDataParallel(_PlainOperator()), os.path.join(tmp, "plain"),
                                    distributed=True)
                self.assertIsNotNone(runner.dist)
            finally:
                dist.destroy_process_group()

    def test_single_process_gino_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner, model = build_phase1_runner(_recipe(epochs=1), _CFG, _pairs(4, 1), _pairs(2, 2), output_dir=tmp)
            self.assertFalse(runner.config.distributed)
            self.assertEqual(runner.run()["global_step"], 1)
            runner = _runner_on(GinoRunnerModule(_CFG, _LATTICE), os.path.join(tmp, "direct"), distributed=False)
            self.assertIsNone(runner.dist)


class TestMeanRelL2(unittest.TestCase):
    def setUp(self) -> None:
        backend = _fake_backend()
        backend.start()
        self.addCleanup(backend.stop)

    def test_is_the_mean_of_the_per_sample_metric(self) -> None:
        torch.manual_seed(11)
        model = GinoRunnerModule(_CFG, _LATTICE)
        source = _Source(7, 4)
        source.u_t[0] *= 50.0  # one large target: the pooled ratio is then another number
        data = RunnerPairDataset(source, _NODES)
        with torch.no_grad():
            pred = model(torch.from_numpy(source.u0.astype(np.float32))).numpy()
        truth = source.u_t.astype(np.float32)
        per_sample = relative_l2(pred, truth, _LATTICE)
        expected = math.fsum(per_sample.tolist()) / 7
        for batch_size in (1, 3, 7, 100):
            self.assertEqual(mean_rel_l2(model, data, batch_size=batch_size), expected)
        error, reference = pred.astype(np.float64) - truth, truth.astype(np.float64)
        pooled = math.sqrt(float((error ** 2).sum() / (reference ** 2).sum()))
        self.assertGreater(abs(pooled - expected), 1e-3)

    def test_runner_metrics_are_other_numbers(self) -> None:
        """val_loss (mean of batch means) and val_nrmse (pooled) are not the selection statistic."""
        source = _Source(6, 2)
        source.u_t[0] *= 50.0
        val = RunnerPairDataset(source, _NODES)
        with tempfile.TemporaryDirectory() as tmp:
            runner, model = build_phase1_runner(_recipe(epochs=1), _CFG, _pairs(8, 1), val, output_dir=tmp)
            report = runner.run()["history"]["val"][-1]
        exact = mean_rel_l2(model, val)
        self.assertEqual(report["n_batches"], 2)  # 6 samples in batches of 4 and 2
        self.assertGreater(abs(report["val_loss"] - exact), 1e-4)
        self.assertGreater(abs(report["val_nrmse"] - exact), 1e-4)

    def test_mode_is_restored_and_inputs_are_checked(self) -> None:
        model = GinoRunnerModule(_CFG, _LATTICE)
        data = _pairs(3, 1)
        model.train()
        mean_rel_l2(model, data)
        self.assertTrue(model.training)
        model.eval()
        mean_rel_l2(model, data)
        self.assertFalse(model.training)
        with self.assertRaisesRegex(TypeError, "GinoRunnerModule and a RunnerPairDataset"):
            mean_rel_l2(model.gino, data)
        with self.assertRaisesRegex(TypeError, "GinoRunnerModule and a RunnerPairDataset"):
            mean_rel_l2(model, _Source(3, 1))
        with self.assertRaisesRegex(ValueError, "empty"):
            mean_rel_l2(model, RunnerPairDataset([], _NODES))


@unittest.skipUnless(_HAS_BACKEND, "optional neuraloperator backend not installed")
class TestRealBackend(unittest.TestCase):
    """The pinned neuraloperator 2.0.0 GINO itself. Synthetic tensors ONLY."""

    def test_backend_state_is_what_the_module_has_to_handle(self) -> None:
        """Premise of the module: the backend's state holds a function the safe loader refuses."""
        backend = build_gino(_CFG, _LATTICE)
        metadata = backend.state_dict()["_metadata"]
        self.assertIsInstance(metadata, dict)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "backend.pt")
            torch.save(backend.state_dict(), path)
            with self.assertRaises(Exception) as refused:
                torch.load(path, weights_only=True)
        self.assertIn("gelu", str(refused.exception))

    def test_module_state_is_tensors_bound_to_the_config(self) -> None:
        module = GinoRunnerModule(_CFG, _LATTICE)
        state = module.state_dict()
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in state.values()))
        backend_tensors = {k: v for k, v in module.gino.state_dict().items() if isinstance(v, torch.Tensor)}
        self.assertEqual(sorted(state), sorted(["gino." + k for k in backend_tensors] + [CONFIG_DIGEST_KEY]))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "module.pt")
            torch.save(state, path)
            loaded = torch.load(path, weights_only=True)
        fresh = GinoRunnerModule(_CFG, _LATTICE)
        fresh.load_state_dict(loaded)
        self.assertTrue(_tensors_equal(fresh.state_dict(), state))
        with self.assertRaisesRegex(RuntimeError, "another GinoPhase1Config"):
            GinoRunnerModule(_OTHER_RADIUS, _LATTICE).load_state_dict(loaded)

    def test_forward_is_predict_gino_bitwise(self) -> None:
        module = GinoRunnerModule(_CFG, _LATTICE).eval()
        field = torch.randn(3, _NODES, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            self.assertTrue(torch.equal(module(field), predict_gino(module.gino, _LATTICE, field, _CFG)))
        coarse = triangle_hex_lattice(16)
        other = GinoRunnerModule(_CFG, coarse).eval()
        other.load_state_dict(module.state_dict())
        coarse_field = torch.randn(2, coarse.num_nodes, generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            self.assertTrue(torch.equal(other(coarse_field), module.predict(coarse, coarse_field)))

    def test_training_run_writes_loadable_checkpoints(self) -> None:
        for use_ema in (False, True):
            with self.subTest(use_ema=use_ema), tempfile.TemporaryDirectory() as tmp:
                runner, model = build_phase1_runner(_recipe(use_ema=use_ema, learning_rate=1e-3), _CFG,
                                                    _pairs(8, 1), _pairs(6, 2), output_dir=tmp)
                initial = {k: v.clone() for k, v in model.state_dict().items()}
                summary = runner.run()
                self.assertEqual(summary["global_step"], 8)
                self.assertTrue(all(math.isfinite(r["mean_loss"]) for r in summary["history"]["train"]))
                self.assertFalse(_tensors_equal(initial, model.state_dict()))
                for name in ("best.pt", "last.pt"):
                    torch.load(os.path.join(tmp, "checkpoints", name), weights_only=True)
                self.assertTrue(math.isfinite(mean_rel_l2(model, _pairs(6, 2))))

    def test_resume_at_an_epoch_boundary_is_bitwise(self) -> None:
        for use_ema in (False, True):
            with self.subTest(use_ema=use_ema):
                _interrupted_equals_uninterrupted(self, use_ema=use_ema, mid_epoch=False)

    def test_resume_inside_an_epoch_is_bitwise(self) -> None:
        for use_ema in (False, True):
            with self.subTest(use_ema=use_ema):
                _interrupted_equals_uninterrupted(self, use_ema=use_ema, mid_epoch=True)

    def test_ddp_on_gloo_refuses_the_complex_spectral_weights(self) -> None:
        """Premise of spec section 8a (torch 2.6.0): gloo broadcasts no complex tensor, DDP broadcasts every
        parameter at construction, and the spectral weights are complex64. If this test turns red, the
        premise has changed (another torch or backend): re-measure with scripts/oracles/gino_ddp_gloo_oracle.py."""
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo backend not available")
        if dist.is_initialized():
            self.skipTest("a process group is already initialized in this process")
        module = GinoRunnerModule(_CFG, _LATTICE)
        complex_names = sorted(n for n, p in module.named_parameters() if p.is_complex())
        self.assertEqual(complex_names, ["gino.fno_blocks.convs.0.weight.tensor",
                                         "gino.fno_blocks.convs.1.weight.tensor"])
        self.assertEqual({p.dtype for n, p in module.named_parameters() if n in complex_names}, {torch.complex64})
        with tempfile.TemporaryDirectory() as tmp:
            dist.init_process_group("gloo", init_method=Path(tmp, "rendezvous").as_uri(), rank=0, world_size=1)
            try:
                values = torch.ones(2, dtype=torch.complex64)
                dist.all_reduce(values)  # all-reduce takes complex tensors (as their real view)
                self.assertTrue(torch.equal(values, torch.ones(2, dtype=torch.complex64)))
                with self.assertRaisesRegex(RuntimeError, "Invalid scalar type"):
                    dist.broadcast(torch.ones(2, dtype=torch.complex64), 0)
                with self.assertRaisesRegex(RuntimeError, "Invalid scalar type"):
                    DistributedDataParallel(module)
                # negative control: the same module without its complex parameters is accepted
                DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(module, complex_names)
                DistributedDataParallel(module)
            finally:
                dist.destroy_process_group()

    def test_every_built_gino_refuses_data_parallel(self) -> None:
        """Owner decision 2026-11-27: build_gino marks its model, so the refusal also holds for the raw GINO in a
        caller's wrapper; refuse_data_parallel is what a caller runs instead of meeting gloo's raw error."""
        backend = build_gino(_CFG, _LATTICE)
        self.assertEqual(getattr(backend, DATA_PARALLEL_REFUSAL_ATTR), GINO_DATA_PARALLEL_REFUSAL)

        class _CallerWrapper(nn.Module):
            def __init__(self, inner: nn.Module) -> None:
                super().__init__()
                self.inner = inner

        for model, where in ((backend, "the model"), (_CallerWrapper(backend), "submodule 'inner'"),
                             (GinoRunnerModule(_CFG, _LATTICE), "the model")):
            with self.subTest(model=type(model).__name__):
                with self.assertRaises(DataParallelRefusedError) as caught:
                    refuse_data_parallel(model, what="caller")
                self.assertIn(where, str(caught.exception))
                self.assertIn("section 8a", str(caught.exception))
        # the limit named in spec section 8a: the mark is on the instance, so a GINO built outside build_gino
        # (and outside GinoRunnerModule) is not recognised
        self.assertIsNone(getattr(type(backend), DATA_PARALLEL_REFUSAL_ATTR, None))

    def test_two_builds_with_one_seed_train_to_the_same_bits(self) -> None:
        states = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                runner, model = build_phase1_runner(_recipe(epochs=2), _CFG, _pairs(8, 1), _pairs(6, 2),
                                                    output_dir=tmp)
                runner.run()
                states.append(model.state_dict())
        self.assertTrue(_tensors_equal(states[0], states[1]))


if __name__ == "__main__":
    unittest.main()
