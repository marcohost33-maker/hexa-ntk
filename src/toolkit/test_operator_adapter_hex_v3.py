"""
test_operator_adapter_hex_v3.py
===============================
Tests for the operator adapter validation layer.

Covers:
- Valid graph operator call path
- Valid GINO call path
- Missing required fields raise
- Wrong tensor shapes raise
- shared_geometry=False raises for GINO
- Output shape mismatch raises
- Unsupported model_type raises
- SDF / boundary_mask validation
- dt validation (scalar vs batch)
"""
from __future__ import annotations

import unittest
from typing import Any

import torch
import torch.nn as nn

from operator_adapter_hex_v3 import OperatorAdapterV3, OperatorBatchContext


# ------------------------------------------------------------------
# Dummy models
# ------------------------------------------------------------------


class DummyGraphOperator(nn.Module):
    """Minimal graph operator: returns x unchanged."""

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return x


class DummyGINOOperator(nn.Module):
    """Minimal GINO operator: returns x unchanged."""

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return x


class DummyBadOutputModel(nn.Module):
    """Returns wrong shape."""

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return x[:, :1, :]  # Wrong N dimension


class DummyNonTensorOutputModel(nn.Module):
    """Returns a non-tensor."""

    def forward(self, x: torch.Tensor, **kwargs: Any) -> str:
        return "not_a_tensor"


# ------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------

B, N, C, D, E = 2, 7, 3, 2, 12


def make_x() -> torch.Tensor:
    return torch.randn(B, N, C)


def make_coords() -> torch.Tensor:
    return torch.randn(N, D)


def make_edge_index() -> torch.Tensor:
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    return torch.stack([src, dst])


def make_dt_scalar() -> torch.Tensor:
    return torch.tensor([0.01])


def make_dt_batch() -> torch.Tensor:
    return torch.tensor([0.01, 0.02])


# ------------------------------------------------------------------
# Tests: Construction
# ------------------------------------------------------------------


class TestAdapterConstruction(unittest.TestCase):
    def test_valid_graph_operator(self) -> None:
        adapter = OperatorAdapterV3(DummyGraphOperator(), "graph_operator_v3")
        self.assertEqual(adapter.model_type, "graph_operator_v3")

    def test_valid_gino(self) -> None:
        adapter = OperatorAdapterV3(DummyGINOOperator(), "gino_hex_v3")
        self.assertEqual(adapter.model_type, "gino_hex_v3")

    def test_valid_gino_v2(self) -> None:
        adapter = OperatorAdapterV3(DummyGINOOperator(), "gino_hex_v3_v2")
        self.assertEqual(adapter.model_type, "gino_hex_v3_v2")

    def test_unsupported_model_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            OperatorAdapterV3(DummyGraphOperator(), "unknown_type")

    def test_non_module_raises(self) -> None:
        with self.assertRaises(TypeError):
            OperatorAdapterV3("not_a_module", "graph_operator_v3")  # type: ignore


# ------------------------------------------------------------------
# Tests: Graph operator call path
# ------------------------------------------------------------------


class TestGraphOperatorPath(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = OperatorAdapterV3(DummyGraphOperator(), "graph_operator_v3")

    def test_valid_call(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(
            edge_index=make_edge_index(),
            coords=make_coords(),
            dt=make_dt_scalar(),
        )
        out = self.adapter(x, ctx)
        self.assertEqual(out.shape, x.shape)

    def test_valid_call_minimal(self) -> None:
        """edge_index is the only required field for graph operator."""
        x = make_x()
        ctx = OperatorBatchContext(edge_index=make_edge_index())
        out = self.adapter(x, ctx)
        self.assertEqual(out.shape, x.shape)

    def test_missing_edge_index_raises(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext()  # No edge_index
        with self.assertRaises(ValueError):
            self.adapter(x, ctx)

    def test_wrong_edge_index_shape_raises(self) -> None:
        x = make_x()
        # shape [3, E] instead of [2, E]
        bad_ei = torch.randint(0, N, (3, E))
        ctx = OperatorBatchContext(edge_index=bad_ei)
        with self.assertRaises(ValueError):
            self.adapter(x, ctx)

    def test_wrong_coords_ndim_raises(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(
            edge_index=make_edge_index(),
            coords=torch.randn(N, D, 1),  # 3D instead of 2D
        )
        with self.assertRaises(ValueError):
            self.adapter(x, ctx)

    def test_coords_node_count_mismatch_raises(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(
            edge_index=make_edge_index(),
            coords=torch.randn(N + 5, D),  # Wrong N
        )
        with self.assertRaises(ValueError):
            self.adapter(x, ctx)

    def test_dt_batch_aligned(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(
            edge_index=make_edge_index(),
            dt=make_dt_batch(),
        )
        out = self.adapter(x, ctx)
        self.assertEqual(out.shape, x.shape)

    def test_dt_wrong_numel_raises(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(
            edge_index=make_edge_index(),
            dt=torch.tensor([0.01, 0.02, 0.03]),  # 3, not 1 or B
        )
        with self.assertRaises(ValueError):
            self.adapter(x, ctx)

    def test_float_edge_index_raises(self) -> None:
        """edge_index must be integer dtype, not float."""
        x = make_x()
        float_ei = torch.randn(2, E)  # float, should be long
        ctx = OperatorBatchContext(edge_index=float_ei)
        with self.assertRaises(TypeError):
            self.adapter(x, ctx)


# ------------------------------------------------------------------
# Tests: GINO call path
# ------------------------------------------------------------------


class TestGINOPath(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = OperatorAdapterV3(DummyGINOOperator(), "gino_hex_v3")

    def test_valid_call(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(
            coords=make_coords(),
            dt=make_dt_scalar(),
            extra={"shared_geometry": True},
        )
        out = self.adapter(x, ctx)
        self.assertEqual(out.shape, x.shape)

    def test_valid_call_with_sdf_and_boundary(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(
            coords=make_coords(),
            extra={
                "sdf": torch.ones(N, 1),
                "boundary_mask": torch.zeros(N, 1),
                "shared_geometry": True,
            },
        )
        out = self.adapter(x, ctx)
        self.assertEqual(out.shape, x.shape)

    def test_missing_coords_raises(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext()  # No coords
        with self.assertRaises(ValueError):
            self.adapter(x, ctx)

    def test_shared_geometry_false_raises(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(
            coords=make_coords(),
            extra={"shared_geometry": False},
        )
        with self.assertRaises(ValueError):
            self.adapter(x, ctx)

    def test_shared_geometry_non_bool_raises(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(
            coords=make_coords(),
            extra={"shared_geometry": "yes"},
        )
        with self.assertRaises(TypeError):
            self.adapter(x, ctx)

    def test_1d_coords_raises(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(coords=torch.randn(N, 1))  # 1D, need >= 2D
        with self.assertRaises(ValueError):
            self.adapter(x, ctx)

    def test_coords_node_mismatch_raises(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(coords=torch.randn(N + 3, D))
        with self.assertRaises(ValueError):
            self.adapter(x, ctx)

    def test_sdf_wrong_shape_raises(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(
            coords=make_coords(),
            extra={"sdf": torch.ones(N + 1, 1)},  # Wrong N
        )
        with self.assertRaises(ValueError):
            self.adapter(x, ctx)

    def test_sdf_wrong_ndim_raises(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(
            coords=make_coords(),
            extra={"sdf": torch.ones(N)},  # 1D instead of 2D
        )
        with self.assertRaises(ValueError):
            self.adapter(x, ctx)

    def test_boundary_mask_wrong_shape_raises(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(
            coords=make_coords(),
            extra={"boundary_mask": torch.zeros(N + 2, 1)},
        )
        with self.assertRaises(ValueError):
            self.adapter(x, ctx)

    def test_boundary_mask_wrong_ndim_raises(self) -> None:
        x = make_x()
        ctx = OperatorBatchContext(
            coords=make_coords(),
            extra={"boundary_mask": torch.zeros(N)},
        )
        with self.assertRaises(ValueError):
            self.adapter(x, ctx)

    def test_default_shared_geometry_is_true(self) -> None:
        """If shared_geometry not in extra, default is True (no error)."""
        x = make_x()
        ctx = OperatorBatchContext(coords=make_coords())
        out = self.adapter(x, ctx)
        self.assertEqual(out.shape, x.shape)


# ------------------------------------------------------------------
# Tests: Output validation
# ------------------------------------------------------------------


class TestOutputValidation(unittest.TestCase):
    def test_wrong_output_shape_raises(self) -> None:
        adapter = OperatorAdapterV3(DummyBadOutputModel(), "graph_operator_v3")
        x = make_x()
        ctx = OperatorBatchContext(edge_index=make_edge_index())
        with self.assertRaises(ValueError):
            adapter(x, ctx)

    def test_non_tensor_output_raises(self) -> None:
        adapter = OperatorAdapterV3(DummyNonTensorOutputModel(), "graph_operator_v3")
        x = make_x()
        ctx = OperatorBatchContext(edge_index=make_edge_index())
        with self.assertRaises(TypeError):
            adapter(x, ctx)


# ------------------------------------------------------------------
# Tests: Input validation
# ------------------------------------------------------------------


class TestInputValidation(unittest.TestCase):
    def test_non_tensor_x_raises(self) -> None:
        adapter = OperatorAdapterV3(DummyGraphOperator(), "graph_operator_v3")
        ctx = OperatorBatchContext(edge_index=make_edge_index())
        with self.assertRaises(TypeError):
            adapter("not_a_tensor", ctx)  # type: ignore

    def test_wrong_x_ndim_raises(self) -> None:
        adapter = OperatorAdapterV3(DummyGraphOperator(), "graph_operator_v3")
        x = torch.randn(N, C)  # 2D instead of 3D
        ctx = OperatorBatchContext(edge_index=make_edge_index())
        with self.assertRaises(ValueError):
            adapter(x, ctx)

    def test_non_context_raises(self) -> None:
        adapter = OperatorAdapterV3(DummyGraphOperator(), "graph_operator_v3")
        x = make_x()
        with self.assertRaises(TypeError):
            adapter(x, {"edge_index": make_edge_index()})  # type: ignore


# ------------------------------------------------------------------
# Tests: GINO v2 variant
# ------------------------------------------------------------------


class TestGINOV2Path(unittest.TestCase):
    def test_gino_v2_valid_call(self) -> None:
        adapter = OperatorAdapterV3(DummyGINOOperator(), "gino_hex_v3_v2")
        x = make_x()
        ctx = OperatorBatchContext(coords=make_coords())
        out = adapter(x, ctx)
        self.assertEqual(out.shape, x.shape)

    def test_gino_v2_same_validation_as_v1(self) -> None:
        """gino_hex_v3_v2 uses same validation as gino_hex_v3."""
        adapter = OperatorAdapterV3(DummyGINOOperator(), "gino_hex_v3_v2")
        x = make_x()
        ctx = OperatorBatchContext()  # No coords
        with self.assertRaises(ValueError):
            adapter(x, ctx)


if __name__ == "__main__":
    unittest.main()
