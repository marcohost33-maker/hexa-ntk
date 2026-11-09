"""CPU Torch behavioural oracles for the *actual transformed AST* of the R4 step.

Synthetic small tensor operations only, no GINO backend or T1 data.  The tests
exercise executable guard semantics; R4 only tested static text/AST structure.
"""

from __future__ import annotations
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import patch_synthetic_cost_oracle_v1 as stage3  # noqa: E402
import harden_r4_synthetic_oracle as stage4  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


def compiled_step(namespace):
    """Extract and execute only the transformed nested 'step', not full script."""
    # The real oracle at the pinned blob (LF as in the object; a Windows checkout
    # carries CRLF). stage3.transform refuses any other bytes.
    original = (ROOT / 'scripts/oracles/gino_epoch_cost_synthetic.py').read_bytes().replace(b'\r\n', b'\n')
    source = stage4.transform(stage3.transform(original)).decode('utf8')
    root = ast.parse(source)
    wrappers = [node for node in root.body if isinstance(node, ast.FunctionDef) and node.name == '_time_config']
    if len(wrappers) != 1:
        raise RuntimeError('fixture no longer has unique _time_config')
    steps = [node for node in ast.walk(wrappers[0]) if isinstance(node, ast.FunctionDef) and node.name == 'step']
    if len(steps) != 1:
        raise RuntimeError('fixture no longer has unique step')
    body = ast.Module(body=[steps[0]], type_ignores=[])
    ast.fix_missing_locations(body)
    exec(compile(body, 'r4_transformed_step', 'exec'), namespace)
    return namespace['step']


@unittest.skipIf(torch is None, 'Torch not installed: test is OPTIONAL but skip must be visible')
class TestActualTorchBehaviour(unittest.TestCase):
    def setup_case(self, badgrad=False, target_override=None):
        class BadGrad(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, go):
                return torch.full_like(go, float('inf'))

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, x):
                w = BadGrad.apply(self.weight) if badgrad else self.weight
                return w * x

        torch.manual_seed(123)
        model = Toy()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        y = torch.zeros_like(x) if target_override is None else target_override
        ctx = {
            'torch': torch,
            'model': model,
            'optimizer': optimizer,
            'lattice': None,
            'config': None,
            'u0': x,
            'target': y,
            'predict_gino': lambda model, lattice, x, config: model(x),
            '_relative_l2': lambda prediction, truth: ((prediction - truth) ** 2).mean(),
        }
        return model, optimizer, compiled_step(ctx)

    def test_finite_loss_gradient_and_parameters_updates_weight(self):
        model, optimizer, step = self.setup_case()
        before = float(model.weight.detach())
        loss = step(torch.tensor([0]))
        self.assertEqual(loss, 1.0)
        self.assertLess(float(model.weight.detach()), before)
        self.assertTrue(torch.isfinite(model.weight).all())

    def test_nan_loss_is_refused_before_optimizer_update(self):
        model, optimizer, step = self.setup_case(target_override=torch.tensor([[float('nan')], [0.0]]))
        before = model.weight.detach().clone()
        with self.assertRaisesRegex(FloatingPointError, 'non-finite minibatch loss'):
            step(torch.tensor([0]))
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)

    def test_infinite_gradients_refused_before_optimizer_update(self):
        model, optimizer, step = self.setup_case(badgrad=True)
        before = model.weight.detach().clone()
        with self.assertRaisesRegex(FloatingPointError, 'non-finite gradient before optimizer'):
            step(torch.tensor([0]))
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        self.assertTrue(torch.isinf(model.weight.grad).any())

    def test_nonfinite_optimizer_result_is_detected_after_update(self):
        model, optimizer, step = self.setup_case()

        def malicious_step():
            with torch.no_grad():
                model.weight.fill_(float('inf'))

        optimizer.step = malicious_step
        with self.assertRaisesRegex(FloatingPointError, 'non-finite parameter after optimizer'):
            step(torch.tensor([0]))
        self.assertTrue(torch.isinf(model.weight).all())  # no rollback is promised

    def test_all_minibatches_checked_not_only_final_one(self):
        model, optimizer, step = self.setup_case(target_override=torch.tensor([[float('nan')], [0.0]]))
        with self.assertRaises(FloatingPointError):
            step(torch.tensor([0]))
        result = step(torch.tensor([1]))
        self.assertTrue(torch.isfinite(torch.tensor(result)).all())


if __name__ == '__main__':
    unittest.main()
