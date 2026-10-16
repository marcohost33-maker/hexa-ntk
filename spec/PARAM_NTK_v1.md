# PARAM_NTK_v1 — empirical parameter NTK in a W-weighted mode basis

Status: implemented in `src/toolkit/param_ntk_hex_v3.py` (2026-10-16). Owner decision E1015M-03: float64 measurement with a float32 cross-check.

## Definition

For a model output field `y(theta)` on M lattice nodes, W-orthonormal modes `phi_k` (k = 1..K, `phi^T W phi = I`) and the projections `P_k = phi_k^T W y`:

    Theta_kl = < dP_k/dtheta , dP_l/dtheta >

The inner product runs over every real degree of freedom. Complex parameters (FNO spectral weights) count as two real coordinates (`torch.view_as_real`). `Theta` is symmetric positive semidefinite by construction. Each row `dP_k/dtheta` comes from one reverse-mode pass (VJP).

## Precision

| `model_dtype` | Model copy | Kernel assembly | Cross-check reported as `float32_relative_deviation` |
|---|---|---|---|
| `float64` (default) | float64 / complex128 | float64 | the same measurement with a float32 model copy |
| `float32` (backends without float64, e.g. neuraloperator 2.0.0 GINO) | float32 / complex64 | float64 from the float32 gradients | the Gram matrix assembled in float32 |

The caller's model is never modified: measurement always runs on a cast copy.

## Fail-closed guards

1. **Complete parameter set.** An explicit `params` list must be exactly the model's trainable parameters, in order. A partial list would report the NTK of a sub-model.
2. **Autograd agrees with the output.** A central difference of one random projection along one random parameter direction must match autograd.
   - float64: step 1e-6 relative, tolerance `FD_REL_TOL = 1e-6`.
   - float32: step 4e-3 (~u^(1/3)), tolerance `FD_REL_TOL_FLOAT32 = 1e-3` (expected error ~u^(2/3) ~ 1.5e-5).
   - A forward that detaches or swaps a parameter outside autograd is refused (its relative error is ~1).
3. **Inputs.** The forward must return a finite 1-D field of length M. Modes must be (M, K), finite. Weights must be finite and positive.

## Evidence

`src/toolkit/test_param_ntk_hex_v3.py`:
- analytic kernels: a linear model gives `P A A^T P^T`; a complex scale gives a Jacobian `[u, u^2]` in real coordinates;
- PSD and symmetry; float32 deviation is nonzero and below 1e-5;
- the caller's model is unchanged;
- negative tests: partial parameter list, detached forward (in both modes), bad shapes and weights.

Real backend: `test_gino_adapter_hex_v3.py::test_parameter_ntk_of_the_real_gino` (neuraloperator 2.0.0, float32 mode). It runs in the GINO CI step.

## Not claimed

This is an empirical NTK at one parameter point and one input. It is not the infinite-width NTK, says nothing about training dynamics, and gives no convergence guarantee. Pre-registered experiments (D1–D8) decide how it is used.
