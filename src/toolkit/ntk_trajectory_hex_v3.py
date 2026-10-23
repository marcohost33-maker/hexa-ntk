"""Parameter-NTK trajectory over declared training checkpoints (NTK_TRAJECTORY_v1).

PARAM_NTK_v1 measures the empirical parameter NTK Theta(theta) of a model at one
parameter point, in a W-orthonormal mode basis. This module measures it at a
declared sequence of checkpoints theta_0, theta_1, ... of one training run, on
one fixed input, and reports how the kernel moves:

* relative drift ||Theta_t - Theta_0||_F / ||Theta_0||_F (magnitude and direction);
* kernel distance S = 1 - <Theta_a, Theta_b>_F / (||Theta_a||_F ||Theta_b||_F)
  (direction only, scale-invariant; Fort et al., NeurIPS 2020), from the first
  and from the previous checkpoint, and the velocity S(prev, t) / (step_t - step_prev);
* mode coupling ||Theta - diag(Theta)||_F^2 / ||Theta||_F^2 in the given basis;
* effective rank (tr Theta)^2 / ||Theta||_F^2 (participation ratio);
* optional target alignment c^T Theta c / (||Theta||_F ||c||^2) for the modal
  coefficients c of a target field (kernel-target alignment, Cristianini et al.,
  NIPS 2001).

Binding: the record carries SHA-256 digests the module computes ITSELF from the
bytes it measured (every checkpoint state dict, the input tensor, the basis and
weights, the parameter signature, every kernel). A caller-supplied hash string is
never trusted. The digests show WHICH bytes were measured; they do not attest
that the checkpoints came from a real training run ("authenticity": "unattested").

Fail-closed: the checkpoint steps must equal the declared steps (no post-hoc
subset), every checkpoint must match the model's state-dict keys, shapes and
dtypes exactly, and a forward that mutates its input is refused. Each state is
loaded from a private snapshot whose digest must equal the preflight digest
(HNT-R85-01; closes a change between preflight and load, not a concurrent writer
racing the copy). Metadata that JSON would encode like another subject (a data dict
posing as a callable, non-str keys) is refused (HNT-R86-01); a drift beyond binary64
is an error, never inf (HNT-R86-02).

Repeated states (HNT-R25-01): two checkpoints with identical bytes are measured,
not refused. Equal bytes prove neither a mislabelled copy nor a genuinely
stationary run (lr = 0, frozen parameters, a plateau); a copy saved under a wrong
step number is a provenance defect that only an authenticated checkpoint history
can expose, not a hash. Each entry therefore says, machine-readably, whether its
state equals the previous one (``state_unchanged_since_previous``) and which
earlier step first had the same bytes (``repeated_state_digest_of``). The kernel
of a repeated state is measured again, under the full PARAM_NTK_v1 oracle; the
drift since the previous checkpoint is a measured zero (``stationary_drift =
"measured_zero"``) only if that re-measurement is bitwise equal to the previous
kernel, otherwise ``"not_measured"``. ``check_trajectory_record`` refuses a record
whose markers disagree with its own digests.

Not claimed: an empirical kernel at one input says nothing about generalisation,
is not the infinite-width NTK, and is no causal statement about learning.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import types
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import param_ntk_hex_v3 as pn

POLICY = "NTK_TRAJECTORY_v1"
# Operational limits (fail-closed budget, not a scientific bound): one checkpoint
# costs K reverse passes plus the float32 cross-check.
MAX_CHECKPOINTS = 1000
MAX_MODES = 256

__all__ = [
    "MAX_CHECKPOINTS",
    "MAX_MODES",
    "NTKTrajectory",
    "NTKTrajectoryError",
    "POLICY",
    "TrajectoryEntry",
    "check_trajectory_record",
    "kernel_distance",
    "ntk_trajectory",
    "state_dict_sha256",
    "tensor_sha256",
    "trajectory_record_sha256",
]


class NTKTrajectoryError(ValueError):
    """The checkpoints, inputs or forward do not form a valid declared trajectory."""


@dataclass(frozen=True)
class TrajectoryEntry:
    step: int
    state_sha256: str
    kernel_sha256: str
    kernel: np.ndarray                      # (K, K) float64, symmetric PSD
    eigenvalues: np.ndarray                 # ascending
    fd_relative_error: float
    float32_relative_deviation: float | None
    trace: float
    effective_rank: float | None            # None for the zero kernel
    offdiag_energy_fraction: float | None   # None for the zero kernel
    relative_drift: float | None            # vs. the first checkpoint; None if Theta_0 = 0
    distance_from_initial: float | None     # None if either kernel is zero
    distance_from_previous: float | None    # None for the first checkpoint or a zero kernel
    velocity: float | None                  # distance_from_previous / step difference
    target_alignment: float | None          # None without target or for the zero kernel
    # HNT-R25-01: identical checkpoint bytes are allowed and marked, never refused.
    state_unchanged_since_previous: bool    # state_sha256 equals the previous entry's
    repeated_state_digest_of: int | None    # earliest earlier step with the same state_sha256
    stationary_drift: str | None            # "measured_zero" | "not_measured"; None if the state changed


@dataclass(frozen=True)
class NTKTrajectory:
    run_id: str
    entries: tuple[TrajectoryEntry, ...]
    declared_steps: tuple[int, ...]
    model_dtype: str
    n_real_params: int
    n_modes: int
    inputs_sha256: str
    basis_sha256: str
    parameter_signature_sha256: str
    target_sha256: str | None
    policy: str = POLICY

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe record (no NaN/inf; undefined quantities are null) with its own digest."""
        record = {
            "policy": self.policy,
            "run_id": self.run_id,
            "authenticity": "unattested",
            "declared_steps": list(self.declared_steps),
            "model_dtype": self.model_dtype,
            "n_real_params": self.n_real_params,
            "n_modes": self.n_modes,
            "inputs_sha256": self.inputs_sha256,
            "basis_sha256": self.basis_sha256,
            "parameter_signature_sha256": self.parameter_signature_sha256,
            "target_sha256": self.target_sha256,
            "entries": [
                {
                    "step": e.step,
                    "state_sha256": e.state_sha256,
                    "kernel_sha256": e.kernel_sha256,
                    "kernel_diagonal": [float(x) for x in np.diag(e.kernel)],
                    "eigenvalue_min": float(e.eigenvalues[0]),
                    "eigenvalue_max": float(e.eigenvalues[-1]),
                    "fd_relative_error": e.fd_relative_error,
                    # PARAM_NTK_v1 reports inf for a zero float64 kernel against a nonzero
                    # float32 one; JSON has no inf, so it is written as the string "inf".
                    "float32_relative_deviation": _json_deviation(e.float32_relative_deviation),
                    "trace": e.trace,
                    "effective_rank": e.effective_rank,
                    "offdiag_energy_fraction": e.offdiag_energy_fraction,
                    "relative_drift": e.relative_drift,
                    "distance_from_initial": e.distance_from_initial,
                    "distance_from_previous": e.distance_from_previous,
                    "velocity": e.velocity,
                    "target_alignment": e.target_alignment,
                    "state_unchanged_since_previous": e.state_unchanged_since_previous,
                    "repeated_state_digest_of": e.repeated_state_digest_of,
                    "stationary_drift": e.stationary_drift,
                }
                for e in self.entries
            ],
        }
        record["record_sha256"] = trajectory_record_sha256(record)
        return record


def _json_deviation(value: float | None) -> float | str | None:
    if value is None or math.isfinite(value):
        return value
    return "inf"


def _canonical_json(value: Any) -> bytes:
    # allow_nan=False: a NaN or inf in the record is a bug, never a value.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def trajectory_record_sha256(record: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a record without its own ``record_sha256`` field."""
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    return hashlib.sha256(_canonical_json({k: v for k, v in record.items() if k != "record_sha256"})).hexdigest()


def _require_little_endian() -> None:
    if sys.byteorder != "little":  # pragma: no cover - every supported runner is little-endian
        raise NTKTrajectoryError("digests are defined on little-endian bytes; this host is big-endian")


def _tensor_bytes(t: torch.Tensor) -> bytes:
    # Raw element bytes (complex as interleaved real/imag, bool as 0/1 bytes). reshape(-1)
    # first: a 0-d tensor cannot be viewed with a different element size.
    return t.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_header(t: torch.Tensor) -> bytes:
    return f"{t.dtype}|{tuple(t.shape)}|".encode("ascii")


def tensor_sha256(t: torch.Tensor) -> str:
    """SHA-256 over dtype, shape and the raw little-endian element bytes of one tensor."""
    if not isinstance(t, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    _require_little_endian()
    h = hashlib.sha256()
    h.update(_tensor_header(t))
    h.update(_tensor_bytes(t))
    return h.hexdigest()


def _metadata_default(obj: Any) -> Any:
    # neuraloperator 2.0.0 stores its constructor kwargs as a "_metadata" state entry,
    # including functions (F.gelu) and classes (SpectralConv). They are bound by their
    # qualified name; anything without a stable name (lambdas, local functions,
    # instances) cannot be bound and is refused.
    # A bound method also carries its instance, which a name does not cover.
    module, qualname = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
    owner = getattr(obj, "__self__", None)
    if (callable(obj) and isinstance(module, str) and isinstance(qualname, str) and "<" not in qualname
            and (owner is None or isinstance(owner, types.ModuleType))):
        return {"__qualified_name__": f"{module}.{qualname}"}
    raise TypeError(f"{type(obj).__name__} has no stable qualified name")


_QUALIFIED_NAME = "__qualified_name__"
_METADATA_MAX_DEPTH = 64


def _metadata_value(value: Any, depth: int = 0) -> Any:
    """The JSON value that NTK_TRAJECTORY_v1 hashes for a non-tensor state entry.

    HNT-R86-01: plain JSON erases types. A data dict ``{"__qualified_name__": name}``
    encoded exactly like the callable of that name, and a dict key ``1`` (or ``True``,
    ``None``, ``2.5``) exactly like the string key ``"1"``, so the digest and the
    architecture check could bind a different subject than the one present. Both
    are refused here instead of re-encoded: every entry that was bindable before
    keeps its bytes, so digests already written stay valid. Not covered: a qualified
    name does not attest the callable's implementation.
    """
    if depth > _METADATA_MAX_DEPTH:
        raise TypeError(f"nested deeper than {_METADATA_MAX_DEPTH} levels (or cyclic)")
    if value is None or isinstance(value, (str, int, float)):  # bool is an int
        return value
    if isinstance(value, dict):
        out = {}
        for k, v in dict.items(value):
            if type(k) is not str:
                raise TypeError(f"key {k!r} is not a str (JSON would turn it into one)")
            if k == _QUALIFIED_NAME:
                raise TypeError(f"key {_QUALIFIED_NAME!r} is reserved for callables")
            out[k] = _metadata_value(v, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_metadata_value(v, depth + 1) for v in value]
    return _metadata_default(value)


def _metadata_bytes(key: str, value: Any) -> bytes:
    """Canonical JSON of a non-tensor state entry (e.g. neuraloperator's ``_metadata``)."""
    try:
        return json.dumps(_metadata_value(value), sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise NTKTrajectoryError(f"state entry {key!r} cannot be bound to bytes: {exc}") from exc


def state_dict_sha256(state: Mapping[str, Any]) -> str:
    """SHA-256 over every entry of a state dict in sorted key order.

    A tensor contributes its key, dtype, shape and raw bytes; a non-tensor entry
    (constructor metadata) its key and canonical JSON, with callables by qualified
    name. An entry that has no canonical form is refused: its content would not be
    covered by the digest.
    """
    if not isinstance(state, Mapping):
        raise NTKTrajectoryError("a checkpoint state must be a mapping of names to tensors")
    _require_little_endian()
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(key, str):
            raise NTKTrajectoryError(f"state key {key!r} is not a str")
        encoded = key.encode("utf-8")
        h.update(len(encoded).to_bytes(8, "little"))
        h.update(encoded)
        if isinstance(value, torch.Tensor):
            h.update(b"tensor|")
            h.update(_tensor_header(value))
            data = _tensor_bytes(value)
        else:
            h.update(b"json|")
            data = _metadata_bytes(key, value)
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a, dtype="<f8")
        h.update(f"{a.shape}|".encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


def _signature_sha256(model: torch.nn.Module) -> str:
    rows = [[name, str(p.dtype), list(p.shape), bool(p.requires_grad)] for name, p in model.named_parameters()]
    buffers = [[name, str(b.dtype), list(b.shape)] for name, b in model.named_buffers()]
    return hashlib.sha256(_canonical_json({"parameters": rows, "buffers": buffers})).hexdigest()


def _scaled(k: np.ndarray) -> np.ndarray | None:
    """Kernel divided by max|k| (None for the zero kernel): Frobenius norms of tiny or
    huge valid kernels would otherwise under- or overflow."""
    scale = float(np.abs(k).max())
    return None if scale == 0.0 else k / scale


def kernel_distance(a: Any, b: Any) -> float | None:
    """S(a, b) = 1 - <a, b>_F / (||a||_F ||b||_F); None if either kernel is zero.

    For PSD kernels <a, b>_F = tr(ab) >= 0, so S lies in [0, 1]; values a rounding
    error outside are clipped into it. S is invariant under positive scaling of either
    kernel and under an orthogonal change of the (W-orthonormal) basis.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("kernels must be square and of equal shape")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("kernels must be finite")
    sa, sb = _scaled(a), _scaled(b)
    if sa is None or sb is None:
        return None
    if np.array_equal(sa, sb):
        return 0.0  # same direction to the last bit; the cosine below may round to 1 - u
    cosine = float(np.sum(sa * sb)) / (float(np.linalg.norm(sa)) * float(np.linalg.norm(sb)))
    return min(1.0, max(0.0, 1.0 - cosine))


def _relative_drift(k: np.ndarray, k0: np.ndarray) -> float | None:
    if not (np.all(np.isfinite(k)) and np.all(np.isfinite(k0))):
        raise NTKTrajectoryError("relative drift of a non-finite kernel")
    scale = float(np.abs(k0).max())
    if scale == 0.0:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        drift = float(np.linalg.norm((k - k0) / scale)) / float(np.linalg.norm(k0 / scale))
    if math.isfinite(drift):
        return drift  # the v1 formula, unchanged wherever it was finite
    # HNT-R86-02: finite kernels, yet k - k0, its division by a tiny anchor scale or the
    # squares inside the norm overflowed. Factor the magnitude out: with d = k/2 - k0/2
    # (cannot overflow) and m = max|d|, ||k - k0|| / ||k0|| = (m / scale) * 2 ||d/m|| / ||k0/scale||,
    # where both norms lie in [1, n].
    d = k * 0.5 - k0 * 0.5
    m = float(np.abs(d).max())
    with np.errstate(over="ignore"):
        drift = (m / scale) * (2.0 * float(np.linalg.norm(d / m)) / float(np.linalg.norm(k0 / scale)))
    if not math.isfinite(drift):
        raise NTKTrajectoryError("relative drift exceeds the binary64 range (finite kernels, ratio > ~1.8e308)")
    return drift


def _shape_metrics(k: np.ndarray, target: np.ndarray | None) -> tuple[float | None, float | None, float | None]:
    s = _scaled(k)
    if s is None:
        return None, None, None
    fro2 = float(np.sum(s * s))
    diag = np.diag(s)
    effective_rank = float(np.sum(diag)) ** 2 / fro2
    offdiag = max(0.0, (fro2 - float(np.sum(diag * diag))) / fro2)
    alignment = None
    if target is not None:
        alignment = float(target @ s @ target) / (math.sqrt(fro2) * float(target @ target))
        alignment = min(1.0, max(0.0, alignment))  # PSD: 0 <= c^T K c <= ||K||_F ||c||^2
    return effective_rank, offdiag, alignment


def _step(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise NTKTrajectoryError(f"{name} must be an int, got {type(value).__name__}")
    if int(value) < 0:
        raise NTKTrajectoryError(f"{name} must be >= 0, got {value}")
    return int(value)


def _declared(declared_steps: Any) -> tuple[int, ...]:
    if isinstance(declared_steps, (str, bytes)) or not isinstance(declared_steps, Sequence):
        raise NTKTrajectoryError("declared_steps must be a sequence of ints")
    steps = tuple(_step(s, name="declared step") for s in declared_steps)
    if not steps:
        raise NTKTrajectoryError("declared_steps must not be empty")
    if len(steps) > MAX_CHECKPOINTS:
        raise NTKTrajectoryError(f"at most {MAX_CHECKPOINTS} checkpoints (got {len(steps)})")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise NTKTrajectoryError("declared_steps must be strictly increasing")
    return steps


def _check_state(state: Any, reference: Mapping[str, torch.Tensor], step: int) -> None:
    if not isinstance(state, Mapping):
        raise NTKTrajectoryError(f"checkpoint at step {step}: state must be a mapping")
    missing = sorted(set(reference) - set(state))
    extra = sorted(set(state) - set(reference))
    if missing or extra:
        raise NTKTrajectoryError(f"checkpoint at step {step}: missing keys {missing}, unexpected keys {extra}")
    for key, ref in reference.items():
        value = state[key]
        if not isinstance(ref, torch.Tensor):
            # Constructor metadata: a checkpoint of another architecture is refused.
            if isinstance(value, torch.Tensor) or _metadata_bytes(key, value) != _metadata_bytes(key, ref):
                raise NTKTrajectoryError(
                    f"checkpoint at step {step}: {key!r} differs from the model's (another architecture)"
                )
            continue
        if not isinstance(value, torch.Tensor):
            raise NTKTrajectoryError(f"checkpoint at step {step}: {key!r} is not a tensor")
        if value.dtype != ref.dtype or value.shape != ref.shape:
            # load_state_dict would cast the dtype silently: a float16 checkpoint of a
            # float32 run is a different measurement subject.
            raise NTKTrajectoryError(
                f"checkpoint at step {step}: {key!r} is {value.dtype}{tuple(value.shape)}, "
                f"the model has {ref.dtype}{tuple(ref.shape)}"
            )
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise NTKTrajectoryError(f"checkpoint at step {step}: {key!r} is not finite")


def _target(target_coefficients: Any, n_modes: int) -> np.ndarray | None:
    if target_coefficients is None:
        return None
    c = np.asarray(target_coefficients, dtype=np.float64)
    if c.shape != (n_modes,) or not np.all(np.isfinite(c)):
        raise ValueError(f"target_coefficients must be a finite vector of length {n_modes}")
    scale = float(np.abs(c).max())
    if scale == 0.0:
        raise ValueError("target_coefficients must not be zero (alignment undefined)")
    return c / scale


def _stationarity(previous: TrajectoryEntry | None, state_digest: str,
                  kernel_digest: str) -> tuple[bool, str | None]:
    """(state unchanged since the previous checkpoint, stationary-drift status).

    The kernel of a repeated state has just been re-measured under the full
    PARAM_NTK_v1 oracle (a failing oracle raises, so no entry exists then). The drift
    is a measured zero only if that re-measurement equals the previous kernel bit for
    bit; a different kernel from identical bytes means the backend did not reproduce,
    and the zero is "not_measured" (the numeric fields still report what was measured).
    """
    if previous is None or previous.state_sha256 != state_digest:
        return False, None
    return True, "measured_zero" if previous.kernel_sha256 == kernel_digest else "not_measured"


def _record_marker_error(record: Any) -> str | None:
    if not isinstance(record, Mapping):
        return "record must be a mapping"
    if record.get("policy") != POLICY:
        return f"policy must be {POLICY!r}"
    if record.get("authenticity") != "unattested":
        return "authenticity must be 'unattested' (no checkpoint provenance is attested)"
    try:
        if record.get("record_sha256") != trajectory_record_sha256(record):
            return "record_sha256 does not match the record"
    except (TypeError, ValueError) as exc:  # NaN/inf or non-JSON content: no canonical form
        return f"record has no canonical JSON form: {exc}"
    try:
        declared = _declared(record.get("declared_steps"))
    except NTKTrajectoryError as exc:
        return str(exc)
    entries = record.get("entries")
    if not isinstance(entries, list) or not all(isinstance(e, Mapping) for e in entries):
        return "entries must be a list of mappings"
    if [e.get("step") for e in entries] != list(declared) or any(type(e["step"]) is not int for e in entries):
        return "entry steps differ from the declared steps"
    seen: dict[str, int] = {}
    for i, e in enumerate(entries):
        step, state, kernel = e["step"], e.get("state_sha256"), e.get("kernel_sha256")
        if not (isinstance(state, str) and isinstance(kernel, str)):
            return f"entry at step {step}: state_sha256 and kernel_sha256 must be strings"
        for key in ("state_unchanged_since_previous", "repeated_state_digest_of", "stationary_drift"):
            if key not in e:
                return f"entry at step {step}: marker {key!r} is missing"
        # Recomputed from the record's own digests, independently of the producer.
        unchanged = i > 0 and entries[i - 1].get("state_sha256") == state
        if unchanged:
            status = "measured_zero" if entries[i - 1].get("kernel_sha256") == kernel else "not_measured"
        else:
            status = None
        first = seen.setdefault(state, step)
        expected = {
            "state_unchanged_since_previous": unchanged,
            "repeated_state_digest_of": None if first == step else first,
            "stationary_drift": status,
        }
        for key, value in expected.items():
            # type() as well: True == 1 and 0 == False must not pass as a marker.
            if e[key] != value or type(e[key]) is not type(value):
                return f"entry at step {step}: {key} is {e[key]!r}, its digests give {value!r}"
        if status == "measured_zero" and e.get("distance_from_previous") not in (0.0, None):
            return f"entry at step {step}: a measured zero drift with distance_from_previous != 0"
    return None


def check_trajectory_record(record: Any) -> None:
    """Refuse a record whose stationarity markers disagree with its own digests.

    Recomputes ``state_unchanged_since_previous``, ``repeated_state_digest_of`` and
    ``stationary_drift`` from the record's ``state_sha256`` / ``kernel_sha256`` sequence
    and requires the declared steps, ``authenticity == "unattested"`` and a matching
    ``record_sha256``. Internal consistency only: ``record_sha256`` is no signature, and
    nothing here attests that the checkpoints came from a real training run.
    """
    error = _record_marker_error(record)
    if error is not None:
        raise NTKTrajectoryError(f"invalid {POLICY} record: {error}")


def ntk_trajectory(
    model: torch.nn.Module,
    apply: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    modes: Any,
    weights: Any,
    checkpoints: Sequence[tuple[int, Mapping[str, torch.Tensor]]],
    *,
    declared_steps: Sequence[int],
    run_id: str,
    target_coefficients: Any = None,
    model_dtype: str = "float64",
    float32_crosscheck: bool = True,
    seed: int = 0,
) -> NTKTrajectory:
    """Measure PARAM_NTK_v1 at every declared checkpoint of one run on one fixed input.

    ``model`` gives the architecture (eval mode); each checkpoint's state dict is loaded
    into a copy, so the caller's model is never changed. ``apply(m, inputs)`` maps a
    model and the fixed input to the output field (M,). ``target_coefficients`` (K,),
    optional, are the coordinates ``phi^T W y*`` of a target field in the same basis.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if any(m.training for m in model.modules()):
        raise NTKTrajectoryError("model (or a submodule) is in training mode; call model.eval() first")
    if not callable(apply):
        raise TypeError("apply must be callable as apply(model, inputs)")
    if not isinstance(run_id, str) or not run_id.strip():
        raise NTKTrajectoryError("run_id must be a non-empty string")
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    if (inputs.is_floating_point() or inputs.is_complex()) and not bool(torch.isfinite(inputs).all()):
        raise NTKTrajectoryError("inputs must be finite")
    steps = _declared(declared_steps)

    projector = pn._projector(modes, weights)  # validates shapes and W-orthonormality (PARAM_NTK_v1)
    n_modes = projector.shape[0]
    if n_modes > MAX_MODES:
        raise NTKTrajectoryError(f"at most {MAX_MODES} modes (got {n_modes})")
    target = _target(target_coefficients, n_modes)

    if isinstance(checkpoints, (str, bytes, Mapping)) or not isinstance(checkpoints, Sequence):
        raise NTKTrajectoryError("checkpoints must be a sequence of (step, state_dict) pairs")
    pairs = list(checkpoints)
    if any(not isinstance(p, tuple) or len(p) != 2 for p in pairs):
        raise NTKTrajectoryError("checkpoints must be a sequence of (step, state_dict) pairs")
    given = tuple(_step(s, name="checkpoint step") for s, _ in pairs)
    if given != steps:
        # A trajectory over a post-hoc subset, superset or reordering of checkpoints is
        # not the declared one.
        raise NTKTrajectoryError(f"checkpoint steps {list(given)} differ from the declared steps {list(steps)}")

    reference = model.state_dict()
    digests = []
    for step, state in pairs:
        _check_state(state, reference, step)
        digests.append(state_dict_sha256(state))
    # HNT-R25-01: identical bytes are NOT refused. They prove neither a mislabelled copy
    # nor a genuinely stationary run; each repeat is measured again and marked below.

    modes64 = np.asarray(modes, dtype=np.float64)
    weights64 = np.asarray(weights, dtype=np.float64)
    inputs_digest = tensor_sha256(inputs)
    signature = _signature_sha256(model)

    entries: list[TrajectoryEntry] = []
    first_step_of: dict[str, int] = {}
    n_real = None
    for step, (_, state), state_digest in zip(steps, pairs, digests):
        # HNT-R85-01: the digest above was taken in the preflight loop; apply() of an
        # earlier step (or a Mapping whose reads differ) can change this state since.
        # Read it once into a private snapshot and load only that snapshot, after its
        # digest matched the preflight one. This closes a change BETWEEN preflight and
        # load. It does not make the copy atomic: a writer running concurrently with
        # this call can still race the copy itself, so exclusive ownership of the
        # checkpoints for the duration of the call stays the caller's duty.
        # The snapshot is also what neuraloperator's load_state_dict pops "_metadata"
        # from, so the caller's checkpoint is not changed. One state copy at a time.
        snapshot = {key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                    for key, value in dict(state).items()}
        if state_dict_sha256(snapshot) != state_digest:
            raise NTKTrajectoryError(f"checkpoint at step {step}: state changed since the preflight digest")
        twin = copy.deepcopy(model)
        twin.load_state_dict(snapshot, strict=True)
        twin.eval()
        result = pn.param_ntk(
            twin, lambda m: apply(m, inputs), modes64, weights64,
            float32_crosscheck=float32_crosscheck, model_dtype=model_dtype, seed=seed,
        )
        if tensor_sha256(inputs) != inputs_digest:
            raise NTKTrajectoryError(f"apply mutated the fixed input while measuring step {step}")
        n_real = result.n_real_params if n_real is None else n_real
        kernel = result.kernel
        k0 = entries[0].kernel if entries else kernel
        effective_rank, offdiag, alignment = _shape_metrics(kernel, target)
        previous = entries[-1] if entries else None
        dist_prev = kernel_distance(previous.kernel, kernel) if previous is not None else None
        kernel_digest = _array_sha256(kernel)
        unchanged, stationary = _stationarity(previous, state_digest, kernel_digest)
        repeated_of = first_step_of.setdefault(state_digest, step)
        entries.append(TrajectoryEntry(
            step=step,
            state_sha256=state_digest,
            kernel_sha256=kernel_digest,
            kernel=kernel,
            eigenvalues=result.eigenvalues,
            fd_relative_error=result.fd_relative_error,
            float32_relative_deviation=result.float32_relative_deviation,
            trace=float(np.trace(kernel)),
            effective_rank=effective_rank,
            offdiag_energy_fraction=offdiag,
            relative_drift=_relative_drift(kernel, k0),
            distance_from_initial=kernel_distance(k0, kernel),
            distance_from_previous=dist_prev,
            velocity=None if dist_prev is None else dist_prev / (step - previous.step),
            target_alignment=alignment,
            state_unchanged_since_previous=unchanged,
            repeated_state_digest_of=None if repeated_of == step else repeated_of,
            stationary_drift=stationary,
        ))

    return NTKTrajectory(
        run_id=run_id,
        entries=tuple(entries),
        declared_steps=steps,
        model_dtype=model_dtype,
        n_real_params=int(n_real),
        n_modes=n_modes,
        inputs_sha256=inputs_digest,
        basis_sha256=_array_sha256(modes64, weights64),
        parameter_signature_sha256=signature,
        target_sha256=None if target_coefficients is None else _array_sha256(np.asarray(target_coefficients,
                                                                                         dtype=np.float64)),
    )
