"""Second-stage offline patch: enforce finite gradients and model parameters.

Called only AFTER exact-blob R3 patch, never on unreviewed PR source or T1.
No experiment runs here. The additional per-step tensor reductions alter runtime;
any resulting benchmark must be measured anew and labelled GUARDED_COST_R4.
"""

from __future__ import annotations
import ast
import hashlib
from pathlib import Path


class HardeningError(ValueError):
    pass


OLD = "        loss.backward()\n        optimizer.step()\n        return float(loss.detach())\n"
NEW = (
    "        loss.backward()\n"
    "        for parameter in model.parameters():\n"
    "            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):\n"
    '                raise FloatingPointError("non-finite gradient before optimizer update")\n'
    "        optimizer.step()\n"
    "        for parameter in model.parameters():\n"
    "            if not bool(torch.isfinite(parameter).all().item()):\n"
    '                raise FloatingPointError("non-finite parameter after optimizer update")\n'
    "        return float(loss.detach())\n"
)
PROTOCOL_ANCHOR = '                     "finite_loss_checked_every_minibatch": True,\n'
PROTOCOL_REPLACEMENT = (
    PROTOCOL_ANCHOR + '                     "finite_gradient_checked_every_minibatch": True,\n'
    '                     "finite_parameters_checked_every_minibatch": True,\n'
    '                     "cost_measurement_kind": "GUARDED_R4_NOT_COMPARABLE_TO_PR91_TIMINGS",\n'
)


def blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def transform(src: bytes) -> bytes:
    if type(src) is not bytes or len(src) > 131072 or b"\r" in src:
        raise HardeningError("source must be bounded LF UTF8 bytes")
    try:
        text = src.decode("utf-8")
    except UnicodeError as exc:
        raise HardeningError("invalid utf8 source") from exc
    if text.count(OLD) != 1 or text.count(PROTOCOL_ANCHOR) != 1:
        raise HardeningError("R3-patched source structure not recognized")
    if 'T_max=args.schedule_epochs' not in text or 'T_max=args.repeats' in text:
        raise HardeningError("R3 schedule guard missing")
    new = text.replace(OLD, NEW).replace(PROTOCOL_ANCHOR, PROTOCOL_REPLACEMENT)
    ast.parse(new, filename="gino_epoch_cost_r4_candidate.py")
    return new.encode("utf-8")


def apply(src: Path, dst: Path) -> dict:
    if src.resolve() == dst.resolve() or dst.exists():
        raise HardeningError("overwrite of source or output forbidden")
    raw = src.read_bytes()
    result = transform(raw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("xb") as f:
        f.write(result)
    return {
        "input_blob_sha1": blob(raw),
        "output_blob_sha1": blob(result),
        "output_sha256": hashlib.sha256(result).hexdigest(),
        "status": "R4_STATIC_PATCH_NOT_MEASURED_NOT_APPROVED",
    }
