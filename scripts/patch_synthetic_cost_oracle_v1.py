"""Hash-pinned, NON-in-place patch for original PR #91 synthetic GINO oracle.

Does not run Torch, generate data or change git.  The output is a candidate
source file for review in a NEW local branch. Never change a bound source JSON.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ORIGINAL_GIT_BLOB = "9b78231175318ff27ef2828d6e4fd591f24ec25f"
CAP = 128 * 1024

REPLACEMENTS = (
    (
        "        loss = _relative_l2(predict_gino(model, lattice, u0[index], config), target[index])\n"
        "        loss.backward()\n",
        "        loss = _relative_l2(predict_gino(model, lattice, u0[index], config), target[index])\n"
        "        if not bool(torch.isfinite(loss).item()):\n"
        "            raise FloatingPointError(\"non-finite minibatch loss before optimizer update\")\n"
        "        loss.backward()\n",
    ),
    (
        "    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.repeats)\n",
        "    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.schedule_epochs)\n",
    ),
    (
        "        step(order[first:first + args.batch_size])\n    warmup_seconds = time.perf_counter() - start\n",
        "        warmup_loss = step(order[first:first + args.batch_size])\n"
        "        if not math.isfinite(warmup_loss):\n"
        "            raise FloatingPointError(\"non-finite warmup minibatch loss\")\n"
        "    warmup_seconds = time.perf_counter() - start\n",
    ),
    (
        "            last = step(order[first:first + args.batch_size])\n        scheduler.step()\n",
        "            last = step(order[first:first + args.batch_size])\n"
        "            if not math.isfinite(last):\n"
        "                raise FloatingPointError(\"non-finite measured minibatch loss\")\n"
        "        scheduler.step()\n",
    ),
    (
        '    parser.add_argument("--repeats", type=_positive(int), default=3)\n',
        '    parser.add_argument("--repeats", type=_positive(int), default=3)\n'
        '    parser.add_argument("--schedule-epochs", type=_positive(int), default=50,\n'
        '                        help="T_max of preregistered cosine schedule; independent of timed epochs")\n',
    ),
    (
        '    if args.batch_size > args.n_train:\n',
        '    if args.schedule_epochs != 50:\n'
        '        parser.error("D4 draft requires --schedule-epochs=50; a different run needs a new protocol")\n'
        '    if args.batch_size > args.n_train:\n',
    ),
    (
        '        "protocol": {"warmup_steps": args.warmup_steps, "warmup_val_passes": 1,\n',
        '        "protocol": {"warmup_steps": args.warmup_steps, "warmup_val_passes": 1,\n'
        '                     "scheduler_t_max": args.schedule_epochs,\n'
        '                     "finite_loss_checked_every_minibatch": True,\n',
    ),
)


class PatchError(ValueError):
    pass


def blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def transform(source: bytes, *, require_blob: bool = True) -> bytes:
    if not isinstance(source, bytes) or len(source) > CAP:
        raise PatchError("invalid source bytes")
    if require_blob and blob(source) != ORIGINAL_GIT_BLOB:
        raise PatchError("source must match immutable GitHub PR91 original blob")
    try:
        text = source.decode("utf-8")
    except UnicodeError as exc:
        raise PatchError("original source is not UTF-8") from exc
    if "\r" in text:
        raise PatchError("source must use canonical LF")
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise PatchError("patch anchor was changed or is ambiguous: " + old[:88].rstrip())
        text = text.replace(old, new)
    if "T_max=args.repeats" in text:
        raise PatchError("old scheduler remains")
    return text.encode("utf-8")


def apply(original: Path, destination: Path) -> dict:
    if original.resolve() == destination.resolve():
        raise PatchError("in-place overwrite refused")
    if destination.exists():
        raise PatchError("destination exists; overwrite refused")
    with original.open("rb") as stream:
        raw = stream.read(CAP + 1)
    patched = transform(raw)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(patched)
    return {
        "original_git_blob": blob(raw),
        "new_git_blob": blob(patched),
        "new_sha256": hashlib.sha256(patched).hexdigest(),
        "status": "PATCHED_SOURCE_NOT_MEASURED_NOT_APPROVED",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="unmodified script from PR91 commit")
    parser.add_argument("output", type=Path, help="new file; never original path")
    args = parser.parse_args(argv)
    try:
        print(apply(args.source, args.output))
        return 0
    except (PatchError, OSError) as exc:
        print(f"PATCH_REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
