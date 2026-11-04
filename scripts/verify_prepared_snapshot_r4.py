"""Read-only second-person audit of a prepared HexaNTK R4 synthetic sandbox.

Recomputes derived source bytes from the original, never trusts hashes written
in the preparation manifest. Even a passing audit means code-copy integrity,
not D5 machine attestation, operator performance, or authorisation to train T1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src/toolkit"))
import prepare_phase1_synthetic_r4 as prep  # noqa: E402
import patch_synthetic_cost_oracle_v1 as r3  # noqa: E402
import harden_r4_synthetic_oracle as r4  # noqa: E402
from preapproval_budget_hex_v3 import _decode_evidence, preflight_pr91  # noqa: E402


class VerificationError(ValueError):
    pass


def _safe_read(root: Path, relative: str, cap: int = 1_048_576) -> bytes:
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise VerificationError("unsafe path")
    # Reject symlink escape, including parent directories.
    current = root
    for bit in part.parts:
        current = current / bit
        if current.is_symlink():
            raise VerificationError("symlink in evidence path")
    # O_NONBLOCK: a FIFO must not hang the audit. The type check runs on the open
    # descriptor, so swapping the path after a separate stat cannot pass it.
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(current, flags)
    except OSError as exc:
        raise VerificationError("verification input cannot be opened") from exc
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise VerificationError("special file in evidence path")
        raw = f.read(cap + 1)
    if len(raw) > cap:
        raise VerificationError("verification input exceeds byte budget")
    return raw


def verify_prepared(prepared_dir: Path) -> dict:
    root = Path(prepared_dir).resolve()
    raw = _safe_read(root, "R4_PREPARATION_MANIFEST.json", 256 * 1024)
    manifest = _decode_evidence(raw)
    if (
        manifest.get("schema_version") != "HEXA_NTK_PHASE1_R4_PREPARE/1"
        or manifest.get("git_pr_head") != prep.PR_HEAD
        or manifest.get("source_tier") != "CANONICAL_GIT_OBJECTS_PR91"
        or manifest.get("run_performed") is not False
        or manifest.get("t1_data_generated_or_fitted") is not False
        or manifest.get("owner_decision") != "HOLD_NO_APPROVAL"
    ):
        raise VerificationError("invalid preparation provenance/approval state")
    snapshot = root / "snapshot"
    for path, _, sha in prep.FILES:
        if hashlib.sha256(_safe_read(snapshot, path)).hexdigest() != sha:
            raise VerificationError("evidence byte hash mismatch: " + path)
    doc = _safe_read(snapshot, prep.PREREG_PATH)
    if hashlib.sha256(doc).hexdigest() != prep.PREREG_SHA256 or b"**Status:** DRAFT" not in doc:
        raise VerificationError("preregistration bytes/approval status changed")
    source = _safe_read(snapshot, prep.ORACLE_PATH)
    if r3.blob(source) != prep.ORIGINAL_GIT_BLOB:
        raise VerificationError("original oracle bytes changed")
    expected_r3 = r3.transform(source)
    expected_r4 = r4.transform(expected_r3)
    a = _safe_read(snapshot, prep.R3_CANDIDATE_PATH)
    b = _safe_read(snapshot, prep.CANDIDATE_PATH)
    if a != expected_r3 or b != expected_r4:
        raise VerificationError("patched oracle differs from reproducible transformation")
    patches = manifest.get("patch")
    if (
        not isinstance(patches, dict)
        or not isinstance(patches.get("r3"), dict)
        or not isinstance(patches.get("r4"), dict)
    ):
        raise VerificationError("missing patch provenance")
    if (
        patches["r3"].get("new_sha256") != hashlib.sha256(a).hexdigest()
        or patches["r4"].get("output_sha256") != hashlib.sha256(b).hexdigest()
    ):
        raise VerificationError("patch manifest differs from recomputed bytes")
    # The strict two-source semantic gate also remains a mandatory invariant.
    checked = preflight_pr91(snapshot / prep.FILES[0][0], snapshot / prep.FILES[1][0])
    if checked.get("evidence_tier") != "TWO_EVIDENCE_FILES_BYTE_PINNED_PR91_CANDIDATE_ONLY":
        raise VerificationError("strict two-file candidate gate unavailable")
    return {
        "schema_version": "HEXA_NTK_PHASE1_R4_VERIFY/1",
        "source_tier": "CANONICAL_PR91_BYTES_AND_TRANSFORMS_VERIFIED",
        "integrity": "PASS",
        "owner_decision": "HOLD_NO_APPROVAL",
        "scientific_result": False,
        "run_performed": False,
        "candidate_sha256": hashlib.sha256(b).hexdigest(),
        "original_oracle_git_blob": r3.blob(source),
        "preregistration_sha256": hashlib.sha256(doc).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_dir", type=Path)
    a = parser.parse_args(argv)
    try:
        print(json.dumps(verify_prepared(a.prepared_dir), indent=2, sort_keys=True))
        return 0
    except (VerificationError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "HEXA_NTK_PHASE1_R4_VERIFY/1",
                    "integrity": "FAIL_CLOSED",
                    "owner_decision": "HOLD_NO_APPROVAL",
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
