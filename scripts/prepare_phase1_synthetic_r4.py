"""Prepare a hermetic-ish, offline PR#91 synthetic-cost sandbox (no training).

Reads immutable Git objects from a LOCAL clone, snapshots the exact PR commit
without modifying the clone or running an experiment, patches a COPY of the
synthetic cost oracle and records exact provenance. Exit 0 means PREPARED/HOLD,
not scientific validation or permission to train on T1 data.

Run from the package directory, with PYTHONPATH=src/toolkit:scripts on Linux,
or simply invoke this script via its absolute path on Windows.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_ROOT / "src" / "toolkit"))

from extract_pr91_source_evidence_v1 import (  # noqa: E402
    PR_HEAD,
    FILES,
    ExtractError,
    git,
    object_bytes,
)
from patch_synthetic_cost_oracle_v1 import (  # noqa: E402
    ORIGINAL_GIT_BLOB,
    apply as patch_oracle,
)
from harden_r4_synthetic_oracle import apply as harden_r4_patch  # noqa: E402
from preapproval_budget_hex_v3 import (  # noqa: E402
    EvidenceError,
    preflight_pr91,
)

PREREG_PATH = "spec/PREREGISTRATION_PHASE1_v1.md"
PREREG_BLOB = "3ca95c2162d16a18932f7f97caed58a7059a346a"
PREREG_SHA256 = "cbfb8212a575b7a34e32fd753046ad3f9098386192aa7afb7612c97eb7cc4a4f"
ORACLE_PATH = "scripts/oracles/gino_epoch_cost_synthetic.py"
R3_CANDIDATE_PATH = "scripts/oracles/gino_epoch_cost_r3_intermediate.py"
CANDIDATE_PATH = "scripts/oracles/gino_epoch_cost_r4_candidate.py"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 12 * 1024 * 1024
MAX_MEMBERS = 5000


class PreparationError(ValueError):
    """Preparing a candidate without trustworthy sources is refused."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha1_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def read_pinned(repo: Path, revision: str = PR_HEAD) -> dict[str, bytes]:
    """Reject a non-canonical revision, even if content happens to match."""
    if revision != PR_HEAD:
        raise PreparationError("only PR91 exact canonical commit is supported")
    try:
        commit = git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}").strip().decode("ascii")
        if commit != revision:
            raise PreparationError("git commit object mismatch")
        obj = {}
        for rel, blob, sha in FILES:
            raw = object_bytes(repo, revision, rel, blob)
            if _sha(raw) != sha:
                raise PreparationError("pinned file SHA256 mismatch: " + rel)
            obj[rel] = raw
        for path, oid, digest in ((PREREG_PATH, PREREG_BLOB, PREREG_SHA256),):
            raw = object_bytes(repo, revision, path, oid)
            if _sha(raw) != digest:
                raise PreparationError("pinned preregistration SHA256 mismatch")
            if not raw.startswith(b"# Preregistration Phase 1 v1") or b"**Status:** DRAFT" not in raw:
                raise PreparationError("preregistration is not the approved candidate draft")
            obj[path] = raw
        raw_script = object_bytes(repo, revision, ORACLE_PATH, ORIGINAL_GIT_BLOB)
        obj[ORACLE_PATH] = raw_script
        return obj
    except (ExtractError, OSError, UnicodeError) as exc:
        raise PreparationError("canonical source objects inaccessible: " + str(exc)) from exc


def _archive_bytes(
    repo: Path, revision: str = PR_HEAD, *, max_archive_bytes: int = MAX_ARCHIVE_BYTES, timeout_seconds: float = 45.0
) -> bytes:
    """Bounded Git archive with a REAL elapsed-time deadline (R5).

    R4 read stdout with a size limit but no deadline; a process writing no
    bytes could hang indefinitely BEFORE reaching communicate(timeout=30).
    A dedicated reader thread has a timeout; kill/wait closes its pipe.
    Stderr is written to a temporary file to prevent pipe-buffer deadlock.
    """
    if not 0 < max_archive_bytes <= MAX_ARCHIVE_BYTES:
        raise PreparationError('invalid archive byte cap')
    if not 0 < timeout_seconds <= 600:
        raise PreparationError('invalid archive deadline')
    # git archive applies the working-tree conversion: under core.autocrlf=true
    # (the Git for Windows default, and the named D5 machine is Windows) every
    # text member comes out CRLF and no longer equals its blob. Per call only,
    # no configuration is written. --no-replace-objects: a replace ref would swap
    # the pinned commit behind its name, tree included.
    cmd = [
        "git",
        "--no-replace-objects",
        "-c",
        "core.autocrlf=false",
        "-c",
        "core.eol=lf",
        "-C",
        str(repo),
        "archive",
        "--format=tar",
        revision,
        "scripts/oracles",
        "src",
        "spec",
        FILES[0][0],
        FILES[1][0],
    ]
    try:
        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
            assert proc.stdout is not None
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(proc.stdout.read, max_archive_bytes + 1)
                    try:
                        data = future.result(timeout=timeout_seconds)
                    except FutureTimeout as exc:
                        proc.kill()
                        proc.wait(timeout=10)
                        raise PreparationError('git archive timed out') from exc
                if len(data) > max_archive_bytes:
                    proc.kill()
                    proc.wait(timeout=10)
                    raise PreparationError('Git archive exceeds byte limit')
                try:
                    code = proc.wait(timeout=max(5.0, min(10.0, timeout_seconds)))
                except subprocess.TimeoutExpired as exc:
                    proc.kill()
                    proc.wait(timeout=10)
                    raise PreparationError('git archive did not exit') from exc
                if code != 0:
                    err.seek(0)
                    raise PreparationError('local git archive refused: ' + err.read(300).decode('utf-8', 'replace'))
                return data
            finally:
                proc.stdout.close()
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreparationError('local git archive failed') from exc


def unpack_checked(
    tar_data: bytes, dest: Path, *, max_file_bytes: int = MAX_FILE_BYTES, max_members: int = MAX_MEMBERS
) -> dict:
    """No tarfile.extract/extractall; writes only bounded regular files/dirs.

    All archive members are inspected for traversal, links, collisions, special
    files, absolute paths and duplicates BEFORE any contents are written.
    """
    dest = dest.resolve()
    if not isinstance(tar_data, bytes) or len(tar_data) > MAX_ARCHIVE_BYTES:
        raise PreparationError("archive byte budget exceeded")
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:") as tf:
            members = tf.getmembers()
            if len(members) > max_members:
                raise PreparationError("too many archive entries")
            seen = set()
            seen_casefold = set()
            aggregate = 0
            canonical = []
            for info in members:
                p = PurePosixPath(info.name)
                if (
                    not info.name
                    or p.is_absolute()
                    or "\\" in info.name
                    or any(s in ("..", ".", "") for s in info.name.rstrip("/").split("/"))
                    or not (info.isdir() or info.isfile())
                ):
                    raise PreparationError("unsafe tar member (path/link/special type)")
                # Directory entry may carry trailing slash and collide with a file.
                key = p.as_posix()
                if key in seen:
                    raise PreparationError("duplicate tar member")
                # The intended D5 machine is Windows: case-insensitive paths
                # and reserved DOS device names must not alias other files.
                if key.casefold() in seen_casefold:
                    raise PreparationError("case-insensitive tar member collision")
                for component in p.parts:
                    upper = component.rstrip(" .").split(".")[0].upper()
                    if (
                        component.endswith((" ", "."))
                        or ":" in component
                        or len(component) > 255
                        or upper in {"CON", "PRN", "AUX", "NUL"}
                        or upper in {f"COM{i}" for i in range(1, 10)}
                        or upper in {f"LPT{i}" for i in range(1, 10)}
                    ):
                        raise PreparationError("nonportable Windows tar path")
                seen.add(key)
                seen_casefold.add(key.casefold())
                if info.isfile():
                    if info.size < 0 or info.size > max_file_bytes:
                        raise PreparationError("archive file size exceeded")
                    aggregate += info.size
                    if aggregate > MAX_ARCHIVE_BYTES:
                        raise PreparationError("total unpacked bytes exceeded")
                canonical.append((p, info))
            # File/dir ancestor collisions are rejected before any write.
            files = {p.as_posix() for p, info in canonical if info.isfile()}
            for p, _ in canonical:
                for i in range(1, len(p.parts)):
                    if PurePosixPath(*p.parts[:i]).as_posix() in files:
                        raise PreparationError("file/dir ancestor collision")
            count = 0
            for rel, info in canonical:
                target = dest.joinpath(*rel.parts)
                if info.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = tf.extractfile(info)
                if stream is None:
                    raise PreparationError("missing tar payload")
                data = stream.read(info.size + 1)
                if len(data) != info.size:
                    raise PreparationError("truncated tar payload")
                with target.open("xb") as f:
                    f.write(data)
                count += 1
            return {
                "extracted_file_count": count,
                "tar_sha256": _sha(tar_data),
                "tar_bytes": len(tar_data),
                "total_extracted_bytes": aggregate,
            }
    except (tarfile.TarError, EOFError) as exc:
        raise PreparationError("invalid tar archive") from exc


def environment_probe() -> dict:
    """Observation only. Never claims the current machine is the pinned D5."""

    def package(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": package("torch"),
        "numpy": package("numpy"),
        "neuraloperator": package("neuraloperator"),
        "cpu_logical_count": os.cpu_count(),
        "required_d5": {
            "os": "Windows 11 build 26200",
            "python": "3.12.10",
            "torch": "2.6.0+cpu",
            "numpy": "2.5.3",
            "neuraloperator": "2.0.0",
            "threads": 4,
            "cpu": "Intel Core i5-1035G1",
        },
        "is_d5_verified": False,
        "requires_named_machine_and_resource_attestation": True,
    }


def prepare(repo: Path, output: Path) -> dict:
    """Exactly pinned candidate: no T1 generation, no source clone writes."""
    repo = Path(repo).resolve()
    output = Path(output).resolve()
    if not repo.is_dir():
        raise PreparationError("local source repository does not exist")
    if output.exists():
        raise PreparationError("output exists: never replace a prior run")
    if not output.parent.is_dir():
        raise PreparationError("output parent directory does not exist")
    canonical = read_pinned(repo, revision=PR_HEAD)
    tar_data = _archive_bytes(repo, revision=PR_HEAD)
    staging = Path(tempfile.mkdtemp(prefix=".hexa-r4-staging-", dir=output.parent))
    try:
        artifact = staging / "snapshot"
        artifact.mkdir()
        archive_stats = unpack_checked(tar_data, artifact)
        # Validate archive snapshot independently against the Git object bytes.
        for rel, original in canonical.items():
            if (artifact / rel).read_bytes() != original:
                raise PreparationError("Git archive snapshot differs from canonical blob: " + rel)
        cost = artifact / FILES[0][0]
        radius = artifact / FILES[1][0]
        strict = preflight_pr91(cost, radius)
        # Preserve the original oracle unmodified, patch only an added candidate.
        patch_r3 = patch_oracle(artifact / ORACLE_PATH, artifact / R3_CANDIDATE_PATH)
        patch_r4 = harden_r4_patch(artifact / R3_CANDIDATE_PATH, artifact / CANDIDATE_PATH)
        # byte-for-byte match original candidate source, not only loosely parsed values
        if _sha((artifact / ORACLE_PATH).read_bytes()) != _sha(canonical[ORACLE_PATH]):
            raise PreparationError("original oracle was modified while patching")
        manifest = {
            "schema_version": "HEXA_NTK_PHASE1_R4_PREPARE/1",
            "git_pr_head": PR_HEAD,
            "source_tier": "CANONICAL_GIT_OBJECTS_PR91",
            "preregistration": {
                "path": PREREG_PATH,
                "blob_sha1": PREREG_BLOB,
                "sha256": PREREG_SHA256,
                "status": "DRAFT_NOT_BINDING",
            },
            "source_evidence": {
                "cost_sha256": FILES[0][2],
                "radius_sha256": FILES[1][2],
                "radius_verifier_policy": strict.get("policy"),
            },
            "patch": {"r3": patch_r3, "r4": patch_r4},
            "archive": archive_stats,
            "runtime_environment": environment_probe(),
            "synthetic_oracle_relative_path": CANDIDATE_PATH,
            "run_performed": False,
            "t1_data_generated_or_fitted": False,
            "owner_decision": "HOLD_NO_APPROVAL",
            "safe_next_action": (
                "review proposed patch and run the candidate ONLY on pinned D5 "
                "with synthetic inputs; collect full overhead separately"
            ),
            "no_data_assertion": "synthetic normal tensors; original benchmark source pinned by blob hash",
        }
        (staging / "R4_PREPARATION_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging)
        raise


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path, help="local Git repository containing PR91 HEAD")
    p.add_argument("--out", required=True, type=Path, help="new directory; no overwrite")
    args = p.parse_args(argv)
    try:
        result = prepare(args.repo, args.out)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (PreparationError, EvidenceError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "HEXA_NTK_PHASE1_R4_PREPARE/1",
                    "status": "HOLD_PREPARATION_REFUSED",
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
