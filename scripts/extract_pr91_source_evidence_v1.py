"""Offline, immutable PR #91 Git-object evidence extractor (no checkout changes).

Requires a LOCAL Git repository containing the exact PR commit object. Reads
objects from its Git object store, never from the dirty working tree, and
writes a new evidence directory only after BOTH byte pins are verified.
No network, training, Git staging, branch switch, commit or push.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PR_HEAD = "1a54cd8d9326e762ee493d3fc8d9a80202174304"
FILES = (
    (
        "results/2026-10-29_gino_epoch_cost_synthetic.json",
        "cc6bfbfcd95127915d520f283740ec7c61b206f5",
        "8d9212597ba5e827f2c1d9fd707dd20d6c5755d5e70d1e693869dd1ec82f61ed",
    ),
    (
        "results/2026-10-20_gino_radius_tie_survey_float32-graphs.json",
        "9073f51b26cf77165cc4cea37bb8377fd170dd05",
        "3111a480fa934e0bf027a092f4fa389fdb4faa500edf60a6a737160df471b786",
    ),
)
SOURCE_CAP_BYTES = 1_048_576


class ExtractError(ValueError):
    pass


def git(repo: Path, *args: str) -> bytes:
    try:
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExtractError(f"local Git object read failed: {type(exc).__name__}") from exc
    if p.returncode != 0 or len(p.stdout) > SOURCE_CAP_BYTES:
        raise ExtractError(f"local Git object missing/too large: {' '.join(args[:2])}")
    return p.stdout


def object_bytes(repo: Path, revision: str, relative: str, oid: str) -> bytes:
    tree_entry = git(repo, "ls-tree", revision, "--", relative)
    # ASCII tokenization of Git ls-tree: `<mode> <type> <oid>\t<path>`
    expected = f"100644 blob {oid}\t{relative}\n".encode()
    if tree_entry != expected:
        raise ExtractError(f"tree/path/mode mismatch for {relative}")
    raw = git(repo, "cat-file", "blob", f"{revision}:{relative}")
    sha1 = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if sha1 != oid:
        raise ExtractError(f"Git blob bytes mismatch for {relative}")
    return raw


def extract(repo: Path, output: Path, *, head: str = PR_HEAD, files=FILES) -> dict:
    root = Path(repo).resolve()
    output = Path(output).resolve()
    if not root.is_dir():
        raise ExtractError("local repo directory does not exist")
    if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise ExtractError("head is not a pinned Git commit")
    commit = git(root, "rev-parse", "--verify", f"{head}^{{commit}}").strip().decode("ascii")
    if commit != head:
        raise ExtractError("Git reference does not identify the exact commit")
    if output.exists():
        raise ExtractError("output directory must not exist: refusing overwrite")
    payload = []
    for relative, oid, sha256 in files:
        if not relative.startswith("results/") or ".." in Path(relative).parts:
            raise ExtractError("unsafe evidence path")
        raw = object_bytes(root, head, relative, oid)
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise ExtractError(f"SHA256 mismatch for {relative}")
        payload.append((Path(relative).name, raw, sha256, oid))
    if len({x[0] for x in payload}) != len(payload):
        raise ExtractError("filename collision")
    result = {
        "pr_head": head,
        "source": "EXACT_LOCAL_GIT_OBJECTS_NOT_WORKTREE",
        "owner_approved": False,
        "scientific_result": False,
        "files": [
            {"name": name, "size_bytes": len(raw), "sha256": sha, "git_blob_sha1": oid}
            for name, raw, sha, oid in payload
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, raw, _, _ in payload:
            with (output / name).open("xb") as f:
                f.write(raw)
        with (output / "EVIDENCE_MANIFEST.json").open("x", encoding="utf-8") as f:
            f.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    except Exception:
        # Remove only files created by this function, not arbitrary user files.
        for name, _, _, _ in payload:
            (output / name).unlink(missing_ok=True)
        (output / "EVIDENCE_MANIFEST.json").unlink(missing_ok=True)
        output.rmdir()
        raise
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(extract(args.repo, args.out), indent=2, sort_keys=True))
        return 0
    except (ExtractError, OSError, UnicodeError) as exc:
        print(f"LOCAL_PR91_EXTRACTION_REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
