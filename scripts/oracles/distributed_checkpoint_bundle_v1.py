"""Commit-marker and digest protocol for multi-file Distributed Checkpoint bundles."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
from typing import Any, Mapping

SCHEMA = "hexantk_distributed_checkpoint_bundle_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


DCP_METADATA = "dcp/.metadata"


def checked_manifest_path(raw: object) -> str:
    """Return a manifest path only if it is a normalized relative POSIX path
    strictly below ``dcp/`` (Codex PR #58): no absolute or drive paths, no
    backslashes, no empty/./.. components, no decoys outside the DCP tree."""
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f"manifest path must be a non-empty string, got {raw!r}")
    if "\\" in raw or ":" in raw:
        raise RuntimeError(f"manifest path must be POSIX without drive or colon: {raw!r}")
    posix = PurePosixPath(raw)
    if posix.is_absolute() or PureWindowsPath(raw).is_absolute():
        raise RuntimeError(f"manifest path must be relative: {raw!r}")
    parts = raw.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise RuntimeError(f"manifest path must be normalized: {raw!r}")
    if len(parts) < 2 or parts[0] != "dcp":
        raise RuntimeError(f"manifest path must lie under dcp/: {raw!r}")
    return raw


def fsync_directory(path: Path) -> None:
    """Persist directory entries (POSIX). No-op where directories cannot be opened."""
    if os.name != "posix":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def fsync_file(path: Path) -> None:
    """Persist a file's CONTENTS to disk (POSIX). No-op on other platforms,
    the same gate as :func:`fsync_directory`: there ``os.replace`` plus the DCP
    writer's own flush provide the durability, so there is no portable
    read-only fsync to perform."""
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def build_manifest(
    bundle_dir: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    dcp_dir = bundle_dir / "dcp"
    if not dcp_dir.is_dir():
        raise FileNotFoundError("bundle is missing dcp directory")
    files = []
    for path in sorted(item for item in dcp_dir.rglob("*") if item.is_file()):
        files.append(
            {
                "path": str(path.relative_to(bundle_dir)).replace(os.sep, "/"),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError("dcp directory contains no checkpoint files")
    if not any(item["path"] == DCP_METADATA for item in files):
        raise ValueError("DCP checkpoint is missing .metadata")
    return {
        "schema": SCHEMA,
        "metadata": dict(metadata),
        "files": files,
    }


def finalize_bundle(
    bundle_dir: str | Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write COMMITTED.json last, after DCP completion and the caller's barrier."""
    bundle_path = Path(bundle_dir)
    manifest = build_manifest(bundle_path, metadata)
    # A shard's directory entry being durable does not make its CONTENTS
    # durable: after a power loss the commit marker could survive while a shard
    # is lost or truncated (issue #59 P2). Persist in the order
    # contents -> directory -> marker: fsync every shard file's contents, then
    # every directory that holds a shard, then the bundle directory, before the
    # manifest and the marker claim them (O4).
    for item in manifest["files"]:
        fsync_file(bundle_path / item["path"])
    shard_dirs = sorted({(bundle_path / item["path"]).parent for item in manifest["files"]})
    for directory in shard_dirs:
        fsync_directory(directory)
    fsync_directory(bundle_path)
    manifest_data = json_bytes(manifest)
    atomic_write_bytes(bundle_path / "manifest.json", manifest_data)
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    marker = {
        "schema": SCHEMA,
        "manifest_sha256": manifest_sha,
        "state": "COMMITTED",
    }
    atomic_write_bytes(bundle_path / "COMMITTED.json", json_bytes(marker))
    return {
        "manifest_sha256": manifest_sha,
        "n_files": len(manifest["files"]),
    }


def validate_bundle(
    bundle_dir: str | Path,
    *,
    expected_source_sha: str | None = None,
) -> dict[str, Any]:
    bundle_path = Path(bundle_dir)
    marker_path = bundle_path / "COMMITTED.json"
    manifest_path = bundle_path / "manifest.json"
    if not marker_path.is_file():
        raise RuntimeError("checkpoint bundle is not committed")
    if not manifest_path.is_file():
        raise RuntimeError("committed bundle is missing manifest.json")

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("state") != "COMMITTED":
        # Codex #54 T2: a syntactically valid marker in any other state
        # (e.g. WRITING) is not a commit.
        raise RuntimeError(f"bundle marker state is {marker.get('state')!r}, not 'COMMITTED'")
    manifest_data = manifest_path.read_bytes()
    manifest = json.loads(manifest_data)
    if marker.get("schema") != SCHEMA or manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported checkpoint bundle schema")

    actual_manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    if marker.get("manifest_sha256") != actual_manifest_sha:
        raise RuntimeError("manifest digest mismatch")

    metadata = manifest.get("metadata", {})
    if expected_source_sha is not None and metadata.get("source_sha") != expected_source_sha:
        raise RuntimeError("checkpoint source_sha does not match expected source_sha")

    files = manifest.get("files")
    # Codex #54 T3: mirror build_manifest -- a bundle without checkpoint files
    # or without the DCP .metadata file is not valid.
    if not isinstance(files, list) or not files:
        raise RuntimeError("manifest lists no checkpoint files")
    paths = [checked_manifest_path(entry.get("path") if isinstance(entry, dict) else None)
             for entry in files]
    if len(set(paths)) != len(paths):
        raise RuntimeError("manifest lists a path more than once")
    metadata_entries = [p for p in paths if PurePosixPath(p).name == ".metadata"]
    if metadata_entries != [DCP_METADATA]:
        raise RuntimeError(
            f"manifest must list exactly the DCP metadata file {DCP_METADATA!r}, "
            f"got {metadata_entries!r}"
        )
    # Closed inventory (Codex PR #58 round 2): every file under dcp/ must be
    # listed, otherwise an omitted shard could be corrupted and still be
    # consumed by dcp.load while the bundle validates.
    on_disk = {
        item.relative_to(bundle_path).as_posix()
        for item in (bundle_path / "dcp").rglob("*")
        if item.is_file() or item.is_symlink()
    }
    unlisted = sorted(on_disk - set(paths))
    if unlisted:
        raise RuntimeError(f"files under dcp/ missing from the manifest: {unlisted}")
    dcp_root = (bundle_path / "dcp").resolve()
    for entry in files:
        path = bundle_path / entry["path"]
        if not path.resolve().is_relative_to(dcp_root):
            raise RuntimeError(f"manifest path escapes dcp/: {entry['path']}")
        if not path.is_file():
            raise RuntimeError(f"checkpoint file missing: {entry['path']}")
        if path.stat().st_size != entry["size"]:
            raise RuntimeError(f"checkpoint size mismatch: {entry['path']}")
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"checkpoint digest mismatch: {entry['path']}")
    return {
        "valid": True,
        "manifest": manifest,
        "manifest_sha256": actual_manifest_sha,
    }
