"""Commit-marker and digest protocol for multi-file Distributed Checkpoint bundles."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
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
    if not any(Path(item["path"]).name == ".metadata" for item in files):
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
    expected_main_sha: str | None = None,
) -> dict[str, Any]:
    bundle_path = Path(bundle_dir)
    marker_path = bundle_path / "COMMITTED.json"
    manifest_path = bundle_path / "manifest.json"
    if not marker_path.is_file():
        raise RuntimeError("checkpoint bundle is not committed")
    if not manifest_path.is_file():
        raise RuntimeError("committed bundle is missing manifest.json")

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    manifest_data = manifest_path.read_bytes()
    manifest = json.loads(manifest_data)
    if marker.get("schema") != SCHEMA or manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported checkpoint bundle schema")

    actual_manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    if marker.get("manifest_sha256") != actual_manifest_sha:
        raise RuntimeError("manifest digest mismatch")

    metadata = manifest.get("metadata", {})
    if expected_main_sha is not None and metadata.get("main_sha") != expected_main_sha:
        raise RuntimeError("checkpoint main_sha does not match expected main_sha")

    for entry in manifest.get("files", []):
        path = bundle_path / entry["path"]
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
