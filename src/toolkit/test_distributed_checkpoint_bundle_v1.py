"""
test_distributed_checkpoint_bundle_v1.py
========================================
Unit tests for the DCP bundle commit/digest protocol in
``scripts/oracles/distributed_checkpoint_bundle_v1.py`` (PR #54).

The two-rank evidence lane exercises the protocol with real DCP shards; these
tests pin the validator's fail-closed rules on synthetic bundles so they run in
``tests (cpu)``: Codex #54 T2 (marker state), T3 (non-empty inventory with the
DCP .metadata file), T0 (source SHA identity) and O4 (shard directories are
fsynced before the manifest and the marker are written).
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ORACLES = Path(__file__).resolve().parents[2] / "scripts" / "oracles"
if str(_ORACLES) not in sys.path:
    sys.path.insert(0, str(_ORACLES))

import distributed_checkpoint_bundle_v1 as bundle  # noqa: E402

SHA = "a" * 40


def _make_bundle(root: Path) -> Path:
    bdir = root / "bundle"
    (bdir / "dcp").mkdir(parents=True)
    (bdir / "dcp" / ".metadata").write_bytes(b"meta")
    (bdir / "dcp" / "__0_0.distcp").write_bytes(b"shard0")
    bundle.finalize_bundle(bdir, {"source_sha": SHA, "world_size": 2})
    return bdir


def _rewrite_manifest(bdir: Path, mutate) -> None:
    manifest = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
    mutate(manifest)
    data = bundle.json_bytes(manifest)
    (bdir / "manifest.json").write_bytes(data)
    marker = json.loads((bdir / "COMMITTED.json").read_text(encoding="utf-8"))
    marker["manifest_sha256"] = hashlib.sha256(data).hexdigest()
    (bdir / "COMMITTED.json").write_bytes(bundle.json_bytes(marker))


class TestValidateBundle(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.bdir = _make_bundle(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_committed_bundle_is_valid(self) -> None:
        report = bundle.validate_bundle(self.bdir, expected_source_sha=SHA)
        self.assertTrue(report["valid"])
        self.assertEqual(len(report["manifest"]["files"]), 2)

    def test_marker_state_must_be_committed(self) -> None:
        marker = json.loads((self.bdir / "COMMITTED.json").read_text(encoding="utf-8"))
        for state in ("WRITING", None):
            with self.subTest(state=state):
                marker["state"] = state
                (self.bdir / "COMMITTED.json").write_bytes(bundle.json_bytes(marker))
                with self.assertRaisesRegex(RuntimeError, "not 'COMMITTED'"):
                    bundle.validate_bundle(self.bdir)

    def test_empty_or_missing_inventory_is_rejected(self) -> None:
        for mutate in (lambda m: m.__setitem__("files", []), lambda m: m.pop("files")):
            with self.subTest(mutate=mutate):
                _rewrite_manifest(self.bdir, mutate)
                with self.assertRaisesRegex(RuntimeError, "no checkpoint files"):
                    bundle.validate_bundle(self.bdir)

    def test_inventory_without_dcp_metadata_is_rejected(self) -> None:
        _rewrite_manifest(self.bdir, lambda m: m.__setitem__(
            "files", [f for f in m["files"] if not f["path"].endswith(".metadata")]))
        with self.assertRaisesRegex(RuntimeError, r"\.metadata"):
            bundle.validate_bundle(self.bdir)

    def _add_entry(self, rel: str, file_path: Path, *, replace_metadata: bool = False) -> None:
        data = file_path.read_bytes()
        entry = {"path": rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}

        def mutate(manifest):
            files = manifest["files"]
            if replace_metadata:
                files[:] = [f for f in files if not f["path"].endswith(".metadata")]
            files.append(entry)
        _rewrite_manifest(self.bdir, mutate)

    def test_absolute_manifest_path_is_rejected(self) -> None:
        outside = Path(self._tmp.name) / "outside.bin"
        outside.write_bytes(b"x")
        self._add_entry(outside.resolve().as_posix(), outside)
        with self.assertRaisesRegex(RuntimeError, "manifest path"):
            bundle.validate_bundle(self.bdir)

    def test_traversal_manifest_path_is_rejected(self) -> None:
        escaped = self.bdir / "outside.bin"
        escaped.write_bytes(b"x")
        for rel in ("dcp/../outside.bin", "../bundle/outside.bin"):
            with self.subTest(rel=rel):
                self._rewrite_files_without(rel)
                self._add_entry(rel, escaped)
                with self.assertRaisesRegex(RuntimeError, "manifest path"):
                    bundle.validate_bundle(self.bdir)

    def _rewrite_files_without(self, rel: str) -> None:
        _rewrite_manifest(self.bdir, lambda m: m.__setitem__(
            "files", [f for f in m["files"] if f["path"] != rel]))

    def test_decoy_metadata_outside_dcp_is_rejected(self) -> None:
        decoy = self.bdir / "other" / ".metadata"
        decoy.parent.mkdir()
        decoy.write_bytes(b"meta")
        self._add_entry("other/.metadata", decoy, replace_metadata=True)
        with self.assertRaisesRegex(RuntimeError, "manifest path must lie under dcp/"):
            bundle.validate_bundle(self.bdir)

    def test_nested_decoy_metadata_inside_dcp_is_rejected(self) -> None:
        decoy = self.bdir / "dcp" / "sub" / ".metadata"
        decoy.parent.mkdir()
        decoy.write_bytes(b"meta")
        self._add_entry("dcp/sub/.metadata", decoy, replace_metadata=True)
        with self.assertRaisesRegex(RuntimeError, "exactly the DCP metadata file"):
            bundle.validate_bundle(self.bdir)

    def test_source_sha_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "source_sha"):
            bundle.validate_bundle(self.bdir, expected_source_sha="0" * 40)


class TestCheckedManifestPath(unittest.TestCase):
    """First layer of the path confinement (Codex PR #58). validate_bundle
    additionally checks that every resolved path stays inside dcp/."""

    def test_unsafe_forms_are_rejected(self) -> None:
        bad = ["/abs/dcp/x", "C:/dcp/x", "c:dcp/x", "dcp" + chr(92) + "x", "dcp/../x",
               "dcp/./x", "dcp//x", "dcp/x/", "other/.metadata", "dcp", "", None, 5]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                bundle.checked_manifest_path(raw)

    def test_normalized_paths_under_dcp_pass(self) -> None:
        for raw in ("dcp/.metadata", "dcp/__0_0.distcp", "dcp/sub/shard.bin"):
            with self.subTest(raw=raw):
                self.assertEqual(bundle.checked_manifest_path(raw), raw)


class TestFinalizeDurabilityOrder(unittest.TestCase):
    def test_shard_directory_fsynced_before_manifest_and_marker(self) -> None:
        events = []
        real_write = bundle.atomic_write_bytes
        with tempfile.TemporaryDirectory() as tmp:
            bdir = Path(tmp) / "bundle"
            (bdir / "dcp").mkdir(parents=True)
            (bdir / "dcp" / ".metadata").write_bytes(b"meta")
            with mock.patch.object(
                bundle, "fsync_directory",
                side_effect=lambda p: events.append(("fsync", Path(p).name)),
            ), mock.patch.object(
                bundle, "atomic_write_bytes",
                side_effect=lambda p, d: (events.append(("write", Path(p).name)),
                                          real_write(p, d)),
            ):
                bundle.finalize_bundle(bdir, {"source_sha": SHA})
        self.assertEqual(
            events,
            [("fsync", "dcp"), ("fsync", "bundle"),
             ("write", "manifest.json"), ("write", "COMMITTED.json")],
        )


if __name__ == "__main__":
    unittest.main()
