"""R4 no-T1-test suite: tar safety, source guards, patch-chain, fail-closed CLI."""

from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "toolkit"))
import prepare_phase1_synthetic_r4 as prep  # noqa: E402
import harden_r4_synthetic_oracle as hard  # noqa: E402
import patch_synthetic_cost_oracle_v1 as r3  # noqa: E402
import verify_prepared_snapshot_r4 as verify  # noqa: E402


def archive(entries):
    """List of (name, kind, content)."""
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w") as t:
        for path, kind, content in entries:
            ti = tarfile.TarInfo(path)
            if kind == "dir":
                ti.type = tarfile.DIRTYPE
                ti.size = 0
            elif kind == "sym":
                ti.type = tarfile.SYMTYPE
                ti.linkname = "../../etc/passwd"
            elif kind == "hard":
                ti.type = tarfile.LNKTYPE
                ti.linkname = "target"
            elif kind == "fifo":
                ti.type = tarfile.FIFOTYPE
            else:
                ti.type = tarfile.REGTYPE
                ti.size = len(content)
            t.addfile(ti, io.BytesIO(content) if kind == "file" else None)
    return b.getvalue()


class ArchiveTests(unittest.TestCase):
    def extract(self, entries, **kwargs):
        with tempfile.TemporaryDirectory() as p:
            return prep.unpack_checked(archive(entries), Path(p), **kwargs)

    def test_positive_snapshot(self):
        with tempfile.TemporaryDirectory() as p:
            d = Path(p)
            tar = archive(
                [("src/", "dir", b""), ("src/a.py", "file", b"print(1)\n"), ("spec/doc.md", "file", b"# spec\n")]
            )
            r = prep.unpack_checked(tar, d)
            self.assertEqual(r["extracted_file_count"], 2)
            self.assertEqual((d / "src/a.py").read_bytes(), b"print(1)\n")
            self.assertEqual((d / "spec/doc.md").read_bytes(), b"# spec\n")
            self.assertEqual(r["total_extracted_bytes"], len(b"print(1)\n# spec\n"))

    def test_reject_path_traversal(self):
        for n in ("../evil", "d/../../evil", "/tmp/evil", "d/./file", "a//b", "a\\b"):
            with self.subTest(n=n), self.assertRaises(prep.PreparationError):
                self.extract([(n, "file", b"bad")])

    def test_reject_external_symlinks_hardlinks_fifo(self):
        for kind in ("sym", "hard", "fifo"):
            with self.subTest(kind=kind), self.assertRaises(prep.PreparationError):
                self.extract([("bad", kind, b"")])

    def test_duplicate_entries_refused_before_write(self):
        with tempfile.TemporaryDirectory() as p:
            d = Path(p)
            with self.assertRaises(prep.PreparationError):
                prep.unpack_checked(archive([("x", "file", b"1"), ("x", "file", b"2")]), d)
            self.assertFalse((d / "x").exists())

    def test_windows_case_collision_refused(self):
        with self.assertRaises(prep.PreparationError):
            self.extract([("src/A.py", "file", b"1"), ("src/a.py", "file", b"2")])

    def test_windows_reserved_and_invalid_names_refused(self):
        for name in ("src/CON", "src/CON.txt", "src/aux.txt", "src/COM1", "src/a:", "src/b. ", "src/c."):
            with self.subTest(name=name), self.assertRaises(prep.PreparationError):
                self.extract([(name, "file", b"1")])

    def test_file_dir_ancestor_collision(self):
        with self.assertRaises(prep.PreparationError):
            self.extract([("src", "file", b"a"), ("src/child.py", "file", b"b")])

    def test_file_size_limit(self):
        with self.assertRaises(prep.PreparationError):
            self.extract([("a", "file", b"123456")], max_file_bytes=5)

    def test_member_count_limit(self):
        with self.assertRaises(prep.PreparationError):
            self.extract([("a", "file", b"1"), ("b", "file", b"2")], max_members=1)

    def test_invalid_tar(self):
        with tempfile.TemporaryDirectory() as p:
            with self.assertRaises(prep.PreparationError):
                prep.unpack_checked(b"not a tar", Path(p))

    def test_non_bytes_tar(self):
        with tempfile.TemporaryDirectory() as p:
            with self.assertRaises(prep.PreparationError):
                prep.unpack_checked("not bytes", Path(p))

    def test_valid_archive_matches_git_archive(self):
        with tempfile.TemporaryDirectory() as p:
            repo = Path(p) / "r"
            out = Path(p) / "out"
            repo.mkdir()
            out.mkdir()
            subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
            (repo / "src").mkdir()
            (repo / "scripts/oracles").mkdir(parents=True)
            (repo / "spec").mkdir()
            (repo / "results").mkdir()
            (repo / "src/x.py").write_text("x = 3\n")
            (repo / "scripts/oracles/a.py").write_text("pass\n")
            (repo / "spec/a.md").write_text("# ok\n")
            (repo / "results/one.json").write_text("{}\n")
            (repo / "results/two.json").write_text("[]\n")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=test",
                    "-c",
                    "user.email=test@example.org",
                    "commit",
                    "-q",
                    "-m",
                    "test",
                ],
                check=True,
            )
            rev = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
            # The test does NOT bypass the production exact-pr-head guard; here
            # we test the tar creation only with a fabricated local repo.
            from unittest.mock import patch

            with patch.object(prep, "FILES", (("results/one.json", "x", "y"), ("results/two.json", "x", "y"))):
                tar = prep._archive_bytes(repo, rev)
            stats = prep.unpack_checked(tar, out)
            self.assertEqual(stats["extracted_file_count"], 5)
            self.assertEqual((out / "src/x.py").read_text(), "x = 3\n")
            self.assertFalse((out / ".git").exists())


class PreparationTests(unittest.TestCase):
    def test_pr_head_is_nonoverridable(self):
        with self.assertRaisesRegex(prep.PreparationError, "canonical commit"):
            prep.read_pinned(Path("/tmp"), revision="a" * 40)

    def test_invalid_source_does_not_create_folder(self):
        with tempfile.TemporaryDirectory() as p:
            d = Path(p)
            out = d / "NEVER_CREATED"
            with self.assertRaises(prep.PreparationError):
                prep.prepare(d, out)
            self.assertFalse(out.exists())

    def test_refuse_to_overwrite_output_even_if_repo_invalid(self):
        with tempfile.TemporaryDirectory() as p:
            d = Path(p)
            out = d / "existing"
            out.mkdir()
            (out / "keep.txt").write_text("keep")
            with self.assertRaisesRegex(prep.PreparationError, "output exists"):
                prep.prepare(d, out)
            self.assertEqual((out / "keep.txt").read_text(), "keep")

    def test_env_never_implies_authorisation(self):
        env = prep.environment_probe()
        self.assertFalse(env["is_d5_verified"])
        self.assertTrue(env["requires_named_machine_and_resource_attestation"])
        self.assertEqual(env["required_d5"]["neuraloperator"], "2.0.0")

    def test_cli_missing_checkout_fails_closed(self):
        with tempfile.TemporaryDirectory() as p:
            path = Path(p) / "missing"
            out = Path(p) / "out"
            done = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/prepare_phase1_synthetic_r4.py"),
                    "--repo",
                    str(path),
                    "--out",
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(done.returncode, 2)
            self.assertIn("HOLD_PREPARATION_REFUSED", done.stderr)
            self.assertFalse(out.exists())

    def test_snapshot_verifier_fails_closed_on_missing_files(self):
        with tempfile.TemporaryDirectory() as td:
            data = verify.main([td])
            self.assertEqual(data, 2)

    def test_snapshot_verifier_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            try:
                (root / "R4_PREPARATION_MANIFEST.json").symlink_to("/etc/passwd")
            except OSError as exc:
                self.skipTest(f"cannot create a symlink here: {exc}")
            with self.assertRaises(verify.VerificationError):
                verify.verify_prepared(root)

    def test_git_blob_hash_helper(self):
        self.assertEqual(prep._sha1_blob(b"hello"), "b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0")


class HardeningPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = (
            """#!/usr/bin/env python3

def _time_config(args, optimizer, model, lattice, u0, target, config, order):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.repeats)
    def step(index):
        optimizer.zero_grad(set_to_none=True)
        loss = _relative_l2(predict_gino(model, lattice, u0[index], config), target[index])
        loss.backward()
        optimizer.step()
        return float(loss.detach())
    for k in range(args.warmup_steps):
        step(order[first:first + args.batch_size])
    warmup_seconds = time.perf_counter() - start
    for _ in range(args.repeats):
        for first in range(0, args.n_train, args.batch_size):
            last = step(order[first:first + args.batch_size])
        scheduler.step()

def build_report(args):
    return {
        "protocol": {"warmup_steps": args.warmup_steps, "warmup_val_passes": 1,
                     "measured_epochs": args.repeats},
    }

def main(args, parser):
    parser.add_argument("--repeats", type=_positive(int), default=3)
    if args.batch_size > args.n_train:
        parser.error("bad batch")
"""
        ).encode("utf-8")
        cls.r3 = r3.transform(cls.sample, require_blob=False)

    def test_r4_candidate_transformation(self):
        result = hard.transform(self.r3)
        self.assertIn(b"non-finite gradient before optimizer update", result)
        self.assertIn(b"non-finite parameter after optimizer update", result)
        self.assertIn(b"T_max=args.schedule_epochs", result)
        self.assertIn(b'"cost_measurement_kind": "GUARDED_R4', result)
        self.assertNotIn(b"T_max=args.repeats", result)

    def test_gradient_guard_precedes_optimizer(self):
        content = hard.transform(self.r3).decode("utf8")
        self.assertLess(content.index('non-finite gradient'), content.index('optimizer.step()'))
        self.assertGreater(content.index('non-finite parameter'), content.index('optimizer.step()'))

    def test_unpatched_original_refused(self):
        with self.assertRaises(hard.HardeningError):
            hard.transform(self.sample)

    def test_second_r4_patch_refused(self):
        once = hard.transform(self.r3)
        with self.assertRaises(hard.HardeningError):
            hard.transform(once)

    def test_source_conflict_refused(self):
        with self.assertRaises(hard.HardeningError):
            hard.transform(self.r3.replace(b'loss.backward()', b'loss.backward__()'))

    def test_duplicate_patch_anchor_refused(self):
        with self.assertRaises(hard.HardeningError):
            hard.transform(self.r3 + self.r3)

    def test_invalid_type_and_newline_refused(self):
        for wrong in (b"\xff", self.r3.replace(b'\n', b'\r\n'), self.r3 + b"0" * 131072):
            with self.subTest(bad=wrong[:3]), self.assertRaises(hard.HardeningError):
                hard.transform(wrong)
        with self.assertRaises(hard.HardeningError):
            hard.transform("string")

    def test_output_copy_and_hashes(self):
        with tempfile.TemporaryDirectory() as p:
            d = Path(p)
            source = d / "r3.py"
            target = d / "r4.py"
            source.write_bytes(self.r3)
            status = hard.apply(source, target)
            self.assertEqual(source.read_bytes(), self.r3)
            self.assertEqual(hard.blob(target.read_bytes()), status["output_blob_sha1"])
            self.assertEqual(status["status"], "R4_STATIC_PATCH_NOT_MEASURED_NOT_APPROVED")
            with self.assertRaises(hard.HardeningError):
                hard.apply(source, target)

    def test_in_place_write_refused(self):
        with tempfile.TemporaryDirectory() as p:
            source = Path(p) / "x.py"
            source.write_bytes(self.r3)
            with self.assertRaises(hard.HardeningError):
                hard.apply(source, source)


class FullPipelineFixtureTests(unittest.TestCase):
    """Positive *fixture-only* orchestration: source pins are rebound only in test scope."""

    def test_fake_git_snapshot_and_dirty_worktree_are_isolated(self):
        import hashlib
        import json
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repo = base / "gitrepo"
            repo.mkdir()
            subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
            (repo / "src").mkdir()
            (repo / "scripts/oracles").mkdir(parents=True)
            (repo / "spec").mkdir()
            (repo / "results").mkdir()
            HardeningPatchTests.setUpClass()
            original = HardeningPatchTests.sample
            src = repo / prep.ORACLE_PATH
            src.write_bytes(original)
            (repo / "src/a.py").write_text("x=1\n")
            draft = b"# Preregistration Phase 1 v1\n**Status:** DRAFT -- NOT binding.\n"
            (repo / prep.PREREG_PATH).write_bytes(draft)
            (repo / "results/cost.json").write_text('{"demo":true}\n')
            (repo / "results/radius.json").write_text('{"demo":true}\n')
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@fixture.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "synthetic fixture only",
                ],
                check=True,
            )
            rev = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()

            def pins(rel):
                # From the commit, not the working file: under core.autocrlf=true the two differ.
                raw = subprocess.run(
                    ["git", "-C", str(repo), "cat-file", "blob", f"{rev}:{rel}"],
                    capture_output=True,
                    check=True,
                ).stdout
                return (rel, prep._sha1_blob(raw), hashlib.sha256(raw).hexdigest())

            files = (pins("results/cost.json"), pins("results/radius.json"))
            # The worktree content now differs from the committed object.
            src.write_text("# MALICIOUS WORKTREE REPLACEMENT\n")
            before = subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout
            self.assertIn("gino_epoch_cost_synthetic.py", before)
            new_output = base / "prepared"
            with (
                patch.object(prep, "PR_HEAD", rev),
                patch.object(prep, "FILES", files),
                patch.object(prep, "PREREG_BLOB", prep._sha1_blob(draft)),
                patch.object(prep, "PREREG_SHA256", hashlib.sha256(draft).hexdigest()),
                patch.object(prep, "ORIGINAL_GIT_BLOB", prep._sha1_blob(original)),
                patch.object(r3, "ORIGINAL_GIT_BLOB", prep._sha1_blob(original)),
                patch.object(prep, "preflight_pr91", return_value={"policy": "TEST_STUB_NOT_EVIDENCE"}),
                patch.object(
                    verify,
                    "preflight_pr91",
                    return_value={"evidence_tier": "TWO_EVIDENCE_FILES_BYTE_PINNED_PR91_CANDIDATE_ONLY"},
                ),
            ):
                result = prep.prepare(repo, new_output)
                audit = verify.verify_prepared(new_output)
                self.assertEqual(audit["integrity"], "PASS")
                self.assertEqual(audit["owner_decision"], "HOLD_NO_APPROVAL")
                target = new_output / "snapshot" / prep.CANDIDATE_PATH
                original_candidate = target.read_bytes()
                target.write_bytes(original_candidate.replace(b"non-finite gradient", b"invalid gradient! "))
                with self.assertRaises(verify.VerificationError):
                    verify.verify_prepared(new_output)
                target.write_bytes(original_candidate)
                self.assertEqual(verify.verify_prepared(new_output)["integrity"], "PASS")
                doc = new_output / "snapshot" / prep.PREREG_PATH
                original_doc = doc.read_bytes()
                doc.write_bytes(original_doc + b"tamper")
                with self.assertRaises(verify.VerificationError):
                    verify.verify_prepared(new_output)
                doc.write_bytes(original_doc)
                self.assertEqual(verify.verify_prepared(new_output)["integrity"], "PASS")
            self.assertEqual(result["run_performed"], False)
            self.assertEqual(result["owner_decision"], "HOLD_NO_APPROVAL")
            self.assertEqual(result["source_evidence"]["radius_verifier_policy"], "TEST_STUB_NOT_EVIDENCE")
            self.assertEqual((new_output / "snapshot" / prep.ORACLE_PATH).read_bytes(), original)
            patched = (new_output / "snapshot" / prep.CANDIDATE_PATH).read_bytes()
            self.assertIn(b"non-finite gradient before optimizer update", patched)
            self.assertIn(b"T_max=args.schedule_epochs", patched)
            self.assertEqual(
                json.loads((new_output / "R4_PREPARATION_MANIFEST.json").read_text())["run_performed"], False
            )
            after = subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout
            self.assertEqual(before, after)
            # One command, one immutable output; repeated preparation is refused.
            with self.assertRaises(prep.PreparationError):
                prep.prepare(repo, new_output)


if __name__ == "__main__":
    unittest.main()
