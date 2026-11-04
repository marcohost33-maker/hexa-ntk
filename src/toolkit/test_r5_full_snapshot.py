"""Independent red/green tests for the R4 -> R5 whole-tree integrity gap.

Test data are fabricated commits, never original PR #91, never T1 data.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src' / 'toolkit'))
import prepare_phase1_synthetic_r4 as prep  # noqa: E402
import patch_synthetic_cost_oracle_v1 as r3  # noqa: E402
import verify_prepared_snapshot_r4 as old  # noqa: E402
import verify_r5_full_snapshot as new  # noqa: E402
import test_phase1_r4_preparation as reference_tests  # noqa: E402


def _git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


@contextmanager
def prepared_fixture():
    with tempfile.TemporaryDirectory(prefix='hexa-r5-fixture-') as td:
        base = Path(td)
        repo = base / 'source'
        repo.mkdir()
        _git(repo, 'init', '-q')
        reference_tests.HardeningPatchTests.setUpClass()
        original = reference_tests.HardeningPatchTests.sample
        files = {
            prep.ORACLE_PATH: original,
            prep.PREREG_PATH: b'# Preregistration Phase 1 v1\n**Status:** DRAFT -- NOT binding.\n',
            'src/a.py': b'answer = 41\n',
            'src/b.py': b'answer = 42\n',
            'spec/instrument.md': b'# synthetic only\n',
            'results/cost.json': b'{"fixture_cost": true}\n',
            'results/radius.json': b'{"fixture_radius": true}\n',
        }
        for rel, content in files.items():
            f = repo / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(content)
        _git(repo, 'add', '.')
        _git(
            repo,
            '-c',
            'user.name=fixture',
            '-c',
            'user.email=fixture@example.org',
            'commit',
            '-q',
            '-m',
            'synthetic fixture',
        )
        head = _git(repo, 'rev-parse', 'HEAD')

        def pin(rel):
            raw = files[rel]
            return (rel, prep._sha1_blob(raw), hashlib.sha256(raw).hexdigest())

        evidence = (pin('results/cost.json'), pin('results/radius.json'))
        output = base / 'prepared'
        with ExitStack() as patches:
            patches.enter_context(patch.object(prep, 'PR_HEAD', head))
            patches.enter_context(patch.object(prep, 'FILES', evidence))
            patches.enter_context(patch.object(prep, 'PREREG_BLOB', prep._sha1_blob(files[prep.PREREG_PATH])))
            patches.enter_context(
                patch.object(prep, 'PREREG_SHA256', hashlib.sha256(files[prep.PREREG_PATH]).hexdigest())
            )
            patches.enter_context(patch.object(prep, 'ORIGINAL_GIT_BLOB', prep._sha1_blob(original)))
            patches.enter_context(patch.object(r3, 'ORIGINAL_GIT_BLOB', prep._sha1_blob(original)))
            # Test-only evidence gate replacement: not real PR#91 evidence!
            stub = {
                'evidence_tier': 'TWO_EVIDENCE_FILES_BYTE_PINNED_PR91_CANDIDATE_ONLY',
                'policy': 'TEST_STUB_NOT_EVIDENCE',
            }
            patches.enter_context(patch.object(prep, 'preflight_pr91', return_value=stub))
            patches.enter_context(patch.object(old, 'preflight_pr91', return_value=stub))
            prep.prepare(repo, output)
            yield repo, output, files


def rewrite_manifest(folder, fn):
    p = folder / 'R4_PREPARATION_MANIFEST.json'
    data = json.loads(p.read_text())
    fn(data)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


class TestRedGreen(unittest.TestCase):
    def test_positive_fixture_and_explicit_no_science(self):
        with prepared_fixture() as (repo, output, _):
            a = old.verify_prepared(output)
            b = new.verify_full(output, repo)
            self.assertEqual(a['integrity'], 'PASS')
            self.assertEqual(b['status'], 'PASS_FULL_TREE_INTEGRITY_ONLY')
            self.assertFalse(b['origin_signature_verified'])
            self.assertFalse(b['d5_machine_attested'])
            self.assertEqual(b['owner_decision'], 'HOLD_NO_APPROVAL')
            self.assertEqual(b['checked_original_files'], 7)
            self.assertEqual(b['checked_derived_files'], 2)
            self.assertEqual(b['checked_files'], 9)

    def test_old_verifier_accepts_mutated_unpinned_executable_source_r5_rejects(self):
        with prepared_fixture() as (repo, output, _):
            f = output / 'snapshot/src/a.py'
            f.write_text('answer = 0\nexec("print(123)")\n')
            self.assertEqual(old.verify_prepared(output)['integrity'], 'PASS')
            with self.assertRaisesRegex(new.FullSnapshotError, 'content differs'):
                new.verify_full(output, repo)

    def test_old_verifier_accepts_added_executable_source_r5_rejects(self):
        with prepared_fixture() as (repo, output, _):
            (output / 'snapshot/src/ghost.py').write_text('print("tampered")\n')
            self.assertEqual(old.verify_prepared(output)['integrity'], 'PASS')
            with self.assertRaisesRegex(new.FullSnapshotError, 'path-set mismatch'):
                new.verify_full(output, repo)

    def test_old_verifier_accepts_extra_empty_directory_r5_rejects(self):
        with prepared_fixture() as (repo, output, _):
            (output / 'snapshot/src/new_empty_dir').mkdir()
            self.assertEqual(old.verify_prepared(output)['integrity'], 'PASS')
            with self.assertRaisesRegex(new.FullSnapshotError, 'directory-set mismatch'):
                new.verify_full(output, repo)

    def test_old_verifier_accepts_deleted_unpinned_source_r5_rejects(self):
        with prepared_fixture() as (repo, output, _):
            (output / 'snapshot/src/b.py').unlink()
            self.assertEqual(old.verify_prepared(output)['integrity'], 'PASS')
            with self.assertRaisesRegex(new.FullSnapshotError, 'path-set mismatch'):
                new.verify_full(output, repo)

    def test_old_verifier_accepts_false_d5_attestation_r5_rejects(self):
        with prepared_fixture() as (repo, output, _):
            rewrite_manifest(output, lambda j: j['runtime_environment'].__setitem__('is_d5_verified', True))
            self.assertEqual(old.verify_prepared(output)['integrity'], 'PASS')
            with self.assertRaisesRegex(new.FullSnapshotError, 'D5 attestation'):
                new.verify_full(output, repo)

    def test_old_verifier_accepts_changed_required_d5_cpu_r5_rejects(self):
        with prepared_fixture() as (repo, output, _):
            rewrite_manifest(
                output, lambda j: j['runtime_environment']['required_d5'].__setitem__('cpu', 'unrelated CPU')
            )
            self.assertEqual(old.verify_prepared(output)['integrity'], 'PASS')
            with self.assertRaisesRegex(new.FullSnapshotError, 'machine/software contract'):
                new.verify_full(output, repo)

    def test_old_verifier_accepts_fake_candidate_path_r5_rejects(self):
        with prepared_fixture() as (repo, output, _):
            rewrite_manifest(output, lambda j: j.__setitem__('synthetic_oracle_relative_path', 'src/other.py'))
            self.assertEqual(old.verify_prepared(output)['integrity'], 'PASS')
            with self.assertRaisesRegex(new.FullSnapshotError, 'runnable candidate path'):
                new.verify_full(output, repo)

    def test_old_verifier_accepts_forged_radius_policy_r5_rejects(self):
        with prepared_fixture() as (repo, output, _):
            rewrite_manifest(output, lambda j: j['source_evidence'].__setitem__('radius_verifier_policy', 'APPROVED'))
            self.assertEqual(old.verify_prepared(output)['integrity'], 'PASS')
            with self.assertRaisesRegex(new.FullSnapshotError, 'source-evidence policy'):
                new.verify_full(output, repo)

    def test_old_verifier_accepts_false_safe_action_r5_rejects(self):
        with prepared_fixture() as (repo, output, _):
            rewrite_manifest(output, lambda j: j.__setitem__('safe_next_action', 'run T1 training now'))
            self.assertEqual(old.verify_prepared(output)['integrity'], 'PASS')
            with self.assertRaisesRegex(new.FullSnapshotError, 'safe next-action'):
                new.verify_full(output, repo)

    def test_old_verifier_accepts_forged_archive_hash_r5_rejects(self):
        with prepared_fixture() as (repo, output, _):
            rewrite_manifest(output, lambda j: j['archive'].__setitem__('tar_sha256', '0' * 64))
            self.assertEqual(old.verify_prepared(output)['integrity'], 'PASS')
            with self.assertRaisesRegex(new.FullSnapshotError, 'archive provenance'):
                new.verify_full(output, repo)

    def test_old_verifier_accepts_forged_r3_git_blob_r5_rejects(self):
        with prepared_fixture() as (repo, output, _):
            rewrite_manifest(output, lambda j: j['patch']['r3'].__setitem__('new_git_blob', '0' * 40))
            self.assertEqual(old.verify_prepared(output)['integrity'], 'PASS')
            with self.assertRaisesRegex(new.FullSnapshotError, 'patch provenance'):
                new.verify_full(output, repo)

    def test_old_verifier_accepts_unknown_scientific_result_claim_r5_rejects(self):
        with prepared_fixture() as (repo, output, _):
            rewrite_manifest(output, lambda j: j.update(scientific_result=True))
            self.assertEqual(old.verify_prepared(output)['integrity'], 'PASS')
            with self.assertRaisesRegex(new.FullSnapshotError, 'manifest keys'):
                new.verify_full(output, repo)

    def test_old_verifier_accepts_evidence_metadata_forgery_r5_rejects(self):
        with prepared_fixture() as (repo, output, _):
            rewrite_manifest(output, lambda j: j['source_evidence'].__setitem__('cost_sha256', '0' * 64))
            self.assertEqual(old.verify_prepared(output)['integrity'], 'PASS')
            with self.assertRaisesRegex(new.FullSnapshotError, 'source-evidence'):
                new.verify_full(output, repo)

    def test_symlink_to_external_source_refused_even_if_other_fields_intact(self):
        with prepared_fixture() as (repo, output, _):
            f = output / 'snapshot/src/b.py'
            f.unlink()
            try:
                f.symlink_to(repo / 'src/b.py')
            except OSError as exc:
                self.skipTest(f'cannot create a symlink here: {exc}')
            self.assertEqual(old.verify_prepared(output)['integrity'], 'PASS')
            with self.assertRaisesRegex(new.FullSnapshotError, 'symbolic link'):
                new.verify_full(output, repo)

    def test_git_archive_export_subst_is_not_mistaken_for_raw_blob(self):
        """A real git archive substitution is NOT exact blob evidence."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repo = base / 'source'
            repo.mkdir()
            _git(repo, 'init', '-q')
            src = repo / 'src/a.py'
            src.parent.mkdir()
            src.write_text('commit = "$Format:%H$"\n')
            (repo / '.gitattributes').write_text('src/a.py export-subst\n')
            (repo / 'spec').mkdir()
            (repo / 'spec/a.md').write_text('# spec\n')
            (repo / 'scripts/oracles').mkdir(parents=True)
            (repo / 'scripts/oracles/a.py').write_text('pass\n')
            (repo / 'results').mkdir()
            (repo / 'results/one.json').write_text('{}\n')
            (repo / 'results/two.json').write_text('{}\n')
            _git(repo, 'add', '.')
            _git(
                repo,
                '-c',
                'user.name=fixture',
                '-c',
                'user.email=fixture@example.org',
                'commit',
                '-q',
                '-m',
                'export-subst fixture',
            )
            head = _git(repo, 'rev-parse', 'HEAD')
            with patch.object(prep, 'FILES', (('results/one.json', 'x', 'y'), ('results/two.json', 'x', 'y'))):
                raw = prep._archive_bytes(repo, revision=head)
                out = base / 'extracted'
                out.mkdir()
                prep.unpack_checked(raw, out)
                self.assertNotEqual((out / 'src/a.py').read_bytes(), src.read_bytes())
                paths = {x.relative_to(out).as_posix() for x in out.rglob('*') if x.is_file()}
                with self.assertRaisesRegex(new.FullSnapshotError, 'diverges from raw Git blob'):
                    new._complete_tree_blobs(repo, head, out, paths)

    def test_dirty_worktree_does_not_change_commit_based_verification(self):
        with prepared_fixture() as (repo, output, _):
            (repo / 'src/a.py').write_text('evil = True\n')
            self.assertEqual(new.verify_full(output, repo)['status'], 'PASS_FULL_TREE_INTEGRITY_ONLY')

    def test_original_pinned_file_modification_rejected_by_both(self):
        with prepared_fixture() as (repo, output, _):
            p = output / 'snapshot' / prep.ORACLE_PATH
            p.write_bytes(p.read_bytes() + b'\n# tampered\n')
            with self.assertRaises(old.VerificationError):
                old.verify_prepared(output)
            with self.assertRaises(old.VerificationError):
                new.verify_full(output, repo)

    def test_r4_candidate_modification_rejected_by_both(self):
        with prepared_fixture() as (repo, output, _):
            p = output / 'snapshot' / prep.CANDIDATE_PATH
            p.write_bytes(p.read_bytes() + b'\n# tampered\n')
            with self.assertRaises(old.VerificationError):
                old.verify_prepared(output)
            with self.assertRaises(old.VerificationError):
                new.verify_full(output, repo)

    def test_r5_archive_enforces_wall_deadline(self):
        """R4 blocked before its timeout when stdout never closed; R5 kills."""
        import time

        original_popen = subprocess.Popen

        def inert(*args, **kwargs):
            return original_popen(
                [sys.executable, '-c', 'import time;time.sleep(10)'], stdout=kwargs['stdout'], stderr=kwargs['stderr']
            )

        with tempfile.TemporaryDirectory() as td, patch.object(prep.subprocess, 'Popen', side_effect=inert):
            start = time.perf_counter()
            with self.assertRaisesRegex(prep.PreparationError, 'timed out'):
                prep._archive_bytes(Path(td), revision='0' * 40, timeout_seconds=0.2)
            self.assertLess(time.perf_counter() - start, 3.0)

    def test_r5_archive_enforces_byte_limit_without_buffering_unbounded(self):
        original_popen = subprocess.Popen

        def noisy(*args, **kwargs):
            return original_popen(
                [sys.executable, '-c', 'import sys;sys.stdout.buffer.write(b"A"*8192)'],
                stdout=kwargs['stdout'],
                stderr=kwargs['stderr'],
            )

        with tempfile.TemporaryDirectory() as td, patch.object(prep.subprocess, 'Popen', side_effect=noisy):
            with self.assertRaisesRegex(prep.PreparationError, 'archive exceeds byte limit'):
                prep._archive_bytes(Path(td), revision='0' * 40, max_archive_bytes=256)

    def test_r5_archive_rejects_invalid_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            for extra in (
                {'timeout_seconds': 0},
                {'max_archive_bytes': -1},
                {'max_archive_bytes': prep.MAX_ARCHIVE_BYTES + 1},
                {'timeout_seconds': 601},
            ):
                with self.subTest(extra=extra), self.assertRaises(prep.PreparationError):
                    prep._archive_bytes(Path(td), revision='0' * 40, **extra)

    def test_filetree_oversize_and_rejected_special_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'large.bin').write_bytes(b'a' * 17)
            with self.assertRaisesRegex(new.FullSnapshotError, 'total file byte limit'):
                new._filetree(root, max_bytes=2)
            (root / 'large.bin').unlink()
            if sys.platform != 'win32':
                import os

                os.mkfifo(root / 'bad_pipe')
                with self.assertRaisesRegex(new.FullSnapshotError, 'special snapshot member'):
                    new._filetree(root)

    def test_missing_local_repo_fails_closed(self):
        with prepared_fixture() as (repo, output, _):
            with self.assertRaises(new.FullSnapshotError):
                new.verify_full(output, repo / 'not-a-repo')

    def test_cli_exit2_and_no_report_on_failed_verification(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            report = d / 'report.json'
            rc = new.main(['--repo', str(d), '--prepared', str(d), '--report', str(report)])
            self.assertEqual(rc, 2)
            self.assertFalse(report.exists())

    def test_positive_cli_report_is_readable(self):
        with prepared_fixture() as (repo, output, _):
            report = output.parent / 'review.json'
            rc = new.main(['--repo', str(repo), '--prepared', str(output), '--report', str(report)])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(report.read_text())['status'], 'PASS_FULL_TREE_INTEGRITY_ONLY')
            self.assertEqual(new.main(['--repo', str(repo), '--prepared', str(output), '--report', str(report)]), 2)


if __name__ == '__main__':
    unittest.main()
