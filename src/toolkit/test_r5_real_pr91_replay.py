"""R5 against reality: line endings, export-ignore and the real PR #91 commit.

The delivered R5 suite only ever saw fabricated Git repositories and its report
says so. These tests close what that left open:

* TestArchiveIsByteFaithful runs everywhere: a repository whose own configuration
  sets core.autocrlf=true must still yield the raw blobs.
* TestRealCommit runs the documented command pair against the real commit, where
  its object exists. A shallow CI checkout does not carry it; the skip is visible
  and is not a pass.

PASS_FULL_TREE_INTEGRITY_ONLY is no approval of anything; see
spec/PHASE1_R5_FULL_TREE_INTEGRITY_CONTRACT.md.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src' / 'toolkit'))
import prepare_phase1_synthetic_r4 as prep  # noqa: E402
import verify_prepared_snapshot_r4 as old  # noqa: E402
import verify_r5_full_snapshot as new  # noqa: E402
from preapproval_budget_hex_v3 import EvidenceError, _decode_evidence, verify_pr91_contract  # noqa: E402

COST_JSON = ROOT / 'results' / '2026-10-29_gino_epoch_cost_synthetic.json'


def _git(repo, *args):
    done = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


def _has_commit(repo: Path, commit: str) -> bool:
    done = subprocess.run(['git', '-C', str(repo), 'cat-file', '-e', f'{commit}^{{commit}}'],
                          capture_output=True, check=False)
    return done.returncode == 0


def _small_repo(base: Path, *, attributes: str = '', autocrlf: bool = False) -> tuple[Path, str]:
    """A fabricated repository with the archive's pathspecs; LF bytes in every blob."""
    repo = base / 'source'
    repo.mkdir()
    _git(repo, 'init', '-q')
    if autocrlf:
        _git(repo, 'config', 'core.autocrlf', 'true')
    files = {
        'src/a.py': b'answer = 41\n',
        'src/b.py': b'answer = 42\n',
        'spec/a.md': b'# spec\n',
        'scripts/oracles/a.py': b'pass\n',
        'results/one.json': b'{}\n',
        'results/two.json': b'{}\n',
    }
    if attributes:
        files['.gitattributes'] = attributes.encode()
    for rel, raw in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    _git(repo, 'add', '.')
    _git(repo, '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.org',
         'commit', '-q', '-m', 'fixture')
    return repo, _git(repo, 'rev-parse', 'HEAD')


class TestArchiveIsByteFaithful(unittest.TestCase):
    def setUp(self):
        files = patch.object(prep, 'FILES', (('results/one.json', 'x', 'y'), ('results/two.json', 'x', 'y')))
        files.start()
        self.addCleanup(files.stop)

    def _archive(self, repo, head, out):
        raw = prep._archive_bytes(repo, revision=head)
        out.mkdir()
        prep.unpack_checked(raw, out)
        return {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}

    def test_autocrlf_true_does_not_rewrite_archive_members(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repo, head = _small_repo(base, autocrlf=True)
            out = base / 'extracted'
            paths = self._archive(repo, head, out)
            self.assertEqual((out / 'src/a.py').read_bytes(), b'answer = 41\n')
            self.assertEqual(new._complete_tree_blobs(repo, head, out, paths), 6)

    def test_export_ignore_is_an_inventory_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repo, head = _small_repo(base, attributes='src/b.py export-ignore\n')
            out = base / 'extracted'
            paths = self._archive(repo, head, out)
            self.assertNotIn('src/b.py', paths)
            with self.assertRaisesRegex(new.FullSnapshotError, r"inventory mismatch: missing=\['src/b.py'\]"):
                new._complete_tree_blobs(repo, head, out, paths)


class TestReplaceRefsAreIgnored(unittest.TestCase):
    """`git replace` swaps a whole commit behind its name, tree and all.

    The strongest local forger: the pinned name stays, ls-tree and archive agree
    with each other on the replacement, so tree-against-archive sees nothing.
    """

    def test_replaced_commit_does_not_change_what_is_verified(self):
        with tempfile.TemporaryDirectory() as td, patch.object(
            prep, 'FILES', (('results/one.json', 'x', 'y'), ('results/two.json', 'x', 'y'))
        ):
            base = Path(td)
            repo, honest = _small_repo(base)
            (repo / 'src/a.py').write_bytes(b'evil = True\n')
            _git(repo, 'add', '.')
            _git(repo, '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.org',
                 'commit', '-q', '-m', 'forged')
            forged = _git(repo, 'rev-parse', 'HEAD')
            _git(repo, 'replace', honest, forged)
            self.assertIn('evil', _git(repo, 'show', f'{honest}:src/a.py'))  # the forgery is live
            out = base / 'extracted'
            out.mkdir()
            prep.unpack_checked(prep._archive_bytes(repo, revision=honest), out)
            self.assertEqual((out / 'src/a.py').read_bytes(), b'answer = 41\n')
            paths = {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
            self.assertEqual(new._complete_tree_blobs(repo, honest, out, paths), 6)
            blob = prep._sha1_blob(b'answer = 41\n')
            self.assertEqual(prep.object_bytes(repo, honest, 'src/a.py', blob), b'answer = 41\n')


class TestContractRefusesMalformedRowsCleanly(unittest.TestCase):
    """Malformed rows stay EvidenceError; the CLIs catch that and exit 2."""

    def setUp(self):
        self.record = _decode_evidence(COST_JSON.read_bytes().replace(b'\r\n', b'\n'))

    def test_real_record_passes_the_contract(self):
        checked = verify_pr91_contract(self.record)
        self.assertEqual(checked['configuration_count'], 8)
        self.assertFalse(checked['research_data'])

    def test_missing_epoch_seconds(self):
        del self.record['configs'][3]['epoch_seconds']
        with self.assertRaises(EvidenceError):
            verify_pr91_contract(self.record)

    def test_unhashable_grid_value(self):
        self.record['configs'][0]['hidden_channels'] = [32]
        with self.assertRaises(EvidenceError):
            verify_pr91_contract(self.record)


@unittest.skipUnless(_has_commit(ROOT, prep.PR_HEAD),
                     'real PR #91 commit object absent (shallow checkout): real replay NOT run')
class TestRealCommit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='hexa-r5-real-')
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.prepared = Path(cls.tmp.name) / 'prepared'
        prep.prepare(ROOT, cls.prepared)

    def _copy(self):
        target = Path(tempfile.mkdtemp(prefix='hexa-r5-copy-')) / 'prepared'
        self.addCleanup(shutil.rmtree, target.parent)
        shutil.copytree(self.prepared, target)
        return target

    def test_full_verification_passes_and_claims_nothing_more(self):
        report = new.verify_full(self.prepared, ROOT)
        self.assertEqual(report['status'], 'PASS_FULL_TREE_INTEGRITY_ONLY')
        self.assertEqual(report['git_head'], '1a54cd8d9326e762ee493d3fc8d9a80202174304')
        self.assertEqual(report['owner_decision'], 'HOLD_NO_APPROVAL')
        self.assertIs(report['scientific_result'], False)
        self.assertIs(report['origin_signature_verified'], False)
        self.assertIs(report['d5_machine_attested'], False)
        self.assertIs(report['run_executed'], False)
        self.assertEqual(report['checked_original_git_blobs'], report['checked_original_files'])
        # Counted from the tree here, not from the verifier's own walk.
        listed = _git(ROOT, 'ls-tree', '-r', '--name-only', prep.PR_HEAD).splitlines()
        expected = [p for p in listed if p.startswith(('src/', 'spec/', 'scripts/oracles/'))
                    or p in (prep.FILES[0][0], prep.FILES[1][0])]
        self.assertEqual(report['checked_original_files'], len(expected))

    def test_manifest_stays_on_hold(self):
        manifest = json.loads((self.prepared / 'R4_PREPARATION_MANIFEST.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['owner_decision'], 'HOLD_NO_APPROVAL')
        self.assertEqual(manifest['preregistration']['status'], 'DRAFT_NOT_BINDING')
        self.assertIs(manifest['run_performed'], False)
        self.assertIs(manifest['t1_data_generated_or_fitted'], False)

    def test_one_changed_byte_in_an_unpinned_module_old_passes_new_refuses(self):
        copy = self._copy()
        module = copy / 'snapshot/src/toolkit/runner_hex_v3.py'
        raw = module.read_bytes()
        module.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        self.assertEqual(old.verify_prepared(copy)['integrity'], 'PASS')
        with self.assertRaisesRegex(new.FullSnapshotError, 'content differs'):
            new.verify_full(copy, ROOT)

    def test_extra_module_old_passes_new_refuses(self):
        copy = self._copy()
        (copy / 'snapshot/src/toolkit/ghost.py').write_bytes(b'print("tampered")\n')
        self.assertEqual(old.verify_prepared(copy)['integrity'], 'PASS')
        with self.assertRaisesRegex(new.FullSnapshotError, 'path-set mismatch'):
            new.verify_full(copy, ROOT)

    def test_candidate_carries_every_guard_and_the_original_is_untouched(self):
        original = (self.prepared / 'snapshot' / prep.ORACLE_PATH).read_bytes()
        candidate = (self.prepared / 'snapshot' / prep.CANDIDATE_PATH).read_bytes()
        self.assertIn(b'T_max=args.repeats', original)
        self.assertNotIn(b'T_max=args.repeats', candidate)
        for text in (b'non-finite minibatch loss', b'non-finite gradient', b'non-finite parameter',
                     b'GUARDED_R4_NOT_COMPARABLE_TO_PR91_TIMINGS'):
            self.assertNotIn(text, original)
            self.assertIn(text, candidate)


if __name__ == '__main__':
    unittest.main()
