#!/usr/bin/env python3
"""Independent, fail-closed FULL-snapshot verification of a prepared PR#91 R4 bundle.

Requires the original immutable PR commit object in a local Git repository.
It recomputes the complete archive from that commit, revalidates EVERY file,
then verifies both R3/R4 generated patches and status/metadata bindings.

Exit 0 = local source INTEGRITY ONLY. Not source authenticity, signed SLSA
attestation, D5 machine proof, T1 approval, benchmark outcome, or execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import stat
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src' / 'toolkit'))
import prepare_phase1_synthetic_r4 as base  # noqa: E402
import patch_synthetic_cost_oracle_v1 as patch3  # noqa: E402
import harden_r4_synthetic_oracle as patch4  # noqa: E402
import verify_prepared_snapshot_r4 as verify4  # noqa: E402


class FullSnapshotError(ValueError):
    """A complete, repo-anchored snapshot audit has failed."""


def _json(path: Path, max_bytes: int = 262144) -> dict:
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise FullSnapshotError('manifest too large')
    obj = json.loads(data, parse_constant=lambda c: (_ for _ in ()).throw(FullSnapshotError('NaN/Infinity')))
    if not isinstance(obj, dict):
        raise FullSnapshotError('manifest not an object')
    return obj


def _filetree(
    root: Path, *, max_files: int = base.MAX_MEMBERS, max_bytes: int = base.MAX_ARCHIVE_BYTES
) -> tuple[dict[str, tuple[int, str]], set[str]]:
    """Inspect every regular file, reject all links, specials, extras and escaped paths.

    `os.walk` and lstat are NOT atomic against concurrent attacker writes:
    call on a stable, restricted, offline workspace and do not execute before
    a separate integrity-checked execution environment is established.
    """
    if root.is_symlink() or not root.is_dir() or getattr(root, 'is_junction', lambda: False)():
        raise FullSnapshotError('snapshot root must be a real directory')
    entries: dict[str, tuple[int, str]] = {}
    directories: set[str] = set()
    total = 0
    for folder, dirs, files in os.walk(root, followlinks=False):
        fpath = Path(folder)
        if fpath != root:
            directories.add(fpath.relative_to(root).as_posix())
        for part in dirs + files:
            p = fpath / part
            mode = p.lstat().st_mode
            if stat.S_ISLNK(mode) or getattr(p, 'is_junction', lambda: False)():
                raise FullSnapshotError('symbolic link or junction: ' + str(p.relative_to(root)))
            if part in dirs:
                if not stat.S_ISDIR(mode):
                    raise FullSnapshotError('non-directory in snapshot: ' + str(p.relative_to(root)))
                continue
            if not stat.S_ISREG(mode):
                raise FullSnapshotError('special snapshot member: ' + str(p.relative_to(root)))
            relative = p.relative_to(root).as_posix()
            if len(entries) >= max_files:
                raise FullSnapshotError('too many files')
            with p.open('rb') as stream:
                raw = stream.read(base.MAX_FILE_BYTES + 1)
            if len(raw) > base.MAX_FILE_BYTES:
                raise FullSnapshotError('oversize file: ' + relative)
            total += len(raw)
            if total > max_bytes:
                raise FullSnapshotError('total file byte limit exceeded')
            if relative in entries:
                raise FullSnapshotError('duplicate snapshot path')
            entries[relative] = (len(raw), hashlib.sha256(raw).hexdigest())
    return entries, directories


def _complete_tree_blobs(repo: Path, revision: str, expected_dir: Path, archive_paths: set[str]) -> int:
    """Verify EVERY archived source against its raw Git tree blob, not just TAR.

    `git archive` can transform files via `export-subst` or exclude them via
    `export-ignore`. Both cases are rejected rather than called a byte-faithful
    copy of the Git tree. A SHA-1 Git repository is required by the PR pins.
    """
    try:
        p = subprocess.run(
            ['git', '-C', str(repo), 'ls-tree', '-r', '-z', revision], capture_output=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FullSnapshotError('cannot read complete canonical git tree') from exc
    if p.returncode != 0 or len(p.stdout) > 4 * 1024 * 1024:
        raise FullSnapshotError('canonical git tree list invalid or too large')
    tree: dict[str, tuple[str, str]] = {}
    for item in p.stdout.split(b'\0'):
        if not item:
            continue
        header, sep, name = item.partition(b'\t')
        parts = header.split()
        if not sep or len(parts) != 3 or parts[1] != b'blob':
            raise FullSnapshotError('unexpected git tree member')
        try:
            rel = name.decode('utf-8')
            mode = parts[0].decode('ascii')
            oid = parts[2].decode('ascii')
        except UnicodeError as exc:
            raise FullSnapshotError('git tree path or id cannot be decoded') from exc
        if len(oid) != 40 or any(c not in '0123456789abcdef' for c in oid):
            raise FullSnapshotError('unexpected Git hash format; SHA-1 pins required')
        if mode not in ('100644', '100755', '120000', '160000'):
            raise FullSnapshotError('unknown git tree entry mode')
        if rel in tree:
            raise FullSnapshotError('duplicate git tree path')
        tree[rel] = (mode, oid)
    required = {
        path
        for path in tree
        if path.startswith(('src/', 'spec/', 'scripts/oracles/')) or path in (base.FILES[0][0], base.FILES[1][0])
    }
    if required != archive_paths:
        missing = sorted(required - archive_paths)
        extras = sorted(archive_paths - required)
        raise FullSnapshotError(f'git tree vs archive inventory mismatch: missing={missing[:5]}, extra={extras[:5]}')
    for relative in sorted(required):
        mode, oid = tree[relative]
        if mode not in ('100644', '100755'):
            raise FullSnapshotError('Git tree has symlink/submodule: ' + relative)
        raw = (expected_dir / relative).read_bytes()
        value = base._sha1_blob(raw)
        if value != oid:
            raise FullSnapshotError('archived source diverges from raw Git blob: ' + relative)
    return len(required)


def verify_full(prepared: Path, repo: Path) -> dict:
    """Verify all original sources against a fresh PR#91 git archive.

    Both the prepared directory and local repo are read-only. An independent
    temporary directory is used to extract the canonical Git archive.
    """
    if prepared.is_symlink() or getattr(prepared, 'is_junction', lambda: False)():
        raise FullSnapshotError('prepared directory must not be a symlink/junction')
    prepared = Path(prepared).absolute()
    repo = Path(repo).absolute()
    if repo.is_symlink() or not repo.is_dir():
        raise FullSnapshotError('local repository missing or link-only')
    # The earlier verifier remains useful, but does NOT attest the whole tree.
    earlier = verify4.verify_prepared(prepared)
    if earlier['integrity'] != 'PASS':
        raise FullSnapshotError('previous verification failed')
    manifest = _json(prepared / 'R4_PREPARATION_MANIFEST.json')
    manifest_fields = {
        'schema_version',
        'git_pr_head',
        'source_tier',
        'preregistration',
        'source_evidence',
        'patch',
        'archive',
        'runtime_environment',
        'synthetic_oracle_relative_path',
        'run_performed',
        't1_data_generated_or_fitted',
        'owner_decision',
        'safe_next_action',
        'no_data_assertion',
    }
    if set(manifest) != manifest_fields:
        raise FullSnapshotError('manifest keys changed or unrecognized')
    if manifest.get('safe_next_action') != (
        'review proposed patch and run the candidate ONLY on pinned D5 '
        'with synthetic inputs; collect full overhead separately'
    ):
        raise FullSnapshotError('safe next-action statement modified')
    if manifest.get('no_data_assertion') != ('synthetic normal tensors; original benchmark source pinned by blob hash'):
        raise FullSnapshotError('synthetic-only claim modified')

    # All these claims were previously mutable with no detection by R4.
    expected_prereg = {
        'path': base.PREREG_PATH,
        'blob_sha1': base.PREREG_BLOB,
        'sha256': base.PREREG_SHA256,
        'status': 'DRAFT_NOT_BINDING',
    }
    if manifest.get('preregistration') != expected_prereg:
        raise FullSnapshotError('preregistration binding modified')
    ev = manifest.get('source_evidence')
    if (
        not isinstance(ev, dict)
        or ev.get('cost_sha256') != base.FILES[0][2]
        or ev.get('radius_sha256') != base.FILES[1][2]
    ):
        raise FullSnapshotError('source-evidence declaration differs from fixed pins')
    rechecked_policy = verify4.preflight_pr91(
        prepared / 'snapshot' / base.FILES[0][0],
        prepared / 'snapshot' / base.FILES[1][0],
    ).get('policy')
    if ev != {
        'cost_sha256': base.FILES[0][2],
        'radius_sha256': base.FILES[1][2],
        'radius_verifier_policy': rechecked_policy,
    }:
        raise FullSnapshotError('source-evidence policy differs from independent recomputation')
    env = manifest.get('runtime_environment')
    if not isinstance(env, dict) or env.get('is_d5_verified') is not False:
        raise FullSnapshotError('untrusted D5 attestation claim')
    if env.get('required_d5') != base.environment_probe()['required_d5']:
        raise FullSnapshotError('required machine/software contract modified')
    if env.get('requires_named_machine_and_resource_attestation') is not True:
        raise FullSnapshotError('required D5 attestation falsely waived')
    if manifest.get('synthetic_oracle_relative_path') != base.CANDIDATE_PATH:
        raise FullSnapshotError('runnable candidate path changed')
    if manifest.get('run_performed') is not False or manifest.get('t1_data_generated_or_fitted') is not False:
        raise FullSnapshotError('run or data claim modified')

    # Original snapshot bytes are reconstructed from the COMMIT, not the dirty worktree.
    canonical = base.read_pinned(repo, revision=base.PR_HEAD)
    git_tar = base._archive_bytes(repo, revision=base.PR_HEAD)
    with tempfile.TemporaryDirectory(prefix='hexa-r5-verify-') as tmp:
        expected_dir = Path(tmp)
        audit_archive = base.unpack_checked(git_tar, expected_dir)
        expected, expected_dirs = _filetree(expected_dir)
        git_blob_verified = _complete_tree_blobs(repo, base.PR_HEAD, expected_dir, set(expected))

    # Verify the original immutable evidence, not just its metadata.
    for rel, original in canonical.items():
        record = expected.get(rel)
        if record != (len(original), hashlib.sha256(original).hexdigest()):
            raise FullSnapshotError('git archive disagrees with git object: ' + rel)

    snap = prepared / 'snapshot'
    observed, observed_dirs = _filetree(snap)
    candidate_paths = (base.R3_CANDIDATE_PATH, base.CANDIDATE_PATH)
    expected_candidates = {
        base.R3_CANDIDATE_PATH: patch3.transform(canonical[base.ORACLE_PATH]),
    }
    expected_candidates[base.CANDIDATE_PATH] = patch4.transform(expected_candidates[base.R3_CANDIDATE_PATH])
    for rel, raw in expected_candidates.items():
        expected[rel] = (len(raw), hashlib.sha256(raw).hexdigest())

    if observed_dirs != expected_dirs:
        raise FullSnapshotError('snapshot directory-set mismatch')
    missing = sorted(set(expected) - set(observed))
    extras = sorted(set(observed) - set(expected))
    if missing or extras:
        raise FullSnapshotError(
            'snapshot path-set mismatch missing=' + repr(missing[:5]) + ', extra=' + repr(extras[:5])
        )
    changed = sorted(rel for rel in expected if observed[rel] != expected[rel])
    if changed:
        raise FullSnapshotError('snapshot content differs from Git source/derived patch: ' + repr(changed[:5]))

    # These fields were not cross-checked by R4's independent verifier.
    arch = manifest.get('archive')
    if not isinstance(arch, dict) or arch != audit_archive:
        raise FullSnapshotError('archive provenance/byte-count metadata altered')
    p = manifest.get('patch')
    if not isinstance(p, dict):
        raise FullSnapshotError('missing patch chain')
    original = canonical[base.ORACLE_PATH]
    r3 = expected_candidates[base.R3_CANDIDATE_PATH]
    r4 = expected_candidates[base.CANDIDATE_PATH]
    expected_r3 = {
        'original_git_blob': patch3.blob(original),
        'new_git_blob': patch3.blob(r3),
        'new_sha256': hashlib.sha256(r3).hexdigest(),
        'status': 'PATCHED_SOURCE_NOT_MEASURED_NOT_APPROVED',
    }
    expected_r4 = {
        'input_blob_sha1': patch4.blob(r3),
        'output_blob_sha1': patch4.blob(r4),
        'output_sha256': hashlib.sha256(r4).hexdigest(),
        'status': 'R4_STATIC_PATCH_NOT_MEASURED_NOT_APPROVED',
    }
    if p != {'r3': expected_r3, 'r4': expected_r4}:
        raise FullSnapshotError('patch provenance metadata was changed')
    return {
        'schema_version': 'HEXA_NTK_R5_FULL_SNAPSHOT_VERIFY/1',
        'status': 'PASS_FULL_TREE_INTEGRITY_ONLY',
        'checked_files': len(expected),
        'checked_original_files': len(expected) - len(candidate_paths),
        'checked_original_git_blobs': git_blob_verified,
        'checked_directories': len(expected_dirs),
        'checked_derived_files': len(candidate_paths),
        'snapshot_content_sha256': hashlib.sha256(
            json.dumps([(k, *expected[k]) for k in sorted(expected)], separators=(',', ':'), ensure_ascii=True).encode(
                'utf8'
            )
        ).hexdigest(),
        'git_head': base.PR_HEAD,
        'owner_decision': 'HOLD_NO_APPROVAL',
        'd5_machine_attested': False,
        'scientific_result': False,
        'origin_signature_verified': False,
        'run_executed': False,
        'report': 'Git object/content integrity; not signed provenance or science validation',
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--report', type=Path, default=None)
    a = parser.parse_args(argv)
    try:
        report = verify_full(a.prepared, a.repo)
        content = json.dumps(report, sort_keys=True, indent=2) + '\n'
        if a.report:
            try:
                with a.report.open('x', encoding='utf-8') as stream:
                    stream.write(content)
            except FileExistsError as exc:
                raise FullSnapshotError('output report exists; no overwrite') from exc
        print(content, end='')
        return 0
    except (FullSnapshotError, verify4.VerificationError, base.PreparationError, OSError, ValueError, KeyError) as e:
        print(
            json.dumps(
                {
                    'schema_version': 'HEXA_NTK_R5_FULL_SNAPSHOT_VERIFY/1',
                    'status': 'FAIL_CLOSED',
                    'owner_decision': 'HOLD_NO_APPROVAL',
                    'reason': str(e),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
