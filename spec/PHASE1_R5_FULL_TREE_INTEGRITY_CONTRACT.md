# Hexa-NTK R5 – Full-tree Verification Contract v1

**Scope:** strict local offline preflight for PR #91's *synthetic cost setup*, not a scientific T1 outcome.

## Source authority
- immutable commit: `1a54cd8d9326e762ee493d3fc8d9a80202174304`.
- all `src/`, `spec/`, `scripts/oracles/` and two pinned JSON files must match **raw Git blobs** and the generated `git archive` path inventory. The exact four extra R4 source pins are prerequisites, not a replacement for the full tree.
- generated patch candidates must be the deterministic result of R3 + R4 transform functions, with no source file overwritten.
- fail-closed for `export-subst`, `export-ignore`, symlink, junction, duplicate, missing, extra, special member, archive size, deadline, mismatching metadata, unexpected claim fields, or unpinned hash.

- `git archive` runs with per-call `-c core.autocrlf=false -c core.eol=lf`. Without them a checkout under `core.autocrlf=true` (Git for Windows default; the named D5 machine is Windows) rewrites every text member and the real commit can never pass.

- Every Git read runs with `--no-replace-objects`: a replace ref would swap the pinned commit behind its name, tree included, and tree and archive would agree on the forgery.

## Operational output
`PASS_FULL_TREE_INTEGRITY_ONLY` means only **local source-to-commit integrity** and patch reproducibility. It does **not** confirm the GitHub author, repository authentication, signed SLSA provenance, D5 identity, real GINO behavior, or T1 approval. No test has trained on T1.

## Required command pair

```bash
python scripts/prepare_phase1_synthetic_r4.py --repo /local/repo --out /new/prepared
python scripts/verify_r5_full_snapshot.py --repo /local/repo --prepared /new/prepared --report /new/verification.json
```

Leave the checkout unmodified. Run in a protected local directory; verification is not atomic under concurrent mutations. Do not execute the candidate by simply observing integrity PASS.

## Scientific stop rules
- PR #91 still DRAFT; Owner D4/D5 approval has not been granted.
- D5 benchmark must first be executed with exact software pins, fully measured overheads, controlled system load and setup/validation/checkpoint/data costs. Separately report the expensive nonfinite-guard variant.
- Only after approval: bounded T1 generation, independent truth/manifests, leakage-safe split, strong CN/CG and POD/Ridge baselines, 5 seeds and preregistered ID/OOD plus NTK observations.

## In-repo binding
- `src/toolkit/test_r5_real_pr91_replay.py` runs the command pair against the real commit where its object is available and skips visibly where it is not (a shallow CI checkout does not carry it). Results of a real run belong into a gate log under `results/`.
- `src/toolkit/test_r5_torch_guard_semantics.py` runs the transformed `step()` of the real oracle at the pinned blob.
- `TRANSCRIBED_VALUES_TEST_FIXTURE_NOT_SOURCE.json` is a test fixture, never source evidence. The pinned source is `results/2026-10-29_gino_epoch_cost_synthetic.json`.

## Remaining limitations
- Git does not re-hash a tree or commit object on read. Bytes changed directly in the local object store under the same name stay undetected for files outside the four pins; run `git fsck` on the clone, or use a fresh clone, when that matters.
- Git SHA-1 and SHA-256 are **content identities, not signatures**. Additional authorization, key verification and signed attestation are separate tasks.
- Archive verification stores a temporary copy and requires an accessible local exact Git object; when absent, exit nonzero.
- The Git-rebuild verification re-reads files and is not race-free against a malicious concurrent writer; use a read-only staging area, or cryptographically validated executable artifact handoff.
- The R4 after-step parameter guard has no automatic rollback.
