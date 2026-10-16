# Hexa-NTK PR #64 — exact-head CI evidence (observed, historical)

- Verified GitHub PR #64 commit: `ac96f8670c246d158ffc3eb00fd99fd9f44061af`.
- `ci` run [38105548067](https://github.com/marcohost33-maker/hexa-ntk/actions/runs/38105548067): **success**; `gate` and `tests (cpu)` successful.
- `phase1-adapter-checks` run [38105548091](https://github.com/marcohost33-maker/hexa-ntk/actions/runs/38105548091): **success**; stacked gate, CPU tests, and actual backend smoke.
- `Workflow Security Audit (zizmor)` run [38105548106](https://github.com/marcohost33-maker/hexa-ntk/actions/runs/38105548106): **success**.
- The actual backend test log shows `test_actual_backend_backward_optimizer_smoke ... ok` and `test_actual_backend_forward_smoke ... ok`; `Ran 11 tests` and `OK` without skips under installed `neuraloperator==2.0.0`.
- Local PR64 full-tree run: `674` Toolkit tests, no failures/errors, **two optional skips** because NeuralOperator was not installed locally; 56/56 SOURCES, 49 Doctests/26 engine selftests, around 93% coverage.

This is a report of the **earlier PR64 SHA**. It does not prove a Wave19 commit passed. GitHub Actions is the authoritative source for complete runner logs, dependency versions and CI conclusions. Synthetic SGD is not authorized T1 training, no D5, no MMS V1 release proof. PR63 and PR64 remain unmerged.
