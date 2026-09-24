# HexaFabric benchmark data

## Seed v1

File: `hexafabric_seed_v1.jsonl`  
SHA-256 prefix (LF-normalized): `DA33D0248A57B8AE`  
Created: 2026-09-24

The seed contains only derived routing metadata from historical 007acc2 task documents.
No full private task body, Drive URL, Drive file ID, response text, or adjudication text is
copied into this repository.

Every case is marked:
- `annotation_status=MODEL_REVIEWED_SEED`
- `requires_human_review=true`

Therefore this dataset is suitable for schema, abstention and benchmark plumbing only.
It MUST NOT be treated as training ground truth or evidence for active routing.

## Source governance

Source identity is currently retained only as the historical task title and date.
Before promotion to a reviewed corpus, add an immutable source-reference manifest in a
location with access controls appropriate to the original 007acc2 material. Do not put
private Drive identifiers into a repository that may later become public.

## Split policy

Near-duplicate tasks and later revisions of one task must remain in the same split group.
Future reviewed versions should provide time-held-out, project-held-out, task-family-held-out
and explicit unsupported/OOD partitions.

## Change policy

Any label, capability vocabulary, source set or split change creates a new corpus version.
Do not silently edit a benchmark after held-out evaluation.
