# CASS rebuttal experiments and artifact policy

The [2026-09-27 experiment directory](2026-09-27-night/) records the optimization
and validation work supporting the [revised author responses](../rebuttal/2026-09-27-optimized/README.md).
Start with the [Chinese results overview](../rebuttal/2026-09-27-optimized/RESULTS_zh.md)
or the [reproduction notes](../rebuttal/2026-09-27-optimized/reproduction_notes.md).
The earlier [submitted-method controls](../rebuttal/2026-09-27/README.md) remain
separate from the new contextual operator.

## What belongs in Git

- Experiment, analysis, plotting and validation scripts.
- Protocols, parameter grids, selected configurations and queue definitions.
- Numerical summaries, per-task CSVs, source manifests and verification receipts.
- Reviewer responses, evidence reports, mathematical proofs and manuscript edits.
- Curated PDF/SVG/PNG figures and their CSV data/source hashes.

Model weights, activation/signature tensors, serialized dictionaries, raw JSONL
generation ledgers, source/data transport archives, download chunks, local
environments, credentials, runtime logs, worker state and completion markers are
ignored. The author ZIP and LaTeX auxiliary files are also rebuildable and ignored.
Ignored files are retained on disk; adding ignore rules does not delete them or
remove any previously tracked result from Git.

The dated artifacts have recorded content hashes. `.gitattributes` preserves
their bytes across checkouts and recognizes generated CSV/SVG whitespace, so
repository housekeeping does not rewrite the evidence or invalidate its hashes.

## Reproducing or restoring a run

The numerical summaries and reports are reviewable from a Git checkout. Re-running
generation-level audits or recomputing all analyses additionally requires the raw
artifacts, task datasets and model checkpoints. Raw predictions and tensor caches
remain in the local and authorized-server project directories. Restore them at
their original relative paths, or regenerate them with the documented runners.
`cache_artifact_inventory.json` and `cache_artifact_verification.json` record tensor
and source-archive hashes; `prediction_ledger_validation.json` records the checked
generation files. JSON source manifests are versioned, while their compressed
source snapshots remain in the artifact archive.

The dated launchers preserve the original server paths, physical GPU selection
and experiment-window deadline. They are historical run definitions: inspect
and adapt that configuration before launching work on another machine or date.
No GPU job is started by committing or checking out these files.

To rebuild the author-facing reading bundle, run from the repository root:

```bash
python optimization/2026-09-27-night/package_rebuttal.py
```

This ZIP contains responses, evidence documents, proofs and figures. It is not
the full raw-data reproduction archive and does not submit anything externally.
