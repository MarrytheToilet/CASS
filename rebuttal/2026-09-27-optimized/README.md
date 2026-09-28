# Rebuttal revision from the authorized optimization window

中文入口：[核心结果与回复路线](RESULTS_zh.md)。

The experimental program and evidence documents are complete. This is an
author-facing response and editing package; no reply has been posted externally.

The submitted implementation and its original rebuttal package remain intact at
`rebuttal/2026-09-27/`. New experiments and complete generation logs are in
`optimization/2026-09-27-night/`. The GPU window is restricted to physical GPUs
2 and 3, from 2026-09-27 19:18:42 UTC until 2026-09-28 03:18:42 UTC.

The revised response will distinguish three sources of evidence:

1. Reanalysis and matched interventions on the submitted method.
2. Implementation optimizations that preserve the mathematical operator.
3. New variants selected on the 12-task development split, then frozen for
   transfer evaluation. These are additional experiments, not unchanged results
   from the submitted method.

The primary known-task transfer comparison uses the 20 LOTO targets that were
not development targets. Novel15 and Compound10 are separate suites. Development
mining can include other known skills; this is not a claim that every activation
from the 20 tasks was unavailable during development. All target-task mining is
excluded when that task is evaluated. Seeds 20/21/22, matched queries, matched
demonstrations, and task-level paired intervals are retained.

The reviewer coverage ledger records each substantive request and its evidence.
Principal numerical claims have been checked against their named source files.
The replies address some concerns through new experiments and others through
corrected theory or a more precise claim; experimental coverage is stated explicitly.

Responses: [BiuA](response_BiuA.md), [Bm1S](response_Bm1S.md), [DuwT](response_DuwT.md), [cajh](response_cajh.md). An additional [compact cajh version](response_cajh_compact.md) is available if needed.

Available evidence companions:

- [Revised three-stage method overview](../../optimization/2026-09-27-night/figures/submitted_method_overview.pdf)
  with an editable SVG and caption in `manuscript_revision.md`.
- [Corrected proofs and operator geometry (PDF)](technical_appendix.pdf)
  and [editable TeX](technical_appendix.tex).
- [Original and additional-query confirmation](confirmation_evidence.md).
- [All broader-task comparisons](broader_evidence.md).
- [Separately frozen follow-up variants](followup_evidence.md).
- [Shared-rank control on both query sets](../../optimization/2026-09-27-night/figures/shared_rank_control.pdf).
- [Qwen factor diagnostics](qwen_evidence.md).
- [Qwen complete-answer and punctuation checks](qwen_format_evidence.md).
- [Scaling and calibration](scaling_evidence.md).
- [Known-task supervised risk calibration](known_calibration_evidence.md).
- [Adaptation/serving cost and retained state](cost_evidence.md).
- [Cached serving and energy on all seven broader NLP tasks](broader_cost_evidence.md).
- [Protocols, source records and reproduction commands](reproduction_notes.md).

The GPU runs finished within the authorized window. Only our verified idle
workers on physical GPUs 2 and 3 were stopped, at 2026-09-28 01:35 UTC.
Original predictions, unsuccessful candidates and source snapshots remain in
the project. The compact reply is an alternative draft, not an additional reply
to post alongside its full version.

Final checks recomputed the metrics in 51 prediction ledgers, covering 2,110,989
recorded predictions across candidates and controls; these are not independent
test cases. All 2,217 activation-cache/source-archive files passed the local
transfer-hash check. Their machine-readable receipts are
`prediction_ledger_validation.json` and `cache_artifact_verification.json` in
the optimization directory. `response_number_audit.json` records the selected
numerical claim checks and their source hashes.
