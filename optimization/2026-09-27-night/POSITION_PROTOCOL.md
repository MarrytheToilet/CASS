# Multi-position refinement

Motivation: an intervention at one final prompt state may not preserve the task
behavior through a longer continuation. Test intervention coverage on the same
twelve development targets, with no choice based on confirmation-query labels.
This is a follow-up informed by the earlier study, not a pristine unseen-benchmark
claim. Keep earlier variants and all their outputs.

Fix the contextual geometry and pure-null contrast already selected in the main
extension. On development seeds10/11, vary final prompt positions1/4/8/all,
gain1/1.5/2, scheduleall/prefill and correction0/uncertainty:48 settings. Padding
positions are excluded. Under all-step steering, continuation tokens receive the
same single-position operation as before. Under prefill, only the specified
prompt positions change.

Independently refine the existing552-setting dictionary-free winner over the same
four coverage choices, gains.5/1/1.5/2 and both schedules:32 settings. Its contrast,
raw-mean direction and layer pair remain frozen. All80 settings use identical
development demonstrations and queries. Freeze the best nonzero contextual,
best contextual overall and best dictionary-free settings by development mean.

Confirm all selected methods, matched correction-off, the earlier frozen CASS
extension and ICL4 on20 non-development LOTO targets, Novel15 and Compound10,
seeds20/21/22. Repeat on fifty additional disjoint queries per target, using the
same frozen settings and demonstrations. No new mining or model-weight training
is involved. Multi-position serving cost is a distinct variant and cannot inherit
single-position latency figures without measurement.
