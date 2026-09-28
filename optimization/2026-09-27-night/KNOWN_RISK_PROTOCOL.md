# Known-task calibration transferred to the expanding library

Specified at approximately00:11 UTC on28 September2026 after observing the
degenerate eight-task contextual calibration split. This is an additional
post-hoc protocol. Earlier calibration analyses are retained in full.

Fit only on the32 known tasks' existing LOTO records, seeds20/21/22:96
task/seed observations. Use `llama_confirm/original` for the submitted operator
and `extension_confirm/combined` for the contextual extension. The language
model stays frozen. This is an explicitly supervised auxiliary risk fit using
known-task execution labels, not an untrained probability guarantee.

For each task/seed, define failure as that seed's submitted-prefix accuracy
below0.5. Features are that same four-example extraction's residual and/or
signature norm. Unlike the earlier task-mean calibration, no three-seed feature
average is required at serving time. This is a different, explicitly declared
failure endpoint and observation unit.

Use standardized logistic regression with L2 penalty and C=1, the same fixed
form as earlier analyses: residual-only, norm-only and joint. Retain a constant
known-training failure-rate predictor. Do not choose features, regularization
or a threshold using Novel/Compound outcomes.

Freeze these known-task mappings and apply each to all25 disjoint Novel and
Compound targets, seeds20/21/22, at32/64/128/256 skills, full/shortlist solvers,
and fixed/recomputed shared direction. Assert target disjointness and that
the size32 full predictions match the corresponding original confirmation.
Report all mappings and conditions. Brier is primary, with paired bootstrap
intervals resampling25 task clusters (all three seeds stay together), conditional
on the fitted known-task mapping. Five-bin ECE and pooled test AUROC are diagnostic.
No held-out execution label enters the fitting code, but the analysis was
designed after earlier evaluation results and is not a blind validation claim.
