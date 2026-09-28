# Demonstration-only comparison of steering and prompting

Added after the completed fixed-residual routing and calibration analyses, before
measuring any ICL teacher-forced losses for this experiment. Keep the original
development-selected contextual setting fixed. For each of four supplied examples,
extract the steering signature using only the other three examples, then score
the held-out answer under that fixed steering operator. Reuse these exact
fourfold losses from the completed adaptive-combined experiment. Separately score
the same held-out answer with the other three examples directly in the prompt.
Targets are scored by average token negative log likelihood using the same helper.

Route to ICL4 only if the mean fourfold ICL3 loss is strictly smaller than the
mean fixed-CASS loss; ties use CASS. No margin, per-task threshold, feature model,
evaluation label or accuracy-based selection is introduced. The adaptive grid's
minimum loss is not used in this rule. Report every task and seed on both original
and additional queries, with identical ICL outputs in the correction-off arm.
The already submitted residual0.7 policy, pure steering and ICL4 remain controls.

This is a separate, more expensive task-adaptation variant. Its minimal logical
work is112 extraction sequences(full task plus four three-example folds), four
steered teacher-forced calls and four ICL teacher-forced calls. Existing losses
and predictions may be reused for this study only after exact input/configuration
identity checks. Reuse does not make the deployment adaptation cost zero or allow
it to inherit the earlier0.2-second measurement. The decision uses only the four
available example labels and does not train model parameters.
