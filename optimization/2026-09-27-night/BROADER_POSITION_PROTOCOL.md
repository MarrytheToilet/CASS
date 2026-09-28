# Transfer of the position refinement to broader probes

Specified before position-development results are available. Freeze the three
winners of the existing 12-target position-development grid, then evaluate all
17 broader tasks with their existing queries, four demonstrations and seeds
20/21/22. Broader results never choose a position, gain or schedule.

Compare the best nonzero correction, its matched correction-off intervention,
the best overall contextual configuration and the independently refined
dictionary-free configuration. Include the previously frozen contextual variant
and the same zero-/one-/four-shot prompting outputs as the other plain-prompt
broader experiments. Reuse predictions only after exact input/seed/configuration
identity checks. Preserve all outcomes. The maximum generation lengths and
answer-delimited stopping remain unchanged. There are 374 task/seed/arm rows.

This experiment tests whether a development-selected intervention over more
prompt positions transfers to broader outputs. It does not inherit single-position
latency or claim that broader-task labels are available for tuning.
