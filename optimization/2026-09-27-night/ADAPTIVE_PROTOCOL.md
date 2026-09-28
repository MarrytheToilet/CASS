# Demonstration-only adaptation experiment

An additional exploratory variant will calibrate injection gain and schedule
using the same four supplied demonstrations. For each of four folds, both the
input and label of the held-out demonstration are excluded from signature
extraction and operator construction. Build the operator from the other three
examples and score the held-out output by length-normalized teacher-forced
negative log likelihood. Average the four losses, select a configuration, then
extract the final operator from all four examples. No evaluation label or extra
demonstration is used.

Candidate gains are 0.5, 1.0, 1.5, 2.0, and 3.0; injection occurs at every step
or at prefill only. The original dictionary correction is kept for CASS and
removed for z-only. Prompt-state replacement with these two schedules is also
eligible for routing. Compare (1) calibrated CASS; (2) z-only under exactly the
CASS-selected gain, schedule, and replacement decisions; (3) independently
calibrated z-only; and (4) original fixed CASS and ICL4. Record all fold losses,
selected configurations, total extraction/model calls, adaptation time, and
generated answers. This is a new calibration variant, not a re-description of
the originally submitted method.

These experiments can establish whether demonstration-only calibration improves
the quality-cost tradeoff. They cannot make every benchmark task successful or
turn support identification into proof of sequential causal execution.

A second, separately logged calibration applies the identical four-fold rule
to the development-selected combined contrast and geometry. This selection
must be frozen before any calibration evaluation. Candidate modes are CASS,
matched no-correction and averaged clean-state replacement, with gains
0.5/1/1.5/2/3 and all/prefill schedules. CASS and no-correction each select from
their own mode plus replacement; the attribution arm shares CASS's entire policy
and replacement outputs. Every held-out demonstration's input and label are
excluded from the fold's extraction. Full-four and three-demo fold signatures
use the same frozen contrast fraction. Pure-null extraction needs 28+4*21=112
sequences and 88 teacher-forcing forwards; this additional adaptation cost is
logged and must accompany any performance claim. This is a supplementary
per-task calibration variant, not the submitted fixed-cost 48-sequence method.
