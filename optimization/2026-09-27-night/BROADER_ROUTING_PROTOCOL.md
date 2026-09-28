# Frozen submitted routing on broader tasks

Evaluate the paper's two published signals on the17 predeclared broader tasks,
without tuning thresholds on those tasks. Use the original label-shuffle
signature's norm and residual for all comparisons. At each demonstration seed,
residual>0.7 selects four-shot ICL; otherwise norm<5.125 selects prompt-state
replacement; otherwise select steering. Also report the norm-only policy without
ICL escalation. These thresholds come from the submitted serving policy and cost
dial, not the broader outcomes.

The contextual operator can be substituted into the same steering branch while
keeping both branch assignments and all fallback predictions identical. This is
a matched-policy diagnostic, not a newly calibrated contextual router. Report
per-task quality, ICL escalation fraction and prompt-token use. The full broader
results remain visible, including zero-shot and one-/four-shot ICL. A route using
four-shot ICL does not count as successful zero-context steering.

Accuracy and token F1 are different metrics and will not be pooled into a single
unqualified mean. Reading comprehension and each entity type remain separately
reported. The controlled code expressions and long structured outputs do not
stand in for unrestricted software engineering, tool use or open-ended prose.
