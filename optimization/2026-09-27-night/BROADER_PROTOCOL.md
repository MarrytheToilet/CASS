# Broader probes (defined before model generation)

Seven held-out natural-language task types use data already distributed with the
repository's Function Vectors dataset: AG News topic classification, sentiment,
CommonsenseQA choice prediction, SQuAD reading comprehension, and three CoNLL
entity extraction categories. These are new task types for the original frozen
32-skill dictionary. The provided dataset conversion contains one reference per
example; token F1 therefore measures overlap against that reference, not the full
official multi-reference benchmark score. Classification uses label accuracy.

Three synthetic long structured-output tasks require title casing, uppercasing,
or reversing 20--30-word sequences. They use whitespace-normalized exact match
of the complete extracted first answer line, plus diagnostic position accuracy.
Scoring clarification added after the first results, with no scorer change:
the existing parser removes an opening code fence when present and ignores later
explanatory lines. They are structured transductions, not open-ended
long-form writing.

Seven synthetic code-expression tasks require sum/min/max/sorting of a list, or
sorting then selecting three entries then aggregating. The primary metric parses
a single expression and evaluates a whitelist of AST operations. Success requires
an operation node and the correct execution result. This is a controlled code and
multi-step probe, not a claim of general code-generation capability or tool use.
Post-hoc interpretation of the unchanged probes: the composed-reference
expressions need not reveal an internal execution order:
in particular, min(sorted(xs)[:3]) simplifies to min(xs) for these nonempty lists.
The scorer accepts any whitelisted expression with the correct execution result.

Splits, examples, and metrics are deterministic in broader_tasks.py. No query
label enters extraction or configuration selection. Compare original CASS,
matched z-only, prompt-state replacement, zero-shot, and four-shot ICL. Any tuned
CASS configuration must be selected using the separate original-task development
split. Use four demonstrations, seeds 20/21/22, all predeclared task types, and
report task-level results regardless of outcome.

The contrast-mixture winner selected exclusively on the original development
tasks will also be evaluated on the same broader split. Its extraction mixture,
gain, and schedule stay frozen. Reuse byte-identical baseline predictions when
queries and demonstrations match, recording their provenance, rather than
repeating unchanged baseline generation. Keep original and updated CASS in
separate result files and report both.

Runtime optimization: once the existing first-answer parser's output cannot
change, generation may stop at that boundary. This never shortens the maximum
budget before the answer is complete and does not alter the scorer. Before use,
check_answer_stop.py verifies every such boundary in saved broader outputs and
compares full versus stopped greedy generation on 96 examples across four tasks
and zero/one/four-shot prompts. Future rows record whether that verification
marker enabled stopping. The original in-progress baseline run retains its
previous generation behavior; parsed scores are matched by the verification.
