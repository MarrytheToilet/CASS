# Exact order averaging with four examples

There are exactly3!=6 orders of the other three examples in each leave-one-out
query. The current extractor samples six orders with replacement. A parameter-free
follow-up enumerates all six instead, keeping the same24 positive sequences and
four zero-shot queries, dictionary, layer pair, gain, correction rule and serving
schedule. Conditional on the four examples, this removes Monte Carlo order
sampling variance in the finite-order mean; it does not guarantee better accuracy.

No parameters or method choice are selected from evaluation outputs. Test the
frozen contextual operator and its matched correction-off counterpart on all20
non-development LOTO targets,15 Novel targets and10 Compound targets, with
seeds20/21/22. Compare with the earlier frozen operator/off and ICL4 on identical
queries/demonstrations. Repeat on the same separate fifty-query validation sets.
Reuse existing baseline generations only after matching inputs and examples.

This is an additional extraction refinement informed by the preceding study.
Keep every outcome. It does not inherit a claim about an independently tuned
dictionary-free optimum or measured wall-clock latency; only its logical28-
sequence extraction count is identical. The newly estimated signature also
changes inferred support, normalization and the fixed uncertainty gate, which
are recomputed by the unchanged pipeline.
