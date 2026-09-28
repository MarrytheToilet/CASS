# Additional-query transfer check for both Qwen models

Specified at approximately 00:02 UTC on28 September2026, after the Qwen3
original-query confirmation and before either additional-query run. Qwen2.5
development and confirmation have not yet run. This expands validation; it
does not select or change any setting using the Qwen3 confirmation results.

For both Qwen3-4B and Qwen2.5-3B, evaluate the same independently
development-frozen configuration on the last50 target dict_pool inputs.
These inputs are disjoint from the target's evaluation and few-shot pools.
Exclude each known target from dictionary mining, including the shared
component, before constructing its LOTO operator. Use all20 non-development
known targets, Novel15 and Compound10, seeds20/21/22, and the same four
demonstrations as the original-query confirmation. Keep all seven arms,
generation settings and both accuracy metrics.

`qwen_contrast.py MODEL fresh` records a separate ledger and completion marker.
No new rank, layer, gain, contrast fraction or correction weight is chosen.
Both model follow-ups are queued irrespective of the outcomes.
