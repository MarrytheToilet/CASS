# Cross-model contrast follow-up

Qwen3-4B and Qwen2.5-3B-Instruct each use the same fixed 12-task development
split, with seeds10/11 and no evaluation queries. Their native tokenizer chat
template renders identical system/user content; Qwen3 thinking is disabled.
The grid contains 180 shuffled/null mixture configurations and 20 contextual
residual variants, defined in qwen_contrast.py before their runs. Both the best
nonzero-correction configuration and best overall configuration are frozen.
Confirmation uses seeds20/21/22 on the 20 known tasks excluded from development,
Novel15, and Compound10, with identical data splits and metrics. For known-task
transfer, the task is excluded from dictionary mining and shared estimation.
Matched no-correction, native shuffled contrast at layer24/gain2, original
plain-prompt CASS at layers14/20/gain1, native ICL4 and zero-shot are retained.
The layer24 native comparator is a fixed transfer choice on Qwen2.5, not a
Qwen2.5-tuned baseline. Model-specific selected configurations are the main
comparisons. No inference on architecture-wide steerability follows from any
single intervention. All comparisons retain the eight-token generation budget.

Support controls separately test the frozen Llama combined variant. Wrong
supports exclude inferred and true skills; each block matches its per-layer
ranks, and within-support mean/max coherence must differ by at most .02/.03.
Three closest alternatives from a deterministic 6000-draw search are retained;
only tolerance-passing alternatives enter the matched comparison. Additional
controls normalize each corrective vector to the reference correction norm at
the same hidden state, retaining direction/support as the changed variable.
The original direction, rescaling, gate, correction strength and schedule remain
fixed. True-correction and full true-support versions are distinguished.
