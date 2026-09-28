# Separating prompt syntax and generic instruction

The earlier native protocol changes both the chat template and the presence of
a task-agnostic system instruction. Its overall gain must not be attributed to
special-token formatting alone. Before the layer/gain/position factorial finishes,
specify an additional2x2x2 known-task oracle diagnostic:

- format: plain Q/A or native Qwen chat (thinking disabled);
- generic instruction: absent or the existing chat_probe.SYSTEM text;
- intervention: layers14+20 or layer24, alwaysgain2, one prompt position and all
  generation steps.

For the plain protocol the same instruction is prepended as ordinary text. For
the native protocol it occupies the system role. It contains no task name, query
answer or task-specific operation. Thus the instruction contrast changes the
presence of the same words; the native-format contrast includes the actual role
syntax and native generation prefix, not a claim about one special token.

Remine all32 known-task dictionaries in allfour prompt protocols on the same GPU,
with seed7000,100 pairs,10 examples and separate positive/negative batches8. This
avoids mixing historical extraction batching into the prompt comparison. Evaluate
all eight configurations on the20 targets excluded from development. This is
oracle intervention diagnosis, not unseen-task execution or inferred support.
Use paired task-bootstrap intervals for conditional format/instruction/layer
effects and report every configuration. Do not select a new serving setting from
these target-task results.
