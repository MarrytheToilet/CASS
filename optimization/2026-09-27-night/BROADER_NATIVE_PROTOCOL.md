# Native-template follow-up on the complete broader suite

Declared before native-template development or broader native generations.
Use the settings selected only by the twelve established development tasks in
LLAMA_NATIVE_PROTOCOL.md. Evaluate all seventeen tasks from BROADER_PROTOCOL.md,
with the identical queries, four demonstrations, three seeds, output budgets and
scoring rules. No selection or tuning uses broader outputs or query labels.

Compare the frozen native contextual setting, its correction-off counterpart,
the independently selected native dictionary-free baseline, native original CASS,
native one-/four-shot ICL and native zero-shot. Zero-shot is generated once;
all other arms use seeds20/21/22. The generic system instruction carries no task
name, task-specific operation, query answer or other new task information.

Extraction uses batch4 for longer inputs. Serving also uses batch4. The same
answer parser and verified marker-based stopping apply to every arm. Include the
system and chat-template tokens in token accounting. Results are a new follow-up;
plain-template latency measurements cannot be claimed for this variant.

Retain all323 cells, including zero scores. This suite contains controlled long
structured transformations and restricted executable expressions, not unrestricted
long-form generation, general software engineering or agentic tool use.
