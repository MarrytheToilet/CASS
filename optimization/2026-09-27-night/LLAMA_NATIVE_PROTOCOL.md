# Native-template Llama follow-up

Motivation: native-template extraction improved Qwen3 in the earlier controlled
probe. Test the same mechanism on Llama-3.1, with a model-specific layer check.
This is a new exploratory follow-up informed by previous aggregate findings,
not a claim that the entire benchmark remained unseen during research.

Before native Llama mining or generations, tokenization was checked against
the native template's token IDs without additional special tokens. The shared
string-based generation helper inserts BOS through the tokenizer post-processor;
render() removes a leading template BOS only when that same token will be
reinserted. All48 checks per model pass for Llama3.1, Qwen3 and Qwen2.5 on both
the local and remote tokenizers. Qwen input IDs are unchanged. See
native_tokenization_checks.json and the [Transformers chat-template guidance](https://huggingface.co/docs/transformers/v4.57.1/chat_templating).

Use the tokenizer's native chat template and the same generic system instruction
as the Qwen probe. No task name or target-query answer appears in that instruction.
Mine known-task activations at layers12/16/20/24 from the same100 pairs and seed.

On the established12 development targets and seeds10/11, evaluate exactly:
- contextual correction: fractions0/.25/1, layer pairs12+16/16+20/20+24,
  gains.5/1/1.5/2, correction0/uncertainty, schedulesall/prefill (144 settings);
- independent dictionary-free baseline: the same fractions/layers/gains/schedules,
  using raw mean or signed PC1 directions (144 settings).

Freeze the best nonzero-correction, best contextual setting overall, and best
dictionary-free setting using only these development outputs. Confirm all three,
matched correction-off, native-template original CASS, and native ICL1/4/zero on
the20 non-development LOTO targets, Novel15 and Compound10 with seeds20/21/22.
Also check an additional50 target-dictionary-pool queries with the same frozen
parameters. All earlier plain-template results remain available for matched-input
comparison. The native-template original baseline uses the submitted layers12/16,
label-shuffle extraction and gamma1; it receives its own matching mined dictionary.

Retain every result, including cases where the dictionary-free or ICL baseline
wins. Native-template prompt tokens count toward serving cost; this variant is
not included in the earlier plain-template latency measurements.
