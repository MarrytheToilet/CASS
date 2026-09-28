# Verified distinctions for the novelty response

Encoding demonstrations as activations and injecting them are established
mechanisms. ICV extracts a direction from demonstrations, applies latent shifts,
and also investigates vector arithmetic; do not claim it cannot compose or
handle a newly supplied task. CASS's distinction is mining and reusing de-shared
cross-task subspaces, selecting a sparse support from new demonstrations, and
using that geometry as a correction to the task-specific direction.
Primary source: https://proceedings.mlr.press/v235/liu24bx.html

ELICIT stores capability vectors and retrieves them for arbitrary queries; it
explicitly evaluates unseen-task transfer. Its official implementation describes
a trained retriever. Thus 'retrieval cannot leave known tasks' is not an accurate
characterization. Contrast the selection and representation mechanisms, not the
existence of unseen-task transfer. Its vector-library premise is important prior
art for CASS and should be acknowledged directly.
Primary sources: https://arxiv.org/html/2410.09343v2 and
https://github.com/LINs-lab/ELICIT

ATV uses a smaller model to produce query-conditioned vectors and transforms them
to the target LLM's representation. It also reports unseen-task generalization.
The comparison should distinguish learned generation from CASS's gradient-free
mining and sparse coding, rather than novelty of encoding/decoding itself.
Primary source: https://arxiv.org/abs/2506.03426

The response should identify nearest-skill, top-PC and anchor-blend controls as
matched mechanism controls; they are not complete reproductions of every system
above. The new dense dictionary-free search strengthens the mechanism comparison
but does not become a full published-system reproduction by renaming it.
