# Weighted correction and rank refinement

Specified before any development result from this refinement. The motivating
issue is structural: a union projector gives a newly selected block full span
membership even when its coefficient weight is tiny. Compare that rule with
coefficient-weighted rank truncation and a convex combination of skill-wise
projected corrections. No confirmation-task label selects a setting.

Keep the pure clean-versus-zero contrast, dictionary and layers12/16 fixed.
Use the prompt-position count selected by the existing position-development
experiment, which uses the same12 development targets. Search120 settings:
five correction geometries (union, weighted rank8/16/32, weighted skill sum),
three gains(1/1.5/2), two schedules(all/prefill), and four uncertainty multipliers
(0/.5/1/1.5, clipped to[0,1] before multiplying the cosine gate).

For the weighted-rank alternatives, take the leading left singular vectors of
[sqrt(w_t)U_t] and keep the existing weighted contextual anchor. For the weighted
skill sum, define M=sum_t w_t U_t U_t^T and b=sum_t w_t U_t U_t^T mu_t^+.
Apply F(h)=h+gamma*d+a*g*(b-M(h+gamma*d)). This combines each skill's own projected
anchor instead of projecting the global mixed anchor into the entire union.
It uses no new training or per-query examples. M is positive semidefinite with
spectrum in[0,1], so the fixed-operator map is nonexpansive when a*g is in[0,1].
This is a state-space statement, not a guarantee of better generation.

Freeze the best nonzero correction and best overall setting using development
means. Confirm on the20 non-development known targets, Novel15 and Compound10,
with seeds20/21/22, on both original and additional queries. Compare matched
correction-off, independently selected dictionary-free position baseline, the
previous position winner, the original contextual variant and ICL4. All inputs
are matched. Previously saved baseline predictions may be reused after exact
query/demo identity checks. Both scoring metrics and all outcomes are retained.

If the refinement is evaluated on broader tasks, transfer its frozen settings
to every one of the same17 probes. This follow-up does not inherit timing or
memory measurements from a different intervention implementation.
