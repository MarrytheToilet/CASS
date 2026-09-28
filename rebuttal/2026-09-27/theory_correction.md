# Proposed theory correction

Working technical note, 2026-09-27. This file does not modify the submitted manuscript.
The counterexample below was constructed independently during this analysis. The
review text supplied by the author contains no explicit mathematical counterexample.

## What needs correction

1. The coherence in Proposition 1 is a **maximum**, not a median. With a maximum
   near one, the published sufficient condition certifies at most one active block.
   A failed sufficient condition does not imply that recovery is impossible.
2. The actual objective weights block t by q_t = sqrt(r_t). For unequal ranks,
   these weights enter the no-false-positive condition. The unweighted condition
   cannot simply be reused.
3. An O(epsilon) coefficient bound requires lambda = O(epsilon), in addition to a
   lower bound protecting against noise. A lower bound alone permits arbitrarily
   large shrinkage and does not establish O(epsilon) error.
4. Orthonormality holds **within** each block. It does not make a joint restricted
   fit a single independent soft-thresholding operation across correlated blocks.
5. The numerical support-capped lambda path is not itself proved to meet a
   noise-calibrated lambda condition. Any theorem is conditional on its assumptions.

## An explicit unequal-rank counterexample

In R^5, let U_1 = [e_1,e_2,e_3,e_4] and U_2 = 0.6e_1 + 0.8e_5.
Each block is orthonormal, r_1 = 4, r_2 = 1, and mu_B = 0.6. Let the true
support be S = {1}, c_1^0 = (1,0,0,0), and e = epsilon e_1. The published
condition holds because 1 < (1/0.6 + 1)/2.

Set epsilon = lambda = 0.01 and z = 1.01e_1. The support-restricted optimum is
c_1 = (0.99,0,0,0), because q_1 = 2. Its residual is 0.02e_1, so

    ||U_2^T residual|| = 0.012 > lambda q_2 = 0.01.

This violates the inactive block KKT condition. Consequently the full optimum
must select block 2. The repository solver confirms c_1[0] = 0.988125 and
c_2 = 0.003125. See weighted_counterexample.json and analyze_evidence.py.
More generally, this failure holds for any positive lambda below (1+epsilon)/2,
so changing the constant relating lambda to sufficiently small epsilon does not
repair the missing rank weighting.

## Corrected deterministic statement for the actual objective

Consider

    min_c (1/2)||z - A c||_2^2 + lambda sum_t q_t ||c_t||_2,
    A = [U_1 ... U_T], q_t = sqrt(r_t) > 0,
    z = A_S c_S^0 + e, ||e||_2 <= epsilon.

Assume A_S has full column rank. Define

    G = A_S^T A_S,
    P = A_S G^{-1} A_S^T,
    W_S = blockdiag(q_t I_{r_t}: t in S),
    theta_S = max_{j not in S} (1/q_j)
              sup_{max_t ||v_t||_2 <= 1}
              ||U_j^T A_S G^{-1} W_S v||_2,
    eta_S(e) = max_{j not in S} ||U_j^T (I-P)e||_2 / q_j.

If theta_S < 1 and lambda > eta_S(e)/(1-theta_S), the unique optimum has
support contained in S. Its coefficient error obeys

    ||c_hat_S - c_S^0||_2
      <= epsilon / sigma_min(A_S)
         + lambda sqrt(sum_{t in S} q_t^2) / sigma_min(A_S)^2.

Thus the error is O(epsilon) for a fixed dictionary and lambda = C epsilon,
where C satisfies the lower-bound condition. Exact support additionally follows
if each true block norm exceeds the displayed global error bound. This is a
sufficient beta-min condition, not a necessary one.

**Proof.** Let c_tilde be the minimizer restricted to S. Full column rank makes
this minimizer unique. Its KKT equations give a block subgradient v with
||v_t|| <= 1 and

    c_tilde - c_S^0 = G^{-1}(A_S^T e - lambda W_S v).

The residual is

    r = z - A_S c_tilde
      = (I-P)e + lambda A_S G^{-1} W_S v.

For j outside S,

    ||U_j^T r||/q_j <= eta_S(e) + lambda theta_S < lambda.

These are the strict inactive-block KKT inequalities, while the active-block
conditions already hold. Extending c_tilde by zeros therefore gives a global
minimizer. All minimizers have the same fitted value, because squared error is
strictly convex in that value. Strict inactive inequalities force every
minimizer to be zero off S; full column rank then gives uniqueness. Finally,
||G^{-1}A_S^T|| = 1/sigma_min(A_S), ||G^{-1}|| = 1/sigma_min(A_S)^2, and
||W_S v|| <= sqrt(sum q_t^2), yielding the bound. The beta-min conclusion
follows because each block error is no greater than the global error. QED.

This is a finite deterministic KKT argument. The rank-weighted correlation
condition is consistent with the group-selection literature; compare Bach
(2008), Section 2.8, Eqs. (9)-(10). Do not present it as new sparse-recovery theory.
[Primary reference](https://www.jmlr.org/papers/v9/bach08b.html).

## Conservative coherence corollary

Let s = |S|, mu = max_{j != t} ||U_j^T U_t||_2, and
R_S = max_{t in S} q_t / min_{j not in S} q_j. If

    (s-1)mu < 1,

the block maximum-norm Neumann bound on G^{-1} gives

    theta_S <= s mu R_S / (1-(s-1)mu).

It is therefore sufficient that

    mu < 1 / (s R_S + s - 1).

For a support-independent condition, replace R_S by max_t q_t / min_t q_t.
The noise term is at most epsilon / min_{j not in S} q_j. Combine these
bounds with the lambda condition above. Equal ranks recover mu < 1/(2s-1).
There is no justified substitution of median coherence for mu.

The joint dictionary is block diagonal across layers. Therefore each joint
pair's coherence equals the **maximum** of its layerwise coherences. Averaging
layerwise values, as some descriptive plots do, cannot certify this condition.
Actual maxima, ranks, and certified support sizes are in coherence_results.json.
Support-specific theta_S can be less pessimistic, but it must be established for
the relevant support before claiming a guarantee. A selected support alone is
not evidence of a true generative support.

## Proposition 2: a rigorous scope-preserving version

Fix a hidden state h and a differentiable real-valued score m, such as the
log-probability of a fixed target continuation at a specified intervention.
Assume throughout the relevant ball that

    m(h+v) = m(h) + w^T v + R(v), |R(v)| <= L||v||^2.

For Delta_rec satisfying ||z-Delta_rec|| = epsilon_rec||z|| and an oracle
direction z_or with delta_z = ||z-z_or||, the triangle inequality gives

    |m(h+Delta_rec) - m(h+z_or)|
      <= ||w||(epsilon_rec||z|| + delta_z)
         + L(||Delta_rec||^2 + ||z_or||^2).

For direct injection Delta = z, the first-order term is instead ||w|| delta_z.
The proof is direct substitution, Cauchy-Schwarz, and the triangle inequality.

Use delta_z as an explicit estimation/representation discrepancy. Davis-Kahan
controls eigenspace perturbation under separate covariance and eigengap
assumptions; it does not by itself give a rate for this task-signature error.
Delete the unconditional Davis-Kahan rate unless a compatible sampling model and
the intervening bounds are supplied. Greedy exact-match accuracy is not a
differentiable score, and this one-state result does not certify the nonlinear,
multi-step, gated and routed deployed system. Describe residual/accuracy
relationships as empirical diagnostics.
