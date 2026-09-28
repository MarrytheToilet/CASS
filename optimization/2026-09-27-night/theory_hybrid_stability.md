# A direct stability statement for the deployed hybrid operator

This supplements the corrected weighted support result in
`rebuttal/2026-09-27/theory_correction.md`. It replaces an indirect claim about
reconstruction-only injection with a statement about the implemented correction.
It does not claim new general sparse-recovery theory or certify discrete accuracy.

Fix a hidden state h, gain gamma>=0, beta>=0 and numerical denominator eta>0.
For an orthogonal projector P, anchor mu, direction d and gate g in [0,1], define

    alpha = min(1, beta ||(I-P)(h-mu)|| / (||h||+eta)),
    F(h;d,mu,P,g) = h + gamma[1-g+g alpha]d + g alpha P(mu-h).

This is the submitted operator with alpha_max=1, including its adaptive additive
weight. Let starred quantities define a reference operator at the SAME h. Write

    delta_d = ||d-d*||, delta_mu = ||mu-mu*||,
    delta_P = ||P-P*||_2, delta_g = |g-g*|,
    R = ||mu*-h||, B = delta_mu + R delta_P,
    L_alpha = beta / (||h||+eta).

Then

    ||F-F*|| <= gamma delta_d + B
                + (gamma ||d*|| + R)(delta_g + L_alpha B).             (1)

**Proof.** Set v=P(mu-h) and v*=P*(mu*-h). Orthogonal projectors have operator
norm at most one, so ||v-v*||<=B and ||v*||<=R. The same decomposition, applied
to (I-P)(h-mu), gives a difference at most B. The norm and scalar clipping at
one are both 1-Lipschitz; hence |alpha-alpha*|<=L_alpha B. As g,alpha lie in
[0,1], both |g alpha-g*alpha*| and
|[1-g+g alpha]-[1-g*+g*alpha*]| are at most
|g-g*|+|alpha-alpha*|. The additive-direction difference is therefore bounded
by gamma delta_d + gamma ||d*||(delta_g+L_alpha B), and the correction difference
by B+R(delta_g+L_alpha B). Adding proves (1).

The inequality above can be applied to each intervened layer, with that layer's
direction, anchor and projector. The implementation computes one shared gate in
the stacked multi-layer space. Denote its stacked anchor by mu_bar, distinct from
the per-layer mu above. For
g=max(0,z^T mu_bar/(||z||||mu_bar||+epsilon_g)), epsilon_g>=0, if both compared
stacked z norms are at least a>0 and both stacked anchor norms at least b>0,
an additional bound is

    delta_g <= min(1, 2||z-z*||/a + 2||mu_bar-mu_bar*||/b).

For epsilon_g=0 this follows from the 1-Lipschitz positive-part map and
||u/||u||-v/||v||||<=2||u-v||/min(||u||,||v||). The implemented epsilon_g>0
has the same bound: change z with mu_bar fixed and then mu_bar with z fixed; in
each step the relevant normalized vector is v/(||v||+c), c>=0, which obeys
the identical two-endpoint Lipschitz bound. Thus numerical denominator
regularization does not invalidate this inequality.
The lower-norm assumptions matter; near-zero signals cannot be called uniformly
stable from this argument. The direct bound (1), with measured delta_g, does not
need those lower-norm assumptions.

If a fixed-context downstream smooth score m has gradient norm at most K on
the segment joining F and F*, the fundamental theorem of calculus gives
|m(F)-m(F*)|<=K times the right-hand side of (1). This is a one-intervention,
fixed-state stability statement. It distinguishes direction error, anchor error,
subspace error and gate error, and exposes where support mistakes enter the
actual hybrid. It is not a guarantee for greedy exact match, multi-step free
rollout, or the data-dependent routing policy. The support theorem alone does
not bound these perturbations without additional assumptions; no such empirical
certificate is asserted for the high-coherence dictionaries.

For the norm-based routing threshold tau, the reverse triangle inequality
ensures an unchanged route whenever ||z-z*|| < abs(||z*||-tau). A fixed-state
steering bound therefore extends to that branch under an explicit routing-margin
condition; it does not imply an unconditional bound at the switching threshold.
