# Geometry of the new contextual correction

This statement concerns the development-selected extension, not the submitted
state-dependent operator. It explains what is composed and preserved without a
support-recovery or language-model accuracy assumption.

Let P be the orthogonal projector onto the union of the selected skill subspaces,
mu their code-weighted positive-context anchor, d the new task's demonstration
direction, gamma >= 0, and a = c g in [0,1]. The selected operator is

    F(h) = h + gamma d + a P(mu - h - gamma d).

Here c is the clipped disagreement of the four extracted signatures, and g is
the nonnegative cosine gate. Both are fixed after task adaptation. The projector
and anchor are assembled from the selected dictionary blocks. Write b=h+gamma d.
Then

    (I-P) F(h) = (I-P) b,
    P F(h) = (1-a) P b + a P mu.                              (1)

Thus the task-specific component orthogonal to the selected span is preserved
exactly. Inside that span, the state moves toward the composed context anchor by
a convex interpolation. This is a geometric corrective operation, not a claim
that the model sequentially executes the constituent task functions.

For any h and h' with the task operator fixed,

    F(h)-F(h') = (I-aP)(h-h'),
    ||F(h)-F(h')||^2 = ||(I-P)(h-h')||^2
                       + (1-a)^2 ||P(h-h')||^2
                     <= ||h-h'||^2.                         (2)

The proof of (1) uses P^2=P and P(I-P)=0. For (2), the projected and orthogonal
components are orthogonal. The correction is therefore nonexpansive at the
intervened state and contracts differences inside the selected span by 1-a.
This does not make the remaining transformer layers nonexpansive or guarantee
greedy generation accuracy.

A simple perturbation statement is also available. For a second operator using
P*, mu*, d*, a*, define delta_P=||P-P*||_2 and analogously delta_mu, delta_d,
delta_a. At the same h, set R=||mu* - h - gamma d*||. A telescoping decomposition
gives

    ||F(h)-F*(h)|| <= gamma delta_d + delta_mu
                     + R(delta_P + delta_a).                (3)

Indeed, write F=(I-aP)(h+gamma d)+aPmu. Changing d first costs at most
gamma delta_d by (2); changing mu costs at most delta_mu. With the starred
direction and anchor fixed, the remaining difference is
(aP-a*P*)(mu*-h-gamma d*), and ||aP-a*P*|| <= delta_P+delta_a.
This directly separates the task-direction error from errors in the composed
geometry. All bounds are conditional on fixed selected operators; no assertion
that the highly coherent dictionary satisfies exact support recovery is needed.

Finally, serving can precompute

    beta = gamma d + a P(mu-gamma d),
    F(h) = h + beta - a P h.

The implementation uses an orthonormal basis Q, so P h is computed as Q(Q^T h)
without materializing a dense d-by-d projector. The latency experiment checks
the collected affine form against the original expression before timing it.
