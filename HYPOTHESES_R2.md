# MoE Problem Hypotheses — Round 2 (failure-mode probes)

Round 1 established: domain structure exists in aggregate (H1/H6), but routing is
~50% example-variable, 32% experts are dead *per dataset*, the top-k margin is
0.0023, and consistency tracks input homogeneity (mnli vs snli). Round 2 targets
the **actual weaknesses** of the MoE methodology, each as a falsifiable probe on
Qwen3-30B-A3B.

---

## P1 — Expert functional redundancy
**Problem.** If many experts compute near-identical functions, the model carries
redundant capacity → the "128 experts" are effectively far fewer.
- **Test.** At several layers, feed the same token hidden-states through *all* 128
  experts; measure pairwise cosine of expert outputs and the **effective rank**
  (participation ratio of the expert-output matrix's singular values).
- **Metric.** mean off-diagonal output cosine; #pairs > 0.9; effective #experts.
- **Why it matters.** High redundancy ⇒ pruning/merging headroom; directly tests
  the compression thesis.

## P2 — Routing is lexical, not contextual
**Problem.** If a token's expert choice is fixed by its *identity* rather than its
*context*, the "specialization" is a learned lookup table, and the domain signal
(H1/H6) is just a token-distribution artifact — not semantic computation.
- **Test.** For frequent token-ids appearing in many contexts, measure top-k set
  overlap across occurrences of the *same* token vs across *random* tokens.
- **Metric.** same-token routing overlap vs random-token baseline (per layer).
- **Why it matters.** If routing is lexical, domain-conditional designs are built
  on sand; the real lever is token-level, not domain-level.

## P3 — Top-k is over-provisioned at the margin
**Problem.** If the 5th–8th selected experts barely change the output, ~half the
routed FLOPs are wasted, and the low margin (H4) is "ties that don't matter."
- **Test.** Recompute each MoE block's output with k=1,2,4,6 (renormalized) and
  compare to the true k=8 output (relative L2).
- **Metric.** ‖out_k − out_8‖ / ‖out_8‖ for k ∈ {1,2,4,6}, per layer.
- **Why it matters.** Quantifies routing waste and the real "effective k".

## P4 — Attention-sink-style routing
**Problem.** If the first token(s) route to a near-deterministic expert set across
all inputs (like attention sinks), that capacity is spent on a structural artifact
rather than content.
- **Test.** Routing consistency of position-0 across examples vs later positions.
- **Metric.** position-0 cross-example top-k overlap vs mean-position overlap.
- **Why it matters.** Identifies fixed-overhead experts; informs where capacity
  is mis-allocated.

## P5 — "Dead" experts: domain-gated vs globally wasted
**Problem.** Round-1's 32% dead is *per dataset*. The real question: are those
experts domain-specialists (alive elsewhere) or **never used by anyone** (truly
wasted capacity)?
- **Test.** Pool selection over all 6 datasets; count experts never selected by
  any token in any domain; build a per-expert **domain-breadth** histogram
  (active in 0/1/2/3 domains).
- **Metric.** #globally-dead experts/layer; breadth distribution.
- **Why it matters.** Breadth-0 = wasted capacity (prune outright); breadth-1 =
  domain specialists (prune *per deployment domain*). Different theses.

## P6 — Routing instability under trivial perturbation
**Problem.** Low margin (H4) predicts fragility. Operationalize it: do
meaning-preserving perturbations flip routing?
- **Test.** Prepend an innocuous prefix to each prompt; compare each content
  token's top-k set with vs without the prefix.
- **Metric.** % of (token,layer) top-k sets that change.
- **Why it matters.** If irrelevant context reshuffles experts, routing is not a
  stable function of meaning — a robustness problem and a moving target for any
  routing-based method.

---

### The thesis these build toward
- **P1 + P3 + P5(breadth-0)** = a *capacity-waste* problem (redundant + over-k +
  unused) → compression/pruning is well-motivated and measurable.
- **P2 + P6** = a *routing-quality* problem (lexical + fragile) → the router, not
  the experts, may be the bottleneck.
- **P4** = a *capacity-allocation* problem (fixed overhead).

Outputs: `results_r2/FINDINGS_R2.md`, per-probe arrays in `results_r2/`.
