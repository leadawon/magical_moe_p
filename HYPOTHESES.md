# MoE Routing — Problem-Definition Hypotheses

We probe a modern, production MoE model (**Qwen3-30B-A3B**: 48 MoE layers,
128 routed experts, top-8 per token) to find a *research problem* worth
attacking. The probe feeds raw text from three domains — math, code, NLI —
and records, at every layer and token, the router's scores over all 128
experts and the final top-8 selection. From the aggregated logs we test the
following hypotheses. Each is phrased so the data can confirm *or* refute it.

> Uniform reference: with top-8 of 128 experts, a perfectly balanced router
> gives every expert a selection frequency of 8/128 = **0.0625**, and any
> single expert a per-slot frequency of 1/128 = **0.0078**.

---

## H1 — Domain-conditional expert specialization
**Claim.** Specific experts are preferentially and *consistently* activated for
a given domain; math text, code text, and NLI text drive distinguishable
expert-selection patterns.

- **Signature if true:** per-layer expert selection-frequency vectors are more
  similar *within* a domain (gsm8k↔svamp, humaneval+↔mbpp+, mnli↔snli) than
  *across* domains.
- **Metric:** cosine similarity of selection-frequency vectors; mean
  within-domain vs mean cross-domain; full 6×6 matrix.
- **Confirm:** within − cross gap > 0.05. **Refute:** gap ≈ 0 (routing is
  domain-agnostic). A very high *absolute* cross-domain similarity (>0.85) is
  itself a finding — it points to redundancy (see H5).

## H2 — Load imbalance: dead and hot experts
**Claim.** Despite load-balancing during training, routing at inference is
imbalanced — some experts are over-used ("hot"), many are near-unused ("dead").

- **Signature if true:** high Gini of per-layer selection counts; a non-trivial
  fraction of (layer, expert) slots far below uniform.
- **Metric:** per-layer Gini; fraction of slots with freq < 0.1×uniform (dead)
  and > 5×uniform (hot).
- **Confirm:** mean Gini > 0.3 or dead-slot fraction > 5%. **Refute:** Gini < 0.18.

## H3 — Depth-dependent specialization
**Claim.** How sharply routing distinguishes domains varies with layer depth
(e.g. early layers process generic/syntactic features uniformly; deeper layers
route by semantics/domain — or the reverse).

- **Signature if true:** per-layer cross-domain divergence trends with depth.
- **Metric:** per-layer mean pairwise Jensen–Shannon divergence between the
  three domain selection distributions; correlation with layer index; peak
  layer; early-third vs late-third means.
- **Confirm:** late-third > 1.3×early-third (or vice-versa). **Refute:** flat.

## H4 — Low router confidence / fragile margin
**Claim.** The router's distribution is flat: the gap between the 8th selected
and the 9th (rejected) expert is tiny, so the top-k boundary is near-arbitrary
and small perturbations would flip which experts run.

- **Signature if true:** small top-1 probability, high normalized entropy, small
  8th-vs-9th margin.
- **Metric:** mean top-1 prob, mean cumulative top-8 mass, mean normalized
  entropy (÷ln128), mean margin = p[8th] − p[9th].
- **Confirm:** margin < 0.01 or normalized entropy > 0.6. **Refute:** decisive
  router (margin > 0.02, low entropy).

## H5 — Cross-domain redundancy (universal core)
**Claim.** A large set of experts is used above-uniform by *every* domain — a
"universal core" — implying limited domain-specialized capacity and redundant
experts for any single domain.

- **Signature if true:** large intersection of above-uniform experts across all
  six datasets, carrying most of the routing mass.
- **Metric:** per-layer universal-core size (fraction of experts active in all
  datasets) and the routing mass it carries.
- **Confirm:** core > 30% of experts. **Refute:** core < 15% (disjoint usage).

## H6 — Domain is decodable from the routing fingerprint
**Claim.** A single example's [layers × experts] selection pattern carries enough
signal to recover its domain — i.e. routing *encodes* the domain even if no
single expert is exclusive to it.

- **Signature if true:** a simple classifier on per-example fingerprints predicts
  the domain well above chance.
- **Metric:** nearest-centroid (cosine) classifier accuracy, 50/50 split,
  3-class (chance = 0.33).
- **Confirm:** accuracy > 0.6. **Refute:** accuracy ≈ chance.

---

## Why these matter (the research framing)
The hypotheses are deliberately in tension, and the *combination* of answers
defines a problem:

- **H1+H6 true but H5 also true** ⇒ domains are decodable, yet most capacity is
  a shared core with only a thin specialized rim. The research question becomes:
  *can we expand/relocate specialized capacity, or prune the redundant core,
  without hurting any domain?*
- **H2+H4 true** ⇒ the router is both imbalanced and low-margin ⇒ the selection
  is wasteful and unstable ⇒ motivates **input/Domain-conditioned routing,
  expert pruning, or router sharpening** as the thesis direction.
- **H3** tells us *where* in depth to intervene.

These map directly onto the prior MAGICAL MoE-P pruning idea but ground it in a
measured problem rather than an assumption.
