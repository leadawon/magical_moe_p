# Experiment Plan — MoE Routing Probe

## Goal
Find and ground a research problem for MoE work on a modern model by **measuring
how Qwen3-30B-A3B routes tokens to experts across three domains**, then testing
the six hypotheses in [HYPOTHESES.md](HYPOTHESES.md).

## Model
- **Qwen3-30B-A3B** (reused local weights, bf16, `device_map="auto"`).
- 48 decoder layers, **all** MoE (`decoder_sparse_step=1`, `mlp_only_layers=[]`),
  128 routed experts, top-8, `norm_topk_prob=true`, hidden 2048.
- GPUs **0,1,2,3** only.

## Datasets (200 examples each, fixed seed 42)
| domain | datasets | source | input text |
|---|---|---|---|
| math | gsm8k, svamp | openai/gsm8k(test), ChilleD/SVAMP(test) | question (svamp: body+question) |
| code | humaneval_plus, mbpp_plus | evalplus/humanevalplus, evalplus/mbppplus | the prompt / function stub |
| nli  | mnli, snli | nyu-mll/glue(mnli, val_matched), stanfordnlp/snli(test) | "Premise: … Hypothesis: …" |

Two datasets per domain let us separate *domain* signal from *dataset* signal
(H1 within-domain pairs).

## What we capture (per layer, per token)
Forward hooks on every `mlp.gate` (a plain Linear → output = router logits over
128 experts). From the logits we reproduce the model's own `softmax → top-8`
and accumulate **per (layer, expert)**:
- selection counts (top-8 membership), pre-selection softmax prob mass,
  renormalized top-8 weight mass;
and **per (layer)**: token count, Σ top-1 prob, Σ top-8 cumulative mass,
Σ entropy, Σ margin (p[8th]−p[9th]).
Plus a **per-example fingerprint** [48×128] = top-8 selection frequency (for H6).

## Design choices (and why)
- **Forward-only, no generation.** Deterministic, fast, hang-free, and isolates
  *domain-input* routing for clean cross-domain comparison. (Generation-time
  routing is logged as future work — it adds the model's own output tokens and
  length variance.) This also sidesteps the Qwen3 thinking-mode generation
  hangs seen earlier.
- **Batch size 1, no padding.** Every routed token is a real content token; no
  padding tokens contaminate the statistics.
- **Raw task text, no chat template.** Routing reflects domain content, not
  shared template tokens. (A chat-template variant is a possible ablation.)
- **Hook the gate, not the block.** Version-independent: a gate is always a
  Linear whose output is exactly the logits, regardless of block internals.

## Pipeline
1. `src/probe.py` — load model once, loop datasets × examples, save
   `results/<dataset>_routing.npz` + `results/manifest.json`; live progress in
   `results/STATUS.txt`.
2. `src/analyze.py` — load all npz, compute the metrics, write
   `results/FINDINGS.md`, `results/findings_summary.json`, and derived arrays in
   `results/derived/` (similarity matrix, layerwise JS, core fraction, …).
3. `scripts/run.sh` runs (1) then (2).

## Hypothesis → metric → decision
| H | Metric computed in analyze.py | Confirm threshold |
|---|---|---|
| H1 | within vs cross-domain cosine of selection-freq | gap > 0.05 |
| H2 | per-layer Gini; dead/hot slot fraction | Gini>0.3 or dead>5% |
| H3 | per-layer cross-domain JS vs depth | late>1.3×early (or inverse) |
| H4 | top-1 prob, entropy, margin(8th−9th) | margin<0.01 or H̄>0.6 |
| H5 | universal-core size & mass | core>30% |
| H6 | nearest-centroid domain accuracy on fingerprints | acc>0.6 (chance .33) |

Thresholds are heuristic decision aids for an exploratory study, stated up
front to avoid post-hoc rationalization. Raw numbers are always reported.

## Runtime / cost
~0.4–0.5 examples/s (batch-1, 48-layer hooks across 4 GPUs) ⇒ ~7–8 min/dataset,
**~45–60 min total** for 6×200, plus ~1 min model load. ~15 GB on each of GPU
0–3. Logs to `logs/probe.log`; `results/STATUS.txt` is the at-a-glance progress
file for monitoring.

## Limitations / threats to validity
- Selection is recomputed from logits; it matches the block's own top-k because
  `norm_topk_prob` only rescales weights, not the argtop-k set.
- Forward-only ≠ task-solving routing; generation routing may differ (future).
- 200 examples/dataset gives ~1M routing decisions/dataset — strong for
  aggregate stats, lighter for rare-expert tails.
- No chat template ⇒ slightly off the instruct distribution; intentional for
  domain purity.

## Follow-ups if hypotheses hold
- Generation-time routing capture; co-activation graphs; per-token-type
  (operator/number/keyword) routing; expert ablation → accuracy to test the
  redundancy/pruning thesis directly.
