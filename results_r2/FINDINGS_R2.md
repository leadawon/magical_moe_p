# MoE Problem Probes — Round 2 Findings (P1–P6)

Model: Qwen3-30B-A3B (48 MoE layers, 128 experts, top-8).

## P1 — Expert functional redundancy

All 128 experts run on the same live hidden states; we measure how
similar their outputs are and the effective dimensionality of the bank.

```
 layer  meanCos  pairs>0.9   effPR  rank90
     4    0.046       0.0%    80.9     76
    12    0.028       0.0%   109.0     91
    23    0.028       0.0%   109.7     90
    35    0.021       0.0%    95.4     82
    44    0.124       0.0%    93.1     78
```

- mean pairwise expert-output cosine : **0.049** (random≈0)
- mean effective #experts (PR)       : **97.6 / 128**
- mean rank for 90% energy           : **83 / 128**
- fraction of expert pairs >0.9 cos  : **0.0%**

**Verdict: NOT SUPPORTED**

## P3 — Top-k over-provisioning

Relative L2 deviation of the MoE output when keeping only k of the 8
experts (renormalized) vs the true top-8.

```
 layer     k=1     k=2     k=4     k=6
     4   1.989   1.156   0.515   0.228
    12   1.886   1.119   0.531   0.254
    23   1.937   1.182   0.578   0.279
    35   1.976   1.169   0.549   0.261
    44   2.040   1.203   0.535   0.235
```

- mean rel-L2 at k=1 : **1.966**
- mean rel-L2 at k=2 : **1.166**
- mean rel-L2 at k=4 : **0.542**
- mean rel-L2 at k=6 : **0.251**

**Verdict: NOT SUPPORTED — all 8 matter**

## P5 — Dead experts: domain-gated vs globally wasted

- globally never-selected experts : **1.30%**
- active in 0 domains (lazy tail) : **42.3%**
- active in exactly 1 domain      : **27.7%** (specialists)
- active in all 3 domains         : **10.1%** (generalists)

Globally-wasted capacity is tiny, but utilization is heavily skewed:
~28% of experts are single-domain specialists ⇒ pruning is
**domain-conditional**, not global.

**Verdict: SUPPORTED — capacity is skewed & domain-gated (not globally dead)**

## P2 — Lexical vs contextual routing

- same-token top-8 overlap : **47.1%**
- random-token overlap     : **12.3%**
- gap                      : **34.7 pp**

by token type:
  - punct/other: 45.0%
  - word: 51.3%
  - num: 38.6%

most lexically-fixed tokens (token, count, overlap):
```
           'ise'  n=28   98.7%
          'Prem'  n=28   93.2%
        'thesis'  n=28   83.7%
             '?'  n=24   70.6%
           'Hyp'  n=28   69.1%
      'function'  n=23   68.2%
             'o'  n=28   60.7%
             ':'  n=69   55.2%
             '.'  n=122  54.8%
             '-'  n=21   48.8%
              ''  n=107  43.8%
             '4'  n=21   42.0%
```

**Verdict: PARTIAL — token identity strongly biases routing**

## P4 — Attention-sink-style routing

- position-0 cross-example overlap : **83.3%**
- other-position overlap           : **13.1%**
- distinct first-tokens            : 24 over 80 examples

**Verdict: SUPPORTED — position 0 routes to a fixed expert set (sink)**

## P6 — Routing instability under trivial perturbation

perturbation: prepend 'Note: '

- mean top-8 sets changed : **12.1%**
- early layers : 8.5%   mid : 11.4%   late : 11.6%

**Verdict: NOT SUPPORTED — routing is stable**

## Summary

```
P1: NOT SUPPORTED
P2: PARTIAL — token identity strongly biases routing
P3: NOT SUPPORTED — all 8 matter
P4: SUPPORTED — position 0 routes to a fixed expert set (sink)
P5: SUPPORTED — skewed/domain-gated
P6: NOT SUPPORTED — routing is stable
```
