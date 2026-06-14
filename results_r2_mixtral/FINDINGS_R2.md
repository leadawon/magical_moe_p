# MoE Problem Probes — Round 2 Findings (P1–P6)

Model: Mixtral-8x7B-Instruct (32 MoE layers, 8 experts, top-2).

## P1 — Expert functional redundancy

All 128 experts run on the same live hidden states; we measure how
similar their outputs are and the effective dimensionality of the bank.

```
 layer  meanCos  pairs>0.9   effPR  rank90
     2    0.687       0.0%     6.9      7
     8    0.549       0.0%     7.0      7
    15    0.376       0.0%     6.9      6
    23    0.623       0.0%     7.0      7
    29    0.907      75.0%     6.8      6
```

- mean pairwise expert-output cosine : **0.628** (random≈0)
- mean effective #experts (PR)       : **6.9 / 128**
- mean rank for 90% energy           : **7 / 128**
- fraction of expert pairs >0.9 cos  : **15.0%**

**Verdict: SUPPORTED — substantial redundancy**

## P3 — Top-k over-provisioning

Relative L2 deviation of the MoE output when keeping only k of the 8
experts (renormalized) vs the true top-8.

```
 layer     k=1     k=2     k=4     k=6
     2   0.767   0.411   0.176   0.069
     8   0.823   0.457   0.187   0.075
    15   0.875   0.465   0.189   0.075
    23   0.653   0.301   0.114   0.041
    29   0.500   0.239   0.088   0.031
```

- mean rel-L2 at k=1 : **0.724**
- mean rel-L2 at k=2 : **0.375**
- mean rel-L2 at k=4 : **0.151**
- mean rel-L2 at k=6 : **0.058**

**Verdict: NOT SUPPORTED — all 8 matter**

## P5 — Dead experts: domain-gated vs globally wasted

- globally never-selected experts : **0.00%**
- active in 0 domains (lazy tail) : **23.0%**
- active in exactly 1 domain      : **29.3%** (specialists)
- active in all 3 domains         : **18.0%** (generalists)

Globally-wasted capacity is tiny, but utilization is heavily skewed:
~28% of experts are single-domain specialists ⇒ pruning is
**domain-conditional**, not global.

**Verdict: SUPPORTED — capacity is skewed & domain-gated (not globally dead)**

## P2 — Lexical vs contextual routing

- same-token top-8 overlap : **59.4%**
- random-token overlap     : **25.5%**
- gap                      : **33.9 pp**

by token type:
  - punct/other: 60.2%
  - word: 62.6%
  - num: 47.5%

most lexically-fixed tokens (token, count, overlap):
```
          'Prem'  n=28   100.0%
           'ise'  n=28   99.4%
           '<s>'  n=80   99.0%
         'othes'  n=28   85.8%
            'yp'  n=28   84.6%
            'is'  n=29   84.4%
             '?'  n=24   80.2%
      'function'  n=22   73.4%
             'H'  n=30   71.8%
             '.'  n=146  69.7%
             ':'  n=72   64.9%
           '"""'  n=20   63.6%
```

**Verdict: PARTIAL — token identity strongly biases routing**

## P4 — Attention-sink-style routing

- position-0 cross-example overlap : **99.0%**
- other-position overlap           : **25.2%**
- distinct first-tokens            : 1 over 80 examples

**Verdict: SUPPORTED — position 0 routes to a fixed expert set (sink)**

## P6 — Routing instability under trivial perturbation

perturbation: prepend 'Note: '

- mean top-8 sets changed : **8.0%**
- early layers : 7.1%   mid : 7.5%   late : 7.3%

**Verdict: NOT SUPPORTED — routing is stable**

## Summary

```
P1: SUPPORTED — substantial redundancy
P2: PARTIAL — token identity strongly biases routing
P3: NOT SUPPORTED — all 8 matter
P4: SUPPORTED — position 0 routes to a fixed expert set (sink)
P5: SUPPORTED — skewed/domain-gated
P6: NOT SUPPORTED — routing is stable
```
