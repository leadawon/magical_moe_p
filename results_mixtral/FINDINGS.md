# MoE Routing Probe — Findings


Datasets analysed: gsm8k, humaneval_plus, mbpp_plus, mnli, snli, svamp


Model: Mixtral-8x7B-Instruct (32 MoE layers, 8 experts, top-2). Statistics from forward passes over raw domain text.


## H1 — Domain-conditional expert specialization

Cosine similarity of per-layer expert *selection-frequency* vectors,
averaged over all layers. Within-domain pairs vs cross-domain pairs.

```
               gsm8k   svamp humanev mbpp_pl    mnli    snli
       gsm8k   1.000   0.992   0.956   0.928   0.960   0.933
       svamp   0.992   1.000   0.947   0.917   0.949   0.927
humaneval_pl   0.956   0.947   1.000   0.921   0.958   0.915
   mbpp_plus   0.928   0.917   0.921   1.000   0.922   0.903
        mnli   0.960   0.949   0.958   0.922   1.000   0.963
        snli   0.933   0.927   0.915   0.903   0.963   1.000
```

- mean within-domain similarity : **0.959**
- mean cross-domain similarity  : **0.934**
- separation gap (within-cross) : **0.024**

**Verdict: PARTIAL — weak but present domain structure.  NOTE: very high cross-domain similarity (0.93) ⇒ large shared/redundant routing.**


## H2 — Load imbalance / dead & hot experts

Per-layer Gini of expert selection counts (0=uniform, 1=one expert),
fraction of (layer,expert) slots that are ~dead (<0.1×uniform) or
hot (>5×uniform). Uniform selection freq = 1/128 = 0.0078.

```
       dataset   gini   dead%   hot%
         gsm8k  0.132   0.00  0.00
         svamp  0.161   0.00  0.00
humaneval_plus  0.112   0.00  0.00
     mbpp_plus  0.211   0.00  0.00
          mnli  0.104   0.39  0.00
          snli  0.183   0.00  0.00
```

- mean per-layer Gini      : **0.150**
- mean dead-slot fraction  : **0.07%**
- mean hot-slot fraction   : **0.00%**

**Verdict: NOT SUPPORTED — load is fairly balanced.**


## H3 — Depth-dependent routing specialization

Per-layer cross-domain divergence = mean pairwise Jensen–Shannon
divergence between the three domain selection distributions.
Higher = the layer routes domains more differently.

```
layer     JS  profile (0..max)
    0  0.004  ##..........................
    1  0.006  ###.........................
    2  0.014  #######.....................
    3  0.008  ####........................
    4  0.011  #####.......................
    5  0.011  #####.......................
    6  0.023  ###########.................
    7  0.020  #########...................
    8  0.061  ############################
    9  0.026  ############................
   10  0.025  ############................
   11  0.009  ####........................
   12  0.039  ##################..........
   13  0.010  ####........................
   14  0.004  ##..........................
   15  0.010  #####.......................
   16  0.023  ##########..................
   17  0.019  #########...................
   18  0.010  #####.......................
   19  0.014  ######......................
   20  0.024  ###########.................
   21  0.031  ##############..............
   22  0.027  ############................
   23  0.040  ##################..........
   24  0.014  #######.....................
   25  0.007  ###.........................
   26  0.017  ########....................
   27  0.031  ##############..............
   28  0.018  ########....................
   29  0.012  #####.......................
   30  0.022  ##########..................
   31  0.006  ###.........................
```

- early-third mean JS : **0.018**
- late-third  mean JS : **0.021**
- peak layer          : **8**  (JS=0.061)
- corr(JS, depth)     : **+0.090**

**Verdict: PARTIAL/FLAT — no strong monotonic depth trend; see peak layer.**


## H4 — Router confidence / decision margin

Averaged over all tokens & datasets:

- mean top-1 softmax prob          : **0.473**
- mean cumulative top-8 prob mass  : **0.661**
- mean normalised entropy (0..1)   : **0.725**
- mean margin (p[8th] − p[9th])    : **0.0815**

Interpretation: a small 8th-vs-9th margin means the bottom of the
selected set is nearly tied with the top of the rejected set — the
top-k boundary is fragile and small perturbations flip experts.

**Verdict: SUPPORTED — routing is low-margin / high-entropy (fragile selection).**


## H5 — Cross-domain redundancy (universal core)

Per layer, an expert is 'active' for a dataset if its selection
frequency exceeds uniform (1/128). The 'universal core' = experts
active across ALL six datasets at that layer.

- mean universal-core size   : **6.6%** of experts
- mean routing mass on core  : **8.4%** of token routing

**Verdict: NOT SUPPORTED — domains use largely disjoint experts.**


## H6 — Domain classifiability from routing fingerprints

Each example → its [layers×experts] top-k selection-frequency vector.
Nearest-centroid (cosine) classifier, 50/50 train/test split.
Predicting **domain** (3 classes, chance=0.33).

- test accuracy : **0.972**  (chance 0.33)

Confusion (rows=true, cols=pred):
```
              code   math    nli
      code      49      2      1
      math       0     64      0
       nli       0      2     62
```

**Verdict: SUPPORTED — routing fingerprints strongly encode domain.**


## Summary table


```

H1: PARTIAL — weak but present domain structure.  NOTE: very high cross-domain similarity (0.93) ⇒ large shared/redundant routing.

H2: NOT SUPPORTED — load is fairly balanced.

H3: PARTIAL/FLAT — no strong monotonic depth trend; see peak layer.

H4: SUPPORTED — routing is low-margin / high-entropy (fragile selection).

H5: NOT SUPPORTED — domains use largely disjoint experts.

H6: SUPPORTED — routing fingerprints strongly encode domain.

```