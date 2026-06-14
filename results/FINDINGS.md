# MoE Routing Probe — Findings


Datasets analysed: gsm8k, humaneval_plus, mbpp_plus, mnli, snli, svamp


Model: Qwen3-30B-A3B (48 MoE layers, 128 experts, top-8). Statistics from forward passes over raw domain text.


## H1 — Domain-conditional expert specialization

Cosine similarity of per-layer expert *selection-frequency* vectors,
averaged over all layers. Within-domain pairs vs cross-domain pairs.

```
               gsm8k   svamp humanev mbpp_pl    mnli    snli
       gsm8k   1.000   0.982   0.370   0.423   0.550   0.576
       svamp   0.982   1.000   0.390   0.438   0.539   0.573
humaneval_pl   0.370   0.390   1.000   0.734   0.261   0.254
   mbpp_plus   0.423   0.438   0.734   1.000   0.351   0.348
        mnli   0.550   0.539   0.261   0.351   1.000   0.807
        snli   0.576   0.573   0.254   0.348   0.807   1.000
```

- mean within-domain similarity : **0.841**
- mean cross-domain similarity  : **0.423**
- separation gap (within-cross) : **0.418**

**Verdict: SUPPORTED — measurable domain structure in routing.**


## H2 — Load imbalance / dead & hot experts

Per-layer Gini of expert selection counts (0=uniform, 1=one expert),
fraction of (layer,expert) slots that are ~dead (<0.1×uniform) or
hot (>5×uniform). Uniform selection freq = 1/128 = 0.0078.

```
       dataset   gini   dead%   hot%
         gsm8k  0.676  32.11  3.03
         svamp  0.687  33.90  3.17
humaneval_plus  0.592  19.82  2.38
     mbpp_plus  0.698  32.98  3.96
          mnli  0.637  32.15  2.02
          snli  0.728  43.42  3.86
```

- mean per-layer Gini      : **0.670**
- mean dead-slot fraction  : **32.40%**
- mean hot-slot fraction   : **3.07%**

**Verdict: SUPPORTED — routing load is imbalanced (dead/hot experts present).**


## H3 — Depth-dependent routing specialization

Per-layer cross-domain divergence = mean pairwise Jensen–Shannon
divergence between the three domain selection distributions.
Higher = the layer routes domains more differently.

```
layer     JS  profile (0..max)
    0  0.122  #######.....................
    1  0.168  ##########..................
    2  0.223  #############...............
    3  0.242  ##############..............
    4  0.273  ################............
    5  0.306  ##################..........
    6  0.448  ##########################..
    7  0.469  ###########################.
    8  0.348  ####################........
    9  0.379  ######################......
   10  0.329  ###################.........
   11  0.322  ###################.........
   12  0.370  #####################.......
   13  0.275  ################............
   14  0.386  ######################......
   15  0.427  ########################....
   16  0.385  ######################......
   17  0.330  ###################.........
   18  0.467  ###########################.
   19  0.488  ############################
   20  0.343  ####################........
   21  0.375  ######################......
   22  0.304  #################...........
   23  0.263  ###############.............
   24  0.352  ####################........
   25  0.262  ###############.............
   26  0.378  ######################......
   27  0.426  ########################....
   28  0.400  #######################.....
   29  0.324  ###################.........
   30  0.470  ###########################.
   31  0.476  ###########################.
   32  0.345  ####################........
   33  0.383  ######################......
   34  0.301  #################...........
   35  0.282  ################............
   36  0.339  ###################.........
   37  0.248  ##############..............
   38  0.338  ###################.........
   39  0.385  ######################......
   40  0.351  ####################........
   41  0.308  ##################..........
   42  0.313  ##################..........
   43  0.352  ####################........
   44  0.366  #####################.......
   45  0.340  ####################........
   46  0.277  ################............
   47  0.194  ###########.................
```

- early-third mean JS : **0.318**
- late-third  mean JS : **0.320**
- peak layer          : **19**  (JS=0.488)
- corr(JS, depth)     : **+0.090**

**Verdict: PARTIAL/FLAT — no strong monotonic depth trend; see peak layer.**


## H4 — Router confidence / decision margin

Averaged over all tokens & datasets:

- mean top-1 softmax prob          : **0.112**
- mean cumulative top-8 prob mass  : **0.388**
- mean normalised entropy (0..1)   : **0.840**
- mean margin (p[8th] − p[9th])    : **0.0023**

Interpretation: a small 8th-vs-9th margin means the bottom of the
selected set is nearly tied with the top of the rejected set — the
top-k boundary is fragile and small perturbations flip experts.

**Verdict: SUPPORTED — routing is low-margin / high-entropy (fragile selection).**


## H5 — Cross-domain redundancy (universal core)

Per layer, an expert is 'active' for a dataset if its selection
frequency exceeds uniform (1/128). The 'universal core' = experts
active across ALL six datasets at that layer.

- mean universal-core size   : **4.3%** of experts
- mean routing mass on core  : **12.0%** of token routing

**Verdict: NOT SUPPORTED — domains use largely disjoint experts.**


## H6 — Domain classifiability from routing fingerprints

Each example → its [layers×experts] top-k selection-frequency vector.
Nearest-centroid (cosine) classifier, 50/50 train/test split.
Predicting **domain** (3 classes, chance=0.33).

- test accuracy : **0.998**  (chance 0.33)

Confusion (rows=true, cols=pred):
```
              code   math    nli
      code     167      0      0
      math       0    197      0
       nli       0      1    217
```

**Verdict: SUPPORTED — routing fingerprints strongly encode domain.**


## Summary table


```

H1: SUPPORTED — measurable domain structure in routing.

H2: SUPPORTED — routing load is imbalanced (dead/hot experts present).

H3: PARTIAL/FLAT — no strong monotonic depth trend; see peak layer.

H4: SUPPORTED — routing is low-margin / high-entropy (fragile selection).

H5: NOT SUPPORTED — domains use largely disjoint experts.

H6: SUPPORTED — routing fingerprints strongly encode domain.

```