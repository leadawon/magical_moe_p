#!/usr/bin/env python3
"""
Post-hoc analysis: load results/<dataset>_routing.npz and test the six routing
hypotheses (see HYPOTHESES.md). Produces results/FINDINGS.md, saves derived
arrays to results/derived/, and prints a summary.

No sklearn dependency — the small classifier (H6) is a nearest-centroid model
implemented in numpy.
"""

import os
import sys
import json
import glob
import numpy as np

BASE = "/data1/ai25170474/workspace/magical_moe_probe"
RES = os.environ.get("PROBE_RESULTS_DIR", os.path.join(BASE, "results"))
DERIVED = os.path.join(RES, "derived")
MODEL_DESC = os.environ.get("PROBE_MODEL_DESC", "Qwen3-30B-A3B (48 MoE layers, 128 experts, top-8)")

DOMAIN_OF = {
    "gsm8k": "math", "svamp": "math",
    "humaneval_plus": "code", "mbpp_plus": "code",
    "mnli": "nli", "snli": "nli",
}
ORDER = ["gsm8k", "svamp", "humaneval_plus", "mbpp_plus", "mnli", "snli"]
WITHIN_PAIRS = [("gsm8k", "svamp"), ("humaneval_plus", "mbpp_plus"), ("mnli", "snli")]
UNIFORM = 1.0 / 128.0     # set from data["E"] at runtime (see main)
NUM_EXPERTS = 128         # set from data at runtime


# ----------------------------- helpers --------------------------------------
def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() - (n + 1) * x.sum()) / (n * x.sum()))


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def js_div(p, q):
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / (b[mask] + 1e-12))))
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)) / np.log(2.0)   # in [0,1]


def load_all():
    data = {}
    for f in sorted(glob.glob(os.path.join(RES, "*_routing.npz"))):
        d = np.load(f, allow_pickle=True)
        name = str(d["dataset"])
        tokens = d["tokens"].astype(np.float64)            # [L]
        k = int(d["top_k"])
        sel = d["sel_counts"].astype(np.float64)           # [L,E]
        sel_freq = sel / (tokens[:, None] * k + 1e-12)     # rows ~sum to 1
        prob = d["prob_mass"].astype(np.float64) / (tokens[:, None] + 1e-12)
        data[name] = {
            "domain": str(d["domain"]),
            "L": int(d["num_layers"]), "E": int(d["num_experts"]), "k": k,
            "tokens": tokens,
            "sel_freq": sel_freq,
            "prob_dist": prob,
            "top1": d["top1_sum"].astype(np.float64) / (tokens + 1e-12),
            "topk": d["topk_sum"].astype(np.float64) / (tokens + 1e-12),
            "entropy": d["entropy_sum"].astype(np.float64) / (tokens + 1e-12),
            "margin": d["margin_sum"].astype(np.float64) / (tokens + 1e-12),
            "fingerprints": d["fingerprints"].astype(np.float32),
        }
    return data


def bar(frac, width=30):
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "#" * n + "." * (width - n)


# ----------------------------- hypotheses -----------------------------------
def h1_specialization(data, out):
    names = [n for n in ORDER if n in data]
    L = data[names[0]]["L"]
    sim = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            s = np.mean([cosine(data[a]["sel_freq"][l], data[b]["sel_freq"][l])
                         for l in range(L)])
            sim[i, j] = s
    within, cross = [], []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            (within if DOMAIN_OF[a] == DOMAIN_OF[b] else cross).append(sim[i, j])
    mw, mc = float(np.mean(within)), float(np.mean(cross))
    gap = mw - mc
    np.save(os.path.join(DERIVED, "h1_similarity_matrix.npy"), sim)

    lines = ["## H1 — Domain-conditional expert specialization", "",
             "Cosine similarity of per-layer expert *selection-frequency* vectors,",
             "averaged over all layers. Within-domain pairs vs cross-domain pairs.", "",
             "```",
             "             " + " ".join(f"{n[:7]:>7}" for n in names)]
    for i, a in enumerate(names):
        lines.append(f"{a[:12]:>12} " + " ".join(f"{sim[i,j]:7.3f}" for j in range(len(names))))
    lines += ["```", "",
              f"- mean within-domain similarity : **{mw:.3f}**",
              f"- mean cross-domain similarity  : **{mc:.3f}**",
              f"- separation gap (within-cross) : **{gap:.3f}**", ""]
    if gap > 0.05:
        verdict = "SUPPORTED — measurable domain structure in routing."
    elif gap > 0.02:
        verdict = "PARTIAL — weak but present domain structure."
    else:
        verdict = "NOT SUPPORTED — routing barely distinguishes domains."
    if mc > 0.85:
        verdict += f"  NOTE: very high cross-domain similarity ({mc:.2f}) ⇒ large shared/redundant routing."
    lines += [f"**Verdict: {verdict}**", ""]
    out["H1"] = {"within": mw, "cross": mc, "gap": gap, "verdict": verdict}
    return "\n".join(lines)


def h2_imbalance(data, out):
    names = [n for n in ORDER if n in data]
    L, E = data[names[0]]["L"], data[names[0]]["E"]
    ginis, dead_frac, hot_frac = [], [], []
    for n in names:
        sf = data[n]["sel_freq"]                          # [L,E]
        gl = np.mean([gini(sf[l]) for l in range(L)])
        dead = np.mean(sf < 0.1 * UNIFORM)
        hot = np.mean(sf > 5.0 * UNIFORM)
        ginis.append(gl); dead_frac.append(dead); hot_frac.append(hot)
    mg = float(np.mean(ginis))
    md = float(np.mean(dead_frac))
    mh = float(np.mean(hot_frac))
    lines = ["## H2 — Load imbalance / dead & hot experts", "",
             "Per-layer Gini of expert selection counts (0=uniform, 1=one expert),",
             "fraction of (layer,expert) slots that are ~dead (<0.1×uniform) or",
             "hot (>5×uniform). Uniform selection freq = 1/128 = 0.0078.", "",
             "```",
             f"{'dataset':>14} {'gini':>6} {'dead%':>7} {'hot%':>6}"]
    for i, n in enumerate(names):
        lines.append(f"{n:>14} {ginis[i]:6.3f} {dead_frac[i]*100:6.2f} {hot_frac[i]*100:5.2f}")
    lines += ["```", "",
              f"- mean per-layer Gini      : **{mg:.3f}**",
              f"- mean dead-slot fraction  : **{md*100:.2f}%**",
              f"- mean hot-slot fraction   : **{mh*100:.2f}%**", ""]
    if mg > 0.3 or md > 0.05:
        verdict = "SUPPORTED — routing load is imbalanced (dead/hot experts present)."
    elif mg > 0.18:
        verdict = "PARTIAL — moderate imbalance."
    else:
        verdict = "NOT SUPPORTED — load is fairly balanced."
    lines += [f"**Verdict: {verdict}**", ""]
    out["H2"] = {"gini": mg, "dead_frac": md, "hot_frac": mh, "verdict": verdict}
    return "\n".join(lines)


def h3_depth(data, out):
    names = [n for n in ORDER if n in data]
    L = data[names[0]]["L"]
    domains = {}
    for n in names:
        domains.setdefault(DOMAIN_OF[n], []).append(n)
    dom_keys = list(domains.keys())
    per_layer_js = []
    for l in range(L):
        dom_dist = {d: np.mean([data[n]["sel_freq"][l] for n in domains[d]], axis=0)
                    for d in dom_keys}
        pj = [js_div(dom_dist[a], dom_dist[b])
              for i, a in enumerate(dom_keys) for b in dom_keys[i+1:]]
        per_layer_js.append(float(np.mean(pj)) if pj else 0.0)
    per_layer_js = np.array(per_layer_js)
    np.save(os.path.join(DERIVED, "h3_layerwise_js.npy"), per_layer_js)
    depth = np.arange(L)
    corr = float(np.corrcoef(depth, per_layer_js)[0, 1]) if L > 2 else 0.0
    peak = int(np.argmax(per_layer_js))
    third = L // 3
    early = float(per_layer_js[:third].mean())
    late = float(per_layer_js[2*third:].mean())

    lines = ["## H3 — Depth-dependent routing specialization", "",
             "Per-layer cross-domain divergence = mean pairwise Jensen–Shannon",
             "divergence between the three domain selection distributions.",
             "Higher = the layer routes domains more differently.", "",
             "```", f"{'layer':>5} {'JS':>6}  profile (0..max)"]
    mx = per_layer_js.max() + 1e-12
    for l in range(L):
        lines.append(f"{l:>5} {per_layer_js[l]:6.3f}  {bar(per_layer_js[l]/mx, 28)}")
    lines += ["```", "",
              f"- early-third mean JS : **{early:.3f}**",
              f"- late-third  mean JS : **{late:.3f}**",
              f"- peak layer          : **{peak}**  (JS={per_layer_js[peak]:.3f})",
              f"- corr(JS, depth)     : **{corr:+.3f}**", ""]
    if late > early * 1.3:
        verdict = "SUPPORTED — domain specialization grows with depth."
    elif early > late * 1.3:
        verdict = "SUPPORTED (inverted) — specialization concentrates in early layers."
    else:
        verdict = "PARTIAL/FLAT — no strong monotonic depth trend; see peak layer."
    lines += [f"**Verdict: {verdict}**", ""]
    out["H3"] = {"early": early, "late": late, "peak": peak, "corr": corr, "verdict": verdict}
    return "\n".join(lines)


def h4_confidence(data, out):
    names = [n for n in ORDER if n in data]
    L = data[names[0]]["L"]
    logE = np.log(float(NUM_EXPERTS))
    t1 = np.mean([data[n]["top1"].mean() for n in names])
    tk = np.mean([data[n]["topk"].mean() for n in names])
    en = np.mean([data[n]["entropy"].mean() for n in names]) / logE
    mg = np.mean([data[n]["margin"].mean() for n in names])
    # per-layer averaged across datasets
    pl_top1 = np.mean([np.stack([data[n]["top1"] for n in names])], axis=1)[0]
    np.save(os.path.join(DERIVED, "h4_layer_top1.npy"), pl_top1)

    lines = ["## H4 — Router confidence / decision margin", "",
             "Averaged over all tokens & datasets:", "",
             f"- mean top-1 softmax prob          : **{t1:.3f}**",
             f"- mean cumulative top-8 prob mass  : **{tk:.3f}**",
             f"- mean normalised entropy (0..1)   : **{en:.3f}**",
             f"- mean margin (p[8th] − p[9th])    : **{mg:.4f}**", "",
             "Interpretation: a small 8th-vs-9th margin means the bottom of the",
             "selected set is nearly tied with the top of the rejected set — the",
             "top-k boundary is fragile and small perturbations flip experts.", ""]
    if mg < 0.01 or en > 0.6:
        verdict = "SUPPORTED — routing is low-margin / high-entropy (fragile selection)."
    elif mg < 0.02:
        verdict = "PARTIAL — moderately low margin."
    else:
        verdict = "NOT SUPPORTED — router is decisive."
    lines += [f"**Verdict: {verdict}**", ""]
    out["H4"] = {"top1": float(t1), "topk_mass": float(tk),
                 "norm_entropy": float(en), "margin": float(mg), "verdict": verdict}
    return "\n".join(lines)


def h5_redundancy(data, out):
    names = [n for n in ORDER if n in data]
    L, E = data[names[0]]["L"], data[names[0]]["E"]
    core_frac, core_mass = [], []
    for l in range(L):
        active = []
        for n in names:
            active.append(data[n]["sel_freq"][l] > UNIFORM)   # above-uniform use
        core = np.logical_and.reduce(active)                  # active in ALL datasets
        core_frac.append(core.mean())
        # mass the core carries, averaged across datasets
        cm = np.mean([data[n]["sel_freq"][l][core].sum() for n in names])
        core_mass.append(cm)
    mcf = float(np.mean(core_frac))
    mcm = float(np.mean(core_mass))
    np.save(os.path.join(DERIVED, "h5_core_fraction.npy"), np.array(core_frac))

    lines = ["## H5 — Cross-domain redundancy (universal core)", "",
             "Per layer, an expert is 'active' for a dataset if its selection",
             "frequency exceeds uniform (1/128). The 'universal core' = experts",
             "active across ALL six datasets at that layer.", "",
             f"- mean universal-core size   : **{mcf*100:.1f}%** of experts",
             f"- mean routing mass on core  : **{mcm*100:.1f}%** of token routing", ""]
    if mcf > 0.30:
        verdict = "SUPPORTED — a large expert core is shared by every domain (redundancy)."
    elif mcf > 0.15:
        verdict = "PARTIAL — moderate shared core."
    else:
        verdict = "NOT SUPPORTED — domains use largely disjoint experts."
    lines += [f"**Verdict: {verdict}**", ""]
    out["H5"] = {"core_frac": mcf, "core_mass": mcm, "verdict": verdict}
    return "\n".join(lines)


def h6_classify(data, out):
    names = [n for n in ORDER if n in data]
    # build fingerprint dataset: X [Ntot, L*E], domain labels
    X, ydom = [], []
    for n in names:
        fp = data[n]["fingerprints"]            # [N,L,E]
        if fp.shape[0] == 0:
            continue
        X.append(fp.reshape(fp.shape[0], -1))
        ydom += [DOMAIN_OF[n]] * fp.shape[0]
    if not X:
        out["H6"] = {"verdict": "NO DATA"}
        return "## H6 — Domain classifiability from fingerprints\n\nNo fingerprint data.\n"
    X = np.concatenate(X, axis=0).astype(np.float64)
    ydom = np.array(ydom)
    # L2-normalise rows for cosine nearest-centroid
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(Xn))
    split = len(perm) // 2
    tr, te = perm[:split], perm[split:]
    classes = sorted(set(ydom))
    cents = {}
    for c in classes:
        mask = (ydom[tr] == c)
        cents[c] = Xn[tr][mask].mean(axis=0)
        cents[c] /= (np.linalg.norm(cents[c]) + 1e-12)
    # predict
    correct = 0
    conf = {c: {c2: 0 for c2 in classes} for c in classes}
    for i in te:
        sims = {c: float(np.dot(Xn[i], cents[c])) for c in classes}
        pred = max(sims, key=sims.get)
        conf[ydom[i]][pred] += 1
        if pred == ydom[i]:
            correct += 1
    acc = correct / len(te)
    chance = 1.0 / len(classes)

    lines = ["## H6 — Domain classifiability from routing fingerprints", "",
             "Each example → its [layers×experts] top-k selection-frequency vector.",
             "Nearest-centroid (cosine) classifier, 50/50 train/test split.",
             f"Predicting **domain** ({len(classes)} classes, chance={chance:.2f}).", "",
             f"- test accuracy : **{acc:.3f}**  (chance {chance:.2f})", "",
             "Confusion (rows=true, cols=pred):", "```",
             "            " + " ".join(f"{c[:6]:>6}" for c in classes)]
    for c in classes:
        row = conf[c]
        lines.append(f"{c:>10}  " + " ".join(f"{row[c2]:6d}" for c2 in classes))
    lines += ["```", ""]
    if acc > 0.6:
        verdict = "SUPPORTED — routing fingerprints strongly encode domain."
    elif acc > chance + 0.1:
        verdict = "PARTIAL — fingerprints carry some domain signal."
    else:
        verdict = "NOT SUPPORTED — routing is near domain-agnostic."
    lines += [f"**Verdict: {verdict}**", ""]
    out["H6"] = {"accuracy": acc, "chance": chance, "verdict": verdict}
    return "\n".join(lines)


def main():
    global UNIFORM, NUM_EXPERTS
    os.makedirs(DERIVED, exist_ok=True)
    data = load_all()
    if not data:
        print("No *_routing.npz found in results/. Run src/probe.py first.")
        sys.exit(1)
    # adapt expert-count-dependent constants to the actual model
    NUM_EXPERTS = int(next(iter(data.values()))["E"])
    UNIFORM = 1.0 / NUM_EXPERTS
    print(f"Loaded {len(data)} datasets: {list(data.keys())}  (num_experts={NUM_EXPERTS})")

    out = {}
    sections = [
        "# MoE Routing Probe — Findings\n",
        f"Datasets analysed: {', '.join(data.keys())}\n",
        f"Model: {MODEL_DESC}. "
        "Statistics from forward passes over raw domain text.\n",
        h1_specialization(data, out),
        h2_imbalance(data, out),
        h3_depth(data, out),
        h4_confidence(data, out),
        h5_redundancy(data, out),
        h6_classify(data, out),
        "## Summary table\n",
        "```",
    ]
    for h in ["H1", "H2", "H3", "H4", "H5", "H6"]:
        if h in out:
            sections.append(f"{h}: {out[h]['verdict']}")
    sections.append("```")

    report = "\n\n".join(sections)
    with open(os.path.join(RES, "FINDINGS.md"), "w") as f:
        f.write(report)
    with open(os.path.join(RES, "findings_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n".join(f"{h}: {out[h]['verdict']}" for h in out))
    print(f"\nWrote {os.path.join(RES, 'FINDINGS.md')}")


if __name__ == "__main__":
    main()
