#!/usr/bin/env python3
"""
P5 (dead experts: domain-gated vs globally wasted) + E2 (domain-conditional
prunability headroom). Both are derived purely from the Round-1 routing dumps
(results/*_routing.npz) — no GPU, no model load.

P5 -> results_r2/p5_dead.npz  (breadth[L,E], dead_global[L,E])
E2 -> results_r2/e2_prunability.json  (per-domain keep counts / prune % at 90/95/99%
       coverage + cross-domain keep-set overlap)

Model-agnostic: reads num_experts/top_k from the npz files, so it works for both
Qwen (128/8) and Mixtral (8/2).

Env:
  PROBE_RESULTS_DIR  (default results/)   — where *_routing.npz live
  PROBE_OUT_DIR      (default results_r2/) — where to write outputs
"""
import os, sys, glob, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "/data1/ai25170474/workspace/magical_moe_probe"
RES = os.environ.get("PROBE_RESULTS_DIR", f"{BASE}/results")
OUT = os.environ.get("PROBE_OUT_DIR", f"{BASE}/results_r2")

DOMAIN_OF = {"gsm8k": "math", "svamp": "math", "humaneval_plus": "code",
             "mbpp_plus": "code", "mnli": "nli", "snli": "nli"}
DOMAINS = ["math", "code", "nli"]


def load_domain_freq():
    """Return dom -> sel_freq[L,E] (selection frequency, rows ~sum to 1) and
    dom -> prob_mass[L,E] (mean routing prob), aggregated over the domain's datasets."""
    sel, prob, meta = {}, {}, {}
    for f in sorted(glob.glob(f"{RES}/*_routing.npz")):
        d = np.load(f, allow_pickle=True)
        name = str(d["dataset"]); dom = DOMAIN_OF.get(name, str(d["domain"]))
        tokens = d["tokens"].astype(np.float64)
        k = int(d["top_k"])
        sf = d["sel_counts"].astype(np.float64) / (tokens[:, None] * k + 1e-12)
        pm = d["prob_mass"].astype(np.float64) / (tokens[:, None] + 1e-12)
        sel.setdefault(dom, []).append(sf)
        prob.setdefault(dom, []).append(pm)
        meta["E"] = int(d["num_experts"]); meta["L"] = int(d["num_layers"]); meta["k"] = k
    sel = {dom: np.mean(v, 0) for dom, v in sel.items()}
    prob = {dom: np.mean(v, 0) for dom, v in prob.items()}
    return sel, prob, meta


def main():
    os.makedirs(OUT, exist_ok=True)
    sel, prob, meta = load_domain_freq()
    L, E, k = meta["L"], meta["E"], meta["k"]
    uniform = 1.0 / E
    doms = [d for d in DOMAINS if d in sel]
    print(f"[setup] L={L} E={E} k={k} domains={doms}")

    # ---------------- P5: breadth & global dead ----------------
    # "active in a domain" = selection freq above uniform in that domain
    active = {d: (sel[d] > uniform) for d in doms}            # [L,E] bool per domain
    breadth = np.zeros((L, E), dtype=np.int64)               # in how many domains active
    for d in doms:
        breadth += active[d].astype(np.int64)
    # globally dead = never selected at all in any domain
    dead_global = np.ones((L, E), dtype=bool)
    for d in doms:
        dead_global &= (sel[d] <= 0)
    np.savez_compressed(f"{OUT}/p5_dead.npz", breadth=breadth,
                        dead_global=dead_global)
    b = {x: float((breadth == x).mean()) for x in range(len(doms) + 1)}
    print(f"[P5] global dead {dead_global.mean()*100:.2f}% | "
          f"breadth0 {b.get(0,0)*100:.1f}% | single-domain {b.get(1,0)*100:.1f}% | "
          f"all-{len(doms)} {b.get(len(doms),0)*100:.1f}%")

    # ---------------- E2: domain-conditional prunability ----------------
    def keep_count(pm_row, coverage):
        tot = pm_row.sum()
        if tot <= 0:
            return len(pm_row)
        order = np.argsort(pm_row)[::-1]
        cum = np.cumsum(pm_row[order]) / tot
        return int(np.searchsorted(cum, coverage) + 1)

    def keep_mask(pm, coverage):
        m = np.zeros((L, E), dtype=bool)
        for l in range(L):
            n = max(1, min(E, keep_count(pm[l], coverage)))
            order = np.argsort(pm[l])[::-1]
            m[l, order[:n]] = True
        return m

    # Two coverage bases (both reported — they answer different questions):
    #   prob_mass : mean softmax prob over ALL experts (E3's keep-set basis; the
    #               real, accuracy-validated prune number — conservative).
    #   sel_freq  : how often an expert is actually in top-k (mass concentrated on
    #               the few chosen → aggressive upper bound).
    bases = {"prob": prob, "sel": sel}
    e2 = {}
    for bname, src in bases.items():
        suffix = "" if bname == "prob" else "_selbasis"
        for cov in (0.90, 0.95, 0.99):
            tag = f"cov{int(cov*100)}"
            for d in doms:
                keep = sum(keep_count(src[d][l], cov) for l in range(L)) / L
                e2[f"{d}_{tag}_keep{suffix}"] = float(keep)
                e2[f"{d}_{tag}_prune_pct{suffix}"] = float((E - keep) / E * 100)
        # union keep-set across all domains at 95% (serve-all scenario)
        union95 = np.zeros((L, E), dtype=bool)
        for d in doms:
            union95 |= keep_mask(src[d], 0.95)
        e2[f"all_cov95_keep{suffix}"] = float(union95.sum(1).mean())
        e2[f"all_cov95_prune_pct{suffix}"] = float((E - union95.sum(1).mean()) / E * 100)
        # cross-domain keep-set overlap (Jaccard) at 95%
        masks95 = {d: keep_mask(src[d], 0.95) for d in doms}
        for i in range(len(doms)):
            for j in range(i + 1, len(doms)):
                a, bb = masks95[doms[i]], masks95[doms[j]]
                inter = np.logical_and(a, bb).sum(); uni = np.logical_or(a, bb).sum()
                e2[f"overlap95_{doms[i]}_{doms[j]}{suffix}"] = float(inter / max(1, uni))

    json.dump(e2, open(f"{OUT}/e2_prunability.json", "w"), indent=2)
    print("[E2 prob-mass basis, E3-consistent] 95% prune%:",
          {d: round(e2[f"{d}_cov95_prune_pct"], 1) for d in doms},
          "| all", round(e2["all_cov95_prune_pct"], 1))
    print("[E2 sel-freq basis, upper bound]      95% prune%:",
          {d: round(e2[f"{d}_cov95_prune_pct_selbasis"], 1) for d in doms},
          "| all", round(e2["all_cov95_prune_pct_selbasis"], 1))
    print(f"saved {OUT}/p5_dead.npz and {OUT}/e2_prunability.json")


if __name__ == "__main__":
    main()
