#!/usr/bin/env python3
"""
E1 — Sink-routing anatomy (deepens P4). Forward-only, gate hooks only
(offload-safe). Characterizes the position-0 / early-token routing sink:

  1) per-position cross-example top-8 overlap (how sink-like each early position is)
  2) the position-0 "sink expert set" per layer + its cross-domain consistency
  3) fraction of all routing that goes to the sink experts
  4) how many early positions are sink-like

Run: CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.probe_sink
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
BASE = "/data1/ai25170474/workspace/magical_moe_probe"
MODEL_PATH = os.environ.get("PROBE_MODEL_PATH", f"{BASE}/model")
OUT_DIR = os.environ.get("PROBE_OUT_DIR", f"{BASE}/results_r2")
N_PER_DS = 30
MAXPOS = 8          # study positions 0..7 plus a deep-position baseline
K = 8               # set to the model's top_k at runtime


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.data_loaders import load, ALL_DATASETS, DOMAIN_OF
    from src.gpu_utils import release

    from src.moe_adapter import detect_moe
    global K
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    dev = next(model.model.embed_tokens.parameters()).device
    info = detect_moe(model)
    K = info.top_k
    nE = info.num_experts
    cap = {}
    for idx, gate in enumerate(info.gates):
        def mk(i):
            def hook(m, inp, out): cap[i] = out.detach()
            return hook
        gate.register_forward_hook(mk(idx))
    nL = info.num_layers
    print(f"[setup] {nL} gate layers, top_k(K)={K}, num_experts={nE}")

    # per position: list of [nL,K] arrays (one per example occurrence); per domain for pos0
    pos_top = {p: [] for p in range(MAXPOS)}
    deep_top = []                                   # positions >= 20
    pos0_by_dom = {}                                # domain -> list of [nL,K]
    sink_counts = np.zeros((nL, nE), dtype=np.float64)   # pos0 selection counts
    all_counts = np.zeros((nL, nE), dtype=np.float64)    # all-token selection counts

    for ds in ALL_DATASETS:
        dom = DOMAIN_OF[ds]
        for ex in load(ds, N_PER_DS):
            enc = tok(ex["text"], return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                cap.clear()
                model.model(input_ids=enc["input_ids"].to(dev),
                            attention_mask=enc["attention_mask"].to(dev), use_cache=False)
            T = enc["input_ids"].shape[1]
            top = np.empty((T, nL, K), dtype=np.int16)
            for l in range(nL):
                probs = torch.softmax(cap[l].float(), dim=1)
                top[:, l, :] = torch.topk(probs, K, 1).indices.cpu().numpy()
                idx = top[:, l, :]
                for t in range(T):
                    all_counts[l, idx[t]] += 1
                sink_counts[l, top[0, l, :]] += 1
            for p in range(min(MAXPOS, T)):
                pos_top[p].append(top[p])
            for t in range(20, T):
                deep_top.append(top[t])
            pos0_by_dom.setdefault(dom, []).append(top[0])

    def pair_overlap(arrs, n=1500, rng=None):
        rng = rng or np.random.RandomState(0)
        if len(arrs) < 2: return float("nan")
        s = 0.0
        for _ in range(n):
            a = arrs[rng.randint(len(arrs))]; b = arrs[rng.randint(len(arrs))]
            s += np.mean([len(set(a[l]) & set(b[l])) / K for l in range(a.shape[0])])
        return s / n

    rng = np.random.RandomState(0)
    out = {"per_position_overlap": {}, }
    print("\n=== per-position cross-example top-8 overlap ===")
    for p in range(MAXPOS):
        ov = pair_overlap(pos_top[p], rng=rng)
        out["per_position_overlap"][p] = ov
        print(f"  position {p}: {ov*100:5.1f}%   (n={len(pos_top[p])})")
    deep_ov = pair_overlap(deep_top, rng=rng)
    out["deep_position_overlap"] = deep_ov
    print(f"  position>=20 (baseline): {deep_ov*100:5.1f}%")

    # pos0 sink set per layer: experts present in >=50% of examples' pos0 top-8
    n_ex = len(pos_top[0])
    prev = sink_counts / max(1, n_ex)
    sink_set_sizes = (prev >= 0.5).sum(1)
    out["pos0_sink_set_size_mean"] = float(sink_set_sizes.mean())
    print(f"\n=== pos-0 sink set (experts in >=50% of examples' top-8) ===")
    print(f"  mean size: {sink_set_sizes.mean():.1f} experts/layer")

    # cross-domain consistency of pos0 sink set
    dom_sets = {}
    for dom, arrs in pos0_by_dom.items():
        c = np.zeros((nL, nE))
        for a in arrs:
            for l in range(nL): c[l, a[l]] += 1
        dom_sets[dom] = (c / max(1, len(arrs))) >= 0.5
    doms = list(dom_sets)
    print("  cross-domain overlap of pos-0 sink set:")
    for i in range(len(doms)):
        for j in range(i + 1, len(doms)):
            a, b = dom_sets[doms[i]], dom_sets[doms[j]]
            inter = np.logical_and(a, b).sum(); uni = np.logical_or(a, b).sum()
            jac = inter / max(1, uni)
            print(f"    {doms[i]} ∩ {doms[j]}: Jaccard {jac*100:.1f}%")
            out[f"pos0_sink_jaccard_{doms[i]}_{doms[j]}"] = float(jac)

    # fraction of all routing mass captured by the pos0 sink experts
    sink_mask = prev >= 0.5
    mass_in_sink = np.mean([all_counts[l][sink_mask[l]].sum() / (all_counts[l].sum() + 1e-9)
                            for l in range(nL)])
    out["routing_mass_in_pos0_sink"] = float(mass_in_sink)
    print(f"\n  routing mass (all tokens) captured by pos-0 sink experts: {mass_in_sink*100:.1f}%")
    n_sinklike = sum(1 for p in range(MAXPOS)
                     if out["per_position_overlap"][p] > deep_ov + 0.2)
    out["n_sinklike_positions"] = n_sinklike
    print(f"  # early positions that are sink-like (overlap > deep+0.2): {n_sinklike}")

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(out, open(f"{OUT_DIR}/e1_sink.json", "w"), indent=2)
    print(f"saved {OUT_DIR}/e1_sink.json")
    release(model, tag="probe_sink")


if __name__ == "__main__":
    main()
