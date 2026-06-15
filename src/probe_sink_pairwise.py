#!/usr/bin/env python3
"""
E1b — Per-DATASET pairwise pos-0 sink Jaccard matrix (extends E1/probe_sink.py).

E1 only reported domain-level Jaccard (math/code/nli, 3 pairs). This computes the
full 6x6 dataset-level matrix so we can show that even *within* a domain
(gsm8k vs svamp) and *across* domains (gsm8k vs mnli) the pos-0 sink expert set
overlaps far above the random baseline — strengthening the "position, not token"
argument against the P2 (lexical) rebuttal.

Same method as probe_sink.py:
  - forward-only, gate forward-hooks (offload-safe)
  - per layer, pos-0 sink set = experts in >=50% of a dataset's pos-0 top-8
  - Jaccard between two datasets' sink sets, averaged over layers
Model-agnostic via src.moe_adapter, so it runs for Qwen and Mixtral.

Run (Qwen):    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.probe_sink_pairwise
Run (Mixtral): PROBE_MODEL_PATH=/data1/ai25170474/models/Mixtral-8x7B-Instruct-v0.1 \
               PROBE_OUT_DIR=.../results_r2_mixtral PROBE_TAG=mixtral \
               CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.probe_sink_pairwise
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
BASE = "/data1/ai25170474/workspace/magical_moe_probe"
MODEL_PATH = os.environ.get("PROBE_MODEL_PATH", f"{BASE}/model")
OUT_DIR = os.environ.get("PROBE_OUT_DIR", f"{BASE}/results_r2")
TAG = os.environ.get("PROBE_TAG", "qwen")
N_PER_DS = int(os.environ.get("PROBE_N_PER_DS", "30"))
MAX_LEN = 128


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.data_loaders import load, ALL_DATASETS, DOMAIN_OF
    from src.gpu_utils import release
    from src.moe_adapter import detect_moe

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    dev = next(model.model.embed_tokens.parameters()).device
    info = detect_moe(model)
    K, nE, nL = info.top_k, info.num_experts, info.num_layers
    cap = {}
    for idx, gate in enumerate(info.gates):
        def mk(i):
            def hook(m, inp, out): cap[i] = out.detach()
            return hook
        gate.register_forward_hook(mk(idx))
    print(f"[setup] tag={TAG} {nL} layers, top_k={K}, num_experts={nE}, N/ds={N_PER_DS}")

    # collect each example's pos-0 top-k expert ids, grouped by dataset.
    # Also collect DEEP-position (>=20) top-k sets for the proper random baseline
    # (matches E1: the baseline is deep tokens, NOT other pos-0 tokens).
    pos0_by_ds = {ds: [] for ds in ALL_DATASETS}   # ds -> list of [nL,K] int arrays
    deep_rows = []                                 # list of [nL,K] for positions >=20
    for ds in ALL_DATASETS:
        for ex in load(ds, N_PER_DS):
            enc = tok(ex["text"], return_tensors="pt", truncation=True, max_length=MAX_LEN)
            with torch.no_grad():
                cap.clear()
                model.model(input_ids=enc["input_ids"].to(dev),
                            attention_mask=enc["attention_mask"].to(dev), use_cache=False)
            T = enc["input_ids"].shape[1]
            top = np.empty((T, nL, K), dtype=np.int16)
            for l in range(nL):
                probs = torch.softmax(cap[l].float(), dim=1)
                top[:, l, :] = torch.topk(probs, K, 1).indices.cpu().numpy()
            pos0_by_ds[ds].append(top[0])          # pos-0 only
            for t in range(20, T):
                deep_rows.append(top[t])
        print(f"  collected {ds}: {len(pos0_by_ds[ds])} examples")

    # per-dataset sink set: expert present in >=50% of that dataset's pos-0 top-k
    sink_set = {}   # ds -> bool [nL,nE]
    set_size = {}
    for ds in ALL_DATASETS:
        c = np.zeros((nL, nE))
        for a in pos0_by_ds[ds]:
            for l in range(nL):
                c[l, a[l]] += 1
        mask = (c / max(1, len(pos0_by_ds[ds]))) >= 0.5
        sink_set[ds] = mask
        set_size[ds] = float(mask.sum(1).mean())

    def jaccard(a, b):
        inter = np.logical_and(a, b).sum(1).astype(float)
        uni = np.logical_or(a, b).sum(1).astype(float)
        per_layer = np.divide(inter, uni, out=np.zeros_like(inter), where=uni > 0)
        return float(per_layer.mean())

    # full 6x6 matrix
    M = {}
    for ds1 in ALL_DATASETS:
        M[ds1] = {}
        for ds2 in ALL_DATASETS:
            M[ds1][ds2] = jaccard(sink_set[ds1], sink_set[ds2])

    # random baseline (matches E1): pair up random DEEP-position (>=20) top-k sets,
    # overlap/K. Deep tokens are NOT sinks, so this is the true ~chance floor.
    rng = np.random.RandomState(0)
    s = 0.0
    for _ in range(2000):
        a = deep_rows[rng.randint(len(deep_rows))]; b = deep_rows[rng.randint(len(deep_rows))]
        s += np.mean([len(set(a[l]) & set(b[l])) / K for l in range(nL)])
    rand_overlap = s / 2000

    # domain-level rollup (for cross-check vs E1)
    dom_of = DOMAIN_OF
    doms = sorted(set(dom_of.values()))
    dom_set = {}
    for d in doms:
        c = np.zeros((nL, nE))
        n = 0
        for ds in ALL_DATASETS:
            if dom_of[ds] == d:
                for a in pos0_by_ds[ds]:
                    n += 1
                    for l in range(nL):
                        c[l, a[l]] += 1
        dom_set[d] = (c / max(1, n)) >= 0.5
    dom_jac = {f"{doms[i]}_{doms[j]}": jaccard(dom_set[doms[i]], dom_set[doms[j]])
               for i in range(len(doms)) for j in range(i + 1, len(doms))}

    out = {"tag": TAG, "model_path": MODEL_PATH, "n_per_ds": N_PER_DS,
           "top_k": K, "num_experts": nE, "num_layers": nL,
           "datasets": ALL_DATASETS, "domain_of": dom_of,
           "pairwise_jaccard": M, "sink_set_size_mean": set_size,
           "random_baseline_overlap": rand_overlap,
           "domain_jaccard_rollup": dom_jac}

    # pretty print matrix
    print("\n=== per-dataset pos-0 sink Jaccard (%) ===")
    hdr = "            " + " ".join(f"{d[:6]:>7}" for d in ALL_DATASETS)
    print(hdr)
    for ds1 in ALL_DATASETS:
        cells = " ".join(f"{M[ds1][ds2]*100:7.1f}" for ds2 in ALL_DATASETS)
        print(f"  {ds1:>10} {cells}")
    print(f"\n  random baseline overlap: {rand_overlap*100:.1f}%")
    print(f"  domain rollup (cross-check vs E1): "
          + ", ".join(f"{k} {v*100:.1f}%" for k, v in dom_jac.items()))

    os.makedirs(OUT_DIR, exist_ok=True)
    fp = f"{OUT_DIR}/e1b_sink_pairwise_{TAG}.json"
    json.dump(out, open(fp, "w"), indent=2, ensure_ascii=False)
    print(f"saved {fp}")
    release(model, tag="probe_sink_pairwise")


if __name__ == "__main__":
    main()
