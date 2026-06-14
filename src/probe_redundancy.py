#!/usr/bin/env python3
"""
Round-2 probe P1 (expert functional redundancy) + P3 (top-k over-provisioning).

Robust to accelerate device_map offload: instead of calling experts manually
(which hits meta tensors when weights are offloaded), we monkey-patch the target
layers' mlp.forward to evaluate all 128 experts ON THE LIVE hidden states during
the real forward pass (when accelerate has materialized that layer). Outputs are
stashed to CPU until enough tokens are gathered, then we restore the forwards.

P1: pairwise cosine of per-expert mean outputs + effective rank (participation
    ratio) of the expert bank.
P3: rebuild the MoE output with k in {1,2,4,6} (renormalized) vs true k=8.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
BASE = "/data1/ai25170474/workspace/magical_moe_probe"
MODEL_PATH = os.environ.get("PROBE_MODEL_PATH", f"{BASE}/model")
OUT_DIR = os.environ.get("PROBE_OUT_DIR", f"{BASE}/results_r2")
TARGET_TOK = 512
N_PROMPTS_PER_DS = 4


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.data_loaders import load, ALL_DATASETS
    from src.moe_adapter import detect_moe

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    embed_dev = next(model.model.embed_tokens.parameters()).device
    layers = model.model.layers
    info = detect_moe(model)
    block_attr = info.block_attr
    nE_total = info.num_experts
    # sample ~5 MoE layers spread across depth (was hardcoded for Qwen's 48)
    nLmoe = info.num_layers
    LAYERS = sorted(set(int(round(f * (nLmoe - 1))) for f in (0.08, 0.25, 0.48, 0.73, 0.92)))
    # map MoE-layer index -> decoder-layer index (they coincide when all layers are MoE)
    moe_decoder_idx = [di for di, l in enumerate(layers)
                       if hasattr(getattr(l, block_attr, None), "gate")]
    LAYERS = [moe_decoder_idx[j] for j in LAYERS]

    store = {i: {"O": [], "G": [], "n": 0} for i in LAYERS}
    orig = {}

    def patch(i):
        mlp = getattr(layers[i], block_attr)
        o = mlp.forward
        orig[i] = o
        def fwd(hidden_states, *a, **kw):
            out = o(hidden_states, *a, **kw)            # real forward (materializes experts)
            if store[i]["n"] < TARGET_TOK:
                with torch.no_grad():
                    hs = hidden_states.reshape(-1, hidden_states.shape[-1])
                    G = mlp.gate(hs).float().cpu()
                    E = len(mlp.experts)
                    chunk = torch.empty(E, hs.shape[0], hs.shape[1], dtype=torch.float32)
                    for e in range(E):
                        chunk[e] = mlp.experts[e](hs).to(torch.float32).cpu()
                    store[i]["O"].append(chunk)
                    store[i]["G"].append(G)
                    store[i]["n"] += hs.shape[0]
            return out
        mlp.forward = fwd

    for i in LAYERS:
        patch(i)

    prompts = []
    for ds in ALL_DATASETS:
        for ex in load(ds, N_PROMPTS_PER_DS):
            prompts.append(ex["text"])
    print(f"[setup] {len(prompts)} prompts, layers={LAYERS}, target {TARGET_TOK} tok")

    with torch.no_grad():
        for p in prompts:
            if all(store[i]["n"] >= TARGET_TOK for i in LAYERS):
                break
            enc = tok(p, return_tensors="pt", truncation=True, max_length=256).to(embed_dev)
            model.model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"], use_cache=False)
    for i in LAYERS:
        getattr(layers[i], block_attr).forward = orig[i]

    results = {"layers": {}}
    for i in LAYERS:
        O = torch.cat(store[i]["O"], dim=1)[:, :TARGET_TOK, :]    # [E, T, hid] cpu fp32
        G = torch.cat(store[i]["G"], dim=0)[:TARGET_TOK, :]       # [T, E]
        E, T, hid = O.shape

        # ---- P1 redundancy ----
        meanO = O.mean(1)                                          # [E, hid]
        mn = torch.nn.functional.normalize(meanO, dim=1)
        cos = mn @ mn.t()
        off = cos[~torch.eye(E, dtype=bool)]
        mean_cos = off.mean().item()
        frac_hi = (off > 0.9).float().mean().item()
        M = (meanO - meanO.mean(0, keepdim=True))
        sv = torch.linalg.svdvals(M)
        pr = (sv.sum() ** 2 / (sv ** 2).sum()).item()
        ev = (sv ** 2); ev = ev / ev.sum()
        rank90 = int((torch.cumsum(ev, 0) < 0.90).sum().item()) + 1

        # ---- P3 top-k sweep ----  (kfull = the model's full routed set per token)
        kfull = min(8, nE_total)
        probs = torch.softmax(G, dim=1)
        topw8, topi8 = torch.topk(probs, kfull, dim=1)
        ar = torch.arange(T)
        def out_for_k(k):
            w, idx = topw8[:, :k], topi8[:, :k]
            w = w / (w.sum(1, keepdim=True) + 1e-9)
            acc = torch.zeros(T, hid)
            for j in range(k):
                acc += w[:, j:j + 1] * O[idx[:, j], ar]
            return acc
        out8 = out_for_k(kfull)
        rel = {k: (torch.norm(out_for_k(k) - out8, dim=1) /
                   (torch.norm(out8, dim=1) + 1e-9)).mean().item()
               for k in [kk for kk in [1, 2, 4, 6] if kk < kfull]}

        results["layers"][str(i)] = {
            "tokens": T, "P1_mean_expert_cos": mean_cos,
            "P1_frac_pairs_over_0.9": frac_hi,
            "P1_participation_ratio": pr, "P1_rank90": rank90,
            "P3_rel_l2_vs_k8": rel}
        relstr = " ".join(f"k{k}={v:.3f}" for k, v in sorted(rel.items()))
        print(f"[L{i:02d}] cos={mean_cos:.3f} hi>0.9={frac_hi*100:.1f}% "
              f"effPR={pr:.1f} rank90={rank90}/{nE_total} | P3 {relstr}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/p1_p3.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {OUT_DIR}/p1_p3.json")
    from src.gpu_utils import release
    release(model, tag="probe_redundancy")


if __name__ == "__main__":
    main()
