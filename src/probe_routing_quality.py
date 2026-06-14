#!/usr/bin/env python3
"""
Round-2 probes P2 (lexical vs contextual routing), P4 (sink routing),
P6 (instability under trivial perturbation). One model load.

Capture, per token, its id/position and the per-layer top-8 expert set.
  P2: overlap of top-8 across occurrences of the SAME token id vs RANDOM tokens.
  P4: overlap of top-8 at position 0 across examples vs other positions.
  P6: prepend an innocuous prefix; compare each content token's top-8 with/without.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
BASE = "/data1/ai25170474/workspace/magical_moe_probe"
# model-agnostic config via env (defaults to Qwen)
MODEL_PATH = os.environ.get("PROBE_MODEL_PATH", f"{BASE}/model")
OUT_DIR = os.environ.get("PROBE_OUT_DIR", f"{BASE}/results_r2")
N_P2 = 80          # prompts for P2/P4 capture
N_P6 = 40          # prompts for P6
K = 8              # overridden from the model's top_k at runtime


def build_capture(model):
    from src.moe_adapter import detect_moe
    info = detect_moe(model)
    cap = {}
    handles = []
    for idx, gate in enumerate(info.gates):
        def mk(i):
            def hook(m, inp, out):
                cap[i] = out.detach()
            return hook
        handles.append(gate.register_forward_hook(mk(idx)))
    return info.gates, cap, handles


@torch.no_grad()
def capture_top8(model, cap, n_layers, input_ids, attn, dev):
    cap.clear()
    model.model(input_ids=input_ids.to(dev), attention_mask=attn.to(dev), use_cache=False)
    T = input_ids.shape[1]
    top8 = np.empty((T, n_layers, K), dtype=np.int16)
    for l in range(n_layers):
        probs = torch.softmax(cap[l].float(), dim=1)      # [T,128]
        idx = torch.topk(probs, K, dim=1).indices          # [T,8]
        top8[:, l, :] = idx.cpu().numpy().astype(np.int16)
    return top8


def overlap(a, b):
    # a,b: [n_layers, K] int -> mean over layers of |set∩|/K
    s = 0.0
    for l in range(a.shape[0]):
        s += len(set(a[l]) & set(b[l])) / a.shape[1]
    return s / a.shape[0]


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.data_loaders import load, ALL_DATASETS

    global K
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    dev = next(model.model.embed_tokens.parameters()).device
    from src.moe_adapter import detect_moe
    K = detect_moe(model).top_k
    gate_layers, cap, handles = build_capture(model)
    nL = len(gate_layers)
    print(f"[setup] {nL} gate layers, top_k(K)={K}")

    # mixed prompts
    pool = []
    for ds in ALL_DATASETS:
        for ex in load(ds, max(N_P2, N_P6)):
            pool.append(ex["text"])
    rng = np.random.RandomState(0)
    rng.shuffle(pool)
    out = {}

    # ---------- P2 + P4 capture ----------
    all_ids, all_top, all_pos = [], [], []
    for p in pool[:N_P2]:
        enc = tok(p, return_tensors="pt", truncation=True, max_length=160)
        t8 = capture_top8(model, cap, nL, enc["input_ids"], enc["attention_mask"], dev)
        ids = enc["input_ids"][0].numpy()
        all_ids.append(ids); all_top.append(t8)
        all_pos.append(np.arange(len(ids)))
    ids = np.concatenate(all_ids)
    top = np.concatenate(all_top, 0)          # [Ntok, nL, 8]
    pos = np.concatenate(all_pos)
    print(f"[P2/P4] captured {len(ids)} tokens")

    # P2: same-id vs random overlap
    uniq, cnt = np.unique(ids, return_counts=True)
    freq_ids = uniq[cnt >= 20]
    freq_ids = freq_ids[np.argsort(-cnt[cnt >= 20])][:60]
    same_ov = []
    detail = []
    for tid in freq_ids:
        loc = np.where(ids == tid)[0]
        pr = [(loc[rng.randint(len(loc))], loc[rng.randint(len(loc))]) for _ in range(40)]
        ov = np.mean([overlap(top[a], top[b]) for a, b in pr if a != b])
        same_ov.append(ov)
        detail.append((tok.decode([int(tid)]).strip()[:10], int((ids == tid).sum()), float(ov)))
    rand_ov = np.mean([overlap(top[rng.randint(len(ids))], top[rng.randint(len(ids))])
                       for _ in range(2000)])
    out["P2"] = {"same_token_overlap": float(np.mean(same_ov)),
                 "random_overlap": float(rand_ov),
                 "gap": float(np.mean(same_ov) - rand_ov),
                 "n_freq_ids": len(freq_ids)}
    print(f"[P2] same-token top8 overlap={np.mean(same_ov)*100:.1f}%  "
          f"random={rand_ov*100:.1f}%  gap={ (np.mean(same_ov)-rand_ov)*100:.1f}pp")
    # token-type breakdown
    def ttype(s):
        s = s.strip()
        if s.isdigit(): return "num"
        if s and all(c.isalpha() for c in s): return "word"
        return "punct/other"
    by = {}
    for nm, c, ov in detail:
        by.setdefault(ttype(nm), []).append(ov)
    out["P2"]["by_type"] = {k: float(np.mean(v)) for k, v in by.items()}
    out["P2"]["examples"] = sorted(detail, key=lambda x: -x[2])[:12]

    # P4: pos0 vs other positions
    p0 = np.where(pos == 0)[0]
    other = np.where(pos >= 3)[0]
    p0_ov = np.mean([overlap(top[p0[rng.randint(len(p0))]], top[p0[rng.randint(len(p0))]])
                     for _ in range(2000)])
    ot_ov = np.mean([overlap(top[other[rng.randint(len(other))]], top[other[rng.randint(len(other))]])
                     for _ in range(2000)])
    # is pos0 routing identical regardless of token id?
    p0_ids = ids[p0]
    out["P4"] = {"pos0_overlap": float(p0_ov), "otherpos_overlap": float(ot_ov),
                 "pos0_distinct_token_ids": int(len(np.unique(p0_ids))),
                 "pos0_count": int(len(p0))}
    print(f"[P4] pos0 overlap={p0_ov*100:.1f}%  other-pos={ot_ov*100:.1f}%  "
          f"(pos0 has {len(np.unique(p0_ids))} distinct first-tokens over {len(p0)} examples)")

    # ---------- P6: perturbation instability ----------
    prefix_ids = tok("Note: ", return_tensors="pt")["input_ids"][0]
    off = len(prefix_ids)
    changed_band = {"early": [], "mid": [], "late": []}
    tot_changed = []
    for p in pool[:N_P6]:
        enc = tok(p, return_tensors="pt", truncation=True, max_length=150)
        cids = enc["input_ids"][0]
        Tc = len(cids)
        # original
        a8 = capture_top8(model, cap, nL, cids.unsqueeze(0),
                          torch.ones(1, Tc, dtype=torch.long), dev)
        # perturbed = prefix + same content tokens
        pids = torch.cat([prefix_ids, cids]).unsqueeze(0)
        b8full = capture_top8(model, cap, nL, pids,
                              torch.ones(1, pids.shape[1], dtype=torch.long), dev)
        b8 = b8full[off:off + Tc]                 # aligned content tokens
        # fraction of (token,layer) top8 sets that changed (1 - overlap)
        for t in range(Tc):
            for band, (lo, hi) in {"early": (0, nL // 3), "mid": (nL // 3, 2 * nL // 3),
                                   "late": (2 * nL // 3, nL)}.items():
                ch = np.mean([1.0 - len(set(a8[t, l]) & set(b8[t, l])) / K
                              for l in range(lo, hi)])
                changed_band[band].append(ch)
        tot_changed.append(np.mean([1.0 - len(set(a8[t, l]) & set(b8[t, l])) / K
                                    for t in range(Tc) for l in range(nL)]))
    out["P6"] = {"mean_top8_changed": float(np.mean(tot_changed)),
                 "early": float(np.mean(changed_band["early"])),
                 "mid": float(np.mean(changed_band["mid"])),
                 "late": float(np.mean(changed_band["late"])),
                 "perturbation": "prepend 'Note: '"}
    print(f"[P6] mean top8 changed by irrelevant prefix={np.mean(tot_changed)*100:.1f}%  "
          f"(early {np.mean(changed_band['early'])*100:.1f} / "
          f"mid {np.mean(changed_band['mid'])*100:.1f} / "
          f"late {np.mean(changed_band['late'])*100:.1f})")

    for h in handles:
        h.remove()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/p2_p4_p6.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {OUT_DIR}/p2_p4_p6.json")
    from src.gpu_utils import release
    release(model, tag="probe_routing_quality")


if __name__ == "__main__":
    main()
