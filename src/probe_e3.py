#!/usr/bin/env python3
"""
E3 — Pruning accuracy validation (tests E2's "30-50% prunable" claim).

E2 gave an UPPER BOUND from routing-mass coverage. E3 actually disables the
experts a domain doesn't use (alive-mask) and measures the impact with
teacher-forced metrics (forward-only, no generation — the model is offloaded
across 4 GPUs and generation would be far too slow).

Method
  - Keep set per domain: from results/*_routing.npz `prob_mass[48,128]`, keep the
    smallest set of experts per layer that covers C% of that layer's routing mass.
    Everything else is masked to -inf at the gate output (offload-safe forward
    hook; the model's own forward is untouched).
  - Metric: on HELD-OUT text per domain, teacher-forced NLL / perplexity and
    next-token top-1 accuracy, vs the unpruned baseline.
  - Conditions per domain D:
      baseline   (no mask)
      matched    (D pruned with D's own keep-set, at 95% and 99% coverage)
      mismatched (D evaluated under ANOTHER domain's keep-set — control showing
                  that the *right* domain's keep-set is what preserves accuracy)

Run: CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.probe_e3
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
BASE = "/data1/ai25170474/workspace/magical_moe_probe"
MODEL_PATH = os.environ.get("PROBE_MODEL_PATH", f"{BASE}/model")
RESULTS_DIR = os.environ.get("PROBE_RESULTS_DIR", f"{BASE}/results")     # routing npz source
OUT_DIR = os.environ.get("PROBE_OUT_DIR", f"{BASE}/results_r2")
N_EVAL = 40            # held-out examples per dataset used for evaluation
MAXLEN = 256
COVERAGES = [0.95, 0.99]
DOMAINS = ["math", "code", "nli"]
# held-out eval dataset per domain; for code use mbpp_plus's tail (less overlap
# with the probe set, and code keep-set is aggregated over both code datasets)
DS_OF_DOMAIN = {"math": "gsm8k", "code": "mbpp_plus", "nli": "mnli"}

# active alive-mask, set per layer just before each forward; None = no masking
_ALIVE = {}     # layer_idx -> bool tensor [128] (True=alive) or absent


def keep_set_from_prob_mass(prob_mass, coverage):
    """prob_mass: [nL,128]. Return bool keep-mask [nL,128] covering `coverage`
    of each layer's routing mass with the fewest experts."""
    nL, nE = prob_mass.shape
    keep = np.zeros((nL, nE), dtype=bool)
    for l in range(nL):
        row = prob_mass[l]
        tot = row.sum()
        if tot <= 0:
            keep[l] = True
            continue
        order = np.argsort(row)[::-1]
        cum = np.cumsum(row[order]) / tot
        n_keep = int(np.searchsorted(cum, coverage) + 1)
        n_keep = max(1, min(nE, n_keep))
        keep[l, order[:n_keep]] = True
    return keep


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.data_loaders import load
    from src.gpu_utils import release

    # ---- load per-domain routing mass (aggregate the two datasets per domain) ----
    dom_prob = {}
    for dom in DOMAINS:
        acc = None
        for ds in [k for k, v in __import__("src.data_loaders", fromlist=["DOMAIN_OF"]).DOMAIN_OF.items() if v == dom]:
            d = np.load(f"{RESULTS_DIR}/{ds}_routing.npz")
            pm = d["prob_mass"].astype(np.float64)
            acc = pm if acc is None else acc + pm
        dom_prob[dom] = acc
    nL = dom_prob["math"].shape[0]
    print(f"[setup] {nL} layers, routing mass loaded for {DOMAINS}")

    # ---- model ----
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
    assert info.num_layers == nL, f"{info.num_layers} gates != {nL}"

    # forward hook on each gate: mask dead experts' logits to -inf (offload-safe)
    def mk(i):
        def hook(m, inp, out):
            am = _ALIVE.get(i)
            if am is None:
                return out
            dead = ~am.to(out.device)
            return out.masked_fill(dead.unsqueeze(0), float("-inf"))
        return hook
    for i, gate in enumerate(info.gates):
        gate.register_forward_hook(mk(i))
    print("[setup] gate masking hooks installed")

    # ---- held-out eval text per domain ----
    eval_text = {}
    for dom, ds in DS_OF_DOMAIN.items():
        items = load(ds, 400)            # seed-42 shuffle, first 400
        held = items[200:200 + N_EVAL]   # disjoint from the first-200 probe set
        if len(held) < N_EVAL:           # small datasets (e.g. humaneval 164): fall back
            held = items[-N_EVAL:]
        eval_text[dom] = [ex["text"] for ex in held]
        print(f"[eval] {dom}: {len(eval_text[dom])} held-out examples ({ds})")

    def set_mask(keep):
        _ALIVE.clear()
        if keep is None:
            return
        for l in range(nL):
            _ALIVE[l] = torch.from_numpy(keep[l])

    @torch.no_grad()
    def evaluate(texts):
        """teacher-forced sum NLL, token count, correct next-token count."""
        tot_nll, tot_tok, tot_correct = 0.0, 0, 0
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=MAXLEN)
            ids = enc["input_ids"].to(dev)
            if ids.shape[1] < 2:
                continue
            out = model(input_ids=ids,
                        attention_mask=enc["attention_mask"].to(dev), use_cache=False)
            logits = out.logits[0, :-1].float()      # [T-1, V]
            tgt = ids[0, 1:]                          # [T-1]
            logp = torch.log_softmax(logits, dim=-1)
            nll = -logp[torch.arange(tgt.shape[0]), tgt]
            tot_nll += nll.sum().item()
            tot_tok += tgt.shape[0]
            tot_correct += (logits.argmax(-1) == tgt).sum().item()
        ppl = float(np.exp(tot_nll / max(1, tot_tok)))
        return {"nll": tot_nll / max(1, tot_tok), "ppl": ppl,
                "next_tok_acc": tot_correct / max(1, tot_tok), "tokens": tot_tok}

    # precompute keep-sets and their prune fractions
    keeps = {}     # (dom, cov) -> keep mask
    for dom in DOMAINS:
        for cov in COVERAGES:
            keeps[(dom, cov)] = keep_set_from_prob_mass(dom_prob[dom], cov)

    results = {"N_eval": N_EVAL, "coverages": COVERAGES, "per_domain": {}}

    for dom in DOMAINS:
        texts = eval_text[dom]
        dres = {}
        # baseline (no mask)
        set_mask(None)
        base = evaluate(texts)
        dres["baseline"] = base
        print(f"\n=== domain={dom} ===")
        print(f"  baseline:  ppl {base['ppl']:.3f}  nll {base['nll']:.4f}  "
              f"nextacc {base['next_tok_acc']*100:.1f}%  ({base['tokens']} tok)")

        for cov in COVERAGES:
            keep = keeps[(dom, cov)]
            prune_pct = float((~keep).mean() * 100)
            set_mask(keep)
            r = evaluate(texts)
            r["prune_pct"] = prune_pct
            r["d_ppl"] = r["ppl"] - base["ppl"]
            r["d_nextacc"] = r["next_tok_acc"] - base["next_tok_acc"]
            dres[f"matched_cov{int(cov*100)}"] = r
            print(f"  matched  cov{int(cov*100)} (prune {prune_pct:4.1f}%): "
                  f"ppl {r['ppl']:.3f} (Δ{r['d_ppl']:+.3f})  "
                  f"nextacc {r['next_tok_acc']*100:.1f}% (Δ{r['d_nextacc']*100:+.1f}pp)")

        # mismatched control: evaluate `dom` under the OTHER domains' keep-sets at 95%
        for other in DOMAINS:
            if other == dom:
                continue
            keep = keeps[(other, 0.95)]
            prune_pct = float((~keep).mean() * 100)
            set_mask(keep)
            r = evaluate(texts)
            r["prune_pct"] = prune_pct
            r["d_ppl"] = r["ppl"] - base["ppl"]
            r["d_nextacc"] = r["next_tok_acc"] - base["next_tok_acc"]
            dres[f"mismatched_{other}_cov95"] = r
            print(f"  MISmatch {other:>4} cov95 (prune {prune_pct:4.1f}%): "
                  f"ppl {r['ppl']:.3f} (Δ{r['d_ppl']:+.3f})  "
                  f"nextacc {r['next_tok_acc']*100:.1f}% (Δ{r['d_nextacc']*100:+.1f}pp)")

        results["per_domain"][dom] = dres
        set_mask(None)

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(results, open(f"{OUT_DIR}/e3_pruning_eval.json", "w"), indent=2)
    print(f"\nsaved {OUT_DIR}/e3_pruning_eval.json")
    release(model, tag="probe_e3")


if __name__ == "__main__":
    main()
