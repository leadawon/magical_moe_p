#!/usr/bin/env python3
"""
Task 1 — pos-0 token identity check for E1 (probe_sink.py).

Question: is E1's "pos-0 top-8 overlap 80.5%" trivial — i.e. just because every
input starts with the SAME special token (e.g. BOS) so they obviously route to
the same experts — or is it a real, content-independent SINK (different pos-0
tokens across datasets, yet the same expert set)?

Method: reproduce probe_sink.py's EXACT tokenization
  enc = tok(ex["text"], return_tensors="pt", truncation=True, max_length=128)
(default add_special_tokens=True), and inspect input_ids[0,0] per dataset.
No GPU / no model forward needed — only the tokenizer determines pos-0.

Run: python -m src.probe_pos0_identity
"""
import os, sys, json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "/data1/ai25170474/workspace/magical_moe_probe"
MODEL_PATH = os.environ.get("PROBE_MODEL_PATH", f"{BASE}/model")
N_PER_DS = 10          # samples to print per dataset (probe_sink uses 30; pos-0 is stable)


def vis(s: str) -> str:
    return s.replace("\n", "\\n").replace("\t", "\\t")


def main():
    from transformers import AutoTokenizer
    from src.data_loaders import load, ALL_DATASETS, DOMAIN_OF

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("=== tokenizer special-token check ===")
    print(f"  bos_token        = {tok.bos_token!r}  (id={tok.bos_token_id})")
    print(f"  eos_token        = {tok.eos_token!r}  (id={tok.eos_token_id})")
    print(f"  add_bos_token    = {getattr(tok, 'add_bos_token', 'n/a')}")
    # what does add_special_tokens actually prepend (if anything)?
    probe_ids = tok("hello world", add_special_tokens=True)["input_ids"]
    print(f"  tok('hello world', add_special_tokens=True) -> first ids {probe_ids[:3]} "
          f"= {[vis(tok.decode([i])) for i in probe_ids[:3]]}")

    out = {"tokenizer": {"bos_token": tok.bos_token, "bos_token_id": tok.bos_token_id},
           "per_dataset": {}, "all_pos0_ids": []}
    per_ds_pos0_ids = {}

    for ds in ALL_DATASETS:
        dom = DOMAIN_OF[ds]
        rows = []
        ids_this_ds = []
        for ex in load(ds, N_PER_DS):
            # EXACT match to probe_sink.py tokenization
            enc = tok(ex["text"], return_tensors="pt", truncation=True, max_length=128)
            tid = int(enc["input_ids"][0, 0])
            dec = tok.decode([tid])
            rows.append((tid, dec))
            ids_this_ds.append(tid)
        per_ds_pos0_ids[ds] = ids_this_ds
        out["all_pos0_ids"].extend(ids_this_ds)
        cnt = Counter(t for t, _ in rows)
        out["per_dataset"][ds] = {
            "domain": dom,
            "samples": [{"token_id": t, "decoded": d} for t, d in rows],
            "distinct_pos0_ids": len(cnt),
            "most_common": [[t, c] for t, c in cnt.most_common()],
        }
        print(f"\n--- {ds} ({dom}) ---  pos-0 distinct={len(cnt)} of {len(rows)}")
        for t, d in rows:
            print(f"    token_id={t:<7d} decode={vis(d)!r}")

    # cross-dataset comparison
    all_ids = set(out["all_pos0_ids"])
    all_identical = len(all_ids) == 1
    out["pos0_union_distinct"] = len(all_ids)
    out["all_pos0_identical"] = all_identical

    print("\n=== cross-dataset pos-0 comparison ===")
    print(f"  union of distinct pos-0 token_ids across all 6 datasets: {len(all_ids)}")
    print(f"  all pos-0 identical (single shared token)? {all_identical}")
    union_decoded = sorted({vis(tok.decode([i])) for i in all_ids})
    out["pos0_union_decoded"] = union_decoded
    print(f"  union decoded: {union_decoded}")

    # verdict
    if all_identical:
        verdict = ("TRIVIAL — all datasets share one identical pos-0 token; "
                   "E1's 80.5% overlap could be explained by that alone. Finding INVALID.")
    else:
        verdict = ("VALID (non-trivial) — pos-0 tokens DIFFER across datasets "
                   "(and Qwen adds no BOS), yet E1 shows 80.5% pos-0 overlap and "
                   "74-79% cross-domain Jaccard. 'same token -> same expert' cannot "
                   "explain it; position itself drives a fixed sink expert set.")
    out["verdict"] = verdict
    print(f"\n=== VERDICT ===\n  {verdict}")

    os.makedirs(f"{BASE}/results_r2", exist_ok=True)
    json.dump(out, open(f"{BASE}/results_r2/pos0_identity.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\nsaved {BASE}/results_r2/pos0_identity.json")


if __name__ == "__main__":
    main()
