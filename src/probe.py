#!/usr/bin/env python3
"""
MoE routing probe — main driver.

Loads Qwen3-30B-A3B once, then for each dataset runs a forward pass over every
example (batch size 1, no generation) while RoutingProbe captures per-layer
router decisions. Aggregated statistics + per-example fingerprints are saved to
results/<dataset>_routing.npz.

Forward-only (no autoregressive generation) is deliberate: it is deterministic,
fast, hang-free, and isolates *domain-input* routing for clean cross-domain
comparison. Generation-time routing is left as a follow-up.

Run:  see scripts/run.sh   (uses GPUs 0,1,2,3)
"""

import os
import sys
import json
import time
import argparse
import logging

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

BASE = "/data1/ai25170474/workspace/magical_moe_probe"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("probe")


def write_status(text: str):
    with open(os.path.join(BASE, "results", "STATUS.txt"), "w") as f:
        f.write(text)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=os.path.join(BASE, "model"))
    p.add_argument("--num_examples", type=int, default=200)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--datasets", nargs="+", default=None,
                   help="subset of dataset names; default = all six")
    p.add_argument("--output_dir", default=os.path.join(BASE, "results"))
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.data_loaders import load, DOMAIN_OF, ALL_DATASETS
    from src.routing_logger import RoutingProbe

    datasets = args.datasets or ALL_DATASETS
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MoE ROUTING PROBE  (Qwen3-30B-A3B)")
    logger.info(f"  GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"  datasets: {datasets}")
    logger.info(f"  examples/dataset: {args.num_examples}  max_len: {args.max_len}")
    logger.info("=" * 60)
    write_status("loading model")

    logger.info(f"Loading tokenizer + model from {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    embed_device = next(model.model.embed_tokens.parameters()).device
    logger.info(f"Model loaded. Input device: {embed_device}")

    from src.moe_adapter import detect_moe
    info = detect_moe(model)
    logger.info(f"MoE detected: block_attr={info.block_attr} "
                f"layers={info.num_layers} experts={info.num_experts} top_k={info.top_k}")
    probe = RoutingProbe(model, num_experts=info.num_experts, top_k=info.top_k)
    logger.info(f"Hooked {probe.num_layers} MoE gate modules.")

    manifest = {"model": args.model_path, "num_examples": args.num_examples,
                "max_len": args.max_len, "datasets": {}}
    t_global = time.time()

    for di, name in enumerate(datasets):
        domain = DOMAIN_OF[name]
        logger.info("-" * 60)
        logger.info(f"[{di+1}/{len(datasets)}] dataset={name} domain={domain}")
        items = load(name, args.num_examples)
        logger.info(f"  loaded {len(items)} examples")
        probe.reset_dataset()

        t0 = time.time()
        for i, ex in enumerate(items):
            inputs = tok(ex["text"], return_tensors="pt",
                         truncation=True, max_length=args.max_len).to(embed_device)
            with torch.no_grad():
                model.model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            use_cache=False)
            probe.consume_example()

            if (i + 1) % 10 == 0 or (i + 1) == len(items):
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (len(items) - (i + 1)) / rate if rate > 0 else 0
                msg = (f"  {name}: {i+1}/{len(items)}  "
                       f"{rate:.2f} ex/s  eta {eta:.0f}s")
                logger.info(msg)
                write_status(
                    f"dataset {di+1}/{len(datasets)} = {name}\n"
                    f"example {i+1}/{len(items)}\n"
                    f"rate {rate:.2f} ex/s  eta {eta:.0f}s\n"
                    f"elapsed_total {time.time()-t_global:.0f}s\n"
                )

        out = os.path.join(args.output_dir, f"{name}_routing.npz")
        probe.save(out, dataset=name, domain=domain)
        dt = time.time() - t0
        tot_tokens = float(probe.tokens.sum().item())
        logger.info(f"  saved {out}  ({dt:.0f}s, {tot_tokens:.0f} tokens routed)")
        manifest["datasets"][name] = {
            "domain": domain, "examples": len(items),
            "tokens_routed": tot_tokens, "seconds": dt, "output": out,
        }
        with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    write_status(f"DONE all datasets in {time.time()-t_global:.0f}s")
    logger.info("=" * 60)
    logger.info(f"PROBE COMPLETE in {time.time()-t_global:.0f}s")
    logger.info("Run: python -m src.analyze  to produce FINDINGS.md")
    logger.info("=" * 60)
    from src.gpu_utils import release
    release(model, tag="probe")


if __name__ == "__main__":
    main()
