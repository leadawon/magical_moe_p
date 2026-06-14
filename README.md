# magical_moe_probe

Diagnostic probe of expert routing in **Qwen3-30B-A3B** (48 MoE layers, 128
experts, top-8) across **math / code / NLI** domains. Goal: measure the model's
routing behavior to define a concrete MoE research problem (see
[HYPOTHESES.md](HYPOTHESES.md), [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)).

This is a fresh start, separate from the prior MAGICAL MoE-P pruning code — the
old methodology is set aside; here we *measure first* to find the real problem.

## Layout
```
HYPOTHESES.md       6 falsifiable hypotheses about MoE routing
EXPERIMENT_PLAN.md  design, datasets, logging spec, metric→hypothesis map
EXPERIMENT_LOG.md   chronological record of what was done
src/
  data_loaders.py   6 datasets (gsm8k, svamp, humaneval+, mbpp+, mnli, snli)
  routing_logger.py RoutingProbe: gate forward-hooks + per-(layer,expert) stats
  probe.py          main: load model once, capture routing per dataset
  analyze.py        test the 6 hypotheses -> results/FINDINGS.md
scripts/run.sh      probe + analyze, GPUs 0,1,2,3
results/            *_routing.npz, manifest.json, STATUS.txt, FINDINGS.md, derived/
logs/               probe.log
model -> ../magical_moe_p/model   (symlink to local Qwen3-30B-A3B weights)
```

## Run
```bash
bash scripts/run.sh            # 200 examples/dataset (default)
bash scripts/run.sh 100        # quicker, 100 examples/dataset
```
Or step by step:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.probe --num_examples 200
python -m src.analyze
```

## Monitor (for a lightweight watcher model)
- **Progress:** `cat results/STATUS.txt` — current dataset, example i/N, ETA.
- **Alive:** `ps aux | grep src.probe | grep -v grep`
- **GPU:** `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -4`
- **Log:** `tail -n 30 logs/probe.log`
- **Done when:** `results/STATUS.txt` starts with `DONE`, all six
  `results/*_routing.npz` exist, and `results/FINDINGS.md` is written.
- **Outputs to read at the end:** `results/FINDINGS.md` (verdicts per hypothesis),
  `results/findings_summary.json` (machine-readable).

## Key outputs
- `results/<dataset>_routing.npz` — aggregated per-(layer,expert) stats +
  per-example fingerprints.
- `results/FINDINGS.md` — per-hypothesis numbers and SUPPORTED/PARTIAL/NOT verdicts.
- `results/derived/` — similarity matrix, layerwise JS divergence, core fraction.
