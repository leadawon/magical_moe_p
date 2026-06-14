#!/bin/bash
# Round-2 probes (P1/P3 redundancy+topk, P2/P4/P6 routing-quality, then assemble).
# Runs on GPUs 0,1,2,3. Launch only when those GPUs are free.
set -e
VENV="/data1/ai25170474/workspace/venvs/qwenmoevenv"
PYTHON="$VENV/bin/python3.10"
PROJECT="/data1/ai25170474/workspace/magical_moe_probe"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT"

echo "============================================================"
echo "ROUND-2 PROBES start $(date '+%F %T')"
echo "============================================================"

echo "[1/3] P1+P3 expert redundancy / top-k waste ..."
$PYTHON -u -m src.probe_redundancy

echo "[2/3] P2+P4+P6 lexical / sink / instability ..."
$PYTHON -u -m src.probe_routing_quality

echo "[3/3] assembling FINDINGS_R2.md (incl. P5) ..."
$PYTHON -u -m src.analyze_r2

echo "============================================================"
echo "ROUND-2 DONE $(date '+%F %T'). See results_r2/FINDINGS_R2.md"
echo "============================================================"
# show GPU ownership so it's clear nothing of ours lingers
bash scripts/gpu_status.sh
