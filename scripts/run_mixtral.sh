#!/bin/bash
# Full Qwen-equivalent experiment suite on Mixtral-8x7B-Instruct.
# Uses all 8 GPUs. Outputs go to results_mixtral/ and results_r2_mixtral/ so the
# Qwen results are never touched.
set -e
VENV="/data1/ai25170474/workspace/venvs/qwenmoevenv"
PY="$VENV/bin/python3.10"
PROJ="/data1/ai25170474/workspace/magical_moe_probe"
cd "$PROJ"

# GPUs 0-3 are held by another user's root vLLM server; we use the free 4-7.
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export TOKENIZERS_PARALLELISM=false
# Mixtral active params (~13B/token) + CPU offload make it far slower than Qwen,
# so we use fewer examples per dataset. Aggregate routing stats (domain
# specialization, prunability) are stable at this N; documented in the notes.
export N_EXAMPLES="${N_EXAMPLES:-60}"
export PROBE_MODEL_PATH="/data1/ai25170474/models/Mixtral-8x7B-Instruct-v0.1"
export PROBE_RESULTS_DIR="$PROJ/results_mixtral"
export PROBE_OUT_DIR="$PROJ/results_r2_mixtral"
export PROBE_MODEL_DESC="Mixtral-8x7B-Instruct (32 MoE layers, 8 experts, top-2)"

mkdir -p "$PROBE_RESULTS_DIR" "$PROBE_OUT_DIR"

echo "############################################################"
echo "MIXTRAL FULL SUITE  start $(date '+%F %T')   GPUs=$CUDA_VISIBLE_DEVICES"
echo "############################################################"

echo "=== [1/7] Round-1 routing probe (H1-H6 data), N=$N_EXAMPLES ==="
$PY -u -m src.probe --model_path "$PROBE_MODEL_PATH" \
    --output_dir "$PROBE_RESULTS_DIR" --num_examples "$N_EXAMPLES"

echo "=== [2/7] analyze H1-H6 ==="
$PY -u -m src.analyze

echo "=== [3/7] P1+P3 redundancy / top-k ==="
$PY -u -m src.probe_redundancy

echo "=== [4/7] P2+P4+P6 lexical / sink / instability ==="
$PY -u -m src.probe_routing_quality

echo "=== [5/7] P5 dead + E2 prunability (no GPU) ==="
$PY -u -m src.analyze_p5_e2
$PY -u -m src.analyze_r2

echo "=== [6/7] E1 sink anatomy ==="
$PY -u -m src.probe_sink

echo "=== [7/7] E3 pruning accuracy validation ==="
$PY -u -m src.probe_e3

echo "############################################################"
echo "MIXTRAL FULL SUITE DONE $(date '+%F %T')"
echo "  Round-1: $PROBE_RESULTS_DIR  (FINDINGS.md, *_routing.npz)"
echo "  Round-2/E: $PROBE_OUT_DIR    (FINDINGS_R2.md, *.json)"
echo "############################################################"
