#!/bin/bash
# Clarify GPU 0-3 ownership: which memory is OURS vs other users'.
# Usage:
#   bash scripts/gpu_status.sh        # report only
#   bash scripts/gpu_status.sh kill   # also kill OUR leftover experiment procs
ME=$(whoami)
KILL=${1:-}

echo "============================================================"
echo "GPU 0-3 status   me=$ME   $(date '+%F %T')"
echo "============================================================"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader | head -4 | \
  awk -F',' '{printf "  GPU%s used:%9s free:%9s util:%5s\n",$1,$2,$3,$4}'

echo ""
echo "compute processes on GPU 0-3:"
mine=0; other=0
my_exp_pids=()
while IFS=',' read -r pid mem; do
  pid=$(echo "$pid" | tr -d ' '); mem=$(echo "$mem" | tr -d ' ')
  [ -z "$pid" ] && continue
  owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
  cmd=$(ps -o args= -p "$pid" 2>/dev/null | tr -s ' ' | cut -c1-58)
  if [ "$owner" = "$ME" ]; then
    mine=$((mine + mem)); tag="[ME]"
    if echo "$cmd" | grep -qE "probe_redundancy|probe_routing_quality|probe_sink|run_r2|src\.probe|train\.py"; then
      my_exp_pids+=("$pid"); tag="[ME/EXPERIMENT]"
    fi
  else
    other=$((other + mem)); tag="[other:${owner:-?}]"
  fi
  printf "  PID %-8s %6s MiB  %-16s %s\n" "$pid" "$mem" "$tag" "$cmd"
done < <(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid,used_memory \
         --format=csv,noheader,nounits 2>/dev/null)

echo ""
echo "  🟢 내($ME) GPU 점유 합계 : ${mine} MiB"
echo "  ⚪ 타 사용자 점유 합계   : ${other} MiB"
if [ ${#my_exp_pids[@]} -eq 0 ]; then
  echo "  ✅ 우리 실험 프로세스 없음 → 잡혀있는 메모리는 우리 것이 아님 (헷갈릴 필요 없음)"
else
  echo "  ⚠️  우리 실험 프로세스 ${#my_exp_pids[@]}개 실행 중: ${my_exp_pids[*]}"
  if [ "$KILL" = "kill" ]; then
    echo "  → 종료 중..."
    for p in "${my_exp_pids[@]}"; do kill "$p" 2>/dev/null && echo "    killed PID $p"; done
    sleep 2
    echo "  재확인:"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | head -4 | \
      awk -F',' '{printf "    GPU%s used:%9s free:%9s\n",$1,$2,$3}'
  else
    echo "     (해제하려면:  bash scripts/gpu_status.sh kill )"
  fi
fi
