# Claude 작업 인수인계 (claude_history.md)

> 이 세션에서 Claude의 도구 호출 형식이 깨져(`call`/`invoke` 잘못된 태그) 명령
> 실행이 반복 실패함. 새 세션에서 이 파일을 읽고 이어서 진행할 것.
> 작성: 2026-06-14

---

## 0. 지금 당장 해야 할 일 (NEXT ACTION)

**E1(sink 해부) 실험을 실행하고 모니터링하기.** GPU 0~3이 비어있음(각 24GB).
실행 명령 (한 줄, 백그라운드):

```bash
cd /data1/ai25170474/workspace/magical_moe_probe && \
CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=false \
nohup /data1/ai25170474/workspace/venvs/qwenmoevenv/bin/python3.10 -u -m src.probe_sink \
> logs/e1_sink.log 2>&1 &
```

확인:
```bash
# 실행 중인지
ps aux | grep "[p]ython3.10 -u -m src.probe_sink"
# 진행 로그 (모델 로드 ~60초 후 측정 시작)
tail -20 /data1/ai25170474/workspace/magical_moe_probe/logs/e1_sink.log
# GPU
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | head -4
```

완료 판정: `results_r2/e1_sink.json` 생성됨 + 로그에 `saved results_r2/e1_sink.json`.
예상 소요: 모델 로드 ~1분 + 측정 ~3~5분 (총 ~6분, 180개 예시 forward).

**모니터링 방식 권장**: 백그라운드 실행 후, `until [ -f results_r2/e1_sink.json ];
do sleep 20; done` 식으로 완료 대기하거나, CronCreate로 3~5분 간격 체크.
(pgrep 패턴 매칭 시 자기 자신 매칭 피하려면 대괄호 트릭: `[p]robe_sink`)

E1 결과가 나오면 **RESULTS.md의 7번 섹션(다음 실험)을 실제 결과로 갱신**하고,
P4(sink) 해석을 구체화할 것.

---

## 1. 프로젝트 개요

- **목적**: 현대 MoE 모델(Qwen3-30B-A3B: 48층 전부 MoE, 128 experts, top-8)의
  라우팅을 측정해 "MoE 연구 문제"를 정의. *measure-first* 접근.
- **폴더**: `/data1/ai25170474/workspace/magical_moe_probe`
- **모델**: `model/` → `../magical_moe_p/model` 심볼릭 링크 (Qwen3-30B-A3B, bf16, 57GB)
- **venv**: `/data1/ai25170474/workspace/venvs/qwenmoevenv/bin/python3.10`
  (datasets 3.1.0, transformers 4.51.0, torch 2.5.1+cu124)
- **GPU 규칙**: 반드시 **0,1,2,3** 사용 (4,5,6,7 아님). 단 노드 공용이라 타 사용자
  (root vLLM 등)가 0~3을 점유하면 대기해야 함. 우리는 일반계정(ai25170474, uid
  1015), sudo 없음 → 남의 프로세스 kill 불가.
- **데이터셋**: math(gsm8k, svamp), code(humaneval_plus, mbpp_plus),
  nli(mnli, snli). 전부 로드 검증됨. 각 200개 샘플(seed 42).

---

## 2. 코드 구조 (전부 작성·검증 완료)

- `src/data_loaders.py` — 6개 데이터셋 로더 (raw 텍스트, seed 42)
- `src/routing_logger.py` — RoutingProbe: 모든 mlp.gate에 forward hook,
  softmax→top-8 재현, per-(layer,expert) 집계 + per-example fingerprint
- `src/probe.py` — Round1 메인 (H1~H6용 데이터 수집), forward-only batch=1
- `src/analyze.py` — H1~H6 검증 → results/FINDINGS.md
- `src/probe_redundancy.py` — P1(중복)+P3(top-k 낭비). **중요**: device_map offload
  시 meta tensor 에러 → mlp.forward를 monkey-patch해 실제 forward 안에서 모든
  expert 평가 (offload-safe)
- `src/probe_routing_quality.py` — P2(어휘)+P4(sink)+P6(불안정), gate hook만 사용
- `src/probe_sink.py` — E1(sink 해부), gate hook만, offload-safe. **아직 미실행**
- `src/analyze_r2.py` — P1~P6 종합 → results_r2/FINDINGS_R2.md
- `src/gpu_utils.py` — `release(model, tag)`: gc+empty_cache+로그 (실험 끝 메모리 해제)
- `scripts/run.sh` — Round1 (probe→analyze)
- `scripts/run_r2.sh` — Round2 (probe_redundancy→probe_routing_quality→analyze_r2→gpu_status)
- `scripts/gpu_status.sh [kill]` — GPU 0~3 소유권 명확화. `kill` 인자 시 우리
  실험 프로세스만 종료 (남의 건 권한상 못 죽이고 대상도 아님)

---

## 3. 실험 결과 요약 (완료된 것)

### Round 1 — H1~H6 (results/FINDINGS.md)
- H1 도메인 특화 ✅: within 0.841 vs cross 0.423 (gap 0.418)
- H2 부하 불균형 ✅: Gini 0.670, dead slot 32%, hot 3%
- H3 깊이별 특화 △ 평탄: early 0.318 ≈ late 0.320, peak layer 19
- H4 라우터 margin ✅: top1 0.112, 정규엔트로피 0.840, margin 0.0023
- H5 도메인 공유 core ❌: universal core 4.3% (도메인끼리 거의 분리)
- H6 fingerprint 분류 ✅: 도메인 분류 99.8%

### 벤치마크 내부/간 분석
- 도메인 간 top-8 겹침(집계): math 81.8%, code 43.5%, nli 51.0%, 교차 4~22%
- 벤치마크 내부 example쌍 겹침: ~50% (gsm8k 51.8, mbpp+ 58.2, mnli 36.2, snli 64.1)
- **핵심**: 도메인→expert는 결정적 아님(개별 50% 유동), 집계해야 드러나는 통계 현상
- mnli(36%) vs snli(64%): 라우팅 일관성은 데이터셋 동질성을 따름 (confound)

### Round 2 — P1~P6 (results_r2/FINDINGS_R2.md)
- P1 전문가 중복 ❌: output cosine 0.049, 유효 expert 97.6/128, 중복쌍 0%
- P2 어휘 라우팅 △: 같은 토큰 47% vs 랜덤 12% (Prem/ise 등 NLI 정형어 93~99% 고정)
- P3 top-k 과잉 ❌: k=4 오차 0.54, k=6 오차 0.25 — 8개 다 필요
- P4 sink 라우팅 ✅강함: pos-0 일치 83% vs 일반 13% (첫토큰 24종인데도)
- P5 dead 분업 ✅: 전역 dead 1.3%, breadth-0 42.3%, 단일도메인 27.7%
- P6 라우팅 불안정 ❌: 무의미 접두어에 top-8 12%만 변동 (안정)

### E2 — 도메인 조건부 pruning 여력 (results_r2/e2_prunability.json)
- 95% 커버리지 기준 prune 가능: math 50%, code 38%, nli 50%, 전 도메인 30%
- 99%: math 34%, code 21%, nli 36%
- 유지셋 겹침(95%): math∩code 74%, math∩nli 76%, code∩nli 50% (code 가장 독립)
- ⚠️ 라우팅 질량 기준 상한. 실제 정확도는 E3 검증 필요

---

## 4. 결론 / thesis 방향 (RESULTS.md에 정리됨)

- ❌ **죽은 방향**: 중복 expert 병합(P1), top-k 축소(P3), 라우터 sharpening(P6)
- ✅ **살아있는 방향**:
  1. **도메인 조건부 pruning** (P5+E2): 배포 도메인 기준 30~50% 잉여
  2. **sink 용량 회수** (P4): 초기 토큰이 고정 expert 점유 ← E1으로 구체화 중
- 추천 thesis: "도메인-조건부 expert pruning(+sink 회수)으로 도메인 성능 지키며
  Qwen3 MoE 압축 가능한가?"

---

## 5. 남은 실험 (계획)

- **E1 (sink 해부)** — `src/probe_sink.py` 작성됨, **지금 실행해야 함** (위 0번 참고).
  위치 0~7별 top-8 겹침, sink expert 집합 식별, 교차도메인 일관성(Jaccard),
  sink가 차지하는 라우팅 질량, sink-like 위치 개수. → results_r2/e1_sink.json
- **E3 (pruning 검증)** — 미작성. 도메인이 안 쓰는 expert를 alive-mask로 끄고
  GSM8k/HumanEval 정확도 측정 → E2의 "30~50% prune"이 정확도 보존하는지 검증.
  dynamic_moe의 alive_mask 마스킹 + 소규모 eval 필요. (구버전 magical_moe_p_jy의
  set_layer_routing(layer, k, alive_mask) 참고)

---

## 6. 결과 파일 위치

- **RESULTS.md** = 단일 통합 결과 (이것만 보면 됨). H1~H6, P1~P6, E2, thesis 전부.
- HYPOTHESES.md (H1~H6), HYPOTHESES_R2.md (P1~P6) — 가설 정의
- results/FINDINGS.md, results_r2/FINDINGS_R2.md — 원시 출력 백업
- results/*_routing.npz — Round1 원시 데이터 (sel_counts, fingerprints 등)
- results_r2/*.json (p1_p3, p2_p4_p6, e2_prunability) — Round2/E2 결과
- EXPERIMENT_LOG.md — 작업 연대기

---

## 7. 이 세션에서 발생한 문제 (새 세션이 피해야 할 것)

- Claude의 도구 호출이 `<call>`/`<invoke>` 같은 깨진 형식으로 나가 "malformed,
  could not be parsed"로 반복 실패함. 그 결과 probe_sink 실행 명령이 여러 번
  안 떴음 (로그 미생성, GPU 그대로 비어있음 확인됨).
- 새 세션에서는 **정상 도구 형식**으로 위 0번 명령을 실행하면 됨.
- 모니터링: 백그라운드 실행 직후 한 번 확인하고, 완료까지 적당한 간격(자동 재호출
  또는 CronCreate)으로 체크. pgrep은 `[p]robe_sink` 대괄호 트릭 사용.
