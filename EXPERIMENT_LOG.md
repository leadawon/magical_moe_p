# Experiment Log

Chronological record of what was done in this project. Newest entries at the
bottom.

---

## 2026-06-11 — Project created (pivot from MAGICAL MoE-P)

### Decision / intent
Set aside the prior MAGICAL MoE-P pruning methodology. New direction: **measure
the routing behavior of a modern MoE model first, then define the research
problem from data.** Same model family (Qwen3-30B-A3B), GPUs 0,1,2,3, new folder
`magical_moe_probe`. The downstream monitoring of the run is delegated to a
cheaper model; this session's job was to design, implement, smoke-test, launch,
and document.

### Stopped the previous run
- Killed the running MAGICAL MoE-P-JY DAPO training (PID 2613130/2613131).
- Verified GPUs 0–3 freed (~15 MiB each).
- Cancelled the in-session 3-minute monitoring cron job and the background
  log-watch task.

### Environment checks
- venv `qwenmoevenv`: datasets 3.1.0, transformers 4.51.0, torch 2.5.1+cu124.
- Internet OK (HF 200). GSM8k already cached.
- Verified all six datasets load: gsm8k, ChilleD/SVAMP (300),
  evalplus/humanevalplus (164), evalplus/mbppplus (378),
  nyu-mll/glue:mnli val_matched (9815), stanfordnlp/snli test (10000).
- Confirmed model config: 48 layers, **all MoE** (`decoder_sparse_step=1`,
  `mlp_only_layers=[]`), 128 experts, top-8, `norm_topk_prob=true`, hidden 2048.

### Built the probe
- `src/data_loaders.py` — six loaders, raw domain text, fixed seed 42 sampling.
- `src/routing_logger.py` — `RoutingProbe`: forward-hooks every `mlp.gate`,
  reproduces softmax→top-8, accumulates per-(layer,expert) selection counts,
  prob mass, weight mass; per-(layer) confidence/entropy/margin; per-example
  [48×128] fingerprints.
- `src/probe.py` — loads model once, forward-only (batch 1, no generation),
  saves `results/<dataset>_routing.npz`, `manifest.json`, live `STATUS.txt`.
- `src/analyze.py` — tests H1–H6, writes `results/FINDINGS.md` +
  `findings_summary.json` + `derived/` arrays.
- `scripts/run.sh` — probe then analyze, CUDA_VISIBLE_DEVICES=0,1,2,3.
- Docs: `HYPOTHESES.md`, `EXPERIMENT_PLAN.md`, `README.md`, this log.

### Smoke test (3 examples × {gsm8k, mnli})
- 48 gates hooked; forwards ran; npz saved. ~0.4–0.5 ex/s steady state.
- Validated: selection-frequency rows sum to 1; fingerprints sum to 1/layer;
  shapes [48,128] / [N,48,128] correct.
- Ran `analyze.py` on the smoke data end-to-end → `FINDINGS.md` generated
  (within-domain nan expected with no same-domain pair). Cleaned up smoke files.
- Early signal already visible: gsm8k↔mnli cross-domain cosine ≈ 0.43, per-layer
  **Gini ≈ 0.70**, ~40% dead slots, low margin, domain-classifiable fingerprints.
  (Indicative only; real verdicts come from the full 6-dataset run.)

### Launched full run
- `nohup bash scripts/run.sh 200` → `logs/probe.log` (run.sh PID 3205213).
- 6 datasets × 200 examples, forward-only. Expected ~45–60 min + ~1 min load.
- Then `analyze.py` runs automatically → `results/FINDINGS.md`.

### How to monitor (handoff to watcher)
- `cat results/STATUS.txt` for progress; done when it starts with `DONE`.
- All six `results/*_routing.npz` present + `results/FINDINGS.md` written = done.
- If the process dies: `tail -n 40 logs/probe.log` for the error; re-run
  `bash scripts/run.sh 200` (it reloads the model and recomputes; cheap to retry).

### After completion — what to read / decide
- `results/FINDINGS.md`: per-hypothesis numbers + SUPPORTED/PARTIAL/NOT verdicts.
- Use the H1/H5/H6 combination to decide the thesis framing (specialized rim vs
  redundant core), H2/H4 for the routing-quality angle, H3 for *where* (depth)
  to intervene. See "Why these matter" in HYPOTHESES.md.

---

## 2026-06-14 — E1 (sink 해부) 완료

- 실행: `CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.probe_sink` (GPU 0~3, pid 1955973)
- 결과: `results_r2/e1_sink.json`, 로그 `logs/e1_sink.log`. 6 ds × 30 = 180 예시.
- 핵심 수치:
  - 위치별 top-8 겹침: pos-0 **80.5%**, pos-1 34%, pos-2~7 28→17%, baseline(≥20) 12%.
  - sink-like 위치 2개(pos 0,1). → sink ≈ 첫 토큰 현상.
  - pos-0 sink 집합 교차도메인 Jaccard 74~79% (도메인 무관).
  - sink 집합 크기 7.5/8 expert/층 (거의 결정적).
  - **pos-0 sink expert의 전체 라우팅 질량 = 8.1%** (회수 ROI 작음).
- 결론: P4 확증되나 thesis는 도메인-조건부 pruning이 메인, sink는 보조.
  RESULTS.md 4/6/7 섹션 갱신 완료.
- 인수인계 노트(claude_history.md)의 "도구 호출 깨짐" 문제는 이번 세션엔 없었음.
- 남은 일: **E3 (pruning 정확도 검증)** — 미작성.

---

## 2026-06-15 — E3 (pruning 정확도 검증) 완료

- 코드: `src/probe_e3.py` (신규). gate forward hook으로 dead expert logit -inf
  마스킹(offload-safe), teacher-forced NLL/PPL/next-tok-acc. held-out 도메인당 40예시.
- 유지셋: results/*_routing.npz의 prob_mass[48,128]에서 도메인별 cov% 최소 expert.
- 실행: `CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.probe_e3` (pid 2044717, ~7분).
- 결과 `results_r2/e3_pruning_eval.json`:
  - matched 95%cov (prune 16~22%): ΔPPL math +0.11 / code +0.29 / nli +0.24 (거의 무손실)
  - matched 99%cov (prune 6~9%): ΔPPL ~0 (노이즈, 일부 음수)
  - MISmatched(틀린 도메인 유지셋): ΔPPL +1.08 ~ **+14.2** (nli를 code셋으로 끌 때)
  - code matched 95% next-acc -7.6pp → code 민감(가장 독립적).
- 검증된 결론: 도메인-조건부 pruning은 정확도 보존, "무엇을 끄느냐"가 도메인별로
  결정적. thesis 확정. RESULTS.md 0/5.5/6/7 섹션 갱신.
- 남은 일: (선택) 실제 생성 정확도, cov 미세 스윕, sink 회수 실험.

---

## 2026-06-15 — Mixtral-8x7B 교차검증 (cross-model replication)

**목적**: Qwen3-30B-A3B에서 얻은 결론(도메인-조건부 pruning)이 단일 모델 특이성이
아니라 현대 MoE의 일반 성질임을 보이기 위해, 구조가 다른 Mixtral로 동일 실험 재현.

**모델 구조 대조**:
| | Qwen3-30B-A3B | Mixtral-8x7B-Instruct |
|--|--|--|
| MoE 층 | 48 | 32 |
| experts/층 | 128 | 8 |
| top-k | 8 | 2 |
| 활성/토큰 | ~3B | ~13B |

**환경/실행 결정 (남들이 납득하도록 명시)**:
- 모델 다운로드: HF `mistralai/Mixtral-8x7B-Instruct-v0.1` (gated 아님), 87GB,
  `/data1/ai25170474/models/Mixtral-8x7B-Instruct-v0.1`.
- **GPU**: 4~7 사용. GPU 0~3은 **타 사용자(root)의 vLLM 서버**(VLLM::Worker_TP0~3)가
  점유 중 — 우리(uid 1015, sudo 없음)는 접근/종료 불가. 우리 프로세스 누수 아님(확인함).
- **예시 수: 데이터셋당 60** (Qwen은 200). 이유: Mixtral은 토큰당 활성 파라미터가
  ~4배(13B) + 일부 CPU offload라 매우 느림(3예시 55초 ≈ 200예시/ds 60분 → Round1만
  6시간). 도메인 특화·prunability 등 **집계 통계는 60예시에서 안정적**이므로 결론
  비교에 충분. 절대 수치 비교 시 N 차이를 감안할 것.
- **코드 적응**: 모델 비종속 어댑터 `src/moe_adapter.py` 신설 (block 속성
  mlp↔block_sparse_moe, num_experts, top_k 자동 감지). 모든 probe/analyze가 이를
  사용하도록 수정. 출력은 `results_mixtral/`, `results_r2_mixtral/`로 분리(Qwen 결과
  불변). 실행: `scripts/run_mixtral.sh`.
- 어댑터 검증: smoke test에서 block_sparse_moe/32층/8 experts/top-2 정확 감지 확인.
- P5/E2 생성기 신설(`src/analyze_p5_e2.py`): routing npz에서 GPU 없이 파생. Qwen
  데이터로 교차검증 통과 (P5 기존값 정확 일치; E2는 prob-mass 기준=E3와 일치,
  sel-freq 기준=기존 e2값과 일치 → 두 기준 모두 기록).

**결과 (스위트 완료 2026-06-15 03:23, 60예시/ds, GPU 4~7)**:
→ results_mixtral/FINDINGS.md, results_r2_mixtral/*

| 항목 | Qwen3 (128 exp, top-8) | Mixtral (8 exp, top-2) |
|------|------------------------|------------------------|
| H1 도메인특화 gap | **0.42** (within .84/cross .42) ✅ | **0.024** (within .96/cross .93) △ 약함 |
| H2 부하 Gini / dead | 0.67 / 32% ✅불균형 | **0.15 / 0.07%** ❌ 균형 |
| P1 expert 출력 cosine | 0.049 (독립) | **0.628** (상당히 유사) |
| P3 top-k 축소 오차 | (k=4) 0.54 | (k=1) 0.72 — 둘 다 다 필요 |
| P5 단일도메인 expert | 27.7% | 29.3% (유사) |
| E2 prune여력(prob-mass 95%) | 16~22% | **0%** |
| E3 matched prune(무손실) | 16~22% ΔPPL+0.1~0.3 | **0% (끌 게 없음)** |
| P4/E1 sink pos0 / mass | 80% / 8% | **99% / 26%** (더 강함) |

**핵심 해석 — 일반화는 "동일 재현"이 아니라 expert 입도(granularity) 의존**:
1. **도메인-조건부 pruning은 fine-grained MoE(Qwen, expert 多)에서 성립, coarse
   MoE(Mixtral, expert 8개)에선 불가.** Mixtral은 8개를 균등·중복(cos 0.63)으로 다
   써서 도메인별로 버릴 것이 없음(E2/E3 0%). → thesis 범위를 "fine-grained MoE"로
   명확히 한정해야 하며, 이는 오히려 문제정의를 더 단단하게 만듦(왜 expert가 많아야
   pruning 여력이 생기는가, 라는 메커니즘 설명 제공).
2. **sink 라우팅은 두 모델 모두 강함(보편적 현상)** — pos0 일치 80~99%. Mixtral은
   sink가 차지하는 질량이 26%로 Qwen(8%)보다 큼 → coarse MoE에서 sink 회수가 상대적
   으로 더 의미 있을 수 있음(후속 가능).
3. E3 prune 0%는 버그 아님: prob-mass 95% 커버에 8 experts가 거의 다 필요(top-2라
   질량이 분산) → keep≈8 → prune 0. 구조가 답을 그대로 보여준 것.

---

## 2026-06-15 — E1 sink: pos-0 토큰 정체성 검증 (trivial 여부 점검)

**동기**: E1의 "pos-0 일치 80.5%, 교차도메인 Jaccard 74~79%"가 *내용 무관 sink*인지,
아니면 단순히 "모든 입력의 pos-0이 동일 BOS라서 같은 expert로 가는 trivial 결과"인지
확인. probe_sink.py와 **동일한 토크나이징**(`tok(ex["text"], ...)`, add_special_tokens
기본 True)으로 6개 데이터셋 × 10예시의 pos-0 token_id/decoded를 추출.

**결과**:
- **Qwen 토크나이저는 BOS를 붙이지 않음** (`bos_token=None`, add_bos 없음).
  → pos-0은 특수토큰이 아니라 실제 첫 단어 토큰.
- **pos-0 토큰은 데이터셋마다 다름** (union 19종):
  - gsm8k/svamp(math): `A`,`The`,`Jordan`,`Tim`,`Joe`,`B`,`In`,`If`… (이름·관사, 다양)
  - humaneval+(code): `\n`,`\n\n`,`from`
  - mbpp+(code): `Write` (고정)
  - mnli/snli(nli): `Prem`("Premise:"의 첫 조각, 고정)
  - 단일 동일 토큰? **아니오** (`all identical = False`).

**판정: 발견 유효 (trivial 아님).** 서로 다른 pos-0 토큰(math의 A/The/Jordan… vs
code의 \n/Write vs nli의 Prem)인데도 pos-0 sink expert 집합이 교차도메인 74~79%
겹친다는 것은, "같은 토큰→같은 expert"로 설명되지 않고 **토큰 정체성과 무관하게
위치(pos-0) 자체가 고정 expert 집합을 부른다**는 sink 주장을 입증한다. 특히 math는
pos-0 토큰이 7종으로 다양한데도 sink 일치가 높아 trivial 반박을 더 강화.

**유의(정직성)**: mbpp+(`Write`)·nli(`Prem`)는 데이터셋 내부에서 pos-0이 사실상 단일
토큰이라, 그 두 데이터셋 *내부*의 pos-0 일치는 토큰 동일성 효과를 일부 포함한다.
그러나 (1) gsm8k/svamp는 pos-0이 다양함에도 sink가 강하고, (2) 핵심 지표인 *교차도메인
Jaccard*는 서로 다른 토큰 간 비교이므로, sink 결론 자체는 토큰 동일성으로 환원되지 않음.

---

## 2026-06-15 — P4 "24종 첫토큰" 정체 확인 + P2(lexical)와의 겹침 평가

**동기**: RESULTS.md P4가 "첫토큰 24종인데도 pos-0 일치 83%"라 적혀 있어, 그 24종이
무엇이고 형식어/특수토큰 위주라면 P4(sink=위치 효과)가 P2(lexical=토큰 효과)와
얼마나 겹치는지 점검.

**재현**: probe_routing_quality.py의 P4 풀을 정확 복원(6ds×80 로드 → seed0 list
shuffle → 앞 80개, max_length=160). pos0_count=80 / distinct=24로 원본과 일치.

**24종 분류** (Qwen 토크나이저는 BOS 미부착 → **특수토큰 0%**):
| 분류 | 토큰 | 예시수(80중) | 종수(24중) |
|------|------|------|------|
| BOS/특수토큰 | (없음) | 0 | 0 |
| 형식어(도메인 정형 시작어) | `Prem`(28),`Write`(16),`\n`(9),`\n\n`(3),`import`(1) | **57 (71%)** | 5 |
| 순수 내용어(이름/관사/숫자) | A,The,All,In,If,And,There,B,3,John,Andy,Rose,Dan,May,Christ,Johnny,Anna,Jordan,Dave | **23 (29%)** | 19 |

→ **예시 비중은 형식어 71%**(nli=Prem, mbpp=Write가 단일 시작어라 쏠림),
  **종 다양성은 내용어 79%**(gsm8k/svamp의 pos-0은 이름·관사로 매우 다양).

**P4 vs P2 겹침 평가 (핵심)**:
- P4 pos-0 일치 **83%**  vs  P2 same-token 일치 **47%**  vs  P2 random **12%**.
- ① **pos-0 일치(83%)가 same-token(47%)을 크게 초과** → "같은 토큰이라 같은 expert"
  (P2)만으로는 83%가 안 나옴. 위치(pos-0) 자체가 *추가* 효과를 준다.
- ② **교차도메인 sink Jaccard 74~79%** (Prem vs Write vs A/John 등 *서로 다른 토큰* 간
  비교). P2 random(다른 토큰)이 12%인 것과 대비 → 다른 토큰인데도 sink 집합이 거의
  동일 = lexical로 환원 불가능한 순수 위치 효과.

**판정**: P4는 형식어 데이터셋(nli·mbpp)에서 토큰효과(P2)와 위치효과가 일부 **얽혀
있으나**, 핵심 주장(sink = 위치 효과)은 P2로 환원되지 않는다. 두 독립 증거: (1) pos-0
일치가 same-token을 36pp 초과, (2) 서로 다른 토큰 간 교차도메인 sink가 74~79%.
- **한계(정직)**: nli·mbpp는 pos-0이 사실상 단일 형식어라, 그 데이터셋 *내부* pos-0
  일치는 P2 기여가 큼. P4를 P2와 완전히 분리하려면 pos-0 토큰이 다양한 gsm8k/svamp
  단독 pos-0 일치를 별도 측정하는 것이 더 깨끗함(후속 가능, 현재 E1 교차도메인
  Jaccard가 그 역할을 부분 수행).
