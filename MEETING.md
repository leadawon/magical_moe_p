# 회의 브리핑 — MoE 라우팅 sink artifact (P4/E1 중심)

> **대상**: 연구자 본인 (발표·토론용)
> **안건**: 논문화할 연구 문제 확정 — 메인은 **MoE routing sink**, 보조는 도메인-조건부 pruning
> **모델**: Qwen3-30B-A3B (메인) + Mixtral-8x7B-Instruct (교차검증)
> **상태**: 핵심 실험 전부 완료 (H1~H6, P1~P6, E1·E2·E3, Mixtral 재현, pos-0 정체 검증)
> **작성일**: 2026-06-15
>
> 본 문서는 RESULTS.md / EXPERIMENT_LOG.md / 기존 MEETING.md + [작업1] pos-0 토큰 정체
> 검증 결과를 통합한 발표 대본이다. 모든 수치의 단일 출처는 RESULTS.md.

---

## 0. 한 장 요약 (회의 첫 슬라이드)

- **무엇을 했나**: 거대 MoE가 토큰을 어느 expert로 보내는지 **48개 층 전부의 라우터를
  직접 측정**(forward-only). 추측이 아니라 측정에서 문제를 정의(measure-first).
- **메인 발견 (P4 → E1)**: MoE 라우팅에 **attention sink와 유사한 구조적 artifact**가
  있다. **문장의 첫 토큰**이 *내용·도메인과 무관하게* 거의 고정된 expert 집합으로 간다.
  - pos-0 예시 간 top-8 일치 **80.5%** vs 깊은 위치 baseline **12.1%**
  - 교차도메인 Jaccard **74~79%** (도메인 무관), 층당 top-8 중 **7.5/8 고정**
  - 두 번째 모델 Mixtral에서 **더 강하게**(pos0 99%) 재현 → 아키텍처 무관 보편 현상
  - **[작업1] 검증 완료**: 이 일치는 trivial(동일 BOS 탓) 아님 — 발견 **유효** (7절)
- **보조 발견 (P5 → E2/E3)**: fine-grained MoE에서 도메인별 16~22% expert를 무손실
  제거 가능(마스킹 실측). 단 기존 pruning 연구와 겹쳐 novelty 약함 → 보조.
- **닫힌 문**: 중복 병합(P1)·top-k 축소(P3)·라우터 sharpening(P6)은 데이터로 기각.
- **제안 thesis**: *"Routing Sinks in MoE — 첫 토큰이 위치만으로 고정 expert를
  점유하는 구조적 artifact의 발견·특성화."*

---

## 1. 이 연구가 왜 시작됐는가 (배경·동기)

### 1.1 출발점 — pruning 방법론에서 measure-first로 피벗
원래 프로젝트(MAGICAL MoE-P)는 **동적 TopK MoE pruning**을 RL로 학습시키는
*방법론*이었다. 그러나 "어떤 expert를 어떻게 끄는가"를 가정하고 들어가는 것이 문제였다.
2026-06-11, **방법을 먼저 만들지 말고 모델의 라우팅을 먼저 측정해 문제를 데이터에서
정의하자**로 피벗 → 본 프로젝트 `magical_moe_probe`. (EXPERIMENT_LOG 2026-06-11)

### 1.2 왜 MoE 라우팅이 연구 가치가 있나
- MoE는 FFN을 여러 expert로 쪼개고 토큰마다 일부(top-k)만 통과시킨다. 총 파라미터는
  크지만 토큰당 계산은 작다(Qwen3-30B-A3B = 총 30B, 토큰당 ~3B 활성).
- 문제: **모든 expert를 메모리에 올려둬야** 한다(쓰든 안 쓰든) → 압축·배포 비용.
- "안 쓰는 expert를 안전히 버릴 수 있나?"는 *추측이 아니라 측정*해야 답할 수 있다.
- 그래서 라우팅을 전수 측정했고, 그 과정에서 **압축과 무관한, 더 근본적인 artifact
  (sink)**가 드러났다 — 이게 본 연구의 메인으로 부상.

### 1.3 핵심 용어 (발표 중 청중 질문 대비)
| 용어 | 뜻 |
|------|----|
| expert | 토큰을 처리하는 작은 FFN. Qwen3는 층마다 128개. |
| router/gate | 토큰을 어느 expert로 보낼지 점수 매기는 부품. |
| top-k | 라우터 점수 상위 k명에게만 토큰 전송. Qwen3=top-8, Mixtral=top-2. |
| routing | "토큰→expert" 배정. 본 연구의 측정 대상. |
| (routing) sink | 내용과 무관하게 특정 위치(특히 첫 토큰)가 고정 expert를 점유하는 현상. |
| dead expert | 거의 토큰을 안 받는 expert. |

---

## 2. 핵심 가설 — P4/E1 sink 중심, 다른 가설과의 관계

### 2.1 메인 가설 (P4 → E1)
> **"MoE 라우팅은 의미 기반이라고 암묵적으로 가정되지만, 실제로는 첫 토큰이
> 내용·도메인과 무관하게 고정된 expert 집합을 점유하는 구조적 artifact(routing
> sink)가 존재한다."**

- P4(Round 2 약점 탐침)에서 처음 포착 → **E1(probe_sink.py)에서 해부·정량화**.
- attention sink(첫 토큰이 attention을 빨아들이는 알려진 현상)의 **라우팅 버전**.

### 2.2 다른 가설과의 관계 (전체 그림)
세 묶음으로 측정했다. **sink는 P4에서 나와 E1로 깊어졌고, 나머지는 sink의 맥락·대조군.**

- **H1~H6 (라우팅 구조)**: "MoE는 도메인별로 다른 expert를 쓰나?"
  → H1/H6: **그렇다**(분류 99.8%). 이 "의미 기반 라우팅" 통념과 **정반대인 것이
  sink**(첫 토큰은 도메인 무관) → H1과 P4의 충돌이 발견의 핵심 긴장.
- **P1·P3·P6 (닫힌 압축 각도)**: 중복 병합/k 축소/라우터 손질 → 전부 기각.
  sink가 "남은 살아있는 각도"로 부상하는 대조군 역할.
- **P2 (어휘 라우팅)**: "라우팅이 단어에 끌려가나?"(부분적). sink가 P2(토큰 효과)로
  환원되는지 아닌지가 **발견의 사활** → 7절에서 직접 검증.
- **P5 → E2/E3 (도메인-조건부 pruning)**: 보조 발견. sink와 독립적으로 측정됨.

### 2.3 ⚠️ 회의에서 반드시 합의할 framing 결정
**RESULTS.md와 MEETING.md(이 문서)의 메인이 다르다.**
- RESULTS.md: thesis = **도메인-조건부 pruning(메인)**, sink는 보조.
- MEETING.md(이 문서): **sink(메인)**, pruning은 보조.
- 이유: pruning은 novelty가 약하고(기존 연구와 겹침) Mixtral에서 0%로 일반성도
  제한적인 반면, **sink는 novelty가 높고 두 모델 모두에서 강함**. → 논문 메인은
  sink가 맞다는 판단. **회의 결론으로 이 framing을 확정하고 RESULTS.md TL;DR를
  정렬할 것.** (현재 두 문서 불일치는 알려진 상태 — 정직하게 안건으로 올림.)

---

## 3. 관련 선행 연구 (novelty 근거)

> 정식 문헌 서베이는 미실시(논문화 시 필수, 8절). 아래는 발견의 novelty를 주장하기
> 위한 위치 설정이며, **회의 후 실제 인용 확인이 필요한 항목**.

- **Attention sink** (예: StreamingLLM 계열): 첫 토큰이 attention을 대량 흡수한다는
  잘 알려진 artifact. 본 연구의 sink는 **attention이 아니라 expert 라우팅**에서의
  대응물 — "둘이 같은 원인인가?"가 열린 질문(6절).
- **MoE 라우팅 분석/load balancing**: 부하 불균형·dead expert·라우터 붕괴는 알려짐
  (본 연구 H2/H5와 정합). 그러나 **"위치(첫 토큰)가 내용을 누르고 라우팅을 지배한다"는
  체계적 측정은 본 연구가 처음**(주장) — 기존은 expert *내용/부하* 관점이지 *위치* 관점이 아님.
- **MoE 압축/expert pruning (태스크·도메인 기반)**: 보조 발견(도메인-조건부 pruning)과
  상당히 겹침 → 그래서 보조로 둔다.
- **novelty 한 줄**: "라우팅은 토큰 의미를 따른다"는 통념에 반해, **첫 토큰은 *위치*로
  라우팅되며 이 현상이 입도가 정반대인 두 MoE에 보편적**임을 측정·특성화한 것이 기여.

---

## 4. 실험 설계 디테일

### 4.1 모델
- **Qwen3-30B-A3B** (메인): 48층 **전부 MoE**(decoder_sparse_step=1, mlp_only_layers=[]),
  층당 **128 routed experts, top-8**, norm_topk_prob=true, hidden 2048. GPU 0~3.
- **Mixtral-8x7B-Instruct** (교차검증): 32층, 층당 **8 experts, top-2**, 토큰당 ~13B
  활성. GPU 4~7(0~3은 타 사용자 root vLLM 점유). 입도가 Qwen과 정반대 → 일반성 시험.

### 4.2 데이터 (3 도메인 × 2 데이터셋)
- math: GSM8k, SVAMP / code: HumanEval+, MBPP+ / nli: MNLI(val_matched), SNLI(test)
- **Qwen 200예시/ds**, **Mixtral 60예시/ds**(속도 — 집계 통계는 이 N에서 안정, 절대값
  비교 시 N차 감안). seed 42 고정 샘플링.
- **raw 도메인 텍스트**(chat 템플릿·instruction scaffolding 없음) → 공유 템플릿 토큰이
  아니라 도메인 내용 자체의 라우팅을 본다. nli는 `"Premise: ... Hypothesis: ..."` 정형.

### 4.3 측정 방법
- **forward-only**(생성 없음), batch=1. 모든 `mlp.gate`(=router)에 **forward hook** →
  router logits 캡처 → softmax → top-k 재현 → 통계 집계. 모델 비종속 어댑터
  `src/moe_adapter.py`(block 속성·num_experts·top_k 자동 감지)로 Qwen/Mixtral 공용.
- 산출물: per-(layer,expert) 선택빈도/확률질량, per-layer confidence/entropy/margin,
  per-example [48×128] fingerprint, `results*/*_routing.npz`.

### 4.4 비교 기준 (baseline)
- **sink(E1)**: pos-0~7 위치별 cross-example top-8 겹침 vs **깊은 위치(≥20) baseline**.
  도메인 무관성은 **교차도메인 Jaccard**(서로 다른 도메인 pos-0 sink 집합 비교).
- **pruning(E3)**: 도메인별 유지셋 밖 expert를 gate logit −inf 마스킹(offload-safe),
  held-out 텍스트(train 200과 분리, 도메인당 40)로 **teacher-forced PPL / next-tok acc**.
  비교군: matched(자기 도메인 유지셋) vs **mismatched(틀린 도메인 유지셋)**.
- **sink trivial 반박([작업1])**: probe_sink와 **동일 토크나이징**으로 pos-0 token_id 추출.

---

## 5. 실험 결과 (수치 전부 + 해석)

### 5.1 라우팅 구조 (H1~H6, Qwen)
| 가설 | 판정 | 핵심 수치 |
|------|------|-----------|
| H1 도메인 특화 | ✅ | within **0.841** vs cross **0.423** (gap 0.418) |
| H2 부하 불균형 | ✅ | Gini **0.670**, dead slot **32%**, hot 3% |
| H3 깊이 특화 | △ 평탄 | early 0.318 ≈ late 0.320 |
| H4 라우터 margin | ✅ | top1 0.112, 정규엔트로피 0.840, margin **0.0023** |
| H5 공유 core | ❌ | universal core **4.3%** |
| H6 fingerprint 분류 | ✅ | 도메인 분류 **99.8%** (chance 33%) |

- 해석: MoE는 도메인별 거의 분리된 expert를 쓰고(H1/H5/H6), 1/3은 거의 안 씀(H2).
- **집계 착시(중요)**: "math 82% 일치"는 200개 풀링 통계. *개별 예시쌍*은 top-8의
  ~50%만 공유(gsm8k 51.8%, snli 64.1%, **mnli 36.2%**) → 도메인→expert는 결정적이 아니라
  **분포적**. mnli(36%) vs snli(64%)는 같은 NLI인데도 2배 차 → 라우팅 일관성은 "의미"가
  아니라 **데이터셋 동질성**을 따라감 = "의미 특화 vs 형식 특화"를 분리 못 하는 confound.

### 5.2 닫힌/열린 문 (P1~P6, Qwen)
| 가설 | 아이디어 | 판정 | 수치 |
|------|---------|------|------|
| P1 중복 병합 | ❌ | output cosine **0.049**, 유효 97.6/128, 중복쌍 0% |
| P3 top-k 축소 | ❌ | k=4 오차 **0.54**, k=6 0.25 → 8개 다 필요 |
| P6 라우터 sharpening | ❌ | 무의미 접두어에 top-8 **12%만** 변동(안정) |
| P2 어휘 라우팅 | △ | 같은 토큰 **47%** vs 랜덤 **12%** (gap 35pp) |
| **P4 sink (메인)** | ✅ **강함** | pos-0 일치 **83%**(R2 풀) / **80.5%**(E1) vs 일반 ~13% |
| P5 도메인 전용 | ✅ | 전역 dead 1.3%, breadth-0 42.3%, **단일도메인 27.7%** |

### 5.3 ★ 메인 — sink 해부 (E1, `results_r2/e1_sink.json`)
위치별 cross-example top-8 겹침 (n=180, 6ds×30):

| 위치 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | ≥20 baseline |
|------|---|---|---|---|---|---|---|---|---|
| 겹침 | **80.5%** | 34.4% | 28.3% | 22.8% | 20.0% | 18.2% | 17.6% | 17.3% | **12.1%** |

- **사실상 첫 토큰 1개 현상**: pos-0 80.5% → pos-1 34%로 급락 → pos-2부터 baseline 수렴.
  sink-like 위치(overlap > deep+0.2)는 **2개**(pos 0,1)뿐.
- **도메인 무관**: pos-0 sink 집합 교차도메인 **Jaccard 74~79%**(math∩code 78.9 /
  math∩nli 77.8 / code∩nli 73.7) — H1(within 0.84)과 정반대.
- **거의 결정적**: pos-0 sink 집합 크기 **7.5/8 expert/층**.
- **저질량**: pos-0 sink expert의 전 토큰 라우팅 질량 = **8.1%** → "일은 별로 안 하면서
  첫 토큰에 의해 자리만 묶이는" 구조적 artifact. (회수 ROI는 작음.)

### 5.4 보조 — 도메인-조건부 pruning (E2 상한 / E3 실측)
E2(라우팅 질량 95% 커버 유지 시 층당 prune 상한): math 50.4% / code 38.0% / nli 50.2%.
E3(마스킹+teacher-forced 실측, held-out, `results_r2/e3_pruning_eval.json`):

| 도메인 | baseline PPL | matched 95%(prune%) | matched 99%(prune%) | mismatched 95% |
|--------|---|---|---|---|
| math | 10.83 | ΔPPL **+0.11** (20.8%) | −0.07 (8.1%) | code셋 **+1.08** |
| code | 10.91 | +0.29 / acc **−7.6pp** (16.1%) | −0.03 (6.1%) | math +3.34, nli +3.89 |
| nli | 32.81 | **+0.24** (22.3%) | −0.10 (8.6%) | code셋 **+14.2** / acc −5.4pp |

- matched(자기 도메인) 16~22% prune ≈ 무손실. mismatched는 PPL +1~**+14** 붕괴
  → "무엇을 끄느냐가 도메인-조건부여야 함" 직접 입증. **code가 가장 민감**(독립적).

### 5.5 교차검증 — Mixtral (sink 우선)
| 항목 | Qwen3 (128/top-8) | Mixtral (8/top-2) |
|------|---|---|
| **sink pos0 / 질량 (메인)** | **80.5% / 8%** | **99.0% / 26%** (더 강함) |
| 도메인 특화 H1 gap | 0.42 ✅ | 0.024 △ 약함 |
| 부하 H2 Gini/dead | 0.67 / 32% | 0.15 / 0.07% (균형) |
| 중복 P1 cosine | 0.049(독립) | 0.628(유사) |
| pruning 여력 E2/E3 (보조) | 16~22% 무손실 | **0%** (끌 게 없음) |

- **(메인) sink는 두 모델 모두 강함 → 보편 현상.** 입도 정반대인데도 pos-0이
  baseline(12%/25%)을 압도. coarse MoE(Mixtral)에서 **더 강함**(질량 26%).
- **(보조) pruning은 입도 의존**: fine-grained(Qwen)만 성립. Mixtral은 8개를
  골고루·중복 사용해 버릴 게 없음(E3 0%, 버그 아님 — 구조가 답을 보여줌).

### 5.6 ★ P2 반박 결정타 — 데이터셋 6×6 sink Jaccard (E1b, 신규)
E1이 도메인 단위(3쌍)만 봤던 걸 **데이터셋 6개 쌍별**로 확장. *같은 도메인이지만 첫
토큰이 다른* 쌍이 겹치면 "토큰이 아니라 위치"가 직접 증명됨.

**Qwen3 (Jaccard %, baseline 12.4%)** `results_r2/e1b_sink_pairwise_qwen.json`:

| | gsm8k | svamp | human+ | mbpp+ | mnli | snli |
|--|--|--|--|--|--|--|
| gsm8k | — | **88.6** | 71.3 | 78.0 | 77.9 | 77.7 |
| svamp | 88.6 | — | 70.4 | 78.5 | 77.5 | 77.3 |
| human+ | 71.3 | 70.4 | — | 72.4 | 72.6 | 72.8 |
| mbpp+ | 78.0 | 78.5 | 72.4 | — | 74.1 | 73.9 |
| mnli | 77.9 | 77.5 | 72.6 | 74.1 | — | **99.8** |
| snli | 77.7 | 77.3 | 72.8 | 73.9 | 99.8 | — |

**Mixtral (baseline 25.8%)**: 모든 쌍 **97.9~100%**(데이터셋·토큰 전부 무관).

- **핵심 한 줄**: **gsm8k↔svamp 88.6%** — 같은 math지만 첫 토큰이 A/Jordan vs Josh/Ed로
  *서로 다른데도* sink 집합이 거의 같다 → "같은 토큰→같은 expert"(P2)로 설명 불가.
- 다른 도메인 쌍(다른 토큰)도 70~78% ≫ baseline 12.4%. 도메인 rollup(80.2/78.3/77.2%)은
  E1 원본(78.9/77.8/73.7)과 정합 → **E1 독립 재현 확인**. Mixtral은 97.9~100%로 더 극단.

---

## 6. 강점·약점 / 예상 반론

### 6.1 강점
- **통념에 반함**: "라우팅=의미 기반"에 반해 첫 토큰은 *위치*로 라우팅 → 신선함.
- **아키텍처 무관 보편성**: 입도 정반대 두 모델에서 재현(Mixtral 더 강함).
- **다증거**: 위치 의존(80.5% vs 12%) + 도메인 무관(Jaccard 74~79%) + 고정성(7.5/8)
  + trivial 반박([작업1]) + 흔한 압축각도가 왜 안 되는지(P1/P3/P6)까지 데이터로 닫음.
- **measure-first**: 가정이 아니라 측정에서 문제 도출.

### 6.2 약점·예상 반론 (회의에서 나올 것)
1. **"sink는 P2(어휘)의 재포장 아닌가?"** → 가장 날카로운 반론. 대응: pos-0 일치(83%)가
   same-token(47%)을 36pp 초과 + 교차도메인 Jaccard는 *다른 토큰* 간 비교(random 12%
   대비 74~79%) + [작업1]에서 math pos-0은 7~9종으로 다양 + **[E1b 신규] 데이터셋 6×6
   매트릭스에서 gsm8k↔svamp Jaccard 88.6%**(같은 math지만 첫 토큰 A/Jordan vs Josh/Ed로
   상이한데도 거의 동일) → "같은 토큰이라서"로 직접 반박됨(§5.6). **단 한계 인정**:
   nli(`Prem`)·mbpp(`Write`)는 데이터셋 *내부* pos-0이 단일 토큰이라 내부 일치엔
   토큰효과가 섞임 — 그러나 매트릭스의 *교차도메인·비대각* 칸은 모두 다른 토큰 비교.
   → **클린 후속**: gsm8k/svamp 단독 pos-0 일치를 별도 측정(8절, E1b가 상당 부분 수행).
2. **"그래서 뭐가 좋아지나? (저질량 8%)"** → sink는 질량 8%만 쓰면서 자리를 묶음.
   응용 이득보다 **현상 존재·특성 자체가 기여**. 실효 검증은 후속(sink 회수 실험).
3. **"PPL은 상관 지표지 실제 정확도 아님"**(pruning 측) → E3는 teacher-forced PPL.
   실제 생성 정확도 재확인 필요(8절).
4. **"표본이 작다"** → Mixtral 60/ds, E1 180예시. 집계 통계는 안정하나 N 명시 필요.
5. **"sink 원인 미규명"** → 학습 동역학? attention sink와 동일 원인? 현재는 **현상 보고**
   단계. 원인은 논문의 열린 질문(메커니즘 분석 필요).
6. **framing 불일치**(2.3절) → RESULTS.md는 pruning 메인. 회의에서 sink 메인으로 확정.

---

## 7. [작업1] 결과 반영 — pos-0 토큰 정체 최종 판정

**점검 질문**: E1의 "pos-0 일치 80.5%"는 *모든 입력의 첫 토큰이 동일 BOS라서 당연히
같은 expert로 가는* trivial 결과인가?

**방법**: 신규 `src/probe_pos0_identity.py` — probe_sink.py와 **동일 토크나이징**
(`tok(ex["text"], …, max_length=128)`, default add_special_tokens=True)으로 6 데이터셋
× 10예시의 `input_ids[0,0]` 추출. GPU 불필요(pos-0은 토크나이저만으로 결정).
산출물 `results_r2/pos0_identity.json`.

**결과**:
- **`bos_token = null`** → Qwen은 BOS를 안 붙임. pos-0 = 실제 첫 단어 토큰(특수토큰 아님).
- pos-0 distinct(샘플 10 중): gsm8k **7**(A,J,Jordan,Tim,Domin,The,Joe) / svamp **9**
  (Josh,Ed,A,The,B,Mary,In,Jack,If) / humaneval+ 3(\n,\n\n,from) / mbpp+ 1(Write) /
  mnli 1(Prem) / snli 1(Prem). **union 19종, `all_pos0_identical = false`.**

**최종 판정: 발견 유효 (trivial 아님).**
- 전부 동일한 BOS가 아니라 데이터셋마다 다른 *일반 토큰*인데도 pos-0 일치 80.5% /
  교차도메인 Jaccard 74~79% → "같은 토큰→같은 expert"로 설명 불가. **위치(pos-0) 자체가
  고정 sink expert 집합을 부른다**는 주장 확증. 특히 math는 pos-0이 7~9종으로 다양함에도
  sink가 강해 trivial 반박을 강화.
- **정직한 한계**: mbpp+(`Write`)·nli(`Prem`)는 데이터셋 *내부* pos-0이 단일 토큰이라
  그 내부 일치엔 토큰 동일성 효과가 일부 섞인다. 단 핵심 지표인 *교차도메인 Jaccard*는
  서로 다른 토큰 간 비교이고 math의 pos-0이 다양하므로, **sink 결론은 토큰 동일성으로
  환원되지 않는다.** → 완전 분리용 클린 후속은 8절.

---

## 8. 논문으로 발전시키려면 (To-do)

### 8.1 sink 강화 (메인 — 우선순위 高)
- **메커니즘 규명**: routing sink는 왜 생기나? 학습 동역학? 라우터가 첫 토큰을
  "기본값"으로 처리? **attention sink와 같은 원인인가**(상관·인과 분석).
- **클린 분리 실험**(반론 1 대응): pos-0이 다양한 **gsm8k/svamp 단독** pos-0 일치를
  별도 측정 → P2(토큰효과)와 완전 분리. 또는 첫 토큰을 임의 토큰으로 치환해도 sink가
  유지되는지(=위치 효과 인과 검증).
- **입도 의존 설명**: 왜 Qwen 80% < Mixtral 99%인가(top-k·expert 수와의 관계 모델링).
- **sink 회수 실효**: sink expert를 첫 토큰 전용으로 분리/고정 시 나머지 토큰 용량이
  늘어 성능이 오르는가(질량 8%의 실효 검증).
- **모델 다양화**: 3번째 MoE(예: DeepSeek-MoE, OLMoE 등)로 보편성 추가 확인.

### 8.2 보조(pruning) 보강
- **실제 생성 정확도**: E3의 teacher-forced PPL → GSM8k/HumanEval 실제 생성 정확도로
  matched pruning 무손실 재확인(offload라 느림).
- **cov 미세 스윕**: code 95%cov next-acc −7.6pp 경계를 좁히는 곡선.

### 8.3 논문 인프라
- **정식 문헌 서베이**(3절을 실인용으로): attention sink, MoE 라우팅 분석, expert
  pruning. sink의 진짜 novelty(선행 부재) 확인.
- **framing 확정 & 문서 정렬**(2.3절): sink=메인으로 RESULTS.md TL;DR 동기화.
- **제목(가안)**: *Routing Sinks in Mixture-of-Experts: A Positional Artifact in
  Modern MoE LLMs* (보조로 domain-conditional pruning).

---

## 9. 부록 — 자료 출처
- 통합 결과(모든 수치 단일 출처): `RESULTS.md`
- 작업 연대기·결정 근거: `EXPERIMENT_LOG.md`
- 가설 정의: `HYPOTHESES.md`(H1~H6), `HYPOTHESES_R2.md`(P1~P6)
- 원시 데이터(Qwen): `results/*_routing.npz`, `results_r2/*.json`(e1_sink, e3, pos0_identity)
- 원시 데이터(Mixtral): `results_mixtral/`, `results_r2_mixtral/`
- 코드: `src/probe.py`(H), `src/probe_redundancy.py`(P1/P3),
  `src/probe_routing_quality.py`(P2/P4/P6), `src/probe_sink.py`(E1),
  `src/probe_e3.py`(E3), `src/probe_pos0_identity.py`([작업1]), `src/moe_adapter.py`
- 재현: `scripts/run.sh`(Qwen), `scripts/run_mixtral.sh`(Mixtral),
  pos-0 검증 `python3.10 -m src.probe_pos0_identity`
  (venv: `/data1/ai25170474/workspace/venvs/qwenmoevenv`, python3.10/transformers 4.51.0)
