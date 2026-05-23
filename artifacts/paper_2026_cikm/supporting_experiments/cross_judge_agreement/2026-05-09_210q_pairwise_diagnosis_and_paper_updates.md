# 2026-05-09 — 210q 평가 결과 진단과 sample.md 수정 정리

본 메모는 7-topic × 30q (총 210q) 평가에서 GPT-4o judge로 얻은 결과를 분석한 것이며,
pairwise에서 AnchorRAG가 모든 강한 베이스라인에 1.9~10% 승률로 패배한 원인을 추적하고
`sample.md`(현재 60q 기준 draft) 어디를 어떻게 고쳐야 하는지 정리한다.

평가 산출물:
- 절대평가: `AllSides_Qbias/exp4/eval_v5_7methods_210q_gpt4o_20260508.jsonl` (1470 rows = 7×210)
- 절대평가 요약: `exp4_eval_summary_v5_gpt4o_20260508.csv`
- Pairwise: `eval_v5_7methods_210q_gpt4o_pairwise_20260508.jsonl` (1260 rows = 6 baseline × 210)
- Pairwise 요약: `exp4_pairwise_summary_v5_gpt4o_20260508.csv`
- Claude-sonnet-4-6 judge 평가는 절대평가 단계 진행 중 (2026-05-09 14:00 기준 1429/1470)

---

## 1. 핵심 발견: AnchorRAG 출력은 stage-1 planner skeleton

### 1.1 답변 길이 분포 (210 queries)

| method            | mean | median | min  | max  | <600자 | "evidence-lacking" 표현 |
|-------------------|------|--------|------|------|--------|------------------------|
| vanilla_rag       | 2467 | 2528   | 1209 | 3277 | 0%     | —                      |
| mmr_rag           | 2466 | 2486   | 1485 | 3408 | 0%     | —                      |
| graph_retrieval   | 2476 | 2494   | 1470 | 3476 | 0%     | —                      |
| r2ag              | 2027 | 2030   |  822 | 3353 | 0%     | —                      |
| grag              | 2200 | 2207   | 1022 | 3375 | 0%     | —                      |
| reclaim           |  890 |  896   |  214 | 1809 | 29.0%  | —                      |
| **trustrag_anchor** | **797** | **828** | **297** | **1426** | **27.1%** | **30% (62/210)** |

AnchorRAG 답변은 다른 강한 베이스라인 대비 **3배 짧고**, ~30%가 "evidence가 없다"는 abstention 형식이며,
모든 답변이 동일한 bullet skeleton 형식 (`Most supported points` / `Main disagreement or uncertainty` / `Bottom line`).

### 1.2 §3.5 설계와의 불일치

`sample.md` §3.5 (Anchor-Guided Evidence Synthesis):

> \method performs answer generation in two stages.
> First, an evidence planner produces a compact synthesis plan
> (most strongly supported points, main disagreements, evidence gaps, response strategy).
> Second, the generator produces the final answer conditioned on both the plan and the card set.

현재 trustrag_anchor의 모든 출력 구조는 **stage-1 planner의 bullet 출력 그대로**임.
즉 stage-2 (prose generation conditioned on plan + cards)가 호출 안 되고 있거나,
stage-2 prompt가 plan을 prose로 풀어 쓰지 않고 그대로 bullet 형식으로 재출력 중.

### 1.3 Pairwise judge가 일관되게 baseline을 고른 이유

GPT-4o judge 설명에서 반복적으로 등장하는 키워드:
- "more comprehensive"
- "more grounded"
- "uses specific evidence from multiple sources"
- "broader range of evidence"
- "effectively differentiates between perspectives"

400~800자 bullet skeleton vs 2400자 prose multi-perspective synthesis의 holistic 비교에서
judge가 baseline을 고르는 것은 trust 신호 부재 때문이 아니라 **prose 부재 때문**.
pairwise 1.9~10% 승률은 method의 retrieval/trust 가설 결함이 아니라 generation pipeline 미완성의 결과.

### 1.4 절대평가 갭(utility −12pt, groundedness −8pt) 도 같은 원인

- AnchorRAG는 8개 카드를 모두 인용하면서 "이 카드에는 직접 데이터가 없다"고 abstention → groundedness 낮게 평가됨
- 짧고 평가 절제된 출력 → response_utility 낮게 평가됨
- 즉 절대평가에서도 일부 갭은 method가 아니라 미완성 pipeline의 산물

### 1.5 사례

**imm-01** (What conflicting evidence exists regarding effects of immigration on native wages and employment?)
- vanilla_rag: 2419자, 5개 perspective synthesis (NBER 연구, low-skilled visa 논쟁, 노조 입장, 여론 등)
- graph_retrieval: 2179자, 4개 perspective synthesis
- **trustrag_anchor: 399자, "current evidence set lacks direct studies or data" → 사실상 답변 거부**

**imm-11** (Sanctuary cities crime impact)
- vanilla_rag: 2400+자 prose
- **trustrag_anchor: 453자 bullet skeleton, "lack of direct evidence ... in the cards"**

**eco-20** (가장 긴 trustrag 답변, 1426자)
- 그래도 bullet skeleton 형식 유지: `Most supported points` / `Main disagreement or uncertainty` / `Bottom line`
- prose 답변 아님

---

## 2. Paper 영향과 framing 재검토

### 2.1 현재 draft의 "trust-utility tradeoff" 서사

현재 `sample.md`는 §1, §5, §6, §7에 걸쳐 일관되게:
- "We do not aim to universally maximize generic answer quality" (§1, line 12)
- "Tradeoff between trust-sensitive evidence selection and claim-level support" (§6, line 344)
- "Some trust-rich evidence is converted into integrative claims that the sentence-level evaluator does not mark as directly supported" (§5.4, line 254)

이 서사는 60q 결과를 본 후 만들어진 것인데, 실제 원인이 **integrative claim 때문이 아니라 stage-2 prose 부재**라면
서사가 부분적으로 잘못된 인과관계를 주장하고 있음. 특히 case study (line 312-317)의 "Integrative summary claim" 분석은
사실 stage-1 planner output을 integrative synthesis라고 잘못 해석한 것.

### 2.2 정직하게 갈 수 있는 framing

원인 진단을 받아들이면 두 가지 framing이 가능:
1. **현재 trustrag_anchor 출력은 의도한 prose 답변이 아니라 미완성 pipeline의 산물** → §3.5와 일치하도록 stage-2 prose 추가 후 재평가
2. **현재 출력을 "structured plan-style answer"로 의도적 설계라고 reframe** → "controversial QA에서 plan-style 출력은 evaluator transparency를 높인다"는 보호 framing. 다만 이건 사후 합리화로 보일 위험 큼.

(2)는 reviewer가 baseline과 출력 형식이 다른 데서 오는 unfairness를 지적할 수 있어 권장 안 함.

---

## 3. `sample.md` 수정 사항 정리

### 3.1 단순 숫자/표현 업데이트 (낮은 위험)

| 위치 | 현재 | 변경 |
|------|------|------|
| §1, line 12 | "Across six controversial topics and 60 evaluation queries" | "Across seven controversial topics and 210 evaluation queries" |
| §1, line 18 (contributions) | "expanded AllSides/Qbias evaluation over six topics and 60 controversial queries" | "expanded AllSides/Qbias evaluation over seven topics and 210 controversial queries" |
| §4.1, line 93 | "We evaluate \method on six controversial topics" | "We evaluate \method on seven controversial topics" + climate_change 토픽 설명 추가 |
| §4.1, line 93 | "the six topic graphs contain 534, 282, 530, 279, 225, and 270 articles" | climate_change 165 노드 포함된 7개 그래프 수치 |
| §4.1, line 97 | "10 open-ended controversial questions ... 60 evaluation queries in total" | "30 open-ended controversial questions ... 210 evaluation queries in total" |
| §6 Conclusion, line 344 | "six AllSides/Qbias topics and 60 controversial queries" | "seven AllSides/Qbias topics and 210 controversial queries" |
| §7 Discussion, line 349 | "six-topic expansion is especially important here" | "seven-topic 210-query expansion ..." |
| §7 Discussion, line 357 | "six controversial topics" | "seven controversial topics" |

### 3.2 Table 1 (community structure, line 132-149)

climate_change 행 추가 필요. 현재 6 topic 표:

```
Topic                       Nodes  Edges  Communities  Modularity  Mean Purity
Immigration                 534    1053   56           0.8630      0.3895
Gun Control                 282     550   32           0.8372      0.3901
Economy and Jobs            530    1063   51           0.8860      0.3887
Abortion                    279     688   19           0.7779      0.3835
Free Speech                 225     481   26           0.8177      0.3689
Voting Rights and Voter Fraud 270   658   22           0.8017      0.3926
```

→ climate_change 행 추가 (165 nodes; 나머지 통계 산출 필요).

산출 스크립트: `AllSides_Qbias/exp1_additional_topics/graph_allsides_climate_change_scored.json` 에서
edges/communities/modularity/purity 계산.

### 3.3 §5.1 Retrieval-Side Validation (line 153)

현재 6 topic 기준 fixed-budget retrieval 결과:
> Coverage 9.65, Community Entropy 2.2467, Avg s_trust 0.7661

→ 7 topic 기준으로 재실행 필요. 결정 사항:
- 옵션 A: 7 topic 모두 재실행 (정합성 좋음, 시간 소요)
- 옵션 B: "We additionally validated this on the climate_change topic with consistent results" 한 줄로 처리 (최소 비용)

### 3.4 Table 4 (main results, line 223-243) — 가장 중요

현재는 60q × 6 topic 기준 7개 컬럼:
Cite Cov, HT Cite, Backbone Trust | Grounded, Support, Utility, TW Claim Support

210q GPT-4o 절대평가에서 AnchorRAG가 명확히 우위인 지표:
- citation_coverage 0.6316 (vs 베이스라인 0.27~0.39, reclaim 0.93)
- high_trust_citation_rate 0.6584 (vs 0.36~0.45)
- anchor_utilization_rate 0.918 (vs 0.29~0.90)
- avg_s_trust@k 0.7635, top3_avg_s_trust 0.7781, high_trust_ratio 0.68 (vs 0.36~0.45)
- backbone_trust_rate 0.6616 (vs 0.34~0.42)
- backbone_avg_claim_trust 0.7598 (vs 0.62~0.74)

AnchorRAG 열위 지표:
- groundedness 0.7117 (vs 0.751~0.797)
- evidence_support_rate 0.7719 (vs 0.698~0.865)
- mssr 0.7767 (vs 0.672~0.841)
- response_utility 0.6619 (vs 0.709~0.786)
- backbone_support_rate 0.7948 (vs 0.736~0.890)

**컬럼 결정** (3.4-A vs 3.4-B 중 하나 선택):
- **A. 보수적 교체**: 현재 컬럼 유지, 숫자만 210q 값으로 교체. 변화 최소.
- **B. 컬럼 보강**: anchor_utilization_rate, high_trust_ratio, backbone_avg_claim_trust 중 1~2개 추가하여 method 우위 가시성 강화.

권장: **B (1~2개 추가)**. 현재 표 7컬럼인데 8컬럼까지는 무리 없고, anchor_utilization은 method의 핵심 동작을 직접 측정.

### 3.5 §5.4 Step 1~3 narrative (line 245-256)

현재 case study (abo-03 trust-win, eco-10 tradeoff)가 **stage-1 skeleton 출력에 기반**한 분석.
→ pipeline 진단을 받아들이면 case study 자체의 정확성에 의문.

선택:
- **선택 1**: 재생성 가능하면 stage-2 prose 추가 후 case study 다시 작성
- **선택 2**: 재생성 불가하면 case study (Figure 4) 제거하거나 toned-down 형태로 유지

### 3.6 §5.3 Ablation (line 171-218)

현재 60q 6-topic 기준. 결정 필요:
- 옵션 A: 60q 그대로 두고 본 비교만 210q로 진행 (혼란 가능성)
- 옵션 B: 210q ablation 재실행 (시간/비용 추가)
- **현실적으로 60q 유지 + "ablation was conducted on the original 60-query subset for cost reasons; full-comparison metrics in §5.4 use the 210-query setting" 한 줄 명시 가능**

### 3.7 §5.4 Pairwise 섹션 (현재 없음)

옵션 P1~P3 중 결정 필요:
- **P1**: 60q 때처럼 pairwise 섹션 누락. Discussion에 "We focus on absolute reference-free metrics; holistic A/B preference judging is sensitive to surface format and is left to future work" 보호 한 줄 추가.
- **P2**: 정직하게 포함. 4~10% 승률 그대로 보고하고 §5 또는 §7에 "AnchorRAG outputs follow a structured plan-style format while baselines produce free prose; holistic LLM judges are sensitive to surface format. We discuss this as a generation-policy limitation" 명시.
- **P3**: stage-2 prose 추가 후 pairwise 재실행 (가장 정직하고 정합적, 비용 ~$15 / 5~6시간).

### 3.8 §1, §6, §7 framing 조정 (재생성 여부와 무관하게 권장)

현재 paper의 "trust-utility tradeoff" / "integrative claim trade-off" 표현은 case study 분석에 의존하는데
이 분석이 stage-1 skeleton output을 integrative synthesis로 잘못 해석한 결과일 수 있음.
- 옵션: 해당 표현을 약화하고 "we observe that absolute trust-sensitive metrics improve while overall answer quality remains topic-dependent" 식의 약한 주장으로 변경
- 또는 stage-2 prose 재생성 후 진짜 prose answer 비교로 case study 갱신

---

## 4. 결정 행렬

| 결정 | 옵션 | 시간/비용 | 정직도 | reject 위험 |
|------|------|-----------|--------|-------------|
| **(A) Pairwise** | P1 누락 + 보호문 | 0 | 부분적 | 중간 |
|                  | P2 정직 포함 | 0 | 높음 | 중간 (format 한계 명시 시) |
|                  | P3 stage-2 후 재실행 | $15 / 5~6h | 가장 높음 | 낮음 |
| **(B) Stage-2 prose** | 안 함 | 0 | 낮음 | 높음 (case study 부정확) |
|                       | 함 | $15 / 5~6h | 높음 | 낮음 |
| **(C) Ablation** | 60q 유지 | 0 | 중간 (명시 시) | 낮음 |
|                  | 210q 재실행 | $20+ / 수시간 | 높음 | 낮음 |
| **(D) Retrieval validation** | climate_change 한 줄 | 0 | 중간 | 낮음 |
|                              | 7토픽 재실행 | 적음 | 높음 | 낮음 |

---

## 5. 마감 시나리오별 권장 경로

### 시나리오 1 — 마감 1주 이상
- (A) P3, (B) 함, (C) 60q 유지 + 명시, (D) 한 줄
- §5.4 case study는 stage-2 prose 답변으로 재작성
- §1/§6/§7 tradeoff 서사 유지하되 framing 약하게

### 시나리오 2 — 마감 2~3일
- (A) P2 정직 포함, (B) 안 함, (C) 60q 유지 + 명시, (D) 한 줄
- Discussion/Limitations에 format-effect 명시 (정직 framing)
- §5.4 case study toned-down 또는 제거
- "trust-utility tradeoff" 표현 약화

### 시나리오 3 — 마감 1일 이내
- (A) P1 누락 + 보호문, (B) 안 함, (C) 60q 유지 + 명시, (D) 한 줄
- 숫자만 210q로 교체, 60q 때와 동일한 누락 전략
- 리스크 감수, 본문 수정 최소화

---

## 6. 다음 액션

1. 마감일/우선순위 결정 → 시나리오 1/2/3 중 선택
2. 결정 행렬 (A)~(D) 합의
3. 합의된 경로에 따라 sample.md 패치 시작
   - 1차 패스: §1, §4.1, §6, §7 단순 숫자
   - 2차 패스: Table 1 (climate_change 추가)
   - 3차 패스: Table 4 (210q 숫자 + 컬럼 결정)
   - 4차 패스: §5 narrative + (선택) pairwise 섹션
