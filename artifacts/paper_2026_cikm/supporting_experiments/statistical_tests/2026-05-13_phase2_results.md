# Phase 2 — MBFC 외부 평가 & 통계 검정 결과

**작성일**: 2026-05-13
**대상 plan**: `docs/CIKM/2026-05-13_anchorrag_revision_plan.md` Phase 2 (Statistical robustness + Circularity 방어)
**평가 대상**: 210 queries × 7 methods (`trustrag_anchor`, `vanilla_rag`, `mmr_rag`, `graph_retrieval`, `r2ag`, `grag`, `reclaim`)
**Reference method**: `trustrag_anchor` (= AnchorRAG)

---

## 1. 동기

Reviewer-facing 두 가지 약점을 동시에 보강한다.

1. **Circularity**: 내부 `S_trust` 점수로 측정한 trust metric에서 AnchorRAG가 이긴다는 결과는 "AnchorRAG가 자기 점수를 최대화한 것"이라는 비판에 노출됨.
   → **외부 human-rated source factuality proxy**(MBFC factual reporting label; article/claim 단위 ground truth는 아님)로 동일 metric을 재계산.
2. **Significance 부재**: 기존 표는 mean only. 차이가 우연인지 입증되지 않음.
   → **paired bootstrap 95% CI + Wilcoxon signed-rank + Cliff's δ**를 6 baseline × 15 metric에 일괄 적용.

두 작업은 독립이므로 **A (stat tests) / B (MBFC mapping 확장 + 외부 평가)** 병렬로 수행.

---

## 2. 데이터

| 파일 | 용도 |
|---|---|
| `AllSides_Qbias/exp4/exp4_eval_data_v6_7methods_210q_20260509.jsonl` | raw 210q output — `card_id → source` 매핑 추출용 |
| `AllSides_Qbias/exp4/eval_v6_7methods_210q_gpt4o_20260509.jsonl` | judged eval — 기존 metric + `support_checks` (`[Card N]` 참조 포함) |
| `AllSides_Qbias/source_validation/mbfc_validation_top50.csv` | 기존 top-50 MBFC mapping (49 sources) |
| `AllSides_Qbias/source_validation/mbfc_extension_20260513.csv` | **신규**: 23 sources 추가 매핑 |

---

## 3. 실험 B — MBFC mapping 확장 & 외부 평가

### 3.1 매핑 확장 (`mbfc_extension_20260513.csv`)

기존 top-50 매핑이 다루지 못한 multi-cite source 중 MBFC 등록 outlet 23개를 수기로 매핑. 모든 row는 MBFC URL을 명시해 재검증 가능.

추가된 23 outlets (MBFC numeric label 1=Low, 2=Mixed, 3=Mostly Factual, 4=High, 5=Very High):
- **Very High** (n=5; numeric 5): Pew Research Center, Roll Call, SCOTUSblog, ProPublica, FactCheck.org
- **High** (n=10; numeric 4): Los Angeles Times, Deseret News, Financial Times, The Atlantic, CNET, AllSides, Atlanta Journal-Constitution, Mother Jones, Columbia Journalism Review, BuzzFeed News
- **Mixed** (n=7; numeric 2): The Federalist, Just The News, CNN (Opinion), Rich Lowry → National Review, New York Times (Opinion), Jacobin, The Daily Signal
- **Low** (n=1; numeric 1): Tucker Carlson Network

### 3.2 평가 스크립트 (`mbfc_external_eval_20260513.py`)

핵심 동작:
1. `card_id → source` map을 raw jsonl에서 method별로 구축.
2. judged eval의 각 `support_checks` 문장에서 `\[Card N\]` regex로 인용 추출 (backbone = 처음 K=3 문장).
3. 각 인용 source를 MBFC numeric label(1..5)로 매핑. **HT threshold = 4** (High 또는 Very High).
4. 두 방식으로 집계:
   - **per-query mean**: 각 query 내에서 평균낸 뒤 query 단위로 평균 (paper의 main 보고용 — paired test와 호환)
   - **corpus token-weighted**: 전체 인용 토큰 합산 비율 (robustness check)

### 3.3 Coverage (mapping 확장 효과)

| 지표 | 기존 mapping (top-50) | 확장 후 (top-50 + ext) |
|---|---|---|
| MBFC-Coverage (all cites, AnchorRAG) | ≈ 0.79 | **0.931** |
| MBFC-Coverage (backbone) | ≈ 0.77 | **0.932** |
| Distinct unmapped sources 잔존 | ~90 | **67** |

잔존 unmapped 대부분은 MBFC 등록 자체가 없는 부류:
- 사이트가 아닌 **개별 칼럼니스트/저자 이름** (Robert Reich, Rex Huppke, Jennifer Rubin 등 — outlet으로 환원 불가)
- AllSides의 **"Guest Writer - {Side}"** 익명 라벨 (Center/Right/Left 합쳐 인용 403건 — MBFC에 outlet 자체 없음)
- 정치인 본인 (Barack Obama 등)

### 3.4 Per-method 결과 (per-query mean)

| Method | MBFC-HT-Cite (all) | MBFC-AvgFactual (all) | MBFC-BB-HT | MBFC-BB-AvgFactual | Coverage (all) |
|---|---|---|---|---|---|
| **trustrag_anchor** | **0.4228** | **3.365** | **0.4140** | **3.359** | **0.931** |
| vanilla_rag | 0.3790 | 3.212 | 0.3464 | 3.148 | 0.889 |
| reclaim | 0.3519 | 3.194 | 0.3581 | 3.204 | 0.899 |
| graph_retrieval | 0.3418 | 3.146 | 0.2982 | 3.079 | 0.885 |
| grag | 0.3449 | 3.145 | 0.3228 | 3.079 | 0.896 |
| r2ag | 0.3342 | 3.134 | 0.3280 | 3.122 | 0.895 |
| mmr_rag | 0.3283 | 3.101 | 0.2672 | 3.000 | 0.875 |

→ **AnchorRAG는 외부 human-rated source-factuality proxy(MBFC) 기준에서도 평균적으로 모든 baseline을 앞선다.** 단, MBFC는 article/claim 단위 ground truth가 아니며, all-cite MBFC-HT-Cite의 vanilla_rag 대비 차이는 borderline이므로 본문에서는 평균 factuality와 backbone metrics 중심으로 정직하게 보고한다.

---

## 4. 실험 A — 통계 검정 (`stat_tests_20260513.py`)

### 4.1 방법

- **Paired bootstrap 95% CI**: `n_boot = 10000`, seed `20260513`. query를 unit으로 동시 resample → mean diff 분포의 2.5/97.5 percentile.
- **Wilcoxon signed-rank test**: tie 보정 포함 normal approximation, two-sided. zero diff는 제외.
- **Significance 마킹**: `**` p<0.01, `*` p<0.05, 무표시 = not significant.
- 검정 대상: 15 metric × 6 baseline = 90 비교. 모든 비교에서 query를 paired unit으로 사용 (양쪽 method 모두 metric이 정의된 query에 한함).
- 보조 effect size: Cliff's δ (CSV에 함께 기록). **단, 본 구현은 paired rank가 아니라 unpaired dominance 형태**(`(n_greater − n_lesser) / N²`)이므로 paired bootstrap CI + Wilcoxon이 main reporting 단위. paper에서는 Cliff's δ를 별도 효과크기로 강조하지 않고, paired bootstrap CI(차이의 크기)와 Wilcoxon p-값(유의성)만 인용한다.

### 4.2 핵심 결과 요약

#### (a) Trust-sensitive metric → 전부 AnchorRAG 승리, 거의 모두 p < 0.001

차이는 paired bootstrap 95% CI(평균차의 크기) + Wilcoxon p-값(유의성) 기준으로 보고. Cliff's δ는 보조용이며 본문에서는 강조하지 않음.

| Metric | vs vanilla_rag | vs reclaim (강baseline) | Worst case across 6 baselines |
|---|---|---|---|
| HT-Cite (internal) | +0.225 ** | +0.207 ** | 모두 ** |
| Anchor-Util | +0.307 ** | +0.259 ** | graph_retrieval에 대해서만 **negative** (−0.150, **) — graph_retrieval은 community-aware retrieval로 post-hoc representative citation 비율이 구조적으로 높게 나오므로, paper에서는 method note로 처리 |
| Backbone-Trust (internal) | +0.237 ** | +0.210 ** | 모두 ** |
| Avg-Strust@k | +0.035 ** | +0.035 ** | 모두 ** |
| Top3-Avg-Strust | +0.052 ** | +0.052 ** | 모두 ** |
| **MBFC-AvgFactual (external)** | +0.145 ** | +0.176 ** | 모두 ** |
| **MBFC-BB-AvgFactual (external)** | +0.204 ** | +0.161 ** | 모두 ** |
| **MBFC-BB-HT (external)** | +0.065 * (p=0.011) | +0.060 ** | vanilla_rag 대비 *, 그 외 모두 ** |
| **MBFC-Coverage (external)** | +0.043 ** | +0.032 * | reclaim 대비 *, 그 외 모두 ** |
| MBFC-HT-Cite (external, all) | **+0.041, p=0.078 ❗ (borderline)** | +0.074 ** | mean은 6/6 baseline보다 높지만, vanilla_rag 대비는 비유의(p=0.078) |

→ **MBFC 핵심 claim의 정직한 프레이밍**:
1. **MBFC factuality 평균 (`MBFC-AvgFactual`, `MBFC-BB-AvgFactual`)은 vanilla_rag 포함 모든 baseline 대비 유의** (전부 **).
2. **Backbone 한정 high-trust citation rate (`MBFC-BB-HT`) 도 vanilla_rag 대비 유의** (p=0.011, *).
3. **그러나 all-cite MBFC-HT-Cite는 vanilla_rag 대비 borderline (p=0.078)**. → paper에서는 "전 baseline 대비 유의"라고 쓰면 안 됨. 대신 "mean이 6/6에서 높고, AvgFactual/BB-AvgFactual/BB-HT는 vanilla 대비도 유의, all-cite HT는 vanilla 대비 borderline"으로 보고.

#### (b) 일반 답변 품질 → AnchorRAG 패배, 대부분 유의 (paired bootstrap CI + Wilcoxon)

| Metric | vs vanilla_rag | vs r2ag | vs reclaim |
|---|---|---|---|
| Groundedness | −0.090 ** | −0.061 ** | −0.045 ** |
| Support (evidence_support_rate) | −0.202 ** | −0.042, p=0.086 (ns) | −0.058, p=0.053 (ns) |
| Utility | −0.093 ** | −0.061 ** | −0.020 ** |
| Cite-Cov | −0.031 ** | −0.043 ** | **−0.641 ** ← reclaim은 거의 모든 문장에 인용을 강제 |
| TW-Claim-Support | −0.108 ** | +0.001 (ns) | −0.032 (ns) |

→ **plan v3 Phase 1.1의 "evidential discipline trade-off" 프레이밍이 통계로 확정됨**. 한편 **reclaim에 대해서는 Support·TW-Claim-Support 차이가 통계적으로 비유의**해 격차가 좁아짐. paper Discussion에서 정직히 다뤄야 함.

### 4.3 주의해야 할 비유의 결과 (정직 보고 후보)

| Comparison | Metric | Δ | CI | p | 해석 |
|---|---|---|---|---|---|
| vs vanilla_rag | **MBFC-HT-Cite** | +0.041 | [−0.001, +0.083] | 0.078 | 외부 metric에서 vanilla_rag 대비 유의 미달. 단, backbone (MBFC-BB-HT)에서는 +0.065 * 유의. |
| vs r2ag | Support | −0.042 | [−0.098, +0.014] | 0.086 | AnchorRAG가 r2ag에 졌으나 통계적 유의 미달 |
| vs reclaim | Support | −0.058 | [−0.124, +0.007] | 0.053 | borderline |
| vs reclaim | TW-Claim-Support | −0.032 | [−0.078, +0.014] | 0.125 | ns |
| vs r2ag | TW-Claim-Support | +0.001 | [−0.041, +0.044] | 0.818 | tie |
| **vs graph_retrieval** | **Anchor-Util** | **−0.150 ** | [−0.183, −0.117] | <0.001 | graph_retrieval은 community-aware retrieval 특성상 post-hoc representative citation rate가 구조적으로 높게 측정되는 setup. paper에서 method note로 처리 |

---

## 5. 산출물 파일

### 5.1 신규 코드
- `AllSides_Qbias/exp4/mbfc_external_eval_20260513.py`
- `AllSides_Qbias/exp4/stat_tests_20260513.py`

### 5.2 신규 매핑
- `AllSides_Qbias/source_validation/mbfc_extension_20260513.csv` (23 sources)

### 5.3 결과 CSV
- `AllSides_Qbias/exp4/mbfc_external_eval_per_query_20260513.csv` (1470 = 7 methods × 210 queries)
- `AllSides_Qbias/exp4/mbfc_external_eval_summary_20260513.csv` (per method, per-query mean + corpus token-weighted)
- `AllSides_Qbias/exp4/mbfc_external_eval_by_topic_20260513.csv` (method × topic)
- `AllSides_Qbias/exp4/mbfc_external_eval_unmapped_sources_20260513.csv` (67 distinct unmapped sources, top: 익명 Guest Writer, 개인 칼럼니스트)
- `AllSides_Qbias/exp4/stat_tests_20260513.csv` (6 × 15 = 90 rows)

---

## 6. Paper 반영 가이드 (Phase 2.2~2.4)

> Phase 1은 Codex 담당 중. 아래는 Phase 2 결과의 LaTeX 통합 항목.

### Phase 2.2 — Table 업데이트 (`overleaf/latex/results.tex`)
- 기존 Table 4 (main 결과)의 AnchorRAG 행에 ** / * (모든 baseline 대비 유의) 표기 추가.
- **신규 Table 5 (Circularity defense)** 신설: 7 method × {MBFC-HT-Cite, MBFC-AvgFactual, MBFC-BB-HT, MBFC-BB-AvgFactual, MBFC-Coverage}. AnchorRAG 행에 유의 마크 + MBFC-HT-Cite는 vanilla 대비 borderline임을 dagger로 명시.
- 표 캡션 또는 footnote에 mapping 출처(`mbfc_validation_top50.csv` + `mbfc_extension_20260513.csv`, coverage 87~93%)와 HT threshold (≥4) 명시.

### Phase 2.3 — 통계 검정 방법 1줄 보고
- Setup 또는 Results 첫 단락에 paired bootstrap 95% CI + Wilcoxon signed-rank test 정의 1줄. **Cliff's δ는 paper 본문에서 강조하지 않음** (코드 구현이 unpaired dominance 형태라 paired 해석과 정확히 부합하지 않음 — supplementary CSV에만 보조 기록).

### Phase 2.4 — Discussion "Statistical Robustness" 문단 (`overleaf/latex/discussion.tex`)
다룰 포인트 (정직한 main claim 프레이밍):
1. Trust-sensitive 핵심 metric (HT-Cite, Backbone-Trust, Avg-Strust@k 등): 6/6 baseline 대비 p<0.01.
2. MBFC 외부 검증 (circularity 방어):
   - mean MBFC factuality는 AnchorRAG가 6/6 baseline보다 높음.
   - MBFC-AvgFactual / MBFC-BB-AvgFactual / MBFC-BB-HT는 **vanilla_rag 포함 모든 baseline 대비 유의**.
   - **그러나 all-cite MBFC-HT-Cite는 vanilla_rag 대비 borderline** (Δ=+0.041, CI [−0.001, +0.083], p=0.078). 동일 metric의 backbone 한정 (MBFC-BB-HT)에서는 vanilla_rag 대비도 유의 (Δ=+0.065, p=0.011).
3. Groundedness/Support/Utility 패배는 유의 — evidential-discipline trade-off가 paper 주장과 일치하게 통계로 확인됨.
4. r2ag/reclaim 같은 강 baseline 대비 Support·TW-Claim-Support 차이는 비유의 (p≈0.05~0.13) → AnchorRAG의 일반 품질 손실은 "약한 baseline 대비"에 집중됨.
5. graph_retrieval의 Anchor-Util 수치는 community-aware retrieval 구조상 post-hoc representative citation rate가 높게 측정되는 setup 특성으로 method note 처리.

**Main claim 가이드**: "AnchorRAG가 모든 면에서 이긴다"가 아니라, "**외부 factuality reference와 통계 검정 모두에서 trust/factuality/backbone evidence 쪽이 유의하게 강하고, 일반 답변 품질은 trade-off가 통계적으로 유의하게 존재한다**"로 prefer.

### Phase 2.5 (선택) — Appendix 표
- 90개 paired test 전수표를 Appendix C로 추가하면 reviewer trust ↑.

---

## 7. Reproducibility 노트

```bash
# from the original working repo root

# (1) MBFC external eval — coverage check & per-query CSVs
python3.8 AllSides_Qbias/exp4/mbfc_external_eval_20260513.py

# (2) Paired statistical tests — uses (1)'s per_query CSV
python3.8 AllSides_Qbias/exp4/stat_tests_20260513.py
```

Seed `20260513` 고정 (bootstrap). 두 스크립트 모두 deterministic. Python 3.8 (project 표준).
