# 2026-05-14 — MSSR / Verifier Trade-off 정리

## 1. 배경

`trustrag_anchor`에서 unsupported 비율이 높게 나온 주된 패턴은 단순 hallucination이라기보다 다음 세 유형에 가까웠다.

- 근거 부족 또는 불확실성을 드러내는 메타 주장
- 여러 evidence를 통합적으로 요약하는 synthesis 문장
- 데이터 부재를 직접 주장하는 문장

기존 sentence-level support judge는 한 문장이 특정 evidence span에 의해 직접 지지되는지를 엄격하게 본다. 따라서 "empirical evidence is limited", "the evidence is mixed", "sources emphasize different aspects" 같은 문장은 신뢰성 관점에서는 바람직할 수 있지만, 단일 cited evidence가 그대로 말하지 않으면 `unsupported` 또는 `unclear`로 떨어질 수 있다.

이 문제를 확인하기 위해 두 방향의 실험/보강을 진행했다.

1. 평가 지표 보강: MSSR
2. generation 보강: verifier / v5 hybrid

---

## 2. MSSR: Synthesis-aware Support Metric

### 목적

MSSR는 기존 support metric이 integrative synthesis 문장을 과하게 벌주는지 확인하기 위한 보조 지표다.

기존 `SUPPORT_SYSTEM_PROMPT`는 문장을 다음 세 label로 평가한다.

- `supported`
- `unsupported`
- `unclear`

하지만 이 기준은 단일 evidence가 직접 말하지 않은 synthesis 문장에 불리했다. 그래서 `MSSR_SYSTEM_PROMPT`에서는 여러 source를 함께 읽었을 때 도출되는 정당한 synthesis claim에도 credit을 주도록 했다.

### MSSR에서 supported로 인정하는 예

- 직접 근거가 있는 claim
- 여러 source를 종합했을 때 드러나는 conflicting findings
- "evidence is mixed" 류의 epistemic summary
- "empirical evidence for X is limited" 같은 evidence-insufficiency claim
- Source A와 Source B의 차이를 비교하는 comparative conclusion

### 구현 위치

- `AllSides_Qbias/exp4/eval_exp4_allsides.py`
  - `SUPPORT_SYSTEM_PROMPT`
  - `MSSR_SYSTEM_PROMPT`

### 최종 210q 결과에서의 의미

최종 GPT-4o 210q 결과 기준:

| Method | Evidence Support Rate | MSSR |
|---|---:|---:|
| vanilla rag | 0.8522 | 0.8406 |
| graph retrieval | 0.8654 | 0.8353 |
| grag | 0.7904 | 0.7734 |
| AnchorRAG | 0.7719 | 0.7767 |

AnchorRAG의 MSSR은 일반 support보다 약간 높다. 이는 일부 문장이 strict support 기준에서는 불리하지만 synthesis-aware 기준에서는 일부 회복된다는 신호다. 다만 MSSR이 전체 순위를 뒤집을 만큼 크지는 않다. 따라서 이 결과는 "AnchorRAG가 general support metric에서도 우월하다"는 주장이 아니라, "strict sentence-level support metric이 trust-aware synthesis의 일부 행동을 완전히 포착하지 못한다"는 방어 논리로 쓰는 것이 적절하다.

---

## 3. Verifier / v5 Hybrid: Unclear와 Unsupported 문장 처리 실험

### 목적

MSSR가 평가 보정이라면, verifier 계열은 generation 자체에서 `unclear` 또는 `unsupported`가 될 만한 문장을 줄이려는 시도였다.

핵심 아이디어:

- 답변 전체를 다시 쓰지 않는다.
- 위험한 문장만 sentence-level로 감사한다.
- `supported` 문장은 유지한다.
- `unclear` 문장은 더 약한 표현으로 hedge/rewrite한다.
- `unsupported` 문장은 삭제하거나 uncertainty sentence로 교체한다.

### 설계 파일

- `docs/2026-04-03_claim_graph_verifier_loop_실험설계.md`
- `AllSides_Qbias/exp4/v5/2026-04-06_exp4_v5_실험설계.md`
- `AllSides_Qbias/exp4/run_verifier_proto_smoke10.py`

특히 `run_verifier_proto_smoke10.py`는 eval rubric을 그대로 재사용하지 않도록 설계했다. 즉 evaluator의 `supported / unsupported / unclear` label을 직접 최적화하지 않고, editorial action인 `keep / hedge / drop`을 사용했다. 이는 metric overfitting 비판을 피하기 위한 장치였다.

---

## 4. Smoke10 결과: Support는 오르지만 다른 지표가 하락

### Anchorfix smoke10

| Metric | Value |
|---|---:|
| Groundedness | 0.725 |
| Evidence Support Rate | 0.920 |
| MSSR | 0.8483 |
| Citation Coverage | 0.6862 |
| High-Trust Citation Rate | 0.8417 |
| Response Utility | 0.675 |
| Trust-Weighted Claim Support | 0.6968 |
| Backbone Trust Rate | 0.8333 |

### Verifier smoke10

| Metric | Value |
|---|---:|
| Groundedness | 0.695 |
| Evidence Support Rate | 1.000 |
| MSSR | 0.9667 |
| Citation Coverage | 0.5048 |
| High-Trust Citation Rate | 0.7467 |
| Response Utility | 0.645 |
| Trust-Weighted Claim Support | 0.7686 |
| Backbone Trust Rate | 0.7667 |

### 해석

Verifier는 의도대로 support 계열을 크게 올렸다.

- ESR: 0.920 → 1.000
- MSSR: 0.8483 → 0.9667
- TW Claim Support: 0.6968 → 0.7686

하지만 동시에 다른 중요한 지표가 내려갔다.

- Citation Coverage: 0.6862 → 0.5048
- High-Trust Citation Rate: 0.8417 → 0.7467
- Response Utility: 0.675 → 0.645
- Backbone Trust Rate: 0.8333 → 0.7667

즉 verifier는 sentence-level support를 개선했지만, 답변이 더 짧고 보수적으로 변하면서 citation behavior와 utility를 깎았다. 이 때문에 최종 메인 모델로 채택하지 않았다.

---

## 5. v5 Hybrid 결과 해석

v5 hybrid는 verifier 아이디어를 더 자연스러운 post-hoc trust editing 형태로 확장하려는 시도였다.

핵심 설계:

- AnchorRAG retrieval은 유지
- baseline처럼 직접적인 초안을 작성
- trust audit으로 위험 문장만 점검
- unsupported / unclear 문장만 약화하거나 교체
- 답 전체를 evidence briefing처럼 재작성하지 않음

하지만 v5 결과도 최종 메인으로 쓰기에는 좋지 않았다.

주요 메모:

- `AllSides_Qbias/exp4/v5/2026-04-06_exp4_v5_run_짧은메모.md`

v5 run memo 기준:

| Method | Groundedness | Evidence Support Rate | Citation Coverage | High-Trust Citation Rate | Utility |
|---|---:|---:|---:|---:|---:|
| trustrag_hybrid_v5 | 0.6590 | 0.8028 | 0.9444 | 0.6521 | 0.6143 |
| trustrag_anchor | 0.7537 | 0.5635 | 0.6994 | 0.5255 | 0.7213 |
| vanilla rag | 0.7907 | 0.8192 | 0.3206 | 0.4152 | 0.7827 |

v5는 citation coverage와 high-trust citation을 높였지만, groundedness와 utility가 크게 낮아졌다. Pairwise에서도 `trustrag_hybrid_v5`는 `trustrag_anchor`에 1승 29패, `vanilla_rag`에 0승 30패였다.

결론적으로 v5는 useful diagnostic run이지만 final improvement는 아니었다.

---

## 6. 교수님 설명용 핵심 요약

이 실험들의 결론은 다음과 같다.

1. AnchorRAG의 unsupported 문장은 단순 hallucination만이 아니라, evidence-insufficiency / synthesis / data absence claim에서 많이 발생했다.
2. 이를 보정하기 위해 MSSR를 도입했고, MSSR는 synthesis-aware claim에 credit을 주도록 설계했다.
3. Generation 쪽에서도 verifier와 v5 hybrid로 unclear/unsupported 문장을 hedge/drop하려 했지만, support는 올라가는 대신 utility, citation coverage, high-trust citation이 내려가는 trade-off가 발생했다.
4. 따라서 최종 논문에서는 verifier를 메인 모델로 채택하지 않고, AnchorRAG를 trust-sensitive evidence organization framework로 framing했다.
5. MSSR와 verifier 결과는 "왜 strict support metric만으로는 trust-aware synthesis를 완전히 평가하기 어렵나"를 설명하는 방어 근거로 쓰는 것이 적절하다.

한 문장 요약:

> Strict support를 올리는 방법은 있었지만, 그 방법은 답변을 더 보수적이고 짧게 만들어 citation behavior와 utility를 깎았다. 그래서 최종 모델은 support-only optimization이 아니라 trust-sensitive evidence use를 목표로 framing했다.

---

## 7. Support 문제와 Groundedness 문제의 차이

기존 MSSR / verifier / v5 실험은 주로 `support` 또는 `unsupported` 문제를 해결하기 위한 실험이었다. 하지만 최종 결과에서 `groundedness`가 낮게 나오는 이유는 support 문제와 완전히 같지 않다.

### 7.1 Support는 문장 단위 정합성 문제

`Evidence Support Rate`는 개별 answer sentence가 cited evidence에 의해 직접 지지되는지를 본다.

따라서 support가 낮아지는 주요 원인은 다음과 같다.

- 문장이 cited evidence보다 더 일반화됨
- 여러 evidence를 통합한 문장이 단일 span으로 직접 지지되지 않음
- evidence-insufficiency claim이 특정 cited span에 딱 매칭되지 않음
- data absence claim이 evidence에 명시적으로 들어 있지 않음

MSSR는 이 문제를 완화하기 위해 도입된 보조 지표다. 즉 valid synthesis claim이나 evidence-insufficiency claim이 여러 evidence를 함께 읽었을 때 정당하다면 supported로 인정하려는 목적이었다.

Verifier / v5 hybrid도 같은 문제를 generation 쪽에서 줄이려는 시도였다. 위험 문장을 `hedge`하거나 `drop`해서 unsupported sentence를 줄이고자 했다.

### 7.2 Groundedness는 answer-level coverage와 specificity 문제

반면 `Groundedness`는 단순히 sentence-level support만 보지 않는다. judge는 답변 전체가 질문에 대해 얼마나 직접적이고, 구체적이고, 충분한 evidence-backed synthesis를 제공하는지도 함께 본다.

AnchorRAG에서 groundedness가 낮게 나오는 이유는 다음 요인이 섞여 있다.

1. **Cautious answer policy**
   - AnchorRAG는 high-trust evidence를 우선하고, 직접 근거가 약하면 강한 결론을 피한다.
   - 이로 인해 "available evidence is limited" 또는 "retrieved evidence does not directly establish X" 같은 cautious claim이 늘어난다.
   - 이는 hallucination을 줄이는 방향이지만, groundedness judge에게는 질문에 충분히 답하지 못한 것으로 보일 수 있다.

2. **Card / excerpt compression**
   - 코드상 generator는 raw article 전체가 아니라 evidence card와 짧은 excerpt를 본다.
   - anchor도 raw article 전체가 아니라 anchor metadata block과 `Community anchor: Yes`가 붙은 evidence card로 제공된다.
   - 따라서 baseline보다 사용할 수 있는 세부 evidence가 줄고, 답변의 specificity와 facet coverage가 약해질 수 있다.

3. **Anchor-centered backbone**
   - AnchorRAG는 community anchor를 중심으로 답변 backbone을 구성한다.
   - 이 구조는 high-trust backbone evidence use에는 유리하지만, baseline처럼 여러 관점과 세부 근거를 넓게 섞는 breadth에는 불리할 수 있다.

4. **Trust-first selection**
   - high-trust evidence를 우선하면서, 더 직접적이지만 lower-trust인 evidence가 덜 사용될 수 있다.
   - 이 경우 trust-sensitive metric은 오르지만, general groundedness judge가 선호하는 direct answer coverage는 낮아질 수 있다.

### 7.3 교수님 설명용 구분

Support 문제와 groundedness 문제는 다음처럼 구분해서 설명하는 것이 좋다.

| 구분 | 핵심 질문 | AnchorRAG에서의 이슈 | 대응 실험 |
|---|---|---|---|
| Support | 이 문장이 cited evidence로 직접 지지되는가? | synthesis / uncertainty / data absence 문장이 strict support에서 불리 | MSSR, verifier, v5 hybrid |
| Groundedness | 답변 전체가 질문에 충분히 직접적이고 구체적인 evidence-backed synthesis를 제공하는가? | trust-first / cautious / card-compressed generation 때문에 breadth와 specificity 감소 | 최종 논문에서는 trade-off로 framing |

핵심 표현:

> Support 하락은 claim-evidence alignment 문제이고, groundedness 하락은 answer-level coverage, directness, specificity 문제에 가깝다. 기존 verifier/MSSR 실험은 주로 전자를 해결하려는 시도였고, 후자는 trust-first generation의 구조적 trade-off로 남았다.

따라서 groundedness 하락을 설명할 때는 "unsupported claim이 많아서"라고만 말하면 부족하다. 더 정확히는 다음과 같이 말하는 것이 좋다.

> AnchorRAG는 hallucination이 늘어서 groundedness가 낮아진 것이 아니라, high-trust evidence와 anchor backbone을 우선하면서 답변이 더 cautious하고 compressed해졌고, 그 결과 general groundedness judge가 선호하는 breadth, specificity, directness가 baseline보다 약해졌다.

---

## 8. 관련 파일

- Trade-off 정리:
  - `docs/2026-04-13_exp4_6topics_tradeoff_analysis.md`
- MSSR 구현:
  - `AllSides_Qbias/exp4/eval_exp4_allsides.py`
- generation trust metric:
  - `AllSides_Qbias/exp4/analyze_generation_trust_metrics.py`
- verifier 설계:
  - `docs/2026-04-03_claim_graph_verifier_loop_실험설계.md`
- verifier smoke:
  - `AllSides_Qbias/exp4/run_verifier_proto_smoke10.py`
  - `AllSides_Qbias/exp4/eval_verifier_smoke10_gpt4o_20260507.csv`
- v5 hybrid 설계/결과:
  - `AllSides_Qbias/exp4/v5/2026-04-06_exp4_v5_실험설계.md`
  - `AllSides_Qbias/exp4/v5/2026-04-06_exp4_v5_run_짧은메모.md`
