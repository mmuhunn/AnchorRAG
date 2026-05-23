# Exp4 케이스 스터디 3종 정리

작성일: 2026-04-23

## 목적

본 문서는 Exp4의 `Downstream Answer Analysis`를 보강하기 위해 사용할 수 있는 세 가지 분석을 정리한 메모다.

정리 대상은 아래 세 가지다.

1. `Clean trust-win case`
2. `Trust-support trade-off case`
3. `Trust-advantaged slice analysis`

핵심 목표는 단순 aggregate 평균만 제시하는 것이 아니라,

- AnchorRAG의 trust 강점이 실제 query-level answer formation에서 어떻게 드러나는지
- 어떤 경우에는 그 강점이 support/utility와 공존하고
- 어떤 경우에는 generation trade-off로 이어지는지

를 보다 직관적으로 보여주는 것이다.

---

## 1. 케이스 1: Clean trust-win case

### 선정 쿼리

- Topic: `abortion`
- Query ID: `abo-03`
- 질문:
  - **What evidence is used to argue that abortion restrictions change overall abortion rates versus only shifting timing or geography?**

### 이 케이스를 고른 이유

이 쿼리는 AnchorRAG가

- `Citation Coverage`
- `High-Trust Citation Rate`

를 크게 끌어올리면서도,

- `Evidence Support Rate`
- `Groundedness`
- `Response Utility`

를 유지한 사례다.

즉, **"trust-aware retrieval이 answer quality를 항상 희생시키는 것은 아니다"**라는 점을 보여주기에 적합하다.

### 방법별 핵심 지표 비교

| Method | Citation Coverage | High-Trust Citation Rate | Evidence Support Rate | Groundedness | Response Utility |
|---|---:|---:|---:|---:|---:|
| `graph_retrieval` | 0.1667 | 0.5000 | 1.0000 | 0.72 | 0.68 |
| `trustrag_anchor` | 0.8000 | 1.0000 | 1.0000 | 0.72 | 0.68 |

### 직접 관찰되는 차이

#### `graph_retrieval`

- support는 높지만, citation이 붙은 evaluable claim 수가 적다.
- judged 결과상 `supported_sentences = 2`, `evaluable_sentences = 2`.
- 즉, **짧고 직접적인 local claim** 위주로 답변이 구성되었다.

#### `trustrag_anchor`

- trust가 더 높은 source를 더 많이 가져오고, 더 많은 문장에 explicit citation을 붙인다.
- judged 결과상 `supported_sentences = 4`, `evaluable_sentences = 4`.
- 즉, **support를 유지한 채 citation density와 citation trust quality를 동시에 높인 사례**다.

### 대표 retrieval 카드 비교

#### `graph_retrieval` 상위 카드

| Card | Source | S_trust | Query Relevance |
|---|---|---:|---:|
| 1 | The Hill | 0.7350 | 0.5248 |
| 2 | The Dallas Morning News | 0.7583 | 0.4705 |
| 3 | New York Post (News) | 0.7228 | 0.4629 |
| 4 | Financial Times | 0.7896 | 0.4505 |
| 5 | Wall Street Journal (News) | 0.7726 | 0.4420 |

#### `trustrag_anchor` 상위 카드

| Card | Source | S_trust | Query Relevance |
|---|---|---:|---:|
| 1 | NBC News (Online) | 0.7783 | 0.5175 |
| 2 | Fox News (Online News) | 0.8638 | 0.4441 |
| 3 | Axios | 0.9293 | 0.4129 |
| 4 | The Dallas Morning News | 0.7583 | 0.4705 |
| 5 | Financial Times | 0.7896 | 0.4505 |

### 해석

이 케이스는 다음 메시지를 잘 뒷받침한다.

- AnchorRAG의 장점은 단순히 citation을 많이 다는 것이 아니다.
- **더 높은 trust의 source를 answer backbone에 끌어오고**, 그 evidence를 answer 전반에 더 명시적으로 사용한다.
- 그럼에도 불구하고, query와 evidence alignment가 충분할 경우에는 support와 utility가 유지될 수 있다.

### 논문에서의 활용 포인트

본문에서는 다음처럼 쓸 수 있다.

- AnchorRAG의 trust-sensitive gain은 단순 aggregate artifact가 아니다.
- 일부 query에서는 graph retrieval과 같은 strong baseline 대비, **citation density와 trust quality를 크게 높이면서도 support를 유지하는 사례**가 관찰된다.

---

## 2. 케이스 2: Trust-support trade-off case

### 선정 쿼리

- Topic: `economy_jobs`
- Query ID: `eco-10`
- 질문:
  - **How do different ideological perspectives frame the balance between market freedom, worker protection, and government intervention in the economy?**

### 이 케이스를 고른 이유

이 쿼리는 AnchorRAG가 더 높은 trust를 갖는 source를 answer에 활용하지만,

- claim이 더 integrative하고 ideology-summary-heavy해지면서
- `support`, `groundedness`, `utility`가 약해지는

전형적인 trade-off 사례다.

즉, 논문의 약점을 숨기지 않으면서도 **trade-off의 원인이 retrieval이 아니라 generation policy에 있음**을 설명하기에 적합하다.

### 방법별 핵심 지표 비교

| Method | Citation Coverage | High-Trust Citation Rate | Evidence Support Rate | Groundedness | Response Utility |
|---|---:|---:|---:|---:|---:|
| `graph_retrieval` | 0.5263 | 0.1818 | 0.8000 | 0.78 | 0.82 |
| `trustrag_anchor` | 0.8571 | 0.6429 | 0.1667 | 0.72 | 0.68 |

### 직접 관찰되는 차이

#### `graph_retrieval`

- judged 결과상 `supported_sentences = 8`, `unsupported_sentences = 2`, `evaluable_sentences = 10`.
- 상대적으로 길지만, 각 문장이 특정 evidence에 직접 연결되는 local claim 비중이 높다.
- 그래서 support와 utility가 높게 유지된다.

#### `trustrag_anchor`

- judged 결과상 `supported_sentences = 1`, `unsupported_sentences = 5`, `evaluable_sentences = 6`.
- citation은 훨씬 많이 붙고, trust 높은 source도 더 많이 사용하지만,
- 문장이 “left/right/center perspective”를 포괄적으로 요약하는 형태로 바뀌면서 evaluator가 direct support를 잘 주지 못한다.

### 대표 retrieval 카드 비교

#### `graph_retrieval` 상위 카드

| Card | Source | S_trust | Query Relevance |
|---|---|---:|---:|
| 1 | Bustle | 0.6950 | 0.3364 |
| 2 | Wall Street Journal (News) | 0.7550 | 0.3093 |
| 3 | Fox Business | 0.6950 | 0.3139 |
| 4 | Slate | 0.7100 | 0.3186 |
| 5 | HuffPost | 0.6950 | 0.2993 |

#### `trustrag_anchor` 상위 카드

| Card | Source | S_trust | Query Relevance |
|---|---|---:|---:|
| 1 | New York Times (News) | 0.8522 | 0.2751 |
| 2 | Bustle | 0.6950 | 0.3364 |
| 3 | Wall Street Journal (News) | 0.7550 | 0.3093 |
| 4 | Wall Street Journal (News) | 0.7550 | 0.3011 |
| 5 | Slate | 0.7100 | 0.3186 |

### 왜 support가 무너졌는가

support check를 보면 AnchorRAG 쪽 answer는 다음과 같은 문장을 많이 만든다.

- “Right-leaning perspectives prioritize market freedom ...”
- “Center perspectives aim to balance economic growth with inflation control ...”
- “There is disagreement on the role of government intervention ...”

이런 문장들은 직관적으로는 plausible해 보이지만,

- 실제 카드가 그 ideological summary를 직접적으로 말하지 않거나
- 여러 카드의 맥락을 종합한 메타 요약이라

LLM judge가 `supported`보다 `unsupported`로 판정하기 쉽다.

즉, 이 사례는 다음 점을 보여준다.

- AnchorRAG의 약점은 **trust-rich retrieval failure**가 아니다.
- 핵심 bottleneck은 **retrieved evidence를 directly supportable claim으로 바꾸는 generation policy**다.

### 논문에서의 활용 포인트

본문에서는 다음처럼 쓸 수 있다.

- AnchorRAG의 trust-sensitive citation gain이 항상 support gain으로 이어지는 것은 아니다.
- 특히 경제/이념 framing처럼 추상적이고 관점 요약이 강한 질문에서는, anchor-guided generation이 더 cautious하고 integrative한 claim을 만들어 support/utility trade-off를 만들 수 있다.

---

## 3. 케이스 3: Trust-advantaged slice analysis

### 목적

개별 case study만 제시하면 “좋은 예시만 골라온 것 아니냐”는 인상을 줄 수 있다.  
이를 보완하기 위해, AnchorRAG의 trust 강점이 크게 발현된 query subset을 따로 묶어 보는 분석이 유용하다.

### 분석 정의

`graph_retrieval` 대비 `High-Trust Citation Rate` 향상이 가장 큰 상위 10개 query를 선택했다.

정렬 기준:

- 1차: `Δ High-Trust Citation Rate`
- 2차: `Δ Citation Coverage`

### 상위 10개 query 목록

| Topic | Query ID | 질문 요약 |
|---|---|---|
| abortion | `abo-07` | abortion policy와 adoption/child welfare |
| gun_control | `gun-02` | gun ownership rates와 public safety |
| gun_control | `gun-05` | permissive concealed-carry laws의 효과 |
| abortion | `abo-09` | parental notification/consent laws와 health outcomes |
| economy_jobs | `eco-08` | manufacturing decline 해석 차이 |
| immigration | `imm-09` | legal immigration pathways와 irregular migration |
| voting_rights_and_voter_fraud | `vote-10` | federal oversight vs state control |
| immigration | `imm-10` | sovereignty vs migrant/asylum seeker rights |
| free_speech | `speech-08` | rapid moderation vs viewpoint neutrality |
| abortion | `abo-03` | abortion restrictions: overall rate vs shift |

### 상위 10개 query 평균 비교

| Metric | `graph_retrieval` | `trustrag_anchor` |
|---|---:|---:|
| High-Trust Citation Rate | 0.1400 | 0.8674 |
| Citation Coverage | 0.1960 | 0.8133 |
| Evidence Support Rate | 0.7000 | 0.7467 |

### 해석

이 결과는 다음을 보여준다.

- AnchorRAG의 trust-sensitive gain은 몇 개의 극단적 outlier에만 의존하지 않는다.
- AnchorRAG의 메커니즘이 강하게 작동하는 query subset에서는,
  - 더 높은 trust source를 인용하고
  - 더 많은 factual sentence에 citation을 붙이며
  - support도 반드시 더 낮아지지 않는다.

즉, “AnchorRAG는 trust를 높이면 support가 항상 떨어진다”는 식으로 단순화하면 안 된다.

오히려 더 정확한 해석은 다음과 같다.

- **trust advantage가 강하게 발현되는 query 중 일부에서는, trust-sensitive evidence use와 support가 함께 유지되거나 동반 개선될 수 있다.**
- trade-off는 모든 query에서 자동으로 발생하는 것이 아니라,
  - 질문의 추상도
  - 관점 요약의 강도
  - 직접 empirical support의 유무

에 따라 달라진다.

### 논문에서의 활용 포인트

이 분석은 본문에서 2~3문장으로 짧게 언급하고, Appendix에 표로 빼는 것이 가장 적절하다.

예시 메시지:

- trust-sensitive gain은 단순 aggregate 평균에서만 보이는 것이 아니라,
- trust advantage가 큰 query subset에서도 support가 유지되거나 개선되는 사례가 존재한다.

---

## 4. 세 케이스를 함께 쓸 때의 메시지

위 세 가지를 묶으면 Exp4의 서술은 더 설득력 있게 정리된다.

### 정리 메시지

1. `abo-03`:
   - **trust gain과 support 유지가 동시에 가능한 positive case**
2. `eco-10`:
   - **trust gain이 generation trade-off로 이어지는 negative / mixed case**
3. Top-10 trust slice:
   - **AnchorRAG의 trust 강점이 outlier가 아니라 subset 수준에서도 반복된다는 보조 근거**

### 최종 해석

- AnchorRAG의 강점은 실제로 존재한다.
- 그 강점은 retrieval trust quality와 answer-level trust-sensitive citation behavior에서 매우 분명하다.
- 다만 support/utility는 generation policy에 의해 추가적인 trade-off를 받는다.
- 따라서 논문의 핵심 contribution은 **trust-aware retrieval + trust-sensitive evidence use**에 두고,
- answer-level mixed behavior는 **generation-stage limitation**으로 정직하게 설명하는 것이 가장 방어적이다.

