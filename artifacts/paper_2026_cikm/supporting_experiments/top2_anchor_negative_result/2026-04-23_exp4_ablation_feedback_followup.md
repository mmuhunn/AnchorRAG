# Exp4 / Ablation 피드백 대응 정리

작성일: 2026-04-23

## 목적

최근 피드백의 핵심은 두 가지다.

1. Exp4에서 AnchorRAG의 `trust` 강점을 더 직관적이고 설득력 있게 보여줄 추가 보강이 필요하다.
2. Ablation study에서 각 지표가 서로 다른 variant에서 최고값을 보이는 이유를 더 명확히 분석해야 한다.

본 문서는 현재 확보된 6-topic AllSides/QBias 결과를 기준으로, **추가 실험 없이 바로 만들 수 있는 보강안**과 **필요 시 가볍게 추가 분석하면 되는 보강안**을 정리한다.

---
# 최종 방향향
- Exp4 본문: abo-03 + top-10 trust slice
- Exp4 appendix: eco-10 trade-off case
- Ablation 본문: behavior decomposition + gun_control 메커니즘 설명
- Ablation appendix: topic-wise winner matrix + top-2 anchor + claim-style examples

## 1. Exp4 보강 방향

### 1.1 핵심 방향

Exp4는 더 이상 “AnchorRAG가 모든 answer metric에서 가장 좋다”는 식으로 밀기 어렵다.  
대신 다음 메시지로 정리하는 것이 가장 설득력 있다.

- AnchorRAG는 **higher-trust evidence retrieval**과 **trust-sensitive evidence use**에서 가장 강하다.
- 이 강점은 단순 aggregate 수치뿐 아니라, **실제 query-level answer formation 과정**에서도 확인된다.
- 다만, trust-first generation policy는 일부 query에서 **support / utility trade-off**를 유발한다.

즉, Exp4 보강은 아래 두 축으로 가는 것이 좋다.

- **Positive trust-win case study**
- **Trade-off case study**

---

## 2. Exp4 케이스 스터디 구성안

### 2.1 추천 구성

본문 혹은 appendix에 아래 2개 케이스를 넣는 것이 좋다.

1. **Clean trust-win case**
2. **Trust-support trade-off case**

각 케이스는 다음 4요소로 구성하면 된다.

- Query
- Top retrieved evidence cards 비교 (`graph_retrieval` vs `trustrag_anchor`)
- Final answer excerpt 비교
- Query-level metric mini-table

이 구성이 좋은 이유는, retrieval과 generation을 한 화면에서 같이 보여줄 수 있기 때문이다.

---

## 3. 추천 케이스 1: Clean trust-win case

### Query 후보

- Topic: `abortion`
- Query ID: `abo-03`
- Question:
  - *What evidence is used to argue that abortion restrictions change overall abortion rates versus only shifting timing or geography?*

### 이 케이스를 추천하는 이유

이 쿼리는 AnchorRAG가 **trust-sensitive citation behavior를 크게 강화하면서도**, `support`, `groundedness`, `utility`를 거의 희생하지 않은 사례다.

### Query-level 비교

`graph_retrieval` vs `trustrag_anchor`

- Citation Coverage:
  - `0.1667 -> 0.8000`
- High-Trust Citation Rate:
  - `0.5000 -> 1.0000`
- Evidence Support Rate:
  - `1.0000 -> 1.0000`
- Groundedness:
  - `0.72 -> 0.72`
- Response Utility:
  - `0.68 -> 0.68`

### 해석

이 예시는 다음 메시지를 강하게 뒷받침한다.

- AnchorRAG는 단지 citation을 많이 다는 것이 아니라,
- **higher-trust source를 더 많이 answer backbone에 끌어오고**,
- 동시에 **query-relevant direct support를 유지할 수 있는 경우가 존재한다.**

즉, “trust-aware retrieval이 answer quality를 항상 희생시키는 것은 아니다”라는 반례로 매우 유용하다.

### 본문에서 강조할 포인트

- `graph_retrieval`도 support는 높지만 citation density가 낮다.
- `trustrag_anchor`는 더 많은 factual claim에 citation을 달면서도 support를 유지한다.
- 따라서 AnchorRAG의 장점은 “citation quantity”가 아니라 **trust-aware explicit evidence use**라고 해석할 수 있다.

---

## 4. 추천 케이스 2: Trade-off case

### Query 후보

1차 추천:

- Topic: `economy_jobs`
- Query ID: `eco-10`
- Question:
  - *How do different ideological perspectives frame the balance between market freedom, worker protection, and government intervention in the economy?*

대안:

- Topic: `immigration`
- Query ID: `imm-01`

### 이 케이스를 추천하는 이유

이 사례는 AnchorRAG가 더 신뢰도 높은 source를 활용하지만, answer가 더 **cautious / integrative / ideology-summary-heavy**해지면서 support나 utility가 약해지는 전형적인 trade-off를 보여준다.

### 해석 포인트

- AnchorRAG는 higher-trust sources를 더 적극적으로 선택한다.
- 그러나 final answer가 “직접 지지 가능한 로컬 claim”보다 “통합적 요약”으로 이동한다.
- 그 결과:
  - `support`는 낮아질 수 있고
  - `utility`도 덜 시원하게 느껴질 수 있다.

이 케이스는 논문의 약점을 드러내는 것이 아니라, 오히려 Discussion의 trade-off 서술을 정당화하는 장치로 쓸 수 있다.

### 본문에서의 역할

이 케이스는 다음 메시지를 보조한다.

- AnchorRAG의 한계는 retrieval failure보다는
- **generation policy가 trust-rich evidence를 직접 supportable claim으로 바꾸는 과정**에 있다.

즉, 문제의 위치를 retrieval이 아니라 generation policy에 두는 데 도움이 된다.

---

## 5. 추가 보강안: Trust-advantaged slice analysis

개별 케이스 스터디만으로는 “좋은 예시 cherry-pick”처럼 보일 수 있다.  
이를 보완하기 위해, 작은 subset analysis를 하나 추가하는 것이 좋다.

### 제안

`graph_retrieval` 대비 `\Delta` High-Trust Citation Rate가 가장 큰 상위 10개 query를 따로 모아 비교한다.

### 현재 관찰값

상위 10개 trust-gain query에서 평균적으로:

- High-Trust Citation Rate:
  - `graph_retrieval`: `0.140`
  - `trustrag_anchor`: `0.867`
- Citation Coverage:
  - `graph_retrieval`: `0.196`
  - `trustrag_anchor`: `0.813`
- Evidence Support Rate:
  - `graph_retrieval`: `0.700`
  - `trustrag_anchor`: `0.747`

### 의미

이 subset에서는 AnchorRAG의 핵심 메커니즘이 강하게 발현될 때,

- trust-sensitive citation gain이 매우 크고,
- support가 반드시 무너지지 않는다는 점을 보일 수 있다.

### 논문에서의 위치

- 본문에 2~3문장 요약
- Appendix에 subset table 추가

이 정도가 가장 안전하다.

---

## 6. Ablation이 중구난방처럼 보이는 이유

현재 ablation table은 “어느 variant가 모든 지표에서 이긴다”는 구조가 아니다.  
그 이유는 각 variant가 서로 다른 objective를 더 강하게 최적화하기 때문이다.

### 핵심 해석

- `trustrag_anchor`
  - trust-sensitive evidence selection과 backbone trust usage에 강함
- `wo_anchor`
  - 짧고 직접적인 claim을 만들기 쉬워 support-oriented metric에 강함
- `wo_trust`
  - relevance-heavy retrieval 덕분에 direct support가 잘 붙는 경우가 있음
- `wo_diversity`
  - answer는 깔끔해 보일 수 있으나 evidence set이 좁아져 support는 약해질 수 있음

즉, ablation은 “full model이 모든 metric에서 최고”인 구조가 아니라,
**각 구성요소가 어떤 metric family를 밀어주는가**를 보여주는 메커니즘 분석으로 읽어야 한다.

---

## 7. Ablation 보강안 1: Behavior decomposition table

현재 table은 최종 metric만 보여주기 때문에 산만해 보인다.  
이를 보완하려면, variant별 answer behavior를 직접 보여주는 보조 표가 필요하다.

### 추천 컬럼

- Supported \%
- Unclear \%
- Unsupported \%
- Avg evaluable claim count
- Factual sentence count
- Citation Coverage
- Backbone Trust Rate
- Trust-Weighted Claim Support

### 현재 6-topic 평균에서 보이는 패턴

#### `wo_anchor`

- Supported \%: `0.8468` (가장 높음)
- Avg evaluable claim count: `3.20`
- Citation Coverage: `0.2791`

해석:

- anchor-guided planning이 빠지면 더 짧고 직접적인 claim을 만들기 쉽다.
- 그 결과 sentence-level evaluator가 `supported`를 주기 쉬워진다.

#### `trustrag_anchor`

- Supported \%: `0.8073`
- Avg evaluable claim count: `3.83`
- Citation Coverage: `0.2933`
- HT Citation / Backbone Trust는 strongest

해석:

- 더 많은 cited claim을 만들고,
- 더 trust-rich evidence를 backbone으로 쓰지만,
- claim이 더 integrative해져서 `supported` 판정은 일부 약해진다.

#### `wo_trust`

- Supported \%: `0.8115`
- Avg evaluable claim count: `4.17` (가장 높음)

해석:

- trust constraint를 제거하면 relevance-heavy retrieval이 direct support에 유리하게 작동한다.

#### `wo_diversity`

- Supported \%: `0.6622` (가장 낮음)
- Unsupported \%: `0.1643` (가장 높음)
- Citation Coverage: `0.2272`

해석:

- diversity 제거로 evidence set이 좁아지고,
- 특정 지역 cluster에 과도하게 집중되면서 answer support robustness가 떨어진다.

### 이 표가 주는 장점

이 보조 표가 들어가면 ablation은 더 이상 “중구난방”이 아니라,

- support-friendly style
- trust-friendly style
- diversity-sensitive style

의 차이로 읽히게 된다.

---

## 8. Ablation 보강안 2: Topic-wise winner matrix

토픽별로 어떤 variant가 어떤 metric에서 strongest였는지를 matrix로 정리하면, ablation의 분산된 결과가 훨씬 자연스럽게 보인다.

### 현재 관찰

#### Immigration

- Support best: `vanilla_rag`
- Utility best: `trustrag_anchor`
- HT Cite best: `trustrag_anchor`

#### Gun Control

- Support best: `wo_anchor`
- Utility best: `trustrag_anchor`
- HT Cite best: `trustrag_anchor`
- Backbone Trust best: `trustrag_anchor`

#### Economy and Jobs

- Support best: `vanilla_rag`
- Utility best: `wo_diversity`
- Backbone Trust best: `trustrag_anchor`

#### Abortion

- Support best: `wo_trust`
- Utility best: `vanilla_rag`
- HT Cite best: `wo_anchor`

#### Free Speech

- Support best: `trustrag_anchor`
- HT Cite best: `wo_anchor`
- Backbone Trust best: `trustrag_anchor`
- TW Claim Support best: `trustrag_anchor`

#### Voting Rights and Voter Fraud

- Support best: `wo_anchor`
- Utility best: `wo_diversity`
- HT Cite best: `trustrag_anchor`
- Backbone Trust best: `trustrag_anchor`

### 해석

이 matrix는 다음 사실을 보여준다.

- trust-sensitive metrics는 full model이 가장 자주 이긴다.
- support/TW는 `wo_anchor` 혹은 `wo_trust`가 자주 이긴다.
- utility는 topic에 따라 `wo_diversity`, `vanilla`, `top2`가 더 좋아질 수 있다.

즉, ablation 결과는 noisy한 것이 아니라,
**metric family마다 winner가 다르게 나오는 구조**다.

---

## 9. Ablation 보강안 3: Claim style analysis

가능하다면 가장 설득력 있는 추가 분석은 이것이다.

### 질문

왜 `wo_anchor`는 support가 높고, `trustrag_anchor`는 backbone trust가 높은가?

### 가설

- `wo_anchor`:
  - shorter
  - local
  - directly supportable claim
- `trustrag_anchor`:
  - more integrative
  - more uncertainty-aware
  - multi-card synthesis claim

### 실행 방식

각 variant에서 20~30개 정도의 `support_checks` 문장을 샘플링해 아래 타입으로 분류한다.

- Direct factual claim
- Integrative summary claim
- Disagreement framing claim
- Evidence-limitation / uncertainty claim

### 기대 효과

이 분석이 들어가면,

- 왜 `support`가 `wo_anchor`에서 높은지
- 왜 `TW Claim Support`가 full model보다 높을 수 있는지
- 왜 full model은 backbone trust는 높지만 sentence-level support는 흔들리는지

를 매우 직관적으로 설명할 수 있다.

---

## 10. 추천 우선순위

시간 대비 효과 기준 우선순위는 아래와 같다.

### 1순위

- Exp4 positive case study 1개
- Exp4 trade-off case study 1개

### 2순위

- Trust-advantaged slice analysis
- Ablation behavior decomposition table

### 3순위

- Ablation topic-wise winner matrix
- Claim style qualitative analysis

---

## 11. 논문 서술에 반영할 메시지

이 피드백 대응 이후 논문의 메시지는 아래처럼 정리되는 것이 가장 좋다.

### Exp4

- AnchorRAG는 trust-sensitive evidence use에서 aggregate와 case level 모두 강점을 보인다.
- 특히 일부 query에서는 citation density와 trust quality를 크게 높이면서 support를 유지할 수 있다.
- 다만 integrative trust-aware generation은 일부 topic/query에서 support와 utility trade-off를 만든다.

### Ablation

- ablation 결과는 noisy failure가 아니라 objective decomposition으로 해석해야 한다.
- trust weighting은 backbone evidence quality를 높이고,
- anchor removal은 sentence-level support를 높이며,
- diversity removal은 retrieval breadth를 약화시킨다.

---

## 12. 바로 실행 가능한 산출물

현재 결과 파일만으로 바로 만들 수 있는 항목:

- Exp4 case-study figure/table
- Trust-advantaged subset table
- Ablation behavior decomposition table
- Ablation topic-wise winner matrix

추가 생성 실험 없이도 충분히 보강 가능한 범위다.

