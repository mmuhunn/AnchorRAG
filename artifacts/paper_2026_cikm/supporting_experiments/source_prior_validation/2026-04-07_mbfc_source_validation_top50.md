# MBFC Source Validation Top50

## 요약

`AllSides_Qbias`의 빈도 상위 50개 source를 MBFC 기준으로 확장 매칭했다.  
이 중 `Guest Writer - Right` 1개 row는 실제 매체 source가 아니어서 제외했고, 최종적으로 `49개 source`를 매칭했다.

사용 파일:

- `artifacts/paper_2026_cikm/supporting_experiments/source_prior_validation/mbfc_validation_top50.csv`
- 원본 작업 repo 기준 생성 스크립트: `AllSides_Qbias/source_validation/build_mbfc_pilot_validation.py`
- 원본 작업 repo 기준 비교 스크립트: `AllSides_Qbias/source_validation/compare_source_reliability_alignment.py`

## coverage

- matched sources: `49`
- unmatched source rows: `1`
  - `Guest Writer - Right`
- covered docs: `1779 / 1983`
- corpus coverage: `89.7%`

즉 top50 확장본은 사실상 corpus의 거의 90%를 MBFC 기준으로 설명한다.

## main validation: source-only score

proxy: `current_source_score_mean`  
target: `mbfc_factual_numeric`

unweighted:

- Spearman: `0.6620`
- Pearson: `0.7021`

document-weighted:

- weighted Pearson: `0.7108`
- expanded Spearman: `0.7284`

## sanity check: full S_trust

proxy: `current_s_trust_mean`  
target: `mbfc_factual_numeric`

document-weighted:

- weighted Pearson: `0.7065`
- expanded Spearman: `0.7450`

## 해석

top30 pilot보다 Spearman은 약간 낮아졌지만, coverage가 `80.3% -> 89.7%`로 커졌다.  
즉 long-tail에 가까운 source들을 더 포함하면서 noise가 늘었음에도, source-only score와 MBFC factuality의 정렬은 여전히 `~0.66 ~ 0.73` 수준으로 유지된다.

이건 다음 주장을 뒷받침한다.

- source score는 완전히 임의적인 내부 지표가 아니다
- source score는 외부 human-rated source factuality와 reasonably aligned 된다
- full `S_trust`도 유사한 수준의 정렬을 보여, graph structural term이 source reliability prior를 크게 망가뜨리진 않는다

## factuality 분포

source count 기준:

- `HIGH`: `18`
- `MOSTLY FACTUAL`: `18`
- `MIXED`: `11`
- `VERY HIGH`: `1`
- `LOW`: `1`

doc-weighted 기준:

- `HIGH`: `377`
- `MOSTLY FACTUAL`: `859`
- `MIXED`: `420`
- `VERY HIGH`: `97`
- `LOW`: `26`

## 예시

낮은 source-only score:

- `Breitbart News`: `0.3562`, MBFC `MIXED`
- `The Blaze`: `0.4000`, MBFC `MIXED`
- `Newsmax (News)`: `0.4200`, MBFC `LOW`
- `Townhall`: `0.4333`, MBFC `MIXED`

높은 source-only score:

- `Washington Post`: `0.8664`, MBFC `MOSTLY FACTUAL`
- `New York Times (News)`: `0.9000`, MBFC `HIGH`
- `Reuters`: `0.9276`, MBFC `VERY HIGH`
- `Associated Press`: `0.9333`, MBFC `HIGH`

## 논문에서의 안전한 표현

`Extending the MBFC validation to the top 50 most frequent sources in AllSides/Qbias, we matched 49 source entries covering 89.7% of the corpus documents. The recovered source-level reliability score remained substantially aligned with MBFC’s human-rated factual reporting labels (Spearman = 0.66; weighted Spearman = 0.73), indicating that our source prior is not arbitrary even under broader source coverage.`

## 메모

- 이번 확장본은 `NewsGuard` 없이도 꽤 강한 1차 validation 역할을 한다.
- 다만 `Guest Writer - Right`처럼 non-source row가 남아 있으므로, 최종 논문에는 `raw source field`의 noise를 짧게 언급하는 편이 안전하다.
