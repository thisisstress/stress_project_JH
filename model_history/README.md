# V1~V8 모델 버전 기록

JH가 직접 개발·검증한 모델을 버전 순서대로 보존합니다. 현재 저장소 루트의
`final_submission.py`와 제출 기준 모델은 **V7 Pair-Neighbor Quantile
Blend**입니다.

모든 `v1`~`v8` 폴더에는 동일한 형식의 `README.md`가 있습니다. 최종 제출
코드가 만들어진 버전은 `final_submission_vN.py`도 함께 보존하며, 승격 전에
폐기한 V5는 코드가 없는 이유와 실험 경로를 README에 명시했습니다.

## 한눈에 보기

| 버전 | 모델 | 대표 Train-only MAE | Public MAE | 상태 |
|---|---|---:|---:|---|
| [V1](v1/README.md) | Weighted Quantile ExtraTrees (`q=0.51`) | 0.147505 | 0.1282776667 | 이전 기준 |
| [V2](v2/README.md) | Weighted Median ExtraTrees (`q=0.50`) | 0.146710 | 0.1284666667 | 미채택 |
| [V3](v3/README.md) | OOF Blended Median ExtraTrees | 0.146666 | 미확인 | 미승격 |
| [V4](v4/README.md) | Conservative Quantile ExtraTrees (`q=0.505`) | 0.147434 | 0.1286866667 | 미채택 |
| [V5](v5/README.md) | 보조 모델 혼합 실험 | 0.148844 (최선) | 미확인 | 검증 후 폐기 |
| [V6](v6/README.md) | Adaptive Feature-Probability Quantile Forest | 0.147678 (Audit2) | 미확인 | V7 주 모델 |
| [V7](v7/README.md) | Pair-Neighbor Quantile Blend | **0.146644 (Audit3)** | **0.1272333333** | **현재 최고·채택** |
| [V8](v8/README.md) | Robust Subspace-Neighbor Blend | **0.150044 (신규 Audit3)** | 제출 대기 | 검증 통과 후보 |

> 검증 구간과 Seed가 다른 MAE의 절대값을 단순 비교하지 않습니다. 후보 승격은
> 같은 Fold에서 기준 모델과 쌍대 비교하고, 새 Audit Seed에서도 개선되는지를
> 함께 확인했습니다.

## 버전 흐름

### V1 — 첫 기준 모델

행 단위 파생변수와 중요 피처 복제를 적용한 ExtraTrees 1,200개의 개별 Tree
예측을 51% 분위수로 집계했습니다. Public MAE 0.1282776667을 기록해 V7 이전
기준 모델로 사용했습니다.

### V2 — 중앙값 집계

V1의 51% 분위수를 50% 중앙값으로 바꿨습니다. Train-only MAE는 좋아졌지만
Public MAE 0.1284666667로 V1보다 나빠 미채택했습니다.

### V3 — OOF 혼합

Train OOF에서 선택한 두 ExtraTrees 구성을 혼합했습니다. 신규 Audit 평균은
V1보다 0.000476 낮았지만 쌍대 95% 신뢰구간이 0을 포함해 승격하지 않았습니다.
상세 비교는 [V1~V4 재감사 보고서](../validation/ROBUST_VALIDATION_REPORT.md)에
있습니다.

### V4 — 보수적 분위수

V1 구조를 유지하고 분위수만 50.5%로 조정했습니다. Public MAE
0.1286866667로 V1보다 나빠 미채택했습니다.

### V5 — 보조 혼합 실험·폐기

상호작용 보조 모델과 다중 Seed pooled 예측 혼합을 실험했습니다. V5A는
V1보다 유의하게 나빴고 V5B는 개선 신뢰구간이 0을 포함해, 사전 승격 기준에
따라 최종 코드와 제출 파일을 만들지 않았습니다. 상세 내용은
[V5 실험 보고서](../validation/EXPERIMENT_V5_REPORT.md)에 있습니다.

### V6 — 피처 선택 확률 최적화

피처별 복제 수로 ExtraTrees의 무작위 피처 선택 확률을 조절하고 Tree 1,200개
예측의 52% 분위수를 0.01 단위로 반올림했습니다. 신규 Audit2 Seed 3개에서
V1을 모두 이겨 독립 후보로 승격했고, 이후 V7의 85% 주 모델이 됐습니다.
상세 내용은 [V6 보고서](../validation/V6_MODEL_REPORT.md)에 있습니다.

### V7 — 현재 최고 모델

V6 예측 85%와 8개 핵심 피처의 모든 2차원 조합에서 찾은 Train 1-NN 분위수
예측 15%를 혼합합니다. 신규 Audit3 Seed 3개에서 모두 V6를 이겼고, Public
MAE **0.1272333333**으로 2026-08-01 14:43 KST 기준 리더보드 1위를
기록했습니다. 상세 내용은 [V7 보고서](../validation/V7_MODEL_REPORT.md)와
[실행 완료 분석 노트북](../validation/V7_TRAIN_ONLY_ANALYSIS.ipynb)에 있습니다.

### V8 — 검증 통과 제출 후보

V7 82.5%에 3차원·4차원 최근접 이웃과 2~4차원 이웃 풀 분위수 17.5%를
혼합합니다. 가중치 고정 후 새 Audit Seed 3개에서 모두 V7을 이겼지만 평균
개선 폭은 0.000090으로 작습니다. Public 결과 확인 전까지 루트 기본 모델은
V7으로 유지합니다. 상세 내용은 [V8 보고서](../validation/V8_MODEL_REPORT.md)에
있습니다.

## 기록 원칙

- Test 데이터로 인코더, 결측 대체값, 경험적 순위 기준 또는 모델을 학습하지 않습니다.
- Test 전체 통계, 행 수, 인덱스, 순서와 다른 Test 행을 모델링에 사용하지 않습니다.
- 데이터 원본, 제출 CSV와 학습된 모델 파일은 Git에 올리지 않습니다.
- Public 점수는 Train-only에서 설정을 고정한 뒤 최종 확인 결과로만 기록합니다.
- 실패한 실험도 삭제하지 않고 미채택·폐기 이유를 남깁니다.
