# 모델 실험 결과

이 문서는 V1~V7의 결과를 빠르게 확인하기 위한 요약입니다. 버전별 코드와 상세
설명은 [`model_history/README.md`](../model_history/README.md)를 기준 문서로
사용합니다.

## 버전별 결과

| 버전 | 핵심 변경 | 대표 Train-only MAE | Public MAE | 판정 |
|---|---|---:|---:|---|
| V1 | 중요 피처 가중 51% 분위수 ExtraTrees | 0.147505 | 0.1282776667 | 이전 기준 |
| V2 | 집계 분위수를 50% 중앙값으로 변경 | 0.146710 | 0.1284666667 | 미채택 |
| V3 | OOF 두 ExtraTrees 혼합 | 0.146666 | 미확인 | 미승격 |
| V4 | 보수적 50.5% 분위수 | 0.147434 | 0.1286866667 | 미채택 |
| V5 | 상호작용·다중 Seed 보조 혼합 | 0.148844 (최선) | 미확인 | 폐기·코드 없음 |
| V6 | 피처별 선택 확률을 조정한 52% 분위수 Forest | 0.147678 (Audit2) | 미확인 | V7 주 모델 |
| V7 | V6 85% + Pair-Neighbor 15% | **0.146644 (Audit3)** | **0.1272333333** | **현재 채택** |

검증 구간과 Seed가 다른 수치를 단순히 절대 비교하지 않습니다. V3 이후 후보는
동일 Fold의 쌍대 비교와 새 Audit Seed에서 기준 모델을 이기는지를 함께 확인했습니다.

## 현재 채택 모델 V7

- ExtraTrees 1,200개, `max_features=1`, 고정 Seed 42
- Train-only 행 단위 파생변수와 고정 피처 복제
- 개별 Tree 예측의 52% 분위수: 85%
- 8개 핵심 피처의 28개 2차원 조합별 Train 1-NN 분위수: 15%
- 최종 예측을 0.01 단위로 반올림
- 신규 Audit3 Seed 3/3에서 V6보다 낮은 MAE
- Public MAE 0.1272333333, 제출 시각 2026-08-01 14:43 KST

## 선택 및 누수 방지 원칙

- 전처리, 피처, 분위수와 혼합 비율은 Train-only 검증으로 선택합니다.
- Public 점수는 고정된 후보의 최종 확인 수단으로만 기록합니다.
- Test에서는 행 내부 파생변수만 계산합니다.
- Test 전체 통계, 행 수, 인덱스와 다른 Test 행을 사용하지 않습니다.
- 동일 입력 중복 그룹은 교차검증에서 같은 Fold에 배치합니다.

## 상세 보고서

- [V1~V4 재감사](../validation/ROBUST_VALIDATION_REPORT.md)
- [V5 폐기 실험](../validation/EXPERIMENT_V5_REPORT.md)
- [V6 모델 보고서](../validation/V6_MODEL_REPORT.md)
- [V7 모델 보고서](../validation/V7_MODEL_REPORT.md)
- [V7 데이터 품질 보고서](../validation/V7_DATA_QUALITY_REPORT.md)
- [V7 실행 완료 분석 노트북](../validation/V7_TRAIN_ONLY_ANALYSIS.ipynb)


> Historical note: 공식 순위는 별도 보존 근거가 없는 경우 주장하지 않습니다. 전체 팀 final은 BS 8/6이며, 점수 해석은 UNIFIED의 EVALUATION_NOTES.md를 따릅니다.
