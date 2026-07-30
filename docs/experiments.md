# 모델 실험 결과

모든 결과는 Test 정보를 사용하지 않은 동일한 Train 5-Fold OOF MAE입니다.

## 최종 V8 결과

- 모델: ExtraTrees 1,200개, `max_features=1`
- 주요 피처: 원본 외 복제본 2개씩 추가
- 집계: 개별 트리 예측의 51% 분위수
- 반복 CV 평균 MAE: **0.147505**
- Public MAE: **0.1282776667**

| 검증 | V6 | V8 |
|---|---:|---:|
| K-Fold Seed 42 | 0.149653 | **0.148840** |
| K-Fold Seed 2026 | 0.145967 | **0.145450** |
| K-Fold Seed 3407 | 0.148323 | **0.148223** |
| 중복 그룹 분할 | 0.150027 | **0.149202** |

분위수와 피처 복제 수는 Public Score가 아니라 세 가지 Train 분할의 평균 MAE로 선택했습니다.

| 모델 | OOF MAE |
|---|---:|
| ExtraTrees + 파생변수 + One-Hot | **0.18061** |
| ExtraTrees + 기본변수 + One-Hot | 0.18388 |
| RandomForest + 파생변수 + One-Hot | 0.19969 |
| LightGBM | 0.21939 |
| XGBoost | 0.22161 |
| HistGradientBoosting | 0.22304 |
| CatBoost | 0.23328 |
| Ridge | 0.24720 |
| 중앙값 기준선 | 0.24944 |

## 안정성 확인

최적 ExtraTrees 설정을 서로 다른 분할 Seed로 다시 검증했습니다.

| 검증 | OOF MAE |
|---|---:|
| Stratified 5-Fold, Seed 42 | 0.18041 |
| Stratified 5-Fold, Seed 2026 | 0.17874 |
| Stratified 5-Fold, Seed 3407 | 0.17842 |
| 중복 특성 표본을 같은 Fold로 묶은 검증 | 0.17865 |

## 최종 설정

- 결측 수치형: Train 중앙값 대체 및 결측 표시 변수
- 결측 범주형: `__MISSING__` 독립 범주
- 범주형: Train-only One-Hot Encoding
- 파생변수: BMI, 맥압, 평균동맥압, 콜레스테롤/혈당 비율, 체중/신장 비율
- 모델: ExtraTreesRegressor
- `n_estimators=1000`, `min_samples_leaf=1`, `max_features=1.0`
- Seed 42, 2026, 3407의 예측 평균
