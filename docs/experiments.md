# 모델 실험 결과

모든 결과는 Test 정보를 사용하지 않은 동일한 Train 5-Fold OOF MAE입니다.

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
