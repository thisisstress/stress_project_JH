# V7 독립 모델 개발 및 검증 보고서

## 결론

V7은 V14의 코드·예측값·혼합 비율을 사용하지 않고 개발한 독립 후보입니다.
V6의 국소 Tree 분위수 예측 85%와 새로 만든 2차원 피처쌍 최근접 이웃 예측
15%를 혼합합니다.

| 검증 구간 | V6 MAE | V7 MAE | V6 대비 | Seed 승률 |
|---|---:|---:|---:|---:|
| Development Seed 42 | 0.151283 | **0.150560** | -0.000723 | 1/1 |
| Development 확인 Seed 2026·3407 | 0.146502 | **0.146205** | -0.000297 | 2/2 |
| 신규 Audit3 | 0.147019 | **0.146644** | -0.000374 | 3/3 |

Audit3 쌍대 95% 신뢰구간은 `[-0.000791, +0.000044]`입니다. 여섯 Seed에서
모두 개선됐지만 Audit3 신뢰구간이 0을 아주 조금 포함하므로 Public 성능을
보장하지 않습니다.

## Train 데이터 품질

- 행 3,000개, 입력 피처 16개, ID 중복 0개
- 동일 입력 중복 행 12개(6개 그룹), 중복 그룹 안의 target 충돌 0개
- target 범위 0~1, 고유값 101개, 0.01 격자 비율 100%
- 결측률: `family_medical_history` 49.53%, `medical_history` 42.97%,
  `mean_working` 34.40%, `edu_level` 20.23%
- 기본 도메인 점검에서 음수 키·체중·혈당, 수축기<이완기 혈압은 발견되지 않음
- IQR 이상치는 오류로 확인되지 않아 임의 제거하지 않음

상세 실행 결과는 `V7_TRAIN_ONLY_ANALYSIS.ipynb`와
`results/v7_data_quality_report.md`에 남겼습니다.

## ExtraTrees 외 모델군 점검

모든 후보는 같은 Train-only Seed 42 Group 5-Fold에서 비교했습니다.

| 모델군 최선 후보 | MAE | 판정 |
|---|---:|---|
| LightGBM L1 | 0.199790 | 폐기 |
| CatBoost | 약 0.220403 | 폐기 |
| XGBoost | 0.221243 | 폐기 |
| 스플라인 분위수 회귀 | 0.245873 | 폐기 |
| KNN | 0.204040 | 폐기 |
| RandomForest 분위수 | 0.164723 | 단독 폐기 |
| 1차원 피처별 이웃 앙상블 | 0.211517 | 폐기 |
| **2차원 피처쌍 1-NN** | **0.155808** | 보조 모델 채택 |
| V6 기준 | 0.151283 | 기준 |

부스팅·KNN·RandomForest 보조 모델을 V6에 1~15% 혼합해도 개선되지 않았습니다.
2차원 피처쌍 1-NN만 V6와 다른 오차를 보여 15% 혼합 후보로 승격했습니다.

## V7 구조

### 1. V6 독립 Tree 구성 85%

- ExtraTrees 1,200개, `max_features=1`, `random_state=42`
- Train-only 행 단위 파생변수와 고정 피처 복제로 선택 확률 조정
- 개별 Tree 예측의 52% 분위수를 0.01로 반올림

### 2. 2차원 피처쌍 최근접 이웃 15%

고정한 8개 피처의 모든 두 개 조합 28개를 사용합니다.

- `mean_working`, `bmi`, `cholesterol`, `height`
- `glucose`, `weight`, `cholesterol_glucose_ratio`, `bone_density`

각 피처를 Train 기준 경험적 순위로 변환합니다. Test 값은 해당 Train 정렬 배열에
한 행씩 독립적으로 투영합니다. 각 피처쌍에서 Manhattan 거리가 가장 가까운 Train
한 행의 target을 가져온 뒤 28개 target의 52% 분위수를 계산합니다.

최종 예측은 다음과 같습니다.

```text
round(0.85 × V6_tree_quantile + 0.15 × pair_neighbor_quantile, 2)
```

## 누수 방지

- 전처리기, 결측 중앙값, 경험적 순위와 모델은 Train에서만 학습합니다.
- Test 통계, 범주 빈도, 평균, 행 수, 인덱스와 순서를 사용하지 않습니다.
- Test의 다른 행을 참조하지 않습니다.
- 외부 데이터, V14 코드와 V14 예측값을 사용하지 않습니다.
- CV에서는 동일 입력 중복 그룹을 같은 Fold에 배치합니다.

## Validation Report

### Overall Assessment: Share with caveats

- 방법론: 동일 Fold의 paired comparison과 완전히 새로운 Audit3 Seed를 사용했습니다.
- 계산 검산: Audit3 3개 Seed 모두 V7이 V6보다 낮은 MAE를 기록했습니다.
- 남은 불확실성: Audit3 95% CI 상한이 `+0.000044`로 0을 아주 조금 포함합니다.
- 결론: 제출할 가치가 있는 후보이나 Public 1등이나 특정 점수를 보장할 수 없습니다.

## 산출물

- 최종 실행 코드: `final_submission_v7.py`
- 제출 파일: `submit_v7_pair_neighbor_blend.csv`
- 제출 CSV SHA-256:
  `C63DDA7F89A1741366D4DD772AE958846D389C5BC196B6010FA88568BB096618`
- 분석 노트북: `V7_TRAIN_ONLY_ANALYSIS.ipynb` (전체 셀 실행 완료, 오류 0개)
