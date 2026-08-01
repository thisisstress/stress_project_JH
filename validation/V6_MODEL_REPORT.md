# V6 독립 모델 개발 보고서

## 결론

V6는 팀원의 V14 예측값을 섞거나 V14를 기준 모델로 삼지 않고 새로 개발한
`Adaptive Feature-Probability Quantile Forest`입니다. 모델 구조 선택은 Train-only
Development Seed에서 수행했고, 최종 설정을 고정한 뒤 처음 보는 Audit2 Seed에서
한 번만 재검증했습니다.

| 구간 | V6 평균 MAE | V1 평균 MAE | V1 대비 | Seed 승률 | 쌍대 95% CI |
|---|---:|---:|---:|---:|---:|
| Development | **0.148096** | 0.148857 | -0.000761 | 3/3 | [-0.001445, -0.000109] |
| Audit2 | **0.147678** | 0.148398 | -0.000720 | 3/3 | [-0.001432, -0.000017] |

Audit2에서도 차이의 신뢰구간 상한이 0보다 작아 사전에 정한 승격 조건을 모두
통과했습니다. Public MAE는 제출 전이므로 아직 알 수 없으며 리더보드 성능을
보장하지 않습니다.

## 독립 후보 탐색

V14의 코드, 예측값, 혼합 비율은 V6의 학습·추론·튜닝에 사용하지 않았습니다.
다음 세 가지 새 후보군을 동일한 Train-only 검증 틀에서 비교했습니다.

1. **Proximity Quantile Forest**: 같은 Tree leaf에 도달한 Train 표본의 target을
   직접 모아 분위수를 계산했습니다. 50-tree 화면에서 최선 후보가 V1보다
   약 0.00025 낮았지만 개선 불확실성이 커서 폐기했습니다.
2. **Mixture of Quantile Experts**: 근로·신체·대사·생활 습관별 피처 전문가의
   Tree 분포를 결합했습니다. 공정한 Tree 예산 안에서 모든 혼합 후보가 V1보다
   나빠 폐기했습니다.
3. **Adaptive Feature-Probability Forest**: 중요 피처의 동일한 복제본을 만들어
   `max_features=1` 무작위 후보 선택 확률을 조절했습니다. Development와 Audit2를
   모두 통과해 V6로 승격했습니다.

## 최종 설정

- 모델: `ExtraTreesRegressor`
- Tree 수: 1,200
- `max_features`: 1
- `random_state`: 42
- 집계: 개별 Tree 예측의 52% 분위수
- 후처리: 소수 둘째 자리 반올림
- 추가 복제본 수:
  - `mean_working`: 1
  - `bmi`: 4
  - `cholesterol`: 2
  - `height`: 2
  - `glucose`: 3
  - `weight`: 2
  - `cholesterol_glucose_ratio`: 2
  - `bone_density`: 4

복제본은 새로운 외부 정보를 추가하지 않고, ExtraTrees가 각 노드에서 어떤 피처를
후보로 선택할지에 대한 확률만 바꿉니다.

## 검증 설계

- 평가 지표: MAE
- Fold: target 10분위 층화 + 동일 입력 중복 그룹을 보존한 5-Fold
- Development Seed: 42, 2026, 3407
- Audit2 Seed: 181081, 196613, 214627
- 비교: 모든 후보와 V1이 같은 Fold를 쓰는 paired comparison
- 불확실성: 중복 그룹 단위 paired bootstrap 95% 신뢰구간
- Audit2에서는 Development에서 고른 최종 후보 하나만 평가

## 누수 방지

- 인코더와 결측치 대체기는 각 학습 Fold 또는 전체 Train에서만 `fit`합니다.
- Test의 통계, 행 수, 인덱스, 순서 또는 다른 Test 행의 정보를 사용하지 않습니다.
- 파생변수는 각 행 안의 입력값만으로 계산합니다.
- Test target, 외부 데이터, V14 예측값은 사용하지 않습니다.

## 재현 및 산출물

- 제출 재현 코드: `model_history/v6/final_submission_v6.py`
- 독립 후보 실험: `validation/experiments/v6_*.py`
- 집계 결과: `validation/results/v6_*_summary.csv`
- 제출 파일명: `submit_v6_adaptive_feature_quantile_forest.csv`
- 제출 CSV SHA-256:
  `92245B66903FEF2E66012B3544A810687D26FFA7ADF8DBD4EAC15703CAEC677B`

대회 데이터와 제출 CSV 자체는 저장소에 올리지 않습니다.
