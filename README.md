# Stress Score Prediction

신체 정보와 생활 패턴을 활용해 `stress_score`를 예측하는 회귀 프로젝트입니다.

## 검증 결과

- 평가 지표: MAE
- 검증: Train-only Stratified Group 5-Fold × Development/Audit Seed
- 최종 모델: V1 Weighted Quantile ExtraTrees
- Public MAE: **0.1282776667**
- 반복 CV 평균 MAE: **0.147505**
- 중복 표본 그룹 검증 MAE: **0.149202**

누수 없는 행 단위 파생변수와 중요 피처 가중치를 적용한
ExtraTrees 1,200개의 예측을 51% 분위수로 집계하여
Public MAE 0.1282776667을 기록한 모델입니다.

모델 및 전처리 선택 과정은 [실험 결과](docs/experiments.md)에 정리했습니다.
같은 Fold의 반복 사용으로 생길 수 있는 선택 과적합을 줄이기 위한 새 검증법과
V1~V4 재감사 결과는 [검증 보고서](validation/ROBUST_VALIDATION_REPORT.md)에
정리했습니다.

## 실행 방법

데이터 파일을 `/data`에 배치합니다.

```text
/data/
  train.csv
  test.csv
  sample_submission.csv
```

환경을 설치하고 학습·추론 코드를 실행합니다.

```bash
pip install -r requirements.txt
python final_submission.py --data-dir /data --output-dir .
```

실행이 끝나면 `submit_v1_weighted_quantile_extratrees.csv`가 생성됩니다.

## 누수 방지 원칙

- Test 데이터로 인코더·결측 대체값·스케일러를 학습하지 않습니다.
- 전처리기는 Train 데이터에서만 `fit`합니다.
- 파생변수는 한 행 안의 입력값만 사용합니다.
- Test 전체 통계, 행 수, 인덱스 또는 순서를 모델링에 사용하지 않습니다.

## 저장소 보안

대회 데이터, 제출 CSV와 학습 모델 파일은 대회 규정 및 보안을 위해 Git에서 제외합니다.
