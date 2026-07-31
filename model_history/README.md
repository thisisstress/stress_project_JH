# 모델 버전 기록

이 폴더는 JH가 실험한 모델 코드를 버전별로 보존합니다.
현재 제출 기준 모델은 저장소 루트의 `final_submission.py`입니다.

| 버전 | 모델 | Train-only 검증 MAE | Public MAE | 상태 |
|---|---|---:|---:|---|
| V1 | Weighted Quantile ExtraTrees (`q=0.51`) | 0.147505 | **0.1282776667** | 현재 채택 |
| V2 | Weighted Median ExtraTrees (`q=0.50`) | 0.146710 | 0.1284666667 | 미채택 |
| V3 | OOF Blended Median ExtraTrees | 0.146666 | 확인 전 | 검증 중 |

## 기록 원칙

- Test 데이터로 인코더, 결측 대체값 또는 모델을 학습하지 않습니다.
- Test 전체 통계, 행 수, 인덱스와 순서를 모델링에 사용하지 않습니다.
- 데이터 원본, 제출 CSV와 학습된 모델 파일은 Git에 올리지 않습니다.
- Public 점수가 확인되지 않은 후보는 `검증 중`으로 표시합니다.
