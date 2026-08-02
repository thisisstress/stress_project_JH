# V1 — Weighted Quantile ExtraTrees

- 상태: 이전 기준 모델
- 최종 코드: [`final_submission_v1.py`](final_submission_v1.py)
- 제출 파일: `submit_v1_weighted_quantile_extratrees.csv`
- Train-only 검증 MAE: 0.147505
- Public MAE: 0.1282776667

누수 없는 행 단위 파생변수와 중요 피처 복제를 적용한 ExtraTrees 1,200개의
개별 트리 예측을 51% 분위수로 집계합니다. V7이 확인되기 전까지 저장소의
기준 모델로 사용했습니다.
