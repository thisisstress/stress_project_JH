# V6 — Adaptive Feature-Probability Quantile Forest

- 상태: V7의 주 모델·단독 후보
- 최종 코드: [`final_submission_v6.py`](final_submission_v6.py)
- 제출 파일: `submit_v6_adaptive_feature_quantile_forest.csv`
- Development MAE: 0.148096
- 신규 Audit2 MAE: 0.147678
- Public MAE: 확인하지 않음

피처별 복제 수로 ExtraTrees의 무작위 피처 선택 확률을 조절하고, 트리
1,200개의 예측을 52% 분위수로 집계한 뒤 0.01 단위로 반올림합니다. 신규
Audit2 Seed 3개에서 V1을 모두 이겼으며, 이후 V7의 85% 주 모델이 됐습니다.

상세 내용은 [`V6_MODEL_REPORT.md`](../../validation/V6_MODEL_REPORT.md)에 있습니다.
