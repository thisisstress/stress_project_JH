# V7 — Pair-Neighbor Quantile Blend

- 상태: **현재 최고·채택 모델**
- 최종 코드: [`final_submission_v7.py`](final_submission_v7.py)
- 제출 파일: `submit_v7_pair_neighbor_blend.csv`
- 신규 Audit3 MAE: 0.146644
- Public MAE: **0.1272333333**
- 제출 시각: **2026-08-01 14:43 KST**

V6 Adaptive Feature-Probability ExtraTrees 예측 85%와 8개 핵심 피처의 모든
2차원 조합에서 구한 Train 1-NN 분위수 예측 15%를 혼합합니다. 최종 예측은
0.01 단위로 반올림합니다. 신규 Audit3 Seed 3개에서 모두 V6를 이겼고,
Public에서도 지금까지의 JH 모델 중 가장 낮은 MAE를 기록했습니다.

상세 내용은 [`V7_MODEL_REPORT.md`](../../validation/V7_MODEL_REPORT.md)와
[`V7_TRAIN_ONLY_ANALYSIS.ipynb`](../../validation/V7_TRAIN_ONLY_ANALYSIS.ipynb)에
있습니다.


**현재 팀 기준:** V7은 역사적 계보 이정표이며, 전체 팀 final은 BS 8/6입니다. 공식 순위는 별도 보존 근거가 없는 경우 주장하지 않습니다.
