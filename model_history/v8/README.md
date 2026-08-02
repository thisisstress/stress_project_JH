# V8 — Robust Subspace-Neighbor Blend

- 상태: **검증 통과·Public 제출 대기 후보**
- 최종 코드: [`final_submission_v8.py`](final_submission_v8.py)
- 제출 파일: `submit_v8_robust_subspace_blend.csv`
- 신규 Audit3 MAE: **0.150044** (V7 0.150134)
- 6개 Split Seed 승률: **6/6**
- Public MAE: 제출 후 기록 예정

V7 예측 82.5%를 유지하면서, Train 경험적 순위 공간에서 만든 3차원·4차원
최근접 이웃 신호와 2~4차원 이웃 풀 분위수를 총 17.5% 혼합합니다. 가중치는
Seed 42·2026·3407의 Train-only OOF에서 고정했고, 완전히 새로운 Audit Seed
271828·314159·424243에서 3/3 개선을 확인했습니다.

다만 신규 Audit 평균 개선 폭은 0.000090으로 작습니다. 따라서 Public 1위나
0.125대 점수를 보장하는 모델로 표현하지 않으며, V7을 교체할지는 실제 Public
결과까지 확인한 뒤 결정합니다.

상세 내용은 [`V8_MODEL_REPORT.md`](../../validation/V8_MODEL_REPORT.md)에
있습니다.
