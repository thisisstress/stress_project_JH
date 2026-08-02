# V5 — 보조 모델 혼합 실험

- 상태: 검증 후 폐기
- 최종 코드: 없음
- 제출 파일: 생성하지 않음
- Public MAE: 확인하지 않음

V5A는 상호작용·Seed 7777 보조 모델을 2.5% 혼합했고, V5B는 같은 피처의
3-Seed pooled 중앙값 예측을 25% 혼합했습니다. V5A는 V1보다 유의하게
나빴고 V5B는 개선 신뢰구간이 0을 포함하여, 사전에 정한 승격 조건에 따라
Audit과 최종 제출 파일 생성을 생략했습니다.

상세 결과는 [`EXPERIMENT_V5_REPORT.md`](../../validation/EXPERIMENT_V5_REPORT.md)에
있으며 실행 코드는 `validation/experiments/v5_*`에 보존했습니다.
