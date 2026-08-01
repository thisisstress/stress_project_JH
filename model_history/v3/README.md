# V3 — OOF Blended Median ExtraTrees

- 상태: 검증 보존·미승격
- 최종 코드: [`final_submission_v3.py`](final_submission_v3.py)
- 제출 파일: `submit_v3_oof_blended_median_extratrees.csv`
- 기존 Train-only 검증 MAE: 0.146666
- 신규 Audit 평균 MAE: 0.148880
- Public MAE: 확인하지 않음

Train OOF에서 선택한 두 ExtraTrees 구성을 중앙값 기준으로 혼합했습니다. 신규
Audit에서는 V1보다 평균 0.000476 낮았지만 쌍대 95% 신뢰구간이 0을 포함해
강한 개선으로 보지 않고 실험 이력으로 보존했습니다.
