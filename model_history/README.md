# V1~V8 모델 버전 기록

**Owner:** JH research workspace  
**Root submission model:** V7 Pair-Neighbor Quantile Blend  
**Version contract:** `v1`~`v8`별 README · 제출 버전은 `final_submission_vN.py` 병행

## 한눈에 보기

| 버전 | 모델 | 대표 Train-only MAE | Public MAE | 상태 |
|---|---|---:|---:|---|
| [V1](v1/README.md) | Weighted Quantile ExtraTrees (`q=0.51`) | 0.147505 | 0.1282776667 | 이전 기준 |
| [V2](v2/README.md) | Weighted Median ExtraTrees (`q=0.50`) | 0.146710 | 0.1284666667 | 미채택 |
| [V3](v3/README.md) | OOF Blended Median ExtraTrees | 0.146666 | 미확인 | 미승격 |
| [V4](v4/README.md) | Conservative Quantile ExtraTrees (`q=0.505`) | 0.147434 | 0.1286866667 | 미채택 |
| [V5](v5/README.md) | 보조 모델 혼합 실험 | 0.148844 (최선) | 미확인 | 검증 후 폐기 |
| [V6](v6/README.md) | Adaptive Feature-Probability Quantile Forest | 0.147678 (Audit2) | 미확인 | V7 주 모델 |
| [V7](v7/README.md) | Pair-Neighbor Quantile Blend | **0.146644 (Audit3)** | **0.1272333333** | **팀 계보 이정표** |
| [V8](v8/README.md) | Robust Subspace-Neighbor Blend | 0.150044 (신규 Audit3) | 0.1274733333 | 미채택 |

**Comparison rule:** 검증 구간/seed가 다른 절대 MAE 직접 순위화 금지. 승격은 same-Fold paired comparison + 신규 Audit seed 기준.

## 버전 흐름

### V1 — Baseline

- ExtraTrees `1200`
- row-level derived features
- 중요 피처 복제
- Tree prediction Q51
- Public `0.1282776667`

### V2 — Median Aggregation

- V1 Q51 → Q50
- Train-only MAE 개선
- Public `0.1284666667`
- **판정:** 미채택

### V3 — OOF Blend

- 두 ExtraTrees 구성 혼합
- 신규 Audit 평균 vs V1: `-0.000476`
- paired 95% CI includes 0
- **판정:** 미승격
- [V1~V4 재감사](../validation/ROBUST_VALIDATION_REPORT.md)

### V4 — Conservative Quantile

- V1 구조 유지
- Q50.5
- Public `0.1286866667`
- **판정:** 미채택

### V5 — Auxiliary Blend

- interaction auxiliary model · multi-seed pooled blend
- V5A: V1 대비 유의 악화
- V5B: improvement CI includes 0
- final code/submission 미생성
- [V5 report](../validation/EXPERIMENT_V5_REPORT.md)

### V6 — Adaptive Feature Probability

- feature replication → ExtraTrees selection probability control
- Trees `1200` · Q52 · 0.01 rounding
- 신규 Audit2 seed 3/3 V1 우세
- V7 main branch `85%`
- [V6 report](../validation/V6_MODEL_REPORT.md)

### V7 — Pair-Neighbor Blend

- V6 `85%` + Pair-Neighbor `15%`
- 8 features · 28 pair spaces · Train 1-NN · Q52
- 신규 Audit3 seed 3/3 V6 우세
- Public **`0.1272333333`**
- 2026-08-01 14:43 KST 당시 leaderboard 1위
- [V7 report](../validation/V7_MODEL_REPORT.md) · [Train-only notebook](../validation/V7_TRAIN_ONLY_ANALYSIS.ipynb)

### V8 — Robust Subspace Neighbor

- V7 `82.5%` + 3D/4D neighbor/pool `17.5%`
- 신규 Audit seed 3/3 V7 우세
- Audit mean delta: `-0.000090`
- Public `0.1274733333` → V7 대비 `+0.0002400000`
- **판정:** 미채택
- [V8 report](../validation/V8_MODEL_REPORT.md)

## 기록 원칙

- 모델 · 인코더 · 결측 대체 · empirical rank: Train-only
- Test 전체 통계 · 행 수 · 인덱스 · 순서 · cross-row 정보: 미사용
- 원본 데이터 · 제출 CSV · trained model artifact: Git 미포함
- Public score: Train-only 설정 freeze 이후 확인값
- negative results: 미채택/폐기 사유 포함 보존
