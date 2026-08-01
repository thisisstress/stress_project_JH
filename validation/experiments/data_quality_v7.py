# -*- coding: utf-8 -*-
"""Train-only data quality and target-structure audit for V7 development."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


TARGET = "stress_score"
ID_COLUMN = "ID"


def format_rate(count: int, total: int) -> str:
    return f"{count:,} ({count / total:.2%})"


def markdown_table(frame: pd.DataFrame, decimals: int = 6) -> str:
    """Render a small DataFrame without an optional tabulate dependency."""
    display = frame.copy()

    def render(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{decimals}f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(map(str, display.columns)) + " |"
    divider = "|" + "|".join(["---"] * len(display.columns)) + "|"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    if TARGET not in train or ID_COLUMN not in train:
        raise ValueError("필수 컬럼이 없습니다.")
    features = train.drop(columns=[TARGET, ID_COLUMN])
    target = train[TARGET].astype(float)
    n_rows = len(train)

    categorical = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numerical = features.select_dtypes(include=["number"]).columns.tolist()

    duplicate_mask = features.duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_groups = int(
        features.loc[duplicate_mask].drop_duplicates().shape[0]
    )
    duplicate_target_spread = (
        train.loc[duplicate_mask]
        .groupby(list(features.columns), dropna=False)[TARGET]
        .agg(["count", "nunique", "min", "max"])
    )
    conflicting_groups = int((duplicate_target_spread["nunique"] > 1).sum())

    missing = features.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    numeric_rows = []
    for column in numerical:
        values = features[column]
        q1, median, q3 = values.quantile([0.25, 0.50, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((values < lower) | (values > upper)).sum())
        numeric_rows.append(
            {
                "feature": column,
                "min": values.min(),
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": values.max(),
                "iqr_outliers": outliers,
                "outlier_rate": outliers / n_rows,
                "pearson": values.corr(target, method="pearson"),
                "spearman": values.corr(target, method="spearman"),
            }
        )
    numeric_profile = pd.DataFrame(numeric_rows)
    numeric_decile_rows = []
    for column in numerical:
        bins = pd.qcut(features[column], 10, duplicates="drop")
        means = target.groupby(bins, observed=True).mean().to_numpy()
        numeric_decile_rows.append(
            {
                "feature": column,
                **{
                    f"d{index + 1}": value
                    for index, value in enumerate(means)
                },
            }
        )
    numeric_deciles = pd.DataFrame(numeric_decile_rows)

    mi_frame = features.copy()
    discrete = []
    for index, column in enumerate(mi_frame.columns):
        if column in categorical:
            mi_frame[column], _ = pd.factorize(
                mi_frame[column].astype("string").fillna("<missing>"),
                sort=True,
            )
            discrete.append(index)
        else:
            mi_frame[column] = mi_frame[column].fillna(
                mi_frame[column].median()
            )
    mutual_information = mutual_info_regression(
        mi_frame,
        target,
        discrete_features=np.array(
            [i in discrete for i in range(mi_frame.shape[1])]
        ),
        random_state=42,
    )
    mi_table = pd.DataFrame(
        {"feature": mi_frame.columns, "mutual_information": mutual_information}
    ).sort_values("mutual_information", ascending=False)

    category_rows = []
    for column in categorical:
        grouped = (
            train.groupby(column, dropna=False)[TARGET]
            .agg(["count", "mean", "median"])
            .reset_index()
        )
        max_gap = grouped["mean"].max() - grouped["mean"].min()
        category_rows.append(
            {
                "feature": column,
                "levels": features[column].nunique(dropna=False),
                "min_level_count": int(grouped["count"].min()),
                "max_target_mean_gap": max_gap,
            }
        )
    category_profile = pd.DataFrame(category_rows).sort_values(
        "max_target_mean_gap", ascending=False
    )

    target_scaled = target * 100
    on_cent_grid = np.isclose(target_scaled, np.round(target_scaled), atol=1e-10)
    target_counts = target.value_counts().head(10)
    baseline_mae = float(np.mean(np.abs(target - target.median())))

    domain_checks = {
        "height_nonpositive": int((features["height"] <= 0).sum()),
        "weight_nonpositive": int((features["weight"] <= 0).sum()),
        "systolic_below_diastolic": int(
            (
                features["systolic_blood_pressure"]
                < features["diastolic_blood_pressure"]
            ).sum()
        ),
        "glucose_nonpositive": int((features["glucose"] <= 0).sum()),
        "target_outside_0_1": int(((target < 0) | (target > 1)).sum()),
    }

    lines = [
        "# V7 Train-only 데이터 품질 및 타깃 구조 점검",
        "",
        "## 데이터와 grain",
        "",
        f"- 행: {n_rows:,}",
        f"- 입력 피처: {features.shape[1]}",
        f"- 수치형/범주형: {len(numerical)}/{len(categorical)}",
        f"- ID 중복: {int(train[ID_COLUMN].duplicated().sum()):,}",
        f"- 입력값이 완전히 같은 행: {format_rate(duplicate_rows, n_rows)}",
        f"- 완전 중복 입력 그룹: {duplicate_groups:,}",
        f"- 같은 입력이지만 target이 다른 그룹: {conflicting_groups:,}",
        "",
        "## Target",
        "",
        f"- 범위: {target.min():.6f} ~ {target.max():.6f}",
        f"- 평균/중앙값/표준편차: {target.mean():.6f} / {target.median():.6f} / {target.std():.6f}",
        f"- 고유값 수: {target.nunique():,}",
        f"- 0.01 격자 비율: {on_cent_grid.mean():.2%}",
        f"- 전체 Train 중앙값 상수 예측 MAE: {baseline_mae:.6f}",
        "",
        "상위 빈도 target:",
        "",
        markdown_table(
            target_counts.rename_axis("target").reset_index(name="count")
        ),
        "",
        "## 결측",
        "",
        (
            missing.rename("count")
            .to_frame()
            .assign(rate=lambda frame: frame["count"] / n_rows)
            .reset_index(names="feature")
            .pipe(markdown_table, decimals=4)
            if len(missing)
            else "결측값 없음."
        ),
        "",
        "## 수치 피처",
        "",
        markdown_table(
            numeric_profile.sort_values(
                "spearman", key=lambda s: s.abs(), ascending=False
            ),
            decimals=5,
        ),
        "",
        "### 수치 피처 분위별 target 평균",
        "",
        markdown_table(numeric_deciles, decimals=4),
        "",
        "## 범주형 피처",
        "",
        markdown_table(category_profile, decimals=5),
        "",
        "## 단변량 상호정보량",
        "",
        markdown_table(mi_table, decimals=6),
        "",
        "## 도메인 유효성 점검",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {"check": key, "count": value}
                    for key, value in domain_checks.items()
                ]
            )
        ),
        "",
        "## 해석 원칙",
        "",
        "- 이 보고서는 Train만 읽습니다. Test 분포와 통계는 확인하지 않습니다.",
        "- IQR 이상치는 오류로 단정하지 않고 모델별 강건성 비교 대상으로만 사용합니다.",
        "- 상호정보량과 범주별 target 차이는 후보 우선순위 결정용이며, 최종 성능은 Fold 밖 예측으로만 판단합니다.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
