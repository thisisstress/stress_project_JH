# -*- coding: utf-8 -*-
"""Build and execute the reader-facing V7 Train-only analysis notebook."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Evaluated by jupyter_core at import time. The managed Windows sandbox blocks
# ACL mutation but still confines all runtime files to the workspace.
os.environ.setdefault("JUPYTER_ALLOW_INSECURE_WRITES", "true")

import nbformat as nbf
from ipykernel.kernelspec import write_kernel_spec
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    os.environ["HACKATHON_DATA_DIR"] = str(args.data_dir.resolve())
    os.environ["HACKATHON_RESULTS_DIR"] = str(args.results_dir.resolve())
    runtime_root = args.output.parent / ".v7_notebook_runtime"
    ipython_dir = runtime_root / "ipython"
    jupyter_data = runtime_root / "share" / "jupyter"
    os.environ["IPYTHONDIR"] = str(ipython_dir.resolve())
    os.environ["JUPYTER_CONFIG_DIR"] = str(
        (runtime_root / "jupyter_config").resolve()
    )
    os.environ["JUPYTER_DATA_DIR"] = str(jupyter_data.resolve())
    os.environ["JUPYTER_PATH"] = str(jupyter_data.resolve())
    os.environ["JUPYTER_RUNTIME_DIR"] = str(
        (runtime_root / "runtime").resolve()
    )
    # The managed Windows sandbox blocks ACL mutation on temporary connection
    # files, so Jupyter must use ordinary workspace writes for this local run.
    os.environ["JUPYTER_ALLOW_INSECURE_WRITES"] = "true"
    kernel_dir = jupyter_data / "kernels" / "stress-v7"
    kernel_dir.parent.mkdir(parents=True, exist_ok=True)
    if not kernel_dir.exists():
        write_kernel_spec(
            path=kernel_dir,
            overrides={"display_name": "Stress V7 Python"},
        )

    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Stress V7 Python",
        "language": "python",
        "name": "stress-v7",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# V7 Train-only 데이터 진단과 모델 선택

## tl;dr

- Train 3,000행, 입력 피처 16개이며 ID 중복은 없습니다.
- target은 0~1 범위의 0.01 격자 101개 값입니다.
- CatBoost·LightGBM·XGBoost·스플라인·KNN·RandomForest는 V6보다 나빴습니다.
- 독립적인 2차원 피처쌍 1-NN 보조 모델을 15% 혼합한 V7은 고정 후 Audit3에서 V6보다 평균 MAE가 0.000374 낮았고 Seed 3/3을 이겼습니다.
- Test는 이 노트북에서 읽지 않습니다."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

목표는 대회 규정을 지키면서 MAE가 낮은 후보를 찾는 것입니다.

### Key Assumptions

- 모델과 전처리 선택은 Train-only Fold 예측으로 수행합니다.
- 동일 입력 중복 행은 같은 Fold에 둡니다.
- Test 통계·행 수·인덱스·다른 Test 행 정보는 사용하지 않습니다.
- 첫 화면 Seed에서 후보를 고른 뒤 남은 Development Seed와 새 Audit3 Seed에서는 설정을 변경하지 않습니다."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import os
import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ["HACKATHON_DATA_DIR"])
RESULTS_DIR = Path(os.environ["HACKATHON_RESULTS_DIR"])
train = pd.read_csv(DATA_DIR / "train.csv")
TARGET = "stress_score"
ID_COLUMN = "ID"
print({"rows": len(train), "columns": len(train.columns), "input_features": len(train.columns) - 2})"""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            """features = train.drop(columns=[TARGET, ID_COLUMN])
quality = pd.DataFrame({
    "value": [
        train[ID_COLUMN].duplicated().sum(),
        features.duplicated(keep=False).sum(),
        train[TARGET].nunique(),
        train[TARGET].min(),
        train[TARGET].max(),
        np.isclose(train[TARGET] * 100, np.round(train[TARGET] * 100)).mean(),
    ]
}, index=[
    "duplicate_id_rows",
    "duplicate_feature_rows",
    "target_unique_values",
    "target_min",
    "target_max",
    "target_on_0.01_grid_rate",
])
quality"""
        ),
        nbf.v4.new_code_cell(
            """missing = features.isna().sum().sort_values(ascending=False)
pd.DataFrame({"missing_count": missing[missing > 0], "missing_rate": missing[missing > 0] / len(train)})"""
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            """boosting = pd.read_csv(RESULTS_DIR / "v7_boosting_screen" / "v7_family_screen_seed42_summary.csv")
local = pd.read_csv(RESULTS_DIR / "v7_local_screen" / "v7_family_screen_seed42_summary.csv")
pair = pd.read_csv(RESULTS_DIR / "v7_pair_neighbor_seed42_summary.csv")

screen = pd.DataFrame([
    {"family": "Boosting best", **boosting.iloc[0][["candidate", "mean_mae"]].to_dict()},
    {"family": "Local baseline best", **local.iloc[0][["candidate", "mean_mae"]].to_dict()},
    {"family": "Pair-neighbor best", **pair.iloc[0][["candidate", "mean_mae"]].to_dict()},
    {"family": "V6 reference on Seed 42", "candidate": "V6", "mean_mae": 0.15128333333333335},
])
screen.sort_values("mean_mae")"""
        ),
        nbf.v4.new_code_cell(
            """audit3 = pd.read_csv(RESULTS_DIR / "v7_pair_blend_audit3_summary.csv")
audit3[[
    "v6_mean_mae",
    "pair_mean_mae",
    "blend_mean_mae",
    "mean_delta_vs_v6",
    "seed_win_rate",
    "paired_ci95_lower",
    "paired_ci95_upper",
]]"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. 이 데이터에서는 평활화가 강한 부스팅·선형 스플라인보다 세밀한 국소 이웃 구조가 유리했습니다.
2. Pair-neighbor 단독은 V6보다 약했지만 오차가 달라 15% 혼합 시 Development 확인 Seed와 Audit3 Seed를 모두 이겼습니다.
3. Audit3 신뢰구간 상한은 0.000044로 0을 아주 조금 포함합니다. 따라서 개선 가능성은 일관되지만 Public 성능을 보장할 수는 없습니다.
4. 이상치는 Train에서 오류로 확인되지 않았으므로 임의 삭제하지 않았습니다.
5. 최종 코드는 Test 각 행을 Train 기준 경험적 순위에만 투영하므로 Test 전체 통계를 사용하지 않습니다."""
        ),
    ]

    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="stress-v7",
        resources={"metadata": {"path": str(args.output.parent.resolve())}},
    )
    client.execute()
    nbf.write(notebook, args.output)
    print(f"Saved and executed: {args.output.resolve()}")


if __name__ == "__main__":
    main()
