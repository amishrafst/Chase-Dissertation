"""Discrimination (AUC) of the 2019 forecast-based predictions.

Input: data/2019 forecast validation.xlsx, sheet "Predictions". Its MAD and
bias values reproduce Table 4 exactly, so it is the data behind the published
forecast validation. Each row is one sampling day: observed prevalence
("Actual", five samples per day) and the predicted probability made 1..7 days
ahead ("Predicted_k"). Every sample on a day gets that day's prediction, so the
sample-level AUC is computed by expanding each day into its five samples.

Needs numpy, pandas, scikit-learn and openpyxl.
Run (from this folder): python forecast_2019_auc.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
WORKBOOK = DATA / "2019 forecast validation.xlsx"
SAMPLES_PER_DAY = 5


def expand_days(prevalence, prediction, n=SAMPLES_PER_DAY):
    """Per-sample outcomes and scores from per-day prevalence and prediction."""
    y, score = [], []
    for prev, pred in zip(prevalence, prediction):
        positives = round(prev * n)
        if abs(positives - prev * n) > 1e-6:
            raise ValueError(f"prevalence {prev} is not a multiple of 1/{n}")
        y += [1] * positives + [0] * (n - positives)
        score += [pred] * n
    return np.array(y), np.array(score)


def auc_by_lead(days):
    rows = []
    for st, g in days.groupby("Sample type"):
        for k in range(1, 8):
            y, s = expand_days(g["Actual"], g[f"Predicted_{k}"])
            rows.append(dict(sample_type=st, lead_days=k, n_days=len(g), n_samples=len(y),
                             positives=int(y.sum()), auc=roc_auc_score(y, s)))
    return pd.DataFrame(rows)


def main():
    days = pd.read_excel(WORKBOOK, sheet_name="Predictions").dropna(subset=["Sample type"])
    out = auc_by_lead(days)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "forecast_2019_auc.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
