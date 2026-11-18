"""Nested, farm-day-grouped tuning of the random forest.

The published hyperparameters were chosen by a randomized search over
sample-level CV folds, which share farm-days with each other, so the search
rewarded settings that memorize a day (e.g. min_samples_leaf=1). Here the
search itself is grouped: outer 10-fold StratifiedGroupKFold by farm-day for
evaluation; inside each outer training set, a randomized search scored by AUC
on a 5-fold grouped inner CV. SMOTE (soil) stays inside the pipeline.

Run (from this folder, after campy_rf_validation.py): python campy_rf_tuned.py
"""

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from campy_rf_data import as_published, load
from campy_rf_validation import (RESULTS, _smote, cluster_bootstrap_auc, farm_day_groups,
                        features, grouped_folds, summarize)

N_ITER = 25
SPACE = {
    "clf__max_depth": [3, 5, 10, 20, None],
    "clf__max_features": ["sqrt", 0.1, 0.3, 0.6, 1.0],
    "clf__min_samples_leaf": [1, 5, 10, 20, 40],
    "clf__min_samples_split": [2, 10, 20],
    "clf__criterion": ["gini", "entropy"],
}


def tuned_search(sample_type, seed):
    """Randomized RF search scored on a 5-fold farm-day-grouped inner CV (fit with groups=)."""
    pipe = Pipeline(_smote(sample_type) + [("clf", RandomForestClassifier(
        n_estimators=300, random_state=120, n_jobs=-1))])
    return RandomizedSearchCV(
        pipe, SPACE, n_iter=N_ITER, scoring="roc_auc", random_state=seed,
        cv=StratifiedGroupKFold(5, shuffle=True, random_state=seed), n_jobs=1)


def nested(sub, sample_type, seed=0):
    X, y = features(sub, "weather"), sub["CampyCef"].map({"+": 1, "-": 0}).values
    groups = farm_day_groups(sub)
    prob = np.full(len(y), np.nan)
    picks = []
    for k, (train, test) in enumerate(grouped_folds(y, groups, 10, seed)):
        search = tuned_search(sample_type, seed + k)
        search.fit(X[train], y[train], groups=groups[train])
        prob[test] = search.predict_proba(X[test])[:, 1]
        picks.append(dict(sample_type=sample_type, fold=k, inner_auc=search.best_score_,
                          outer_auc=roc_auc_score(y[test], prob[test]),
                          **{p.replace("clf__", ""): v for p, v in search.best_params_.items()}))
        print(f"  {sample_type} fold {k}: inner {search.best_score_:.3f} "
              f"outer {picks[-1]['outer_auc']:.3f} {search.best_params_}", flush=True)
    s = summarize(y, prob)
    lo, hi = cluster_bootstrap_auc(y, prob, groups)
    return dict(sample_type=sample_type, model="RF (grouped nested tuning)",
                auc=s["auc"], auc_ci_lo=lo, auc_ci_hi=hi, sens=s["sens"], spec=s["spec"],
                fold_auc_mean=np.mean([p["outer_auc"] for p in picks])), picks


def main():
    frame = as_published(load())
    rows, picks = [], []
    for st in ("Feces", "Soil"):
        r, p = nested(frame[frame.SampleType == st].reset_index(drop=True), st)
        rows.append(r)
        picks += p
    pd.DataFrame(picks).to_csv(RESULTS / "tuned_rf_folds.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "tuned_rf.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
