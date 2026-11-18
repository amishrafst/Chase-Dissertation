"""Evaluate the Campylobacter models with validation that keeps each farm visit whole.

All samples from one farm on one day share the same 80 weather predictors, and
the 80/20 split in campy_rf_feces.py / campy_rf_soil.py is at the sample level,
so a farm-day could sit in both training and test sets. This script evaluates
the same pipeline under splits that keep each farm-day (or farm, or year) whole:

  sample-level CV     10-fold stratified, 5 repeats   (the leaky scheme, for contrast)
  farm-day grouped CV 10-fold StratifiedGroupKFold by farm-day, 5 repeats
  leave-one-farm-out  11 folds, one farm held out at a time
  temporal            train 2014-2015, test 2016-2017; and train 2014-2016, test 2017

SMOTE (soil only, as published) runs inside each training fold via an imblearn
Pipeline. Models: RF with the published hyperparameters, L2 logistic
regression, gradient boosting, a no-skill baseline, and a calendar-only
baseline (day of year) as a check on how much the weather adds beyond season.

AUC for repeated CV is the pooled out-of-fold AUC averaged over repeats; the 95%
CI is a farm-day cluster bootstrap on the first repeat. Sensitivity and
specificity use the 0.5 probability threshold (what predict() does).

Run (from this folder): python campy_rf_validation.py
(needs the workbook in data/; see campy_rf_data.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from campy_rf_data import as_published, load
from campy_rf_original import CONFIG

RESULTS = Path(__file__).resolve().parent / "results"
REPEATS = 5
N_BOOT = 1000

# Data-entry slips in the 2019 workbook, keyed by the raw SamplingDate string
DATE_TYPOS = {"71520141": "7152014"}  # feces I4; the soil rows read 7152014


def parse_sampling_date(value):
    """SamplingDate is M(M)DDYYYY with no separators, e.g. 3312014 -> 2014-03-31."""
    s = str(value).strip()
    s = DATE_TYPOS.get(s, s)
    if not s.isdigit() or len(s) not in (7, 8):
        raise ValueError(f"unparseable SamplingDate {value!r}")
    return pd.Timestamp(int(s[-4:]), int(s[:-6]), int(s[-6:-4]))


def farm_day_groups(frame):
    """Integer group id per row; rows from the same farm on the same day share one."""
    days = frame["SamplingDate"].map(parse_sampling_date).dt.strftime("%Y-%m-%d")
    return pd.factorize(frame["Farm"].astype(str).values + "|" + days.values)[0]


def grouped_folds(y, groups, n_splits=10, seed=0):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def temporal_mask(dates, train_years, test_years):
    years = pd.Series(dates).map(parse_sampling_date).dt.year
    return years.isin(train_years).values, years.isin(test_years).values


def summarize(y, prob, threshold=0.5):
    """Confusion counts, sensitivity, specificity and AUC; NaN where undefined."""
    y = np.asarray(y)
    pred = (np.asarray(prob) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return dict(n=len(y), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                sens=tp / (tp + fn) if tp + fn else np.nan,
                spec=tn / (tn + fp) if tn + fp else np.nan,
                auc=roc_auc_score(y, prob) if both else np.nan)


def cluster_bootstrap_auc(y, prob, groups, n_boot=N_BOOT, seed=0):
    """95% percentile CI for AUC, resampling whole farm-days."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in ids}
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[g] for g in rng.choice(ids, len(ids))])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], prob[idx]))
    if not aucs:
        return np.nan, np.nan
    return tuple(np.percentile(aucs, [2.5, 97.5]))


# ---------------------------------------------------------------- models

def features(frame, kind):
    if kind == "weather":
        return frame.iloc[:, 8:].values
    doy = frame["SamplingDate"].map(parse_sampling_date).dt.dayofyear.values
    angle = 2 * np.pi * doy / 365.25
    return np.column_stack([np.sin(angle), np.cos(angle)])


MODELS = {
    # name: (feature set, builder(sample_type, seed))
    "RF (published settings)": ("weather", lambda st, seed: Pipeline(
        _smote(st) + [("clf", RandomForestClassifier(random_state=120, n_jobs=-1,
                                                     **CONFIG[st]["rf"]))])),
    "Logistic regression (L2)": ("weather", lambda st, seed: Pipeline(
        [("scale", StandardScaler())] + _smote(st)
        + [("clf", LogisticRegression(C=0.1, max_iter=5000))])),
    "Gradient boosting": ("weather", lambda st, seed: Pipeline(
        _smote(st) + [("clf", HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
            random_state=seed))])),
    "Calendar only (day of year)": ("calendar", lambda st, seed: Pipeline(
        _smote(st) + [("clf", LogisticRegression(max_iter=5000))])),
    "No-skill (prevalence)": ("weather", lambda st, seed: DummyClassifier(strategy="prior")),
}


def _smote(sample_type):
    return [("smote", SMOTE(random_state=1213))] if CONFIG[sample_type]["smote"] else []


# ---------------------------------------------------------------- schemes

def oof(build, X, y, splits):
    prob = np.full(len(y), np.nan)
    fold_aucs = []
    for train, test in splits:
        m = build().fit(X[train], y[train])
        prob[test] = m.predict_proba(X[test])[:, 1]
        if len(np.unique(y[test])) == 2:
            fold_aucs.append(roc_auc_score(y[test], prob[test]))
    return prob, fold_aucs


def evaluate(sub, sample_type):
    y = sub["CampyCef"].map({"+": 1, "-": 0}).values
    groups = farm_day_groups(sub)
    farms = sub["Farm"].values
    rows, lofo_rows = [], []
    for name, (kind, builder) in MODELS.items():
        X = features(sub, kind)
        build = lambda seed=0: builder(sample_type, seed)

        for scheme, make_splits in (
                ("Sample-level CV (leaky)", lambda r: list(StratifiedKFold(
                    10, shuffle=True, random_state=r).split(X, y))),
                ("Farm-day grouped CV", lambda r: grouped_folds(y, groups, 10, r))):
            reps = []
            for r in range(REPEATS):
                prob, fold_aucs = oof(lambda: build(r), X, y, make_splits(r))
                reps.append((summarize(y, prob), fold_aucs, prob))
            lo, hi = cluster_bootstrap_auc(y, reps[0][2], groups)
            rows.append(_row(sample_type, scheme, name, reps, lo, hi))

        prob, fold_aucs = oof(build, X, y, list(LeaveOneGroupOut().split(X, y, farms)))
        lo, hi = cluster_bootstrap_auc(y, prob, groups)
        rows.append(_row(sample_type, "Leave-one-farm-out", name,
                         [(summarize(y, prob), fold_aucs, prob)], lo, hi))
        for farm in sorted(set(farms)):
            m = farms == farm
            s = summarize(y[m], prob[m])
            lofo_rows.append(dict(sample_type=sample_type, model=name, farm=farm,
                                  n=int(m.sum()), positives=int(y[m].sum()),
                                  predicted_pos=int(s["tp"] + s["fp"]),
                                  accuracy=(s["tp"] + s["tn"]) / s["n"], auc=s["auc"]))

        for scheme, tr_years, te_years in (
                ("Temporal: train 2014-15, test 2016-17", (2014, 2015), (2016, 2017)),
                ("Temporal: train 2014-16, test 2017", (2014, 2015, 2016), (2017,))):
            tr, te = temporal_mask(sub["SamplingDate"], tr_years, te_years)
            m = build().fit(X[tr], y[tr])
            prob = m.predict_proba(X[te])[:, 1]
            lo, hi = cluster_bootstrap_auc(y[te], prob, groups[te])
            rows.append(_row(sample_type, scheme, name, [(summarize(y[te], prob), [], prob)], lo, hi))
        print(f"  done: {sample_type} / {name}", flush=True)
    return rows, lofo_rows


def _row(sample_type, scheme, model, reps, lo, hi):
    first = reps[0][0]
    fold_aucs = [a for _, fa, _ in reps for a in fa]
    mean = lambda k: float(np.nanmean([s[k] for s, _, _ in reps])) if any(
        not np.isnan(s[k]) for s, _, _ in reps) else np.nan
    return dict(sample_type=sample_type, scheme=scheme, model=model, n=first["n"],
                positives=first["tp"] + first["fn"], auc=mean("auc"), auc_ci_lo=lo,
                auc_ci_hi=hi, sens=mean("sens"), spec=mean("spec"),
                tp=first["tp"], fp=first["fp"], fn=first["fn"], tn=first["tn"],
                fold_auc_mean=float(np.mean(fold_aucs)) if fold_aucs else np.nan,
                fold_auc_sd=float(np.std(fold_aucs)) if fold_aucs else np.nan)


def main():
    frame = as_published(load())
    RESULTS.mkdir(exist_ok=True)
    rows, lofo = [], []
    for st in ("Feces", "Soil"):
        r, l = evaluate(frame[frame.SampleType == st].reset_index(drop=True), st)
        rows += r
        lofo += l
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "validation_schemes.csv", index=False)
    pd.DataFrame(lofo).to_csv(RESULTS / "lofo_by_farm.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200,
                           "display.float_format", "{:.3f}".format):
        print(res[["sample_type", "scheme", "model", "n", "positives", "auc",
                   "auc_ci_lo", "auc_ci_hi", "sens", "spec"]].to_string(index=False))


if __name__ == "__main__":
    main()
