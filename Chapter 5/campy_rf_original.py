"""Reproduce the original Table 3 (sample-level split) on the rebuilt data.

Mirrors Chapter 5/campy_rf_{feces,soil}.py: stratified 80/20 sample-level split
(random_state 151 feces / 120 soil), SMOTE(random_state=1213) on the soil
training set, RandomForestClassifier(random_state=120). Instead of re-running
the 1,000-combination randomized search, it fits the best hyperparameters
reported in section 2.4.

Run (from this folder): python campy_rf_original.py
"""

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from campy_rf_data import as_published, load

# Table 3 of manuscript v3: tp, fp, fn, tn, AUC
PUBLISHED = {"Feces": (89, 13, 4, 49, 0.959), "Soil": (26, 13, 10, 106, 0.886)}

CONFIG = {
    "Feces": dict(split_seed=151, smote=False,
                  rf=dict(max_depth=None, max_features=80, min_samples_split=8,
                          min_samples_leaf=1, criterion="entropy", n_estimators=500)),
    "Soil": dict(split_seed=120, smote=True,
                 rf=dict(max_depth=None, max_features=41, min_samples_split=4,
                         min_samples_leaf=1, criterion="entropy", n_estimators=800)),
}


def run(frame, sample_type):
    cfg = CONFIG[sample_type]
    sub = frame[frame.SampleType == sample_type]
    X = sub.iloc[:, 8:].values
    y = sub["CampyCef"].map({"+": 1, "-": 0}).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg["split_seed"], stratify=y)
    if cfg["smote"]:
        X_train, y_train = SMOTE(random_state=1213).fit_resample(X_train, y_train)
    rf = RandomForestClassifier(random_state=120, n_jobs=-1, **cfg["rf"]).fit(X_train, y_train)
    tn, fp, fn, tp = confusion_matrix(y_test, rf.predict(X_test)).ravel()
    auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    return dict(n=len(sub), test_pos=int(y_test.sum()), test_neg=int((1 - y_test).sum()),
                tp=tp, fp=fp, fn=fn, tn=tn, sens=tp / (tp + fn), spec=tn / (tn + fp), auc=auc)


def variants(frame):
    """Row-selection guesses for how the final file differed from the 2019 copy."""
    yield "as_published (blank weather filled from same farm-date; minus feces H1-1)", as_published(frame)
    complete = frame.dropna(subset=list(frame.columns[8:]) + ["CampyCef"])
    yield "complete rows only", complete
    no_h11 = complete[~((complete.SampleID == "H1-1") & (complete.SampleType == "Feces"))]
    yield "complete rows, minus H1-1 (blank Capetown label)", no_h11
    yield "complete rows, minus H1-1, duplicate SampleIDs de-duplicated", \
        no_h11.drop_duplicates(subset=["SampleID", "SampleType"], keep="first")


def main():
    frame = load()
    for name, f in variants(frame):
        print(f"\n=== Variant: {name}")
        for st in ("Feces", "Soil"):
            r = run(f, st)
            ptp, pfp, pfn, ptn, pauc = PUBLISHED[st]
            print(f"  {st:5s} n={r['n']}  test +/-={r['test_pos']}/{r['test_neg']} "
                  f"(published {ptp + pfn}/{pfp + ptn})")
            print(f"        ours      TP={r['tp']:3d} FP={r['fp']:3d} FN={r['fn']:3d} TN={r['tn']:3d} "
                  f"sens={r['sens']:.3f} spec={r['spec']:.3f} AUC={r['auc']:.3f}")
            print(f"        published TP={ptp:3d} FP={pfp:3d} FN={pfn:3d} TN={ptn:3d} "
                  f"sens={ptp / (ptp + pfn):.3f} spec={ptn / (ptn + pfp):.3f} AUC={pauc:.3f}")


if __name__ == "__main__":
    main()
