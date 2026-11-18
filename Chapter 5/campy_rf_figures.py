"""Table 3, Table S1, variable importance and Figs. 2, 3 and S1.

Everything is evaluated on the same 10 farm-day-grouped outer folds (seed 0) used
by campy_rf_tuned.py, so the RF numbers match tuned_rf.csv. Per outer fold:
  - RF: grouped nested randomized search (campy_rf_tuned.tuned_search)
  - RF with the published hyperparameters (the model behind the 2019 forecast
    validation, Table 4), evaluated on the same grouped folds
  - L2 logistic regression and gradient boosting (campy_rf_validation.MODELS)
  - no-skill baseline
Variable importance is a grouped permutation importance on the held-out fold:
all 8 columns of one weather variable (1 day, 2 days, averages 1-2 .. 1-7) are
permuted together and the drop in AUC is recorded, then averaged over folds.
Permuting the 8 near-duplicate columns as a block avoids the dilution that
per-column importance suffers when predictors are this correlated.

Fig. S1 contrasts sample-level and grouped evaluation of the same model (RF with
the published settings).

Outputs: results/table3_revised.{csv,md}, results/table_s1_validation_schemes.md,
results/importance_by_variable.csv, results/fig2_roc_grouped.png,
results/figS1_roc_leaky_vs_grouped.png, results/fig3_importance.png;
per-sample out-of-fold predictions go to data/oof_predictions.csv.

Run (from this folder, after campy_rf_validation.py): python campy_rf_figures.py
Redraw Figs. 2 and S1 from the saved predictions, without refitting:
python campy_rf_figures.py --replot
"""

import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from campy_rf_data import as_published, load
from campy_rf_validation import (MODELS, RESULTS, cluster_bootstrap_auc, farm_day_groups,
                        features, grouped_folds, summarize)
from campy_rf_tuned import tuned_search

DATA = RESULTS.parent / "data"

LABELS = {
    "MaximumWindSpeed": "Maximum wind speed", "AverageWindSpeed": "Average wind speed",
    "MaxGustSpeed": "Maximum gust speed", "MinHumidity": "Minimum humidity",
    "MaxHumidity": "Maximum humidity", "AverageHumidity": "Average humidity",
    "MinTemperature": "Minimum temperature", "MaxTemperature": "Maximum temperature",
    "AverageTemperature": "Average temperature", "Precipitation": "Precipitation",
}

# Okabe-Ito palette plus distinct line styles, so curves read without colour
STYLE = {
    "Random forest": dict(color="#0072B2", ls="-"),
    "Random forest, published settings": dict(color="#CC79A7", ls=(0, (6, 2, 1, 2))),
    "Logistic regression": dict(color="#E69F00", ls="--"),
    "Gradient boosting": dict(color="#009E73", ls="-."),
}
FIG2 = {  # legend label: (out-of-fold column, style)
    "Random forest, published settings": ("grouped_random_forest_published_settings",
                                          STYLE["Random forest, published settings"]),
    "Random forest, re-tuned": ("grouped_random_forest", STYLE["Random forest"]),
    "Logistic regression": ("grouped_logistic_regression", STYLE["Logistic regression"]),
    "Gradient boosting": ("grouped_gradient_boosting", STYLE["Gradient boosting"]),
}
FIGS1 = {
    "Sample-level CV": ("rf_sample_level_published", dict(color="#D55E00", ls="--")),
    "CV grouped by farm visit": ("grouped_random_forest_published_settings", dict(color="#0072B2", ls="-")),
}
GROUPED_MODELS = {"Random forest, published settings": "RF (published settings)",
                  "Logistic regression": "Logistic regression (L2)",
                  "Gradient boosting": "Gradient boosting",
                  "No-skill baseline": "No-skill (prevalence)"}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def variable_blocks(columns):
    """{weather variable stem: [column indices]} for the 8-column predictor blocks."""
    blocks = {}
    for start in range(0, len(columns), 8):
        first = columns[start]
        if not first.endswith("OneDayBefore"):
            raise ValueError(f"block at column {start} starts with {first!r}")
        blocks[first[: -len("OneDayBefore")]] = list(range(start, start + 8))
    return blocks


def permute_block(X, cols, rng):
    """Copy of X with the rows of columns `cols` shuffled together."""
    out = X.copy()
    out[:, cols] = X[rng.permutation(len(X))][:, cols]
    return out


def block_permutation_drop(model, X, y, blocks, n_repeats=10, seed=0):
    """Mean AUC drop when each block of columns is permuted (held-out data)."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    return {name: base - np.mean([roc_auc_score(y, model.predict_proba(
                permute_block(X, cols, rng))[:, 1]) for _ in range(n_repeats)])
            for name, cols in blocks.items()}


def run_sample_type(sub, sample_type):
    X, y = features(sub, "weather"), sub["CampyCef"].map({"+": 1, "-": 0}).values
    groups = farm_day_groups(sub)
    blocks = variable_blocks(list(sub.columns[8:]))
    prob = {name: np.full(len(y), np.nan) for name in ["Random forest", *GROUPED_MODELS]}
    importance = []
    for k, (train, test) in enumerate(grouped_folds(y, groups, 10, seed=0)):
        search = tuned_search(sample_type, k).fit(X[train], y[train], groups=groups[train])
        prob["Random forest"][test] = search.predict_proba(X[test])[:, 1]
        drops = block_permutation_drop(search.best_estimator_, X[test], y[test], blocks, seed=k)
        importance += [dict(sample_type=sample_type, fold=k, variable=LABELS[v], auc_drop=d)
                       for v, d in drops.items()]
        for name, key in GROUPED_MODELS.items():
            m = MODELS[key][1](sample_type, 0).fit(X[train], y[train])
            prob[name][test] = m.predict_proba(X[test])[:, 1]
        print(f"  {sample_type} fold {k} done", flush=True)

    leaky = np.full(len(y), np.nan)
    for train, test in StratifiedKFold(10, shuffle=True, random_state=0).split(X, y):
        m = MODELS["RF (published settings)"][1](sample_type, 0).fit(X[train], y[train])
        leaky[test] = m.predict_proba(X[test])[:, 1]

    rows = []
    for name, p in prob.items():
        s = summarize(y, p)
        lo, hi = cluster_bootstrap_auc(y, p, groups)
        rows.append(dict(sample_type=sample_type, model=name, n=len(y), positives=int(y.sum()),
                         auc=s["auc"], auc_ci_lo=lo, auc_ci_hi=hi, sens=s["sens"],
                         spec=s["spec"], tp=s["tp"], fp=s["fp"], fn=s["fn"], tn=s["tn"]))
    oof = pd.DataFrame({"SampleID": sub["SampleID"], "Farm": sub["Farm"],
                        "SamplingDate": sub["SamplingDate"], "y": y, "group": groups,
                        "rf_sample_level_published": leaky,
                        **{f"grouped_{slug(n)}": p
                           for n, p in prob.items()}})
    return rows, importance, oof


def roc_panel(ax, y, curves, title):
    for name, (p, style) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {roc_auc_score(y, p):.2f})", **style)
    ax.plot([0, 1], [0, 1], color="0.5", ls=":", lw=1.5, label="No skill (AUC 0.50)")
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8, frameon=False)


def fmt_table(res):
    out = res.assign(**{
        "AUC (95% CI)": res.apply(lambda r: f"{r.auc:.2f} ({r.auc_ci_lo:.2f}-{r.auc_ci_hi:.2f})", axis=1),
        "Sensitivity": res.sens.map("{:.2f}".format), "Specificity": res.spec.map("{:.2f}".format),
        "TP / FP / FN / TN": res.apply(lambda r: f"{r.tp} / {r.fp} / {r.fn} / {r.tn}", axis=1)})
    return out.rename(columns={"sample_type": "Sample type", "model": "Model"})[
        ["Sample type", "Model", "AUC (95% CI)", "Sensitivity", "Specificity", "TP / FP / FN / TN"]]


def validation_schemes_table():
    r = pd.read_csv(RESULTS / "validation_schemes.csv")
    r = r[r.model != "Calendar only (day of year)"]
    # Pooled AUC is biased low under leave-one-farm-out (the no-skill model
    # scores ~0.2 pooled); report the mean of per-farm AUCs there instead.
    lofo = r.scheme == "Leave-one-farm-out"
    r.loc[lofo, "auc"] = r.loc[lofo, "fold_auc_mean"]
    t = r.pivot_table(index=["sample_type", "model"], columns="scheme", values="auc", sort=False)
    return t.map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")


def draw_roc(oof, spec, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (letter, st) in zip(axes, (("a", "Feces"), ("b", "Soil"))):
        o = oof[oof.sample_type == st]
        roc_panel(ax, o.y, {label: (o[col], style) for label, (col, style) in spec.items()},
                  f"({letter}) {st}")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def draw_roc_figures(oof):
    draw_roc(oof, FIG2, RESULTS / "fig2_roc_grouped.png")
    draw_roc(oof, FIGS1, RESULTS / "figS1_roc_leaky_vs_grouped.png")


def main():
    if "--replot" in sys.argv[1:]:
        draw_roc_figures(pd.read_csv(DATA / "oof_predictions.csv"))
        return
    frame = as_published(load())
    rows, importance, oofs = [], [], []
    for st in ("Feces", "Soil"):
        r, i, o = run_sample_type(frame[frame.SampleType == st].reset_index(drop=True), st)
        rows += r
        importance += i
        oofs.append(o.assign(sample_type=st))
    res = pd.DataFrame(rows)
    oof = pd.concat(oofs)
    oof.to_csv(DATA / "oof_predictions.csv", index=False)
    res.to_csv(RESULTS / "table3_revised.csv", index=False)
    (RESULTS / "table3_revised.md").write_text(fmt_table(res).to_markdown(index=False) + "\n")
    (RESULTS / "table_s1_validation_schemes.md").write_text(
        "AUC by validation scheme (leave-one-farm-out: mean of per-farm AUCs)\n\n"
        + validation_schemes_table().to_markdown() + "\n")

    imp = pd.DataFrame(importance)
    imp_summary = (imp.groupby(["sample_type", "variable"]).auc_drop
                   .agg(["mean", "std"]).reset_index().sort_values(["sample_type", "mean"]))
    imp_summary.to_csv(RESULTS / "importance_by_variable.csv", index=False)

    draw_roc_figures(oof)
    figI, axesI = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for axI, (letter, st) in zip(axesI, (("a", "Feces"), ("b", "Soil"))):
        s = imp_summary[imp_summary.sample_type == st]
        axI.barh(s.variable, s["mean"], xerr=s["std"], color="#0072B2", ecolor="0.3", capsize=3)
        axI.axvline(0, color="0.3", lw=1)
        axI.set(title=f"({letter}) {st}", xlabel="Drop in AUC when the variable is permuted")
    figI.tight_layout()
    figI.savefig(RESULTS / "fig3_importance.png", dpi=300)

    with pd.option_context("display.width", 200):
        print(fmt_table(res).to_string(index=False))
        print(imp_summary.to_string(index=False))


if __name__ == "__main__":
    main()
