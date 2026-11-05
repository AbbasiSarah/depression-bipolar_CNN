"""Part 2 - forecasting depressive episodes in bipolar patients.

    python run_forecasting.py

Question: given the last 14 days of a patient's daily self-report questionnaire,
is the patient in a depressive episode today (horizon 0), and will they be in one
7 days from now (horizon 7)?

Two evaluations:
* **New patient** (leave-one-patient-out): train on two patients, test on the
  third. Hardest setting, since bipolar disorder looks different in every person.
* **Known patients, unseen weeks** (blocked time-series CV): each patient's
  timeline is cut into 4 consecutive blocks; we test on one block and train on
  all other data, leaving a 21-day buffer around the test block so overlapping
  windows cannot leak information.

The figures, `results/forecasting_metrics.json` and `results/forecasting_*.png`
are written to `results/`.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))
from depresjon_cnn.bipolar_data import (FEATURES, FORECAST_PATIENTS, load_diary,  # noqa: E402
                                        load_episodes, load_scale, make_windows)
from depresjon_cnn.cnn import DiaryCNN  # noqa: E402

DATA = Path("data/bdcp/bip4cast_patients.xlsx")
FIGS = Path("results/figures")
WINDOW, HORIZON, SEEDS = 14, 7, 5
PINK, GREY, ORANGE = "#C2185B", "#90A4AE", "#F57C00"


# ------------------------------------------------------------------ models
def fit_predict(name, Xtr, ytr, Xte):
    if name == "CNN":  # average of 5 differently-initialised networks
        return np.mean([DiaryCNN(Xtr.shape[1], epochs=60, seed=s).fit(Xtr, ytr).predict_proba(Xte)
                        for s in range(SEEDS)], axis=0)
    if name == "Mood rule (no training)":
        return -Xte[:, FEATURES.index("mood"), -7:].mean(axis=1)  # low recent mood -> high risk
    flat_tr, flat_te = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    if name == "Logistic Regression":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))
    else:
        m = RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1)
    return m.fit(flat_tr, ytr).predict_proba(flat_te)[:, 1]


MODELS = ["CNN", "Logistic Regression", "Random Forest", "Mood rule (no training)"]


def lopo_splits(w):
    for p in FORECAST_PATIENTS:
        yield w.patient != p, w.patient == p


def blocked_splits(w, n_blocks=4, buffer_days=WINDOW + HORIZON):
    dates = pd.to_datetime(w.date)
    for p in FORECAST_PATIENTS:
        idx = np.where(w.patient == p)[0]
        bounds = np.array_split(np.sort(dates[idx].unique()), n_blocks)
        for b in bounds:
            lo, hi = b[0], b[-1]
            te = (w.patient == p) & (dates >= lo) & (dates <= hi)
            near = (w.patient == p) & (dates >= lo - pd.Timedelta(days=buffer_days)) & \
                   (dates <= hi + pd.Timedelta(days=buffer_days))
            yield ~near, np.asarray(te)


def evaluate(w, splits):
    oof = {m: np.full(len(w.y), np.nan) for m in MODELS}
    for tr, te in splits:
        if w.y[tr].sum() == 0:
            continue
        for m in MODELS:
            oof[m][te] = fit_predict(m, w.X[tr], w.y[tr], w.X[te])
    res = {}
    onset = w.current_dep == 0  # not depressed yet: does the model see it coming?
    for m, p in oof.items():
        ok = ~np.isnan(p)
        # threshold: CNN/LR/RF output probabilities; rule gets the training-free median split
        thr = 0.5 if m != "Mood rule (no training)" else np.nanpercentile(p, 100 * (1 - w.y.mean()))
        pred = (p >= thr).astype(int)
        res[m] = {
            "auc": roc_auc_score(w.y[ok], p[ok]),
            "balanced_accuracy": balanced_accuracy_score(w.y[ok], pred[ok]),
            "recall_depression": recall_score(w.y[ok], pred[ok]),
            "auc_new_episodes_only": (roc_auc_score(w.y[ok & onset], p[ok & onset])
                                      if len(set(w.y[ok & onset])) == 2 else None),
            "per_patient_auc": {pt: (roc_auc_score(w.y[ok & (w.patient == pt)], p[ok & (w.patient == pt)])
                                     if len(set(w.y[ok & (w.patient == pt)])) == 2 else None)
                                for pt in FORECAST_PATIENTS},
        }
    return res, oof


# -------------------------------------------------------- scales analysis
def scales_vs_diary(diary_all):
    out = {}
    for sheet, name in [("HDRS", "HDRS"), ("Young", "YMRS")]:
        sc = load_scale(DATA, sheet)
        moods = []
        for r in sc.itertuples():
            g = diary_all[(diary_all.patient == r.patient) &
                          diary_all.date.between(r.date - pd.Timedelta(days=6), r.date)]
            moods.append(g["mood"].mean() if len(g) else np.nan)
        sc["mood7"] = moods
        sc = sc.dropna()
        rho, pval = spearmanr(sc["total"], sc["mood7"])
        out[name] = {"n_ratings": int(len(sc)), "spearman_rho_vs_7day_mood": float(rho), "p_value": float(pval)}
        out[name + "_data"] = sc
    return out


# ----------------------------------------------------------------- figures
def fig_timelines(diary_all, episodes):
    hd, ym = load_scale(DATA, "HDRS"), load_scale(DATA, "Young")
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for ax, p in zip(axes, FORECAST_PATIENTS):
        g = diary_all[diary_all.patient == p].sort_values("date")
        for e in episodes[episodes.patient == p].itertuples():
            ax.axvspan(e.start, e.end, color=PINK if e.type == "depression" else ORANGE, alpha=0.18, lw=0)
        ax.plot(g.date, g["mood"].rolling(3, min_periods=1).mean(), color="#37474F", lw=1.6, label="Self-reported mood (3-day avg)")
        ax.set_ylim(-3.4, 3.4); ax.set_ylabel(f"Patient {p}\nmood")
        ax2 = ax.twinx()
        h, y = hd[hd.patient == p], ym[ym.patient == p]
        ax2.plot(h.date, h.total, "v", color=PINK, ms=7, label="HDRS (depression)")
        ax2.plot(y.date, y.total, "^", color=ORANGE, ms=7, label="YMRS (mania)")
        ax2.set_ylim(-1, 20); ax2.set_ylabel("clinician score")
        ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
    from matplotlib.patches import Patch
    handles = [plt.Line2D([], [], color="#37474F", lw=1.6), Patch(color=PINK, alpha=0.3), Patch(color=ORANGE, alpha=0.3),
               plt.Line2D([], [], marker="v", color=PINK, ls=""), plt.Line2D([], [], marker="^", color=ORANGE, ls="")]
    axes[0].legend(handles, ["Self-reported mood (3-day avg)", "Depressive episode", "Manic episode",
                             "HDRS score", "YMRS score"], ncol=5, fontsize=8.5, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, 1.02))
    axes[0].set_title("Bipolar patients: daily mood, clinician scales and episodes", pad=28)
    fig.tight_layout(); fig.savefig(FIGS / "forecasting_patient_timelines.png", dpi=150); plt.close(fig)


def fig_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), sharey=True)
    titles = {0: "Detect a depressive episode today", HORIZON: f"Forecast it {HORIZON} days ahead"}
    y = np.arange(len(MODELS))[::-1]
    for ax, h in zip(axes, [0, HORIZON]):
        for off, key, lab, alpha in [(0.18, "known_patients_blocked_cv", "Known patients, unseen weeks", 1.0),
                                     (-0.18, "new_patient_leave_one_out", "New, unseen patient", 0.45)]:
            vals = [results[h][key][m]["auc"] for m in MODELS]
            cols = [PINK if m == "CNN" else GREY for m in MODELS]
            ax.barh(y + off, vals, height=0.34, color=cols, alpha=alpha, label=lab)
            for yi, v in zip(y + off, vals):
                ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8.5)
        ax.axvline(0.5, color="k", lw=0.8, ls=":")
        ax.set(xlim=(0.2, 1.0), title=titles[h], xlabel="ROC AUC (0.5 = chance)", yticks=y, yticklabels=MODELS)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Bipolar patients: 14 days of self-reports -> depressive episode (solid = known patients, faded = new patient)",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIGS / "forecasting_model_comparison.png", dpi=150); plt.close(fig)


def fig_forecast_example(w, oof, episodes):
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    dates = pd.to_datetime(w.date)
    for ax, p in zip(axes, FORECAST_PATIENTS):
        m = w.patient == p
        for e in episodes[(episodes.patient == p) & (episodes.type == "depression")].itertuples():
            ax.axvspan(e.start, e.end, color=PINK, alpha=0.18, lw=0)
        target = dates[m] + pd.Timedelta(days=HORIZON)  # plot each forecast on the day it is about
        ax.plot(target, oof["CNN"][m], ".-", color="#37474F", lw=1, ms=3)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.set(ylim=(0, 1), ylabel=f"Patient {p}\nP(depressed)")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_title("CNN forecasts made 7 days in advance (shaded = actual depressive episode)")
    fig.tight_layout(); fig.savefig(FIGS / "forecasting_cnn_timeline.png", dpi=150); plt.close(fig)


def fig_scales(scales):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, name, col in [(axes[0], "HDRS", PINK), (axes[1], "YMRS", ORANGE)]:
        d = scales[name + "_data"]
        ax.scatter(d["mood7"], d["total"], color=col, alpha=0.75)
        ax.set(xlabel="Mean self-reported mood, previous 7 days", ylabel=f"{name} total",
               title=f"{name}: Spearman ρ = {scales[name]['spearman_rho_vs_7day_mood']:.2f} (n={scales[name]['n_ratings']})")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "forecasting_scales_vs_mood.png", dpi=150); plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(DATA)
    diary_all = load_diary(DATA)
    results, report = {}, {"data": {}, "results": {}}
    for h in [0, HORIZON]:
        w, _ = make_windows(DATA, WINDOW, h)
        print(f"horizon {h} d: {len(w.y)} windows, {w.y.sum()} depressive "
              f"({((w.y == 1) & (w.current_dep == 0)).sum()} new-episode)")
        res_block, oof_block = evaluate(w, blocked_splits(w))
        res_lopo, _ = evaluate(w, lopo_splits(w))
        results[h] = {"known_patients_blocked_cv": res_block, "new_patient_leave_one_out": res_lopo}
        report["data"][f"horizon_{h}d"] = {"windows": int(len(w.y)), "depressive_windows": int(w.y.sum())}
        report["results"][f"horizon_{h}d"] = results[h]
        if h == HORIZON:
            fig_forecast_example(w, oof_block, episodes)
    scales = scales_vs_diary(diary_all)
    fig_timelines(diary_all, episodes)
    fig_comparison(results)
    fig_scales(scales)
    report.update({"window_days": WINDOW, "patients": FORECAST_PATIENTS, "features": FEATURES,
                   "clinician_scales_vs_self_report": {k: v for k, v in scales.items() if not k.endswith("_data")}})
    Path("results/forecasting_metrics.json").write_text(json.dumps(report, indent=2, default=float))
    for h in results:
        for ev, res in results[h].items():
            print(f"h={h:<2d} {ev:28s}", {m: round(v["auc"], 3) for m, v in res.items()})
    print(report["clinician_scales_vs_self_report"])


if __name__ == "__main__":
    main()
