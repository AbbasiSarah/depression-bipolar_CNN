"""Run every experiment and write metrics + figures to ``results/``.

    python run_experiments.py              # full run (~5 min on a laptop CPU)
    python run_experiments.py --quick      # 1 repeat, fewer epochs (smoke test)

Evaluation is *subject-independent*: all days of a participant are either in
the training folds or in the test fold, never both. Mixing one person's days
across train and test inflates accuracy to ~99% and is a common mistake in
published work on this dataset.
"""

from __future__ import annotations

import argparse
import warnings
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, mean_absolute_error, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent / "src"))
from depresjon_cnn.cnn import ActivityCNN  # noqa: E402
from depresjon_cnn.data import load_daily_dataset, preprocess_for_cnn  # noqa: E402
from depresjon_cnn.features import daily_features  # noqa: E402

RESULTS = Path("results")
FIGS = RESULTS / "figures"
COLORS = {"CNN": "#C2185B", "base": "#90A4AE"}


def baseline_models(seed):
    scaled = lambda m: make_pipeline(StandardScaler(), m)  # noqa: E731
    return {
        "Naive Bayes": scaled(GaussianNB()),
        "Logistic Regression": scaled(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "MLP": scaled(MLPClassifier((64, 32), max_iter=600, alpha=1e-3, random_state=seed)),
        "SVM (SMO)": scaled(SVC(C=1.0, class_weight="balanced", probability=True, random_state=seed)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
    }


def day_metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "auc": roc_auc_score(y, prob),
    }


def subject_vote(subject, prob, y):
    df = pd.DataFrame({"s": subject, "p": prob, "y": y}).groupby("s").mean()
    return accuracy_score(df.y, (df.p >= 0.5).astype(int))


# ------------------------------------------------------------- experiment 1
def detection_experiment(d, Xc, F, repeats, epochs):
    """Depressed vs healthy control, day-level, 5-fold subject-wise CV."""
    rows, oof_cnn, oof_all = [], [], {}
    for rep in range(repeats):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
        probs = {name: np.zeros(len(d.label)) for name in ["CNN", *baseline_models(rep)]}
        for tr, te in cv.split(Xc, d.label, d.subject):
            cnn = ActivityCNN(epochs=epochs, seed=rep).fit(Xc[tr], d.label[tr])
            probs["CNN"][te] = cnn.predict_proba(Xc[te])
            for name, model in baseline_models(rep).items():
                model.fit(F[tr], d.label[tr])
                probs[name][te] = model.predict_proba(F[te])[:, 1]
        for name, p in probs.items():
            m = day_metrics(d.label, p)
            m["subject_accuracy"] = subject_vote(d.subject, p, d.label)
            pat = d.label == 1
            m["recall_bipolar"] = ((p >= 0.5) & (d.bipolar == 1)).sum() / (d.bipolar == 1).sum()
            m["recall_unipolar"] = ((p >= 0.5) & pat & (d.bipolar == 0)).sum() / (pat & (d.bipolar == 0)).sum()
            rows.append({"model": name, "repeat": rep, **m})
            oof_all.setdefault(name, []).append(p)
        oof_cnn.append(probs["CNN"])
        print(f"  repeat {rep + 1}/{repeats}: CNN acc {rows[-len(probs)]['accuracy']:.3f}")
    return pd.DataFrame(rows), oof_all


# ------------------------------------------------------------- experiment 2
def early_detection_experiment(d, Xc, repeats, epochs, max_days=7):
    """How many days of wearing the watch are needed to flag depression?

    Out-of-fold CNN day probabilities are averaged over a participant's first
    k days only; we report subject-level accuracy for k = 1..max_days.
    """
    order = pd.Series(range(len(d.subject))).groupby(d.subject).cumcount().to_numpy()
    acc = np.zeros((repeats, max_days))
    for rep in range(repeats):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=100 + rep)
        p = np.zeros(len(d.label))
        for tr, te in cv.split(Xc, d.label, d.subject):
            p[te] = ActivityCNN(epochs=epochs, seed=rep).fit(Xc[tr], d.label[tr]).predict_proba(Xc[te])
        for k in range(1, max_days + 1):
            keep = order < k
            acc[rep, k - 1] = subject_vote(d.subject[keep], p[keep], d.label[keep])
    return acc


# ------------------------------------------------------------- experiment 3
def severity_experiment(d, Xc, F, repeats, epochs):
    """Predict MADRS depression severity for patients (regression)."""
    pat = d.label == 1
    Xp, Fp, yp, sp = Xc[pat], F[pat], d.madrs[pat], d.subject[pat]
    out = {"CNN": [], "Random Forest": [], "Ridge": [], "Mean (no model)": []}
    for rep in range(repeats):
        perm = np.random.default_rng(rep).permutation(np.unique(sp))
        groups = pd.Series(sp).map({s: i for i, s in enumerate(perm)}).to_numpy()
        preds = {k: np.zeros(len(yp)) for k in out}
        for tr, te in GroupKFold(5).split(Xp, yp, groups):
            preds["CNN"][te] = ActivityCNN("regression", epochs=epochs, seed=rep).fit(Xp[tr], yp[tr]).predict(Xp[te])
            preds["Random Forest"][te] = RandomForestRegressor(300, min_samples_leaf=3, random_state=rep,
                                                               n_jobs=-1).fit(Fp[tr], yp[tr]).predict(Fp[te])
            preds["Ridge"][te] = make_pipeline(StandardScaler(), Ridge(10.0)).fit(Fp[tr], yp[tr]).predict(Fp[te])
            preds["Mean (no model)"][te] = yp[tr].mean()
        for k, p in preds.items():  # subject-level: average the daily predictions
            df = pd.DataFrame({"s": sp, "p": p, "y": yp}).groupby("s").mean()
            out[k].append(mean_absolute_error(df.y, df.p))
    return {k: {"mae_mean": float(np.mean(v)), "mae_std": float(np.std(v))} for k, v in out.items()}


# ------------------------------------------------------------------ figures
def make_figures(d, det, oof, early, cnn_final):
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    # 1. average daily activity profile
    hourly = np.log1p(d.X).reshape(len(d.X), 24, 60).mean(axis=2)
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, name, col in [(0, "Healthy controls", "#546E7A"), (1, "Depressed patients", COLORS["CNN"])]:
        h = hourly[d.label == lab]
        mu, se = h.mean(0), h.std(0) / np.sqrt(len(h))
        ax.plot(range(24), mu, color=col, lw=2, label=name)
        ax.fill_between(range(24), mu - se, mu + se, color=col, alpha=0.2)
    bip = hourly[d.bipolar == 1].mean(0)
    ax.plot(range(24), bip, color="#F57C00", lw=1.5, ls="--", label="Bipolar patients (subset)")
    ax.set(xlabel="Hour of day", ylabel="log(1 + activity)", title="Average 24-hour motor activity",
           xticks=range(0, 24, 3))
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "activity_profile.png", dpi=150); plt.close(fig)

    # 2. model comparison
    summary = det.groupby("model")[["accuracy", "balanced_accuracy"]].agg(["mean", "std"])
    summary = summary.sort_values(("balanced_accuracy", "mean"))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = summary.index.tolist()
    vals = summary[("balanced_accuracy", "mean")].to_numpy() * 100
    errs = summary[("balanced_accuracy", "std")].to_numpy() * 100
    cols = [COLORS["CNN"] if n == "CNN" else COLORS["base"] for n in names]
    ax.barh(names, vals, xerr=errs, color=cols, capsize=3)
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(v + e + 1.0, i, f"{v:.1f}%", va="center")
    ax.axvline(50, color="k", lw=0.8, ls=":")
    ax.set(xlim=(40, 100), xlabel="Balanced accuracy (%) — subject-wise 5-fold CV",
           title="Depression detection from one day of activity")
    fig.tight_layout(); fig.savefig(FIGS / "model_comparison.png", dpi=150); plt.close(fig)

    # 3. ROC curves
    fig, ax = plt.subplots(figsize=(5, 5))
    for name in ["CNN", "Random Forest", "Logistic Regression", "Decision Tree"]:
        p = np.mean(oof[name], axis=0)
        fpr, tpr, _ = roc_curve(d.label, p)
        ax.plot(fpr, tpr, lw=2.2 if name == "CNN" else 1.2,
                color=COLORS["CNN"] if name == "CNN" else None,
                label=f"{name} (AUC {roc_auc_score(d.label, p):.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC, day level (avg. of 5 repeats)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGS / "roc_curves.png", dpi=150); plt.close(fig)

    # 4. confusion matrix
    p = np.mean(oof["CNN"], axis=0)
    cm = confusion_matrix(d.label, (p >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="RdPu")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Control", "Depressed"],
           yticklabels=["Control", "Depressed"], xlabel="Predicted", ylabel="Actual",
           title="CNN confusion matrix (days)")
    fig.tight_layout(); fig.savefig(FIGS / "confusion_matrix.png", dpi=150); plt.close(fig)

    # 5. early detection
    fig, ax = plt.subplots(figsize=(6.5, 4))
    k = np.arange(1, early.shape[1] + 1)
    ax.errorbar(k, early.mean(0) * 100, yerr=early.std(0) * 100, color=COLORS["CNN"],
                marker="o", lw=2, capsize=3)
    ax.set(xlabel="Days of recording used", ylabel="Participant-level accuracy (%)",
           title="Early detection: how many days are needed?", xticks=k, ylim=(50, 100))
    fig.tight_layout(); fig.savefig(FIGS / "early_detection.png", dpi=150); plt.close(fig)

    # 6. training curve
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(cnn_final.history) + 1), cnn_final.history, color=COLORS["CNN"], lw=2)
    ax.set(xlabel="Epoch", ylabel="Weighted BCE loss", title="CNN training loss (all data)")
    fig.tight_layout(); fig.savefig(FIGS / "training_loss.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    repeats, epochs = (1, 15) if args.quick else (5, 40)

    d = load_daily_dataset(args.data)
    Xc, F = preprocess_for_cnn(d.X), daily_features(d.X)
    print(f"{len(d.subjects)} participants, {len(d.label)} days "
          f"({d.label.sum()} patient days, {d.bipolar.sum()} bipolar days)")

    print("Experiment 1: depression detection")
    det, oof = detection_experiment(d, Xc, F, repeats, epochs)
    print("Experiment 2: early detection")
    early = early_detection_experiment(d, Xc, repeats, epochs)
    print("Experiment 3: severity (MADRS) regression")
    sev = severity_experiment(d, Xc, F, repeats, epochs)

    RESULTS.mkdir(exist_ok=True)
    summary = det.drop(columns="repeat").groupby("model").agg(["mean", "std"]).round(4)
    summary.to_csv(RESULTS / "detection_metrics.csv")
    final = ActivityCNN(epochs=epochs, seed=0).fit(Xc, d.label)
    np.savez(RESULTS / "cnn_weights.npz", mu=final.mu, sd=final.sd,
             **{f"{i}_{k}": v for i, l in enumerate(final.layers) for k, v in l.params.items()})
    make_figures(d, det, oof, early, final)

    report = {
        "dataset": {"participants": int(len(d.subjects)), "days": int(len(d.label)),
                    "patients": int(d.scores.label.sum()), "bipolar_patients": int(d.scores.bipolar.sum())},
        "settings": {"repeats": repeats, "epochs": epochs, "cv": "5-fold StratifiedGroupKFold by participant"},
        "detection": {m: {c: float(summary.loc[m, (c, "mean")]) for c in summary.columns.levels[0]}
                      for m in summary.index},
        "early_detection_subject_accuracy": {f"{k + 1}_days": float(early[:, k].mean())
                                             for k in range(early.shape[1])},
        "severity_mae_madrs_points": sev,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
