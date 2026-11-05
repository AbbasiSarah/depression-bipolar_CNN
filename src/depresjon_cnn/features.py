"""Hand-crafted daily features for the conventional machine-learning baselines.

The baselines mirror the classic algorithms compared in the proposal
(Naive Bayes, Logistic Regression, MLP, SVM/SMO, Decision Tree) and get a
fixed feature vector per day instead of the raw signal.
"""

from __future__ import annotations

import numpy as np


def daily_features(X: np.ndarray) -> np.ndarray:
    """Return a (n_days, 34) feature matrix from raw 1440-minute activity.

    Features: mean, std, median, max, fraction of inactive minutes,
    coefficient of variation, night (00-06) mean, day (08-20) mean,
    day/night ratio, plus 24 hourly means and the hour of peak activity.
    """
    hourly = X.reshape(len(X), 24, 60).mean(axis=2)
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    feats = [
        mean,
        std,
        np.median(X, axis=1),
        X.max(axis=1),
        (X == 0).mean(axis=1),
        std / (mean + 1e-6),
        hourly[:, 0:6].mean(axis=1),
        hourly[:, 8:20].mean(axis=1),
        hourly[:, 8:20].mean(axis=1) / (hourly[:, 0:6].mean(axis=1) + 1.0),
    ]
    F = np.column_stack(feats + [hourly, hourly.argmax(axis=1)])
    return np.log1p(np.abs(F)) * np.sign(F)
