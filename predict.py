"""Score a new actigraphy recording with the trained CNN.

    python predict.py data/condition/condition_5.csv
    python predict.py my_watch_export.csv        # needs columns: timestamp, activity

Prints the probability of a depressive pattern for every full day and the
average over all days. This is a research prototype, NOT a diagnostic tool.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from depresjon_cnn.cnn import ActivityCNN  # noqa: E402
from depresjon_cnn.data import MINUTES_PER_DAY, preprocess_for_cnn  # noqa: E402


def load_model(path="results/cnn_weights.npz"):
    w = np.load(path)
    model = ActivityCNN()
    model._build()
    model.mu, model.sd = float(w["mu"]), float(w["sd"])
    for i, layer in enumerate(model.layers):
        for k in layer.params:
            layer.params[k] = w[f"{i}_{k}"]
    return model


def main(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    days, dates = [], []
    for day, chunk in df.groupby(df.index.normalize()):
        if len(chunk) == MINUTES_PER_DAY:
            days.append(chunk["activity"].to_numpy(np.float32))
            dates.append(day.date())
    if not days:
        sys.exit("No complete 24-hour days (1440 minutes) found in the file.")
    prob = load_model().predict_proba(preprocess_for_cnn(np.stack(days)))
    for d, p in zip(dates, prob):
        print(f"{d}  P(depressive pattern) = {p:.2f}")
    print(f"\nAverage over {len(prob)} days: {prob.mean():.2f} "
          f"-> {'depressive pattern' if prob.mean() >= 0.5 else 'no depressive pattern'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
