"""Loading the Depresjon actigraphy dataset and turning it into daily samples.

Each participant wore an Actiwatch that recorded motor activity once per minute.
We cut every recording into whole calendar days (00:00-23:59, 1440 minutes),
keep at most the number of days listed in ``scores.csv`` (the clinically
supervised period), and use each day as one sample for the models.

Labels
------
* ``label``     : 1 = depressed patient (condition group), 0 = healthy control
* ``bipolar``   : 1 = bipolar I or II patient (afftype 1 or 3), 0 = otherwise
* ``madrs``     : mean of the two MADRS ratings (start and end of recording),
                  only defined for patients
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440

# afftype codes from the dataset documentation
AFFTYPE_NAMES = {1: "bipolar II", 2: "unipolar depressive", 3: "bipolar I"}


@dataclass
class DailyDataset:
    X: np.ndarray          # (n_days, 1440) raw activity counts
    subject: np.ndarray    # (n_days,) participant id, e.g. "condition_3"
    label: np.ndarray      # (n_days,) 1 = depressed, 0 = control
    bipolar: np.ndarray    # (n_days,) 1 = bipolar patient
    madrs: np.ndarray      # (n_days,) mean MADRS score, NaN for controls
    scores: pd.DataFrame   # one row per participant

    @property
    def subjects(self) -> np.ndarray:
        return np.unique(self.subject)


def load_scores(data_dir: Path) -> pd.DataFrame:
    scores = pd.read_csv(Path(data_dir) / "scores.csv")
    scores["label"] = scores["number"].str.startswith("condition").astype(int)
    scores["bipolar"] = scores["afftype"].isin([1, 3]).astype(int)
    scores["madrs"] = scores[["madrs1", "madrs2"]].mean(axis=1)
    return scores


def _daily_matrix(csv_path: Path, max_days: int) -> np.ndarray:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    days = []
    for day, chunk in df.groupby(df.index.normalize()):
        if len(chunk) != MINUTES_PER_DAY:
            continue  # skip partial first/last days
        days.append(chunk["activity"].to_numpy(dtype=np.float32))
        if len(days) >= max_days:
            break
    return np.stack(days) if days else np.empty((0, MINUTES_PER_DAY), np.float32)


def load_daily_dataset(data_dir: str | Path) -> DailyDataset:
    """Read every participant file and return one row per full recorded day."""
    data_dir = Path(data_dir)
    scores = load_scores(data_dir)
    X, subj, lab, bip, mad = [], [], [], [], []
    for row in scores.itertuples(index=False):
        group = "condition" if row.label == 1 else "control"
        days = _daily_matrix(data_dir / group / f"{row.number}.csv", int(row.days))
        n = len(days)
        X.append(days)
        subj += [row.number] * n
        lab += [row.label] * n
        bip += [row.bipolar] * n
        mad += [row.madrs] * n
    return DailyDataset(
        X=np.concatenate(X),
        subject=np.array(subj),
        label=np.array(lab, dtype=np.int64),
        bipolar=np.array(bip, dtype=np.int64),
        madrs=np.array(mad, dtype=np.float64),
        scores=scores,
    )


def preprocess_for_cnn(X: np.ndarray, bin_minutes: int = 5) -> np.ndarray:
    """log1p-compress the counts and average them into ``bin_minutes`` bins.

    1440 one-minute values -> 288 five-minute values per day. The log transform
    tames the very skewed activity counts (most minutes are 0-50, a few are
    several thousand).
    """
    Xl = np.log1p(X)
    n, t = Xl.shape
    Xb = Xl.reshape(n, t // bin_minutes, bin_minutes).mean(axis=2)
    return Xb.astype(np.float32)
