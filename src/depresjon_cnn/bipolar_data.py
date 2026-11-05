"""Bipolar patient records (Bip4cast / Junestrand 2018) for episode forecasting.

Source: https://github.com/AxelJunes/BDCP (Apache-2.0), collected by
psychiatrists at Clínica Nuestra Señora de la Paz, Madrid, within the
Bip4cast project. The Excel workbook has five sheets:

* ``Episodios`` – start/end dates of depressive and manic episodes
* ``Diario``    – daily self-report questionnaire filled in by the patient
* ``Young``     – Young Mania Rating Scale (YMRS) items, rated by a psychiatrist
* ``HDRS``      – Hamilton Depression Rating Scale (HDRS-17) items
* ``Intervenciones`` – appointments (not used here)

Patient "O" had constantly mixed episodes and is excluded from forecasting,
as recommended by the treating psychiatrist in the original study.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_START, STUDY_END = pd.Timestamp("2017-06-01"), pd.Timestamp("2018-01-31")
FORECAST_PATIENTS = ["D", "G", "M"]

# Spanish column -> English feature name
DIARY_COLUMNS = {
    "Estado de ánimo": "mood",                                # -3 .. 3
    "Motivación": "motivation",                               # -3 .. 3
    "Problemas de concentración y atención": "attention_problems",  # 0 .. 4
    "Irritabilidad": "irritability",                          # 0 .. 4
    "Ansiedad": "anxiety",                                    # 0 .. 4
    "Calidad del sueño": "sleep_problems",                    # 0 .. 4 (higher = worse)
    "Número de cigarrillos": "cigarettes",
    "Cafeína": "caffeine_mg",
}
FEATURES = list(DIARY_COLUMNS.values()) + ["wake_hour", "bed_hour", "sleep_hours"]


def _hour(t, evening_wrap=False):
    if pd.isna(t):
        return np.nan
    h = t.hour + t.minute / 60
    return h + 24 if evening_wrap and h < 12 else h  # 00:30 bedtime -> 24.5


def load_episodes(path) -> pd.DataFrame:
    ep = pd.read_excel(path, sheet_name="Episodios")
    ep = ep[ep["EPISODIO"].notna()].copy()
    ep["start"] = pd.to_datetime(ep["INICIO"])
    # the workbook contains the typo 2917-09-11 for 2017-09-11
    ep["end"] = ep["FIN"].astype(str).str.replace("2917", "2017").pipe(pd.to_datetime)
    ep["type"] = ep["EPISODIO"].map({"DEPRESIÓN": "depression", "MANIA": "mania"})
    return ep.rename(columns={"PACIENTE": "patient"})[["patient", "start", "end", "type"]]


def load_diary(path) -> pd.DataFrame:
    """One row per patient per calendar day inside the study window."""
    d = pd.read_excel(path, sheet_name="Diario")
    d["date"] = pd.to_datetime(d["Fecha"], errors="coerce")
    d = d[(d["date"] >= STUDY_START) & (d["date"] <= STUDY_END)]  # drops typo dates (2007, 2020...)
    out = d[list(DIARY_COLUMNS)].rename(columns=DIARY_COLUMNS).apply(pd.to_numeric, errors="coerce")
    out["wake_hour"] = d["Hora de despertar"].map(_hour)
    out["bed_hour"] = d["Hora a la que te dormiste"].map(lambda t: _hour(t, evening_wrap=True))
    out["sleep_hours"] = (out["wake_hour"] + 24 - out["bed_hour"]).clip(0, 16)
    out["patient"], out["date"] = d["Código"].values, d["date"].values
    return out.groupby(["patient", "date"], as_index=False).mean()


def load_scale(path, sheet) -> pd.DataFrame:
    """Clinician rating scale with a ``total`` column (YMRS or HDRS)."""
    s = pd.read_excel(path, sheet_name=sheet)
    s["date"] = pd.to_datetime(s["Fecha"], errors="coerce")
    items = s.columns[s.columns.get_loc("Fecha") + 1:-1]
    items = [c for c in items if c != "Observaciones"]
    s["total"] = s[items].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    return s.rename(columns={"Código": "patient"})[["patient", "date", "total"]].dropna()


def daily_state(diary: pd.DataFrame, episodes: pd.DataFrame) -> pd.Series:
    """Label each diary row as 'depression', 'mania' or 'euthymic'."""
    state = pd.Series("euthymic", index=diary.index)
    for e in episodes.itertuples():
        inside = (diary["patient"] == e.patient) & diary["date"].between(e.start, e.end)
        state[inside] = e.type
    return state


@dataclass
class ForecastWindows:
    X: np.ndarray          # (n, n_features, window) past questionnaires
    y: np.ndarray          # (n,) 1 = in a depressive episode `horizon` days later
    patient: np.ndarray    # (n,)
    date: np.ndarray       # (n,) last day of the input window
    current_dep: np.ndarray  # (n,) already depressed on the last input day


def make_windows(path, window=14, horizon=7, max_gap=3) -> tuple[ForecastWindows, pd.DataFrame]:
    """Sliding windows of `window` days -> depressive state `horizon` days later.

    Missing diary days are linearly interpolated if the gap is at most
    `max_gap` days; windows still containing missing values are dropped.
    """
    path = Path(path)
    diary, episodes = load_diary(path), load_episodes(path)
    diary = diary[diary["patient"].isin(FORECAST_PATIENTS)].reset_index(drop=True)
    Xs, ys, ps, ds, cur = [], [], [], [], []
    for p, g in diary.groupby("patient"):
        g = g.set_index("date").sort_index()
        full = g[FEATURES].reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
        full = full.interpolate(limit=max_gap, limit_area="inside")
        lab = daily_state(full.reset_index().assign(patient=p).rename(columns={"index": "date"}),
                          episodes).values
        values = full.to_numpy()
        for t in range(window - 1, len(full) - horizon):
            x = values[t - window + 1:t + 1]
            if np.isnan(x).any():
                continue
            Xs.append(x.T)
            ys.append(int(lab[t + horizon] == "depression"))
            cur.append(int(lab[t] == "depression"))
            ps.append(p)
            ds.append(full.index[t])
    w = ForecastWindows(np.stack(Xs).astype(np.float32), np.array(ys), np.array(ps),
                        np.array(ds), np.array(cur))
    return w, diary
