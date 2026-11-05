# Bip4cast bipolar patient records

`bip4cast_patients.xlsx` holds anonymised records of 4 patients with bipolar disorder (codes D, G,
M and O). The data were collected by psychiatrists at Clínica Nuestra Señora de la Paz, Madrid, as
part of the Bip4cast project, during 2017 and early 2018.

| Sheet | Contents |
|---|---|
| `Episodios` | start and end dates of depressive (`DEPRESIÓN`) and manic (`MANIA`) episodes |
| `Diario` | daily self-report: mood (−3..3), motivation (−3..3), attention problems, irritability, anxiety, sleep quality (0..4), cigarettes, caffeine (mg), alcohol, other drugs, wake-up time, bedtime |
| `Young` | Young Mania Rating Scale (YMRS), 11 items rated by a psychiatrist |
| `HDRS` | Hamilton Depression Rating Scale (HDRS-17) |
| `Intervenciones` | appointments, GAF score, perceived relief |

**Cleaning done in code** (`src/depresjon_cnn/bipolar_data.py`, the original file is unchanged):
* The episode end date `2917-09-11` is corrected to `2017-09-11`.
* Diary entries dated outside June 2017 to January 2018 (typos such as 2007 and 2020) are dropped.
* Days with more than one diary entry are averaged.
* Patient O ("constantly mixed") is excluded from forecasting.

**Source:** https://github.com/AxelJunes/BDCP (commit 36bc50b), Apache License 2.0 (see `LICENSE`
in this folder). Original work: A. Junestrand Leal, *Application of Machine Learning Algorithms
for Bipolar Disorder Crisis Prediction*, Bachelor's thesis, Universidad Complutense de Madrid, 2018.
