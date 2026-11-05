# Depresjon dataset

Motor-activity (actigraphy) recordings from 23 unipolar and bipolar depressed patients
(`condition/`) and 32 healthy controls (`control/`), collected with an Actiwatch AW4
(one activity count per minute).

* Source: https://datasets.simula.no/depresjon/ (also on Zenodo, doi:10.5281/zenodo.1219550)
* Each CSV has the columns `timestamp`, `date` and `activity`.
* `scores.csv` has one row per participant:
  * `number`: file name
  * `days`: number of days used
  * `gender`: 1 = female, 2 = male
  * `age`: age group
  * `afftype`: 1 = bipolar II, 2 = unipolar depressive, 3 = bipolar I
  * `melanch`: melancholia (1 = yes, 2 = no)
  * `inpatient`: 1 = inpatient, 2 = outpatient
  * `edu`: years of education
  * `marriage`: 1 = married or cohabiting, 2 = single
  * `work`: 1 = working or studying, 2 = unemployed, sick leave or pension
  * `madrs1`, `madrs2`: MADRS score at the start and end of the recording

**Terms of use:** the dataset may be used for research and educational purposes only.
Any work that uses it must cite:

> E. Garcia-Ceja et al. *Depresjon: A Motor Activity Database of Depression Episodes in
> Unipolar and Bipolar Patients.* ACM MMSys 2018. doi:10.1145/3204949.3208125
