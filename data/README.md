# Data

Cached copies of three public OpenML datasets (pickled `{"X": DataFrame, "y": labels}`):

| File | OpenML ID | Name | Records | Target |
|---|---|---|---|---|
| `raw/dataset_194.pkl` | 194 | Cleveland heart disease | 303 | `num > 0` |
| `raw/dataset_43672.pkl` | 43672 | Heart Disease Dataset (Comprehensive) | 1190 | `target` |
| `raw/dataset_45547.pkl` | 45547 | Cardiovascular Disease | 70,000 | `cardio` |

`src/qfmb/data.py` builds the analysis cohorts from these files; `qfmb.data.audit()` prints duplicates,
zero-coded missing values, Cleveland overlap and class balance.
