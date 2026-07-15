# Spaceship Titanic — Kaggle Competition

FLAML AutoML pipeline — no manual preprocessing, raw data directly, ensemble stacking.

## Result

| Approach | Public Score |
|----------|:------------:|
| FLAML + ensemble (no preproc) | 0.79682 |
| FLAML + ensemble + FE | 0.80009 |
| Manual FE + Stacking (baseline) | 0.80734 |

## Usage

```bash
pip install pandas numpy flaml scikit-learn

python flaml_automl.py
```

## Directory

```
Spaceship-Titanic/
├── data/               # train.csv / test.csv
├── submission/         # submission output
├── flaml_automl.py     # FLAML AutoML pipeline
└── README.md
```
