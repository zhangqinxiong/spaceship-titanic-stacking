# Spaceship Titanic - Kaggle Competition

Predict which passengers were transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly.

## Approach

### Feature Engineering
- **Cabin Parsing**: Split `Cabin` into `Deck` (ordinal), `CabinNum` (numeric), `Side` (P/S)
- **CabinPct**: Cabin number percentile within each deck
- **TotalSpend**: Sum of all amenity spending (`RoomService + FoodCourt + ShoppingMall + Spa + VRDeck`)
- **Route OHE**: One-hot encoding of `HomePlanet × Destination` interaction
- **Target Encoding**: CV-safe mean encoding of `HomePlanet` and `Deck`

### Encoding Strategy
- **Ordinal features**: `CryoSleep`, `VIP`, `Deck` → `OrdinalEncoder`
- **Nominal features**: `HomePlanet`, `Destination`, `Side` → one-hot encoding

### Missing Values
- Numeric features with <50% missing → filled with median
- No features had >50% missing, so none were dropped

### Model: Stacking Ensemble

| Base Model | CV Accuracy |
|------------|-------------|
| CatBoost (tuned) | 0.8220 |
| XGBoost (tuned) | 0.8101 |
| LightGBM (tuned) | 0.8094 |
| Logistic Regression | 0.7901 |
| **Ridge Meta-model** | **0.8192** |
| **Public LB** | **0.81271** |

### Hyperparameter Tuning
Optuna (100 trials) was used for CatBoost hyperparameter search. The best parameters:
- `learning_rate`: 0.188, `depth`: 6, `l2_leaf_reg`: 6.02
- `subsample`: 0.78, `colsample_bylevel`: 0.99
- `min_data_in_leaf`: 33, `border_count`: 123

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training script: preprocessing, model training, stacking, submission |
| `submission.csv` | Latest Kaggle submission file |

## How to Run

```bash
python train.py
```

The script will:
1. Download data via kagglehub
2. Preprocess with feature engineering and target encoding
3. Train 5-fold stacking ensemble (CatBoost + XGBoost + LightGBM + LogisticRegression)
4. Train Ridge meta-model on out-of-fold predictions
5. Generate `submission.csv`

## Requirements

- catboost, xgboost, lightgbm
- scikit-learn, pandas, numpy
- optuna, kagglehub
