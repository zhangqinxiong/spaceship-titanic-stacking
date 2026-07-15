"""
FLAML AutoML - Spaceship Titanic
No preprocessing, raw data directly, ensemble=True, eval_method='cv', n_splits=5.

Metric: accuracy (matches Kaggle competition eval).
"""

import os
import pandas as pd
import numpy as np
from flaml import AutoML

DATA_DIR = 'data'
SUBMISSION_DIR = 'submission'
SEED = 42

os.makedirs(SUBMISSION_DIR, exist_ok=True)

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

X = train.drop('Transported', axis=1)
y = train['Transported'].astype(int)

automl = AutoML()
automl.fit(
    X, y,
    task='classification',
    time_budget=300,
    ensemble=True,
    metric='accuracy',
    eval_method='cv',
    n_splits=5,
    split_type='stratified',
    seed=SEED,
    log_type='all',
)

print("\n" + "=" * 60)
print("FLAML Results (No Preprocessing, Ensemble=True, CV=5)")
print("=" * 60)
print(f"Best CV loss (1-Accuracy): {automl.best_loss:.5f}")
print(f"Best CV Accuracy:         {1 - automl.best_loss:.5f}")
print(f"Best estimator:           {automl.best_estimator}")
print(f"Best config:              {automl.best_config}")
print(f"metric used:              accuracy")
print("=" * 60)

print("\nGenerating submission...")
test_preds = automl.predict(test)
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': test_preds.astype(bool),
})
sub_path = os.path.join(SUBMISSION_DIR, 'submission_flaml.csv')
submission.to_csv(sub_path, index=False)
print(f"Submission saved: {sub_path}")
print(f"Predictions: 正类={(test_preds == 1).sum()}, 负类={(test_preds == 0).sum()}")
