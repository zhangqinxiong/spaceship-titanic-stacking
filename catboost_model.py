import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

DATA_DIR = 'data'
SUBMISSION_DIR = 'submission'
SEED = 42

os.makedirs(SUBMISSION_DIR, exist_ok=True)

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

X_train = train.drop(['PassengerId', 'Transported'], axis=1)
y_train = train['Transported'].astype(int)
X_test = test.drop(['PassengerId'], axis=1)

cat_cols = X_train.select_dtypes(include='object').columns.tolist()
num_cols = X_train.select_dtypes(include='float64').columns.tolist()

combined = pd.concat([X_train[cat_cols], X_test[cat_cols]], axis=0)
for col in cat_cols:
    le = LabelEncoder()
    combined[col] = combined[col].fillna('MISSING').astype(str)
    combined[col] = le.fit_transform(combined[col])
    X_train[col] = combined[col][:len(X_train)]
    X_test[col] = combined[col][len(X_train):]

for col in num_cols:
    med = X_train[col].median()
    X_train[col] = X_train[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
models = []
test_probas = np.zeros(len(X_test))
cv_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n{'='*60}\nFold {fold+1}/5\n{'='*60}")
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    model = CatBoostClassifier(
        random_seed=SEED,
        task_type='GPU',
        loss_function='Logloss',
        auto_class_weights='Balanced',
        early_stopping_rounds=50,
        od_type='Iter',
        verbose=100,
    )

    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

    models.append(model)
    val_preds = model.predict(X_val)
    acc = (val_preds == y_val.values).mean()
    cv_scores.append(acc)
    print(f"Fold {fold+1} val accuracy: {acc:.5f}")

    test_probas += model.predict_proba(X_test)[:, 1] / 5

print("\n" + "=" * 60)
print("CatBoost 5-Fold CV Ensemble Results")
print("=" * 60)
print(f"CV accuracy: {np.mean(cv_scores):.5f} +/- {np.std(cv_scores):.5f}")
for i, s in enumerate(cv_scores):
    print(f"  Fold {i+1}: {s:.5f}")
print("=" * 60)

print("\nGenerating submission (5-fold average)...")
test_preds = (test_probas >= 0.5).astype(int)
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': test_preds.astype(bool),
})
sub_path = os.path.join(SUBMISSION_DIR, 'submission_catboost.csv')
submission.to_csv(sub_path, index=False)
print(f"Submission saved: {sub_path}")
print(f"Predictions: 正类={(test_preds == 1).sum()}, 负类={(test_preds == 0).sum()}")
