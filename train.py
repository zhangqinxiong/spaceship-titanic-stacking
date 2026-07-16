import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5

CB_PARAMS = {
    'random_seed': RANDOM_STATE, 'task_type': 'CPU', 'auto_class_weights': 'Balanced',
    'eval_metric': 'Accuracy', 'verbose': 0, 'early_stopping_rounds': 50,
    'learning_rate': 0.18800165198036325, 'depth': 6, 'l2_leaf_reg': 6.0194787155733795,
    'subsample': 0.7793815608377906, 'colsample_bylevel': 0.9923014273002074,
    'min_data_in_leaf': 33, 'border_count': 123,
}

XGB_PARAMS = {
    'random_state': RANDOM_STATE, 'n_estimators': 1000, 'device': 'cpu',
    'eval_metric': 'logloss', 'verbose': False, 'early_stopping_rounds': 50,
    'learning_rate': 0.01344, 'max_depth': 6, 'subsample': 0.630, 'colsample_bytree': 0.575,
    'reg_lambda': 0.752, 'reg_alpha': 0.123, 'min_child_weight': 1, 'scale_pos_weight': 0.940,
}

LGB_PARAMS = {
    'random_state': RANDOM_STATE, 'n_estimators': 1000, 'verbose': -1,
    'metric': 'binary_logloss', 'class_weight': 'balanced',
    'learning_rate': 0.263, 'num_leaves': 13, 'max_depth': 5, 'subsample': 0.567,
    'colsample_bytree': 0.923, 'reg_lambda': 0.227, 'reg_alpha': 3.686, 'min_child_samples': 39,
}

LR_PARAMS = {
    'random_state': RANDOM_STATE, 'max_iter': 2000, 'class_weight': 'balanced',
    'C': 1.68, 'solver': 'lbfgs', 'penalty': 'l2',
}

def load_data():
    base = '/home/ivi/.cache/kagglehub/competitions/spaceship-titanic'
    train = pd.read_csv(f'{base}/train.csv')
    test = pd.read_csv(f'{base}/test.csv')
    return train, test

def extract_cabin(df):
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce')
    df['Side'] = cabin_split[2]
    return df

def preprocess_te(train, test, y, fold_train_idx, fold_val_idx):
    for df in [train, test]:
        extract_cabin(df)
        df['TotalSpend'] = df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].sum(axis=1, skipna=False)

    train['TotalSpend'] = train['TotalSpend'].fillna(train['TotalSpend'].median())
    test['TotalSpend'] = test['TotalSpend'].fillna(train['TotalSpend'].median())

    t_route = train['HomePlanet'].fillna('MISSING').astype(str) + '_' + train['Destination'].fillna('MISSING').astype(str)
    s_route = test['HomePlanet'].fillna('MISSING').astype(str) + '_' + test['Destination'].fillna('MISSING').astype(str)
    for v in pd.concat([t_route, s_route]).unique():
        train[f'Route_{v}'] = (t_route == v).astype(int)
        test[f'Route_{v}'] = (s_route == v).astype(int)

    all_decks = pd.concat([train['Deck'], test['Deck']]).dropna().unique()
    for deck in all_decks:
        train_mask = train['Deck'] == deck
        test_mask = test['Deck'] == deck
        deck_max = max(train.loc[train_mask, 'CabinNum'].max(), test.loc[test_mask, 'CabinNum'].max())
        if pd.notna(deck_max) and deck_max > 0:
            train.loc[train_mask, 'CabinPct'] = train.loc[train_mask, 'CabinNum'] / deck_max
            test.loc[test_mask, 'CabinPct'] = test.loc[test_mask, 'CabinNum'] / deck_max
    train['CabinPct'] = train['CabinPct'].fillna(0)
    test['CabinPct'] = test['CabinPct'].fillna(0)

    train_te = train.iloc[fold_train_idx].copy()
    train_te['target'] = y.iloc[fold_train_idx].values
    global_mean = y.iloc[fold_train_idx].mean()

    for col in ['HomePlanet', 'Deck']:
        te_map = train_te.groupby(col)['target'].mean()
        train.loc[train.index[fold_val_idx], f'TE_{col}'] = train.loc[train.index[fold_val_idx], col].map(te_map)
        train.loc[train.index[fold_train_idx], f'TE_{col}'] = train.loc[train.index[fold_train_idx], col].map(
            lambda x: te_map.get(x, global_mean))
        test[f'TE_{col}'] = test[col].map(te_map).fillna(global_mean)
        train[f'TE_{col}'] = train[f'TE_{col}'].fillna(0.5)
        test[f'TE_{col}'] = test[f'TE_{col}'].fillna(0.5)

    drop_cols = ['PassengerId', 'Name', 'Transported', 'Cabin']
    for col in drop_cols:
        if col in train.columns: train.drop(columns=[col], inplace=True)
        if col in test.columns: test.drop(columns=[col], inplace=True)

    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum', 'TotalSpend', 'CabinPct']
    for col in numeric_cols:
        if col not in train.columns: continue
        med = train[col].median()
        train[col] = train[col].fillna(med)
        test[col] = test[col].fillna(med)

    ordinal_cols = ['CryoSleep', 'VIP', 'Deck']
    for col in ordinal_cols:
        if col not in train.columns: continue
        train[col] = train[col].astype(str).fillna('MISSING')
        test[col] = test[col].astype(str).fillna('MISSING')

    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train[ordinal_cols] = ordinal_encoder.fit_transform(train[ordinal_cols])
    test[ordinal_cols] = ordinal_encoder.transform(test[ordinal_cols])

    nominal_cols = ['HomePlanet', 'Destination', 'Side']
    for col in nominal_cols:
        if col not in train.columns: continue
        train[col] = train[col].astype(str).fillna('MISSING')
        test[col] = test[col].astype(str).fillna('MISSING')

    train = pd.get_dummies(train, columns=nominal_cols, drop_first=False)
    test = pd.get_dummies(test, columns=nominal_cols, drop_first=False)

    for c in set(train.columns) - set(test.columns): test[c] = 0
    test = test[train.columns]

    return train, test

def train_and_predict():
    train_raw, test_raw = load_data()
    y = train_raw['Transported'].astype(int)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_models = 4
    oof_preds = np.zeros((len(train_raw), n_models))
    test_preds = np.zeros((len(test_raw), n_models))

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_raw)):
        print(f'Fold {fold + 1}')
        train_f = train_raw.copy()
        test_f = test_raw.copy()
        train_pp, test_pp = preprocess_te(train_f, test_f, y, train_idx, val_idx)
        X_tr = train_pp.iloc[train_idx]
        X_val = train_pp.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        cb = CatBoostClassifier(**CB_PARAMS)
        cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        oof_preds[val_idx, 0] = cb.predict(X_val).ravel()
        test_preds[:, 0] += cb.predict(test_pp).ravel() / N_FOLDS
        print(f'  CatBoost: {(cb.predict(X_val).ravel() == y_val.values).mean():.4f}')

        xgb_m = xgb.XGBClassifier(**XGB_PARAMS)
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds[val_idx, 1] = xgb_m.predict(X_val).ravel()
        test_preds[:, 1] += xgb_m.predict(test_pp).ravel() / N_FOLDS
        print(f'  XGBoost:  {(xgb_m.predict(X_val).ravel() == y_val.values).mean():.4f}')

        lgb_m = lgb.LGBMClassifier(**LGB_PARAMS)
        lgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        oof_preds[val_idx, 2] = lgb_m.predict(X_val).ravel()
        test_preds[:, 2] += lgb_m.predict(test_pp).ravel() / N_FOLDS
        print(f'  LightGBM: {(lgb_m.predict(X_val).ravel() == y_val.values).mean():.4f}')

        lr = LogisticRegression(**LR_PARAMS)
        lr.fit(X_tr, y_tr)
        oof_preds[val_idx, 3] = lr.predict(X_val).ravel()
        test_preds[:, 3] += lr.predict(test_pp).ravel() / N_FOLDS
        print(f'  Logistic: {(lr.predict(X_val).ravel() == y_val.values).mean():.4f}')

    print('\n=== Individual Model CV ===')
    for i, name in enumerate(['CatBoost', 'XGBoost', 'LightGBM', 'Logistic']):
        print(f'{name}: {(oof_preds[:, i] == y.values).mean():.4f}')

    ridge = RidgeClassifier(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(oof_preds, y)
    final_preds = ridge.predict(test_preds)
    print(f'Ridge coeffs: {ridge.coef_[0].round(3)}')

    test_raw = pd.read_csv('/home/ivi/.cache/kagglehub/competitions/spaceship-titanic/test.csv')
    submission = pd.DataFrame({
        'PassengerId': test_raw['PassengerId'],
        'Transported': final_preds.astype(bool),
    })
    submission.to_csv('submission.csv', index=False)
    print(submission.head())

if __name__ == '__main__':
    train_and_predict()
