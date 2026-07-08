"""
============================================================================
Spaceship Titanic - 主流程 Pipeline
============================================================================
将原始数据处理、特征工程、模型训练、集成、提交整合为单一流程。

执行顺序：
  1. 读取数据 (data/train.csv, data/test.csv)
  2. 缺失值填充 (impute_missing)
  3. 特征工程，含交叉特征 (engineer_features)
  4. 标签编码 + 数值化 (preprocess)
  5. LightGBM 默认参数计算特征重要性 → Optuna 搜索最佳 top-k
  6. 用最佳 top-k 特征子集训练 Stacking 集成（10 种不同架构模型）
  7. 生成提交文件 → submission/submission.csv
============================================================================
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd


import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================
DATA_DIR = 'data'               # 原始数据目录
SUBMISSION_DIR = 'submission'   # 提交文件输出目录
SEED = 42                       # 随机种子（保证可复现）
N_FOLDS = 5                     # 交叉验证折数
N_TRIALS = 20                   # Optuna 搜索 top-k 的尝试次数

os.makedirs(SUBMISSION_DIR, exist_ok=True)


def log(msg: str):
    """带时间戳的统一日志输出"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# 步骤 1: 缺失值填充
# ============================================================
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于领域知识的缺失值填充策略（不修改原始数据）。

    填充顺序：
      1. CryoSleep — 若所有消费均为 0 → True, 否则 False
      2. 消费字段 — CryoSleep==True 填 0, 否则填中位数
      3. HomePlanet — 根据 Cabin 所在甲板映射
      4. Cabin — 根据 HomePlanet 映射补全甲板
      5. Destination — 根据 HomePlanet 映射（默认 TRAPPIST-1e）
      6. Age — 按 HomePlanet 中位数补全
      7. VIP — 众数填充
      8. Name — 缺失填 'Unknown'
    """
    data = df.copy()
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    log(f"  填充缺失值: 共 {data.isna().sum().sum()} 个缺失值")

    # ---- 解析 Cabin ----
    data['CabinDeck'] = data['Cabin'].str.split('/').str[0]
    data['CabinSide'] = data['Cabin'].str.split('/').str[2]

    # ---- Step 1: 从消费推断 CryoSleep ----
    spend_sum = data[spend_cols].sum(axis=1)
    cryo_mask = data['CryoSleep'].isna()
    data.loc[cryo_mask & (spend_sum > 0), 'CryoSleep'] = False
    data.loc[cryo_mask & (spend_sum == 0), 'CryoSleep'] = True
    log(f"  Step 1/8: CryoSleep 填充 ({cryo_mask.sum()} 个)")

    # ---- Step 2: 从 CryoSleep 推断消费字段 ----
    for c in spend_cols:
        mask = data[c].isna()
        data.loc[mask & (data['CryoSleep'] == True), c] = 0
        data.loc[mask & (data['CryoSleep'] == False), c] = \
            data.loc[data['CryoSleep'] == False, c].median()
    log(f"  Step 2/8: 消费字段填充")

    # ---- Step 3: 从 Cabin 甲板推断 HomePlanet ----
    deck_to_planet = {
        'A': 'Europa', 'B': 'Europa', 'C': 'Europa',
        'D': 'Mars', 'E': 'Mars',
        'F': 'Earth', 'G': 'Earth', 'T': 'Earth',
    }
    hp_mask = data['HomePlanet'].isna()
    data.loc[hp_mask, 'HomePlanet'] = data.loc[hp_mask, 'CabinDeck'].map(deck_to_planet)
    data['HomePlanet'] = data['HomePlanet'].fillna(data['HomePlanet'].mode()[0])
    log(f"  Step 3/8: HomePlanet 填充 ({hp_mask.sum()} 个)")

    # ---- Step 4: 从 HomePlanet 推断 Cabin 甲板 ----
    planet_to_deck = {'Earth': 'F', 'Europa': 'B', 'Mars': 'F'}
    deck_mask = data['CabinDeck'].isna()
    data.loc[deck_mask, 'CabinDeck'] = data.loc[deck_mask, 'HomePlanet'].map(planet_to_deck)
    data['CabinDeck'] = data['CabinDeck'].fillna('F')
    # 还原 Cabin 字段
    data['CabinNum'] = data['Cabin'].str.split('/').str[1].fillna('0')
    data['CabinSide'] = data['CabinSide'].fillna('P')
    data['Cabin'] = data['CabinDeck'] + '/' + data['CabinNum'] + '/' + data['CabinSide']
    log(f"  Step 4/8: Cabin 甲板填充 ({deck_mask.sum()} 个)")

    # ---- Step 5: 从 HomePlanet 推断 Destination ----
    planet_to_dest = {'Earth': 'TRAPPIST-1e', 'Europa': 'TRAPPIST-1e', 'Mars': 'TRAPPIST-1e'}
    dest_mask = data['Destination'].isna()
    data.loc[dest_mask, 'Destination'] = data.loc[dest_mask, 'HomePlanet'].map(planet_to_dest)
    data['Destination'] = data['Destination'].fillna('TRAPPIST-1e')
    log(f"  Step 5/8: Destination 填充 ({dest_mask.sum()} 个)")

    # ---- Step 6: 按 HomePlanet 中位数填充 Age ----
    age_medians = {'Earth': 23.0, 'Europa': 33.0, 'Mars': 28.0}
    age_mask = data['Age'].isna()
    data.loc[age_mask, 'Age'] = data.loc[age_mask, 'HomePlanet'].map(age_medians)
    data['Age'] = data['Age'].fillna(data['Age'].median())
    log(f"  Step 6/8: Age 填充 ({age_mask.sum()} 个)")

    # ---- Step 7: VIP 众数填充 ----
    data['VIP'] = data['VIP'].fillna(data['VIP'].mode()[0])
    log(f"  Step 7/8: VIP 填充")

    # ---- Step 8: Name 填充 ----
    data['FirstName'] = data['Name'].str.split().str[0]
    data['LastName'] = data['Name'].str.split().str[-1]
    data['Name'] = data['Name'].fillna('Unknown')
    log(f"  Step 8/8: Name 填充")

    # ---- 类型转换 ----
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
    for c in cat_cols:
        if c in ('VIP', 'CryoSleep'):
            data[c] = data[c].astype(bool)
        else:
            data[c] = data[c].astype(str)
    for c in spend_cols + ['Age']:
        data[c] = data[c].astype(float)

    # 清理辅助列
    drop_cols = ['CabinDeck', 'CabinNum', 'CabinSide', 'FirstName', 'LastName']
    data.drop(columns=[c for c in drop_cols if c in data.columns], inplace=True)

    return data


# ============================================================
# 步骤 2: 特征工程
# ============================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成全部特征（包括交叉特征、聚合特征等）。

    生成的特征分类：
      - Group 特征: GroupSize, IsAlone, GroupMean/Max/Sum 消费, GroupVIPCount, GroupAge 统计
      - Cabin 解析: Deck, Side, CabinNum（qcut 分箱）
      - Name 特征: FamilySize
      - 消费聚合: TotalSpend, HasSpent, SpendCategoryCount, HasXxx, LogXxx
      - Age 特征: IsChild, IsSenior, AgeGroup
      - 交叉特征: HomePlanet_Destination, HomePlanet_Deck, Deck_Side, CryoSleep_VIP, CryoSleep_Deck
      - 衍生特征: SpendPerYear, DeckNum, SideNum, VIP_TotalSpend
    """
    data = df.copy()
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    log(f"  特征工程: 输入 {data.shape[1]} 列")

    # ---- Group 特征（同一 GroupId 的乘客为同行组） ----
    data['GroupId'] = data['PassengerId'].str.split('_').str[0]
    group_size_map = data.groupby('GroupId')['PassengerId'].count().to_dict()
    data['GroupSize'] = data['GroupId'].map(group_size_map)
    data['IsAlone'] = (data['GroupSize'] == 1).astype(bool)

    # Group 级别的消费聚合（mean / max / sum）
    group_spend = data.copy()
    group_spend[spend_cols] = group_spend[spend_cols].fillna(0)
    group_agg = group_spend.groupby('GroupId')[spend_cols].agg(['mean', 'max', 'sum'])
    group_agg.columns = [
        '_'.join(c).replace('mean', 'GroupMean').replace('max', 'GroupMax').replace('sum', 'GroupSum')
        for c in group_agg.columns
    ]
    data = data.merge(group_agg.reset_index(), on='GroupId', how='left')

    # Group VIP 人数
    group_spend['VIP_num'] = group_spend['VIP'].fillna(False).astype(bool).astype(int)
    grp_vip = group_spend.groupby('GroupId')['VIP_num'].sum().rename('GroupVIPCount').reset_index()
    data = data.merge(grp_vip, on='GroupId', how='left')
    data['GroupVIPCount'] = data['GroupVIPCount'].fillna(0).astype(int)

    # Group Age 统计
    group_spend['Age_num'] = pd.to_numeric(group_spend['Age'], errors='coerce')
    grp_age = group_spend.groupby('GroupId')['Age_num'].agg(['mean', 'max', 'min']).rename(
        columns={'mean': 'GroupMeanAge', 'max': 'GroupMaxAge', 'min': 'GroupMinAge'}
    ).reset_index()
    data = data.merge(grp_age, on='GroupId', how='left')
    for c in ['GroupMeanAge', 'GroupMaxAge', 'GroupMinAge']:
        data[c] = data[c].fillna(data['Age'])

    # ---- Cabin 解析 ----
    data['Deck'] = data['Cabin'].str.split('/').str[0]
    data['Side'] = data['Cabin'].str.split('/').str[2]
    data['CabinNum'] = data['Cabin'].str.split('/').str[1].astype(float)
    data['CabinNum'] = pd.qcut(data['CabinNum'], q=15, labels=False, duplicates='drop').astype(float)

    # ---- Name 特征 ----
    data['LastName'] = data['Name'].str.split().str[-1]
    fam_size_map = data.groupby('LastName')['PassengerId'].count().to_dict()
    data['FamilySize'] = data['LastName'].map(fam_size_map)
    data['FirstName'] = data['Name'].str.split().str[0]

    # ---- 消费聚合 ----
    for c in spend_cols:
        data[c] = data[c].fillna(0)
    data['TotalSpend'] = data[spend_cols].sum(axis=1)
    data['HasSpent'] = (data['TotalSpend'] > 0).astype(bool)
    data['SpendCategoryCount'] = (data[spend_cols] > 0).sum(axis=1)
    for c in spend_cols:
        data[f'Has{c}'] = (data[c] > 0).astype(bool)
    for c in spend_cols:
        data[f'Log{c}'] = np.log1p(data[c])

    # ---- Age 特征 ----
    data['IsChild'] = (data['Age'] < 13).astype(bool)
    data['IsSenior'] = (data['Age'] > 65).astype(bool)
    data['AgeGroup'] = pd.cut(
        data['Age'],
        bins=[-1, 12, 18, 30, 45, 60, 100],
        labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'MiddleAge', 'Senior'],
    )

    # ---- 交叉特征 ----
    data['HomePlanet_Destination'] = data['HomePlanet'].astype(str) + '_' + data['Destination'].astype(str)
    data['HomePlanet_Deck'] = data['HomePlanet'].astype(str) + '_' + data['Deck']
    data['Deck_Side'] = data['Deck'] + '_' + data['Side']
    data['CryoSleep_VIP'] = data['CryoSleep'].astype(str) + '_' + data['VIP'].astype(str)
    data['CryoSleep_Deck'] = data['CryoSleep'].astype(str) + '_' + data['Deck']

    # ---- 数值衍生 ----
    data['SpendPerYear'] = data['TotalSpend'] / (data['Age'] + 1)
    deck_order = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 0}
    data['DeckNum'] = data['Deck'].map(deck_order)
    data['SideNum'] = (data['Side'] == 'P').astype(int)
    data['VIP_TotalSpend'] = data['VIP'].astype(float) * data['TotalSpend']

    # ---- 清理辅助列 ----
    data.drop(columns=['GroupId', 'LastName', 'FirstName'], errors='ignore', inplace=True)

    log(f"  特征工程: 输出 {data.shape[1]} 列 ({data.select_dtypes(include='number').shape[1]} 数值 / "
        f"{data.select_dtypes(include=['object', 'category', 'bool']).shape[1]} 类别)")
    return data


# ============================================================
# 步骤 3: 预处理（标签编码 + 数值化）
# ============================================================
def preprocess(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = 'Transported',
) -> tuple:
    """
    对特征工程后的数据进行统一预处理。

    操作：
      1. 收集所有非数值列，用 LabelEncoder 编码
      2. 测试集中未见过的类别映射到 -1
      3. 将所有列转为 float

    返回:
      X_train, y_train, X_test, feature_names, label_encoders
    """
    log("预处理: 标签编码 + 数值化")

    # 提取 PassengerId 供后续使用，然后移除标识列
    test_passenger_ids = test['PassengerId'].copy()

    # 确定特征列（排除 PassengerId / Name / Cabin / 目标列）
    exclude_cols = {'PassengerId', 'Name', 'Cabin', target_col}
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X_train = train[feature_cols].copy()
    y_train = train[target_col].astype(int).copy()
    X_test = test[feature_cols].copy() if target_col not in test.columns else test[feature_cols].copy()

    # 找出所有非数值列
    non_numeric_cols = []
    for c in feature_cols:
        if X_train[c].dtype in ('object', 'category', 'bool'):
            non_numeric_cols.append(c)

    log(f"  发现 {len(non_numeric_cols)} 个类别列: {non_numeric_cols}")

    # Label Encoding
    encoders = {}
    for c in non_numeric_cols:
        le = LabelEncoder()
        # 统一转为字符串以确保一致性
        train_vals = X_train[c].astype(str)
        le.fit(train_vals)
        X_train[c] = le.transform(train_vals)
        # 测试集：已知类别映射到编码，未知类别映射到 -1
        test_vals = X_test[c].astype(str)
        X_test[c] = test_vals.map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        ).astype(int)
        encoders[c] = le
        log(f"    编码 {c}: {len(le.classes_)} 个类别")

    # 全部转为 float（sklearn 要求输入全数值）
    for c in feature_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors='coerce').fillna(0)
        X_test[c] = pd.to_numeric(X_test[c], errors='coerce').fillna(0)

    log(f"  预处理完成: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, y_train, X_test, feature_cols, encoders, test_passenger_ids


# ============================================================
# 步骤 4: LightGBM 默认参数搜索最佳 top-k 特征子集
# ============================================================
def compute_feature_importance(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    使用 LightGBM 默认参数 + 5-Fold CV 平均特征重要性。

    注意：此处只使用默认参数，不进行超参调优。
    返回值: 与 X.columns 对应的 feature_importances_ 数组
    """
    log("计算特征重要性 (LightGBM 默认参数, 5-Fold CV)...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    importances = np.zeros(X.shape[1])

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(random_state=SEED, verbosity=-1, force_col_wise=True)
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[va], y.iloc[va])],
            callbacks=[lgb.log_evaluation(0)],
        )
        importances += model.feature_importances_ / N_FOLDS
        log(f"    第 {fold + 1}/{N_FOLDS} 折完成")

    return importances


def cv_score_topk(X: pd.DataFrame, y: pd.Series, top_k: int, importance_idx: np.ndarray) -> float:
    """
    使用 top_k 个最重要的特征，5-Fold CV 计算 LightGBM 的 ROC-AUC。
    """
    top_cols = [X.columns[i] for i in importance_idx[:top_k]]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []

    for tr, va in skf.split(X, y):
        model = lgb.LGBMClassifier(random_state=SEED, verbosity=-1, force_col_wise=True)
        model.fit(X.iloc[tr][top_cols], y.iloc[tr])
        scores.append(
            roc_auc_score(y.iloc[va], model.predict_proba(X.iloc[va][top_cols])[:, 1])
        )

    return np.mean(scores)


def objective_topk(trial, X, y, importance_idx):
    """Optuna 目标函数：搜索最佳 top-k"""
    top_k = trial.suggest_int('top_k', 10, len(importance_idx))
    score = cv_score_topk(X, y, top_k, importance_idx)
    log(f"      Trial {trial.number}: top_k={top_k}, CV ROC-AUC={score:.5f}")
    return score


def search_best_topk(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    完整 top-k 搜索流程：
      1. 计算全部特征的 feature importance
      2. Optuna 搜索最优 k
      3. 返回 (selected_features_list, best_top_k)

    使用 LightGBM 默认参数，不进行超参调优。
    """
    log("=" * 60)
    log("开始搜索最佳 top-k 特征子集")
    log("=" * 60)

    # Step 1: 特征重要性
    importances = compute_feature_importance(X, y)
    importance_idx = np.argsort(importances)[::-1]

    # 打印 Top 20 特征
    feat_imp_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances,
    }).sort_values('importance', ascending=False)
    log("Top 20 最重要的特征:")
    for i, row in feat_imp_df.head(20).iterrows():
        log(f"    {row['feature']}: {row['importance']:.2f}")

    # Step 2: Optuna 搜索
    log(f"Optuna 搜索最佳 top-k (随机种子={SEED}, trials={N_TRIALS})...")
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=3),
    )
    study.optimize(
        lambda trial: objective_topk(trial, X, y, importance_idx),
        n_trials=N_TRIALS,
    )

    best_k = study.best_params['top_k']
    best_score = study.best_value
    log(f"最优 top-k: {best_k}, CV ROC-AUC: {best_score:.5f}")

    selected_cols = [X.columns[i] for i in importance_idx[:best_k]]
    log(f"选择的特征 ({len(selected_cols)} 个): {selected_cols}")

    return selected_cols, best_k


# ============================================================
# 步骤 5: Stacking 集成
# ============================================================
def train_stacking(X_train, y_train, X_test):
    """
    Stacking 集成训练。
    使用 StackingClassifier（内部 5-Fold CV，n_jobs=-1 并行）。
    """
    log(f"基模型: 9 models (LGB/XGB/CB/RF/Ada/KNN/SVM/MLP/NB)")
    log(f"Meta-learner: LogisticRegression")

    estimators = [
        ('lgb', lgb.LGBMClassifier(random_state=SEED, verbosity=-1, force_col_wise=True)),
        ('xgb', xgb.XGBClassifier(random_state=SEED, verbosity=0)),
        ('cb', CatBoostClassifier(random_seed=SEED, verbose=False, allow_writing_files=False)),
        ('rf', RandomForestClassifier(random_state=SEED)),
        ('ada', AdaBoostClassifier(random_state=SEED)),
        ('knn', KNeighborsClassifier()),
        ('svm', SVC(random_state=SEED, probability=True, cache_size=1000)),
        ('mlp', MLPClassifier(random_state=SEED, max_iter=500, early_stopping=True)),
        ('nb', GaussianNB()),
    ]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=5, n_jobs=-1, stack_method='predict_proba',
    )
    stacking.fit(X_train, y_train)
    log("  Stacking 训练完成")
    return stacking.predict(X_test)


# ============================================================
# 主流程
# ============================================================
def main():
    """执行完整 Pipeline"""
    start_time = time.time()
    log("=" * 60)
    log("Spaceship Titanic — 完整 Pipeline 开始")
    log("=" * 60)

    # ---- 数据加载 ----
    log("Step 0: 加载原始数据")
    train_path = os.path.join(DATA_DIR, 'train.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    log(f"  train: {train_raw.shape}, test: {test_raw.shape}")

    # ---- Step 1: 缺失值填充 ----
    log("Step 1: 缺失值填充")
    train_imp = impute_missing(train_raw)
    test_imp = impute_missing(test_raw)

    # ---- Step 2: 特征工程 ----
    log("Step 2: 特征工程（含交叉特征）")
    train_feat = engineer_features(train_imp)
    test_feat = engineer_features(test_imp)

    # ---- Step 3: 预处理 ----
    log("Step 3: 预处理（标签编码 + 数值化）")
    X_train, y_train, X_test, feature_names, encoders, test_ids = preprocess(
        train_feat, test_feat
    )

    # ---- Step 4: 搜索最佳 top-k ----
    log("Step 4: 搜索 LightGBM 最佳 top-k 特征子集")
    selected_features, best_k = search_best_topk(X_train, y_train)

    # ---- Step 5: Stacking 集成 ----
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    log(f"使用特征 ({len(selected_features)} 个)")
    test_preds = train_stacking(X_train_selected, y_train, X_test_selected)

    # ---- 预测 ----
    log("\nStep 6: 预测测试集")
    preds = test_preds
    log(f"预测完成, 预测分布: 正类={(preds == 1).sum()}, 负类={(preds == 0).sum()}")

    # ---- 保存提交文件 ----
    submission_path = os.path.join(SUBMISSION_DIR, 'submission.csv')
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Transported': preds.astype(bool),
    })
    submission.to_csv(submission_path, index=False)
    log(f"提交文件已保存至: {submission_path}")

    # ---- 用时统计 ----
    elapsed = time.time() - start_time
    log("=" * 60)
    log(f"Pipeline 完成！总耗时: {elapsed / 60:.2f} 分钟 ({elapsed:.0f} 秒)")
    log(f"最优 top-k: {best_k}")
    log(f"输出文件: {submission_path}")
    log("=" * 60)


if __name__ == '__main__':
    main()
