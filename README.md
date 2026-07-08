# Spaceship Titanic — Kaggle Competition

## Pipeline

```
train.csv / test.csv  →  缺失值填充  →  特征工程（含交叉特征）
    →  标签编码 + 数值化
    →  LightGBM 默认参数计算 feature importance
    →  Optuna 搜索最佳 top-k 特征子集
    →  Stacking 集成（9 种不同架构模型 + LR meta-learner）
    →  submission/submission.csv
```

## 模型

| 模型 | 架构 |
|------|------|
| LightGBM | 叶节点 GBDT |
| XGBoost | 层级 GBDT |
| CatBoost | 对称树 + Ordered Boosting |
| Random Forest | Bagging |
| AdaBoost | 自适应 Boosting |
| KNN | 基于实例 |
| SVM (RBF) | 间隔 + 核技巧 |
| MLP | 神经网络 |
| GaussianNB | 概率生成式 |

Meta-learner: LogisticRegression

## 使用方法

```bash
# 安装依赖
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna

# 运行
python main.py

# 或使用 Jupyter Notebook
jupyter notebook main.ipynb
```

## 目录结构

```
Spaceship-Titanic/
├── data/               # 原始数据（train.csv, test.csv）
├── submission/         # 提交文件输出目录
├── main.py             # 完整 Pipeline 脚本
├── main.ipynb          # Jupyter Notebook 版本
├── .gitignore
└── README.md
```
