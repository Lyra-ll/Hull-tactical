# =================================================================
# validate_target_engineering_v4.py (Fair Showdown)
# 目的: 为不同任务模式匹配专属的超参数，以进行一场真正公平的对决。
# =================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, roc_auc_score
import warnings
import json
import os

warnings.filterwarnings('ignore')

# ================= 1. 文件路径与核心配置 =================
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
REG_PARAMS_FILE = 'best_params_v3_original_only.json' # 回归模式的专属参数
N_SPLITS = 5
PURGE_SIZE = 1
EMBARGO_SIZE = 40 
HANDMADE_SUFFIXES = ['_rol_', '_diff', '_rank']
AI_PREFIX = 'AE_'

# ================= 2. 通用化核心验证函数 (保持不变) =================
def run_validation(X, y_reg, y_cls, sample_weights, params, task_type):
    """
    根据给定的task_type，执行一次完整的净化禁运时序交叉验证。
    """
    print(f"\n{'='*20} 开始测试: {task_type} 模式 {'='*20}")
    
    if task_type == 'REGRESSION':
        y = y_reg
        model_class = lgb.LGBMRegressor
        params.update({'objective': 'regression_l1', 'metric': 'rmse'})
        score_func = lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))
        score_name = "RMSE"
    else: 
        y = y_cls
        model_class = lgb.LGBMClassifier
        params.update({'objective': 'binary', 'metric': 'auc'})
        score_func = roc_auc_score
        score_name = "AUC"

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        purged_train_idx = train_idx[:-PURGE_SIZE]
        X_train_raw, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
        X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        fit_params = {}
        if task_type == 'WEIGHTED_CLASSIFICATION':
            fit_params['sample_weight'] = sample_weights.iloc[purged_train_idx]

        mean = np.nanmean(X_train_raw.values, axis=0); std = np.nanstd(X_train_raw.values, axis=0); std[std == 0] = 1.0
        X_train_scaled = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
        X_val_scaled = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)
        
        model = model_class(**params)
        model.fit(X_train_scaled, y_train, 
                  eval_set=[(X_val_scaled, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)],
                  **fit_params)
                  
        preds = model.predict_proba(X_val_scaled)[:, 1] if task_type != 'REGRESSION' else model.predict(X_val_scaled)
        score = score_func(y_val, preds)
        fold_scores.append(score)

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"    ✅ {task_type} 模式测试完成。平均OOF {score_name}: {mean_score:.8f}")
    return {'模式': task_type, '评估指标': score_name, '平均分数': mean_score, '分数标准差': std_score}

# ================= 3. 数据加载与准备 =================
print("--- 步骤1：加载数据并准备“纯原始”特征集 ---")
raw_df = pd.read_csv(RAW_DATA_FILE)
with open(REG_PARAMS_FILE, 'r') as f:
    reg_base_params = json.load(f) # 加载回归专属参数
df = raw_df
modern_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)

all_feature_names = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
ai_features = [f for f in all_feature_names if f.startswith(AI_PREFIX)]
handmade_features = [f for f in all_feature_names if any(suffix in f for suffix in HANDMADE_SUFFIXES)]
original_features = [f for f in all_feature_names if f not in ai_features and f not in handmade_features]

X = modern_df[original_features]
y_reg = modern_df['forward_returns'].fillna(0) 
y_cls = (modern_df['forward_returns'] > 0).astype(int)
sample_weights = np.abs(modern_df['forward_returns'].fillna(0))

# ================= 4. 自动化执行“三模”对决 =================
print("\n--- 步骤2：启动自动化“三模”对决 ---")

# --- 核心修改：为不同模式定义专属参数 ---
# 为回归模式加载专属优化参数
params_reg = reg_base_params.copy()
params_reg.update({'random_state': 42, 'n_jobs': -1})

# 为分类模式定义一套通用的、稳健的基础参数
params_cls = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

results = []
task_params_map = {
    'REGRESSION': params_reg,
    'CLASSIFICATION': params_cls,
    'WEIGHTED_CLASSIFICATION': params_cls
}

for task in task_params_map:
    result = run_validation(X, y_reg, y_cls, sample_weights, task_params_map[task], task)
    results.append(result)

# ================= 5. 生成最终“战力排行榜” =================
print(f"\n\n{'='*25} 目标工程最终对决报告 {'='*25}")
print(f"{'作战模式':<25} | {'评估指标':<10} | {'平均分数':<20} | {'分数标准差':<20}")
print("-" * 85)

for res in results:
    print(f"{res['模式']:<25} | {res['评估指标']:<10} | {res['平均分数']:<20.8f} | {res['分数标准差']:<20.8f}")
print("=" * 85)

