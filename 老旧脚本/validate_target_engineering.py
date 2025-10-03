# =================================================================
# validate_target_engineering.py (Target Engineering Showdown)
# 目的: 在最严格的验证框架下，科学对比“回归”与“分类”两种任务模式的性能。
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

# ================= 1. 核心改造：任务模式切换 =================
# 可选: 'REGRESSION' 或 'CLASSIFICATION'
TASK_TYPE = 'CLASSIFICATION' 

# --- 文件路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
AE_FEATURES_FILE = 'train_v8_tuned_ae_features_rigorous_gpu.csv' 
PARAMS_FILE = 'best_params_v3_original_only.json' 

# --- 验证策略 ---
N_SPLITS = 5
PURGE_SIZE = 1
EMBARGO_SIZE = 40 

# --- 特征“兵种”识别指纹 ---
HANDMADE_SUFFIXES = ['_rol_', '_diff', '_rank']
AI_PREFIX = 'AE_'

# ================= 2. 数据加载与特征准备 =================
print(f"--- 步骤1：加载数据并准备“纯原始”特征集 ---")
raw_df = pd.read_csv(RAW_DATA_FILE)
ae_features_df = pd.read_csv(AE_FEATURES_FILE)
with open(PARAMS_FILE, 'r') as f:
    best_params = json.load(f)
df = pd.merge(raw_df, ae_features_df, on='date_id', how='left', suffixes=('_orig', None))
if 'forward_returns_orig' in df.columns: df.drop(columns=['forward_returns_orig'], inplace=True)
modern_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)

# 识别并筛选出“纯原始”特征
all_feature_names = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
ai_features = [f for f in all_feature_names if f.startswith(AI_PREFIX)]
handmade_features = [f for f in all_feature_names if any(suffix in f for suffix in HANDMADE_SUFFIXES)]
original_features = [f for f in all_feature_names if f not in ai_features and f not in handmade_features]

X = modern_df[original_features]
y_reg = modern_df['forward_returns'].fillna(0) # 回归任务的目标
y_cls = (modern_df['forward_returns'] > 0).astype(int) # 分类任务的目标

# ================= 3. “终极武器”验证循环 =================
print(f"\n--- 步骤2：启动“终极武器”验证流程 ---")
print(f"--- 当前作战模式: {TASK_TYPE} ---")

# 根据模式选择目标
y = y_cls if TASK_TYPE == 'CLASSIFICATION' else y_reg

# 加载最优参数
final_params = best_params.copy()
final_params['random_state'] = 42
final_params['n_jobs'] = -1

# 根据模式选择模型和评估指标
if TASK_TYPE == 'CLASSIFICATION':
    model_class = lgb.LGBMClassifier
    final_params['objective'] = 'binary'
    final_params['metric'] = 'auc'
    score_func = roc_auc_score
    score_name = "AUC"
else: # REGRESSION
    model_class = lgb.LGBMRegressor
    final_params['objective'] = 'regression_l1'
    final_params['metric'] = 'rmse'
    score_func = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
    score_name = "RMSE"

tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
fold_scores = []
oof_predictions = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"    --- 正在处理第 {fold + 1}/{N_SPLITS} 折 ---")
    purged_train_idx = train_idx[:-PURGE_SIZE]
    X_train_raw, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
    X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    mean = np.nanmean(X_train_raw.values, axis=0); std = np.nanstd(X_train_raw.values, axis=0); std[std == 0] = 1.0
    X_train_scaled = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
    X_val_scaled = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)
    
    model = model_class(**final_params)
    model.fit(X_train_scaled, y_train, 
              eval_set=[(X_val_scaled, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
              
    # 分类任务需要预测概率
    if TASK_TYPE == 'CLASSIFICATION':
        preds = model.predict_proba(X_val_scaled)[:, 1]
    else:
        preds = model.predict(X_val_scaled)
        
    score = score_func(y_val, preds)
    fold_scores.append(score)
    print(f"        -> 第 {fold + 1} 折的 {score_name}: {score:.8f}")

# ================= 4. 最终裁决 =================
mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)

print(f"\n\n{'='*25} {TASK_TYPE} 模式实验结束 {'='*25}")
print(f"✅ 在5折“终极武器”交叉验证中，最终评估结果如下：")
print(f"   平均 OOF {score_name}: {mean_score:.8f}")
print(f"   OOF {score_name} 标准差: {std_score:.8f}")
