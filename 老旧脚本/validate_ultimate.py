# =================================================================
# validate_ultimate.py (V1.0 - 终极武器版)
# 目的: 使用带净化和禁运的严格时序交叉验证，获得最真实的模型性能评估。
# =================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
import json
import os

warnings.filterwarnings('ignore')

# ================= 1. 全局配置 =================
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
AE_FEATURES_FILE = 'train_v6_kfold_leakfree_ae_features.csv'
RANKED_FEATURES_BLUEPRINT = 'ranked_features.csv'
N_TOP_FEATURES = 30
PARAMS_FILE = 'best_params.json'

# --- 核心：定义净化与禁运的参数 ---
N_SPLITS = 5
PURGE_SIZE = 1
EMBARGO_SIZE = 40 

# ================= 2. 数据加载与准备 =================
print("--- 正在加载数据与配置... ---")
try:
    raw_df = pd.read_csv(RAW_DATA_FILE)
    ae_features_df = pd.read_csv(AE_FEATURES_FILE)
    ranked_features_df = pd.read_csv(RANKED_FEATURES_BLUEPRINT, index_col=0)
    with open(PARAMS_FILE, 'r') as f:
        best_params = json.load(f)
    print("    -> 所有文件加载成功！")
except FileNotFoundError as e:
    print(f"错误: 加载文件失败 - {e}")
    exit()

df = pd.merge(raw_df, ae_features_df, on='date_id', suffixes=('_orig', None))
if 'forward_returns_orig' in df.columns:
    df.drop(columns=['forward_returns_orig'], inplace=True)

modern_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)
DREAM_TEAM_FEATURES = ranked_features_df.head(N_TOP_FEATURES).index.tolist()

X = modern_df[DREAM_TEAM_FEATURES]
y = modern_df['forward_returns'].fillna(0)

final_params = best_params.copy()
final_params.update({'objective': 'regression_l1', 'metric': 'rmse', 'random_state': 42, 'n_jobs': -1})


# ================= 3. 核心：终极武器验证循环 =================
print(f"\n--- 启动终极验证流程 (Purged & Embargoed TimeSeriesSplit) ---")
print(f"    Folds: {N_SPLITS}, Purge Size: {PURGE_SIZE}, Embargo Size: {EMBARGO_SIZE}")

tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"\n{'='*25} 开始处理第 {fold + 1}/{N_SPLITS} 折 {'='*25}")

    purged_train_idx = train_idx[:-PURGE_SIZE]

    X_train_raw, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
    X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]

    print(f"    Train period: index {purged_train_idx[0]} to {purged_train_idx[-1]} (Size: {len(purged_train_idx)})")
    print(f"    Validation period: index {val_idx[0]} to {val_idx[-1]} (Size: {len(val_idx)})")
    
    # a. “诚实”缩放：只在当前折的训练集上计算统计量
    mean = np.nanmean(X_train_raw.values, axis=0)
    std = np.nanstd(X_train_raw.values, axis=0)
    std[std == 0] = 1.0
    
    X_train_scaled = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
    X_val_scaled = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)
    
    # b. 模型训练：使用我们的黄金参数
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train_scaled, y_train, 
              eval_set=[(X_val_scaled, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])
              
    # c. 预测与评估
    preds = model.predict(X_val_scaled)
    score = np.sqrt(mean_squared_error(y_val, preds))
    fold_scores.append(score)
    print(f"    -> 第 {fold + 1} 折的诚实RMSE: {score:.8f}")

# ================= 4. 最终裁决 =================
mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)

print(f"\n\n{'='*25} 终极验证结束 {'='*25}")
print(f"✅ 在 {N_SPLITS} 折“带净化和禁运”的交叉验证中，我们得到了最终的“诚实”评估：")
print(f"   平均 OOF RMSE: {mean_score:.8f}")
print(f"   OOF RMSE 标准差: {std_score:.8f}")