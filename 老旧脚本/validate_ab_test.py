# =================================================================
# validate_multi_group_v3.py (Feature Set Analysis Platform)
# 目的: 提供一个标准化的多组实验平台，用于全面分析不同类型特征组合的性能。
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
# --- 文件路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
AE_FEATURES_FILE = 'train_v9_deep_ae_features.csv'
PARAMS_FILE = 'best_params_v2_causal.json'

# --- 验证策略 ---
N_SPLITS = 5
PURGE_SIZE = 1
EMBARGO_SIZE = 40 

# --- 特征“兵种”识别指纹 ---
HANDMADE_SUFFIXES = ['_rol_', '_diff', '_rank']
AI_PREFIX = 'AE_'

# ================= 2. 核心验证函数 (与V2版完全相同) =================
def run_validation(X, y, params, group_name):
    """
    对给定的特征集X和目标y，执行一次完整的净化禁运时序交叉验证。
    返回平均RMSE和RMSE的标准差。
    """
    if X.shape[1] == 0:
        print(f"\n{'='*20} 跳过测试: {group_name} (无特征) {'='*20}")
        return np.nan, np.nan
        
    print(f"\n{'='*20} 开始测试: {group_name} {'='*20}")
    print(f"    包含 {X.shape[1]} 个特征。")
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # print(f"    --- 正在处理第 {fold + 1}/{N_SPLITS} 折 ---") # 可以取消注释以显示更详细的过程
        purged_train_idx = train_idx[:-PURGE_SIZE]
        X_train_raw, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
        X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        mean = np.nanmean(X_train_raw.values, axis=0)
        std = np.nanstd(X_train_raw.values, axis=0)
        std[std == 0] = 1.0
        
        X_train_scaled = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
        X_val_scaled = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_scaled, y_train, 
                  eval_set=[(X_val_scaled, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
                  
        preds = model.predict(X_val_scaled)
        score = np.sqrt(mean_squared_error(y_val, preds))
        fold_scores.append(score)

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"    ✅ {group_name} 测试完成。平均RMSE: {mean_score:.8f}")
    return mean_score, std_score

# ================= 3. 数据加载与特征“兵种”识别 =================
print("--- 步骤1：加载数据并自动识别三大“兵种” ---")
# ... (省略数据加载合并代码) ...
raw_df = pd.read_csv(RAW_DATA_FILE); ae_features_df = pd.read_csv(AE_FEATURES_FILE);
with open(PARAMS_FILE, 'r') as f: best_params = json.load(f)
df = pd.merge(raw_df, ae_features_df, on='date_id', suffixes=('_orig', None))
if 'forward_returns_orig' in df.columns: df.drop(columns=['forward_returns_orig'], inplace=True)
modern_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)
y = modern_df['forward_returns'].fillna(0)
all_feature_names = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]# --------------------------------------------------------------------

# 自动化识别三大兵种
ai_features = [f for f in all_feature_names if f.startswith(AI_PREFIX)]
handmade_features = [f for f in all_feature_names if any(suffix in f for suffix in HANDMADE_SUFFIXES)]
original_features = [f for f in all_feature_names if f not in ai_features and f not in handmade_features]

print(f"兵种识别完毕：{len(original_features)}个原始特征, {len(ai_features)}个AI特征, {len(handmade_features)}个手工特征。")

# ================= 4. 组建七大“方面军”并开始大阅兵 =================
print("\n--- 步骤2：开始对七大特征“方面军”进行实战检验 ---")
final_params = best_params.copy()
final_params.update({'objective': 'regression_l1', 'metric': 'rmse', 'random_state': 42, 'n_jobs': -1})

results = {}
# 方面军1: 原始 + AI
X1 = modern_df[original_features + ai_features]
results['原始+AI'] = run_validation(X1, y, final_params, "方面军1: 原始+AI")

# 方面军2: 原始 + AI + 手工
X2 = modern_df[original_features + ai_features + handmade_features]
results['原始+AI+手工'] = run_validation(X2, y, final_params, "方面军2: 原始+AI+手工")

# 方面军3: 原始 + 手工
X3 = modern_df[original_features + handmade_features]
results['原始+手工'] = run_validation(X3, y, final_params, "方面军3: 原始+手工")

# 方面军4: 纯手工 + AI
X4 = modern_df[handmade_features + ai_features]
results['纯手工+AI'] = run_validation(X4, y, final_params, "方面军4: 纯手工+AI")

# 方面军5: 纯原始
X5 = modern_df[original_features]
results['纯原始'] = run_validation(X5, y, final_params, "方面军5: 纯原始")

# 方面军6: 纯手工
X6 = modern_df[handmade_features]
results['纯手工'] = run_validation(X6, y, final_params, "方面军6: 纯手工")

# 方面军7: 纯AI
X7 = modern_df[ai_features]
results['纯AI'] = run_validation(X7, y, final_params, "方面军7: 纯AI")


# ================= 5. 生成最终“战力排行榜” =================
print(f"\n\n{'='*25} 特征组合最终战力排行榜 {'='*25}")
print(f"{'方面军 (特征组合)':<25} | {'特征数':<10} | {'平均RMSE':<20} | {'RMSE标准差':<20}")
print("-" * 85)

# 对结果按RMSE升序排序
sorted_results = sorted(results.items(), key=lambda item: item[1][0] if not np.isnan(item[1][0]) else np.inf)

for name, (rmse, std) in sorted_results:
    feature_count_map = {
        '原始+AI': len(X1.columns), '原始+AI+手工': len(X2.columns), '原始+手工': len(X3.columns),
        '纯手工+AI': len(X4.columns), '纯原始': len(X5.columns), '纯手工': len(X6.columns), '纯AI': len(X7.columns)
    }
    feature_count = feature_count_map.get(name, 0)
    
    if not np.isnan(rmse):
        print(f"{name:<25} | {feature_count:<10} | {rmse:<20.8f} | {std:<20.8f}")
    else:
        print(f"{name:<25} | {feature_count:<10} | {'N/A':<20} | {'N/A':<20}")
print("=" * 85)