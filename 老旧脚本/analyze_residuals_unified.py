# =================================================================
# analyze_residuals_unified.py (Unified Residual Analysis Platform)
# 目的: 将残差生成与验证合二为一，一键式完成对AI和手工特征边际价值的终极检验。
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
print("--- 步骤1：加载所有数据和配置 ---")
raw_df = pd.read_csv(RAW_DATA_FILE)
ae_features_df = pd.read_csv(AE_FEATURES_FILE)
with open(PARAMS_FILE, 'r') as f:
    best_params = json.load(f)
df = pd.merge(raw_df, ae_features_df, on='date_id', how='left', suffixes=('_orig', None))
if 'forward_returns_orig' in df.columns: df.drop(columns=['forward_returns_orig'], inplace=True)
modern_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)

# 准备好所有“兵种”
all_feature_names = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
ai_features = [f for f in all_feature_names if f.startswith(AI_PREFIX)]
handmade_features = [f for f in all_feature_names if any(suffix in f for suffix in HANDMADE_SUFFIXES)]
original_features = [f for f in all_feature_names if f not in ai_features and f not in handmade_features]

X_original = modern_df[original_features]
X_engineered = modern_df[ai_features + handmade_features]
y_true = modern_df['forward_returns'].fillna(0)

# ================= 3. 【第一阶段】生成OOF预测与残差 =================
print("\n--- 阶段1：为“纯原始”部队生成OOF预测，寻找“认知盲区” ---")
final_params = best_params.copy()
final_params.update({'objective': 'regression_l1', 'metric': 'rmse', 'random_state': 42, 'n_jobs': -1})

tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
oof_predictions = np.zeros(len(X_original))

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_original)):
    print(f"    --- 正在处理第 {fold + 1}/{N_SPLITS} 折 ---")
    purged_train_idx = train_idx[:-PURGE_SIZE]
    X_train_raw, y_train = X_original.iloc[purged_train_idx], y_true.iloc[purged_train_idx]
    X_val_raw, y_val = X_original.iloc[val_idx], y_true.iloc[val_idx]
    
    mean = np.nanmean(X_train_raw.values, axis=0); std = np.nanstd(X_train_raw.values, axis=0); std[std == 0] = 1.0
    X_train_scaled = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
    X_val_scaled = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)
    
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train_scaled, y_train, 
              eval_set=[(X_val_scaled, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])
              
    oof_predictions[val_idx] = model.predict(X_val_scaled)

# 在内存中计算残差
valid_indices = oof_predictions != 0
residuals = pd.Series(np.nan, index=modern_df.index)
residuals.iloc[valid_indices] = y_true[valid_indices] - oof_predictions[valid_indices]
print("✅ “认知盲区”（残差）已在内存中生成完毕。")

# ================= 4. 【第二阶段】用“AI+手工”特征攻击残差 =================
print("\n--- 阶段2：派遣“AI+手工”混合部队，攻击“认知盲区” ---")

# 准备新的目标和特征集
y_residuals = residuals
valid_residual_indices = y_residuals.notna()
X_engineered_filtered = X_engineered[valid_residual_indices].reset_index(drop=True)
y_residuals_filtered = y_residuals.dropna().reset_index(drop=True)

print(f"准备使用 {X_engineered_filtered.shape[1]} 个“AI+手工”特征，攻击 {len(y_residuals_filtered)} 个残差目标。")

# 再次执行验证循环
oof_preds_on_residuals = np.zeros(len(X_engineered_filtered))
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_engineered_filtered)):
    # print(f"    --- 正在处理第 {fold + 1}/{N_SPLITS} 折 ---") # 如需详细信息可取消注释
    purged_train_idx = train_idx[:-PURGE_SIZE]
    X_train_raw, y_train = X_engineered_filtered.iloc[purged_train_idx], y_residuals_filtered.iloc[purged_train_idx]
    X_val_raw, y_val = X_engineered_filtered.iloc[val_idx], y_residuals_filtered.iloc[val_idx]
    
    mean = np.nanmean(X_train_raw.values, axis=0); std = np.nanstd(X_train_raw.values, axis=0); std[std == 0] = 1.0
    X_train_scaled = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
    X_val_scaled = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)
    
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_preds_on_residuals[val_idx] = model.predict(X_val_scaled)

# ================= 5. 【最终审判】 =================
print("\n--- 阶段3：最终审判 ---")
valid_indices_final = oof_preds_on_residuals != 0
rmse_on_residuals = np.sqrt(mean_squared_error(y_residuals_filtered[valid_indices_final], oof_preds_on_residuals[valid_indices_final]))
std_of_residuals = y_residuals_filtered[valid_indices_final].std()

print(f"\n{'='*25} 残差分析最终报告 {'='*25}")
print(f"基准误差 (预测残差的平均值): {std_of_residuals:.8f}")
print(f"模型误差 (预测残差):         {rmse_on_residuals:.8f}")
print("=" * 55)

improvement = (std_of_residuals - rmse_on_residuals) / std_of_residuals
if improvement > 0.001: 
    print(f"✅ 审判：胜利！我们的AI和手工特征成功预测了原始模型的“认知盲区”。")
    print(f"   它们提供了全新的、独立的预测价值，将预测误差降低了 {improvement:.4%}")
else:
    print(f"❌ 审判：失败。我们的AI和手工特征未能有效预测残差。")
    print(f"   它们提供的新信息非常有限，误差改善为 {improvement:.4%}")
