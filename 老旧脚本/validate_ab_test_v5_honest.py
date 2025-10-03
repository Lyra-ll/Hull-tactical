import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import warnings
import json
import os

# =================================================================
# validate_ab_test_v5_honest.py (诚实评估版)
# 目的: 使用与主模型完全一致的、最可靠的预处理方法，
#       进行一次绝对“诚实”的特征集A/B测试。
# =================================================================

warnings.filterwarnings('ignore')

# ================= 1. 全局配置 (保持不变) =================
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
AE_FEATURES_FILE = 'train_v11_autotune_clf_ae_features.csv' 
PARAMS_FILE = 'best_params_v4_weighted_clf.json'
N_SPLITS = 5; PURGE_SIZE = 1; EMBARGO_SIZE = 40 
AI_PREFIX = 'AE_'; HANDMADE_SUFFIXES = ['_rol_', '_diff', '_rank']

# ================= 2. 新增：我们最可靠的预处理函数 =================
### 新增 ###
def preprocess_data(X_train, X_val):
    """
    使用统一的、无数据泄露的逻辑来处理训练集和验证集。
    """
    X_train_filled = X_train.ffill(limit=3)
    median_filler = X_train_filled.median()
    X_train_filled.fillna(median_filler, inplace=True)
    X_train_filled.fillna(0, inplace=True)
    
    X_val_filled = X_val.ffill(limit=3)
    X_val_filled.fillna(median_filler, inplace=True)
    X_val_filled.fillna(0, inplace=True)
    
    return X_train_filled, X_val_filled

# ================= 3. 核心验证函数 (已升级) =================
def run_validation(X, y, sample_weight, params, group_name):
    print(f"\n{'='*20} 开始测试: {group_name} {'='*20}")
    print(f"    包含 {X.shape[1]} 个特征。")
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        purged_train_idx = train_idx[:-PURGE_SIZE]
        X_train_raw, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
        X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
        sw_train = sample_weight.iloc[purged_train_idx]
        sw_val = sample_weight.iloc[val_idx]
        
        ### 升级 ###
        # 替换掉旧的、有问题的 np.nan_to_num 逻辑
        X_train_filled, X_val_filled = preprocess_data(X_train_raw, X_val_raw)
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_filled, y_train, 
                  sample_weight=sw_train,
                  eval_set=[(X_val_filled, y_val)],
                  eval_sample_weight=[sw_val],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
                  
        preds = model.predict_proba(X_val_filled)[:, 1]
        score = roc_auc_score(y_val, preds)
        fold_scores.append(score)

    mean_score = np.mean(fold_scores); std_score = np.std(fold_scores)
    print(f"    ✅ {group_name} 测试完成。平均AUC: {mean_score:.8f}")
    return mean_score, std_score

# ================= 4. 数据加载与特征识别 (保持不变) =================
print("--- 步骤1：加载数据并准备分类任务 ---")
raw_df = pd.read_csv(RAW_DATA_FILE)
ae_features_df = pd.read_csv(AE_FEATURES_FILE)
with open(PARAMS_FILE, 'r') as f: best_params = json.load(f)
df = pd.merge(raw_df, ae_features_df, on='date_id', suffixes=('_orig', None))
modern_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)
y = (modern_df['forward_returns'] > 0).astype(int)
sample_weight = modern_df['forward_returns'].abs()
all_feature_names = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
ai_features = [f for f in all_feature_names if f.startswith(AI_PREFIX)]
handmade_features = [f for f in all_feature_names if any(suffix in f for suffix in HANDMADE_SUFFIXES)]
original_features = [f for f in all_feature_names if f not in ai_features and f not in handmade_features]

# ================= 5. A/B测试与最终审判 (保持不变) =================
print("\n--- 步骤2：开始AI特征价值的“诚实”对决 ---")
final_params = best_params.copy()
final_params.update({'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_jobs': -1})
results = {}
X_control = modern_df[original_features]
results['对照组 (纯原始)'] = run_validation(X_control, y, sample_weight, final_params, "对照组: 纯原始特征")
X_test = modern_df[original_features + ai_features]
results['实验组 (原始+AI)'] = run_validation(X_test, y, sample_weight, final_params, "实验组: 原始+AI特征")
# ... (后续的报告生成部分无需修改) ...
print(f"\n\n{'='*25} AI特征A/B测试最终战报 {'='*25}")
print(f"{'测试组':<25} | {'特征数':<10} | {'平均AUC':<20} | {'AUC标准差':<20}")
print("-" * 85)
sorted_results = sorted(results.items(), key=lambda item: item[1][0] if not np.isnan(item[1][0]) else -np.inf, reverse=True)
feature_count_map = {'对照组 (纯原始)': len(X_control.columns), '实验组 (原始+AI)': len(X_test.columns)}
for name, (auc, std) in sorted_results:
    feature_count = feature_count_map.get(name, 0)
    if not np.isnan(auc): print(f"{name:<25} | {feature_count:<10} | {auc:<20.8f} | {std:<20.8f}")
print("=" * 85)
control_auc = results.get('对照组 (纯原始)', (np.nan,))[0]; test_auc = results.get('实验组 (原始+AI)', (np.nan,))[0]
print("\n--- 最终审判 ---")
if np.isnan(control_auc) or np.isnan(test_auc): print("审判无法进行，至少有一组成绩无效。")
elif test_auc > control_auc:
    improvement = ((test_auc - control_auc) / control_auc) * 100
    print(f"🏆 实验组胜出！"); print(f"   AI特征的加入，使得模型的平均AUC从 {control_auc:.6f} 提升至 {test_auc:.6f}。")
    print(f"   这是一个 {improvement:+.2f}% 的相对性能提升！")
else:
    print(f"⚖️ 对照组胜出或持平。"); print(f"   对照组AUC: {control_auc:.6f} vs 实验组AUC: {test_auc:.6f}")
print("=" * 85)