# =================================================================
# validate_ab_test_v4_clf.py (Feature Set A/B Test Platform - Classification)
# 目的: 提供一个标准化的特征集A/B测试平台，专门用于“加权分类”任务，
#       科学地、决定性地评估一组新特征（如AI特征）的真实边际贡献。
# =================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score # <--- 核心升级 #1: 评估指标改为AUC
import warnings
import json
import os

warnings.filterwarnings('ignore')

# ================= 1. 全局配置 =================
# --- 文件路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
# 确保这里使用的是您最新的、经过因果验证生成的AI特征文件
AE_FEATURES_FILE = 'train_v11_autotune_clf_ae_features.csv' 
# <--- 核心升级 #2: 加载为“加权分类”模式找到的专属黄金参数
PARAMS_FILE = 'best_params_v4_weighted_clf.json'

# --- 验证策略 ---
N_SPLITS = 5
PURGE_SIZE = 1
EMBARGO_SIZE = 40 

# --- 特征“兵种”识别指纹 ---
# 我们暂时不评估手工特征，所以这个列表可以为空，但保留框架
HANDMADE_SUFFIXES = ['_rol_', '_diff', '_rank']
AI_PREFIX = 'AE_'

# ================= 2. 核心验证函数 (已为分类任务全面升级) =================
def run_validation(X, y, sample_weight, params, group_name):
    """
    对给定的特征集X和目标y，在“加权分类”模式下，
    执行一次完整的净化禁运时序交叉验证。
    返回平均AUC和AUC的标准差。
    """
    if X.shape[1] == 0:
        print(f"\n{'='*20} 跳过测试: {group_name} (无特征) {'='*20}")
        return np.nan, np.nan
        
    print(f"\n{'='*20} 开始测试: {group_name} {'='*20}")
    print(f"    包含 {X.shape[1]} 个特征。")
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        purged_train_idx = train_idx[:-PURGE_SIZE]
        X_train_raw, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
        X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 提取对应的样本权重
        sw_train = sample_weight.iloc[purged_train_idx]
        sw_val = sample_weight.iloc[val_idx]
        
        mean = np.nanmean(X_train_raw.values, axis=0)
        std = np.nanstd(X_train_raw.values, axis=0)
        std[std == 0] = 1.0
        
        X_train_scaled = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
        X_val_scaled = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)
        
        # <--- 核心升级 #3: 使用LGBMClassifier
        model = lgb.LGBMClassifier(**params)
        
        # <--- 核心升级 #4: 在fit方法中传入样本权重
        model.fit(X_train_scaled, y_train, 
                  sample_weight=sw_train,
                  eval_set=[(X_val_scaled, y_val)],
                  eval_sample_weight=[sw_val],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
                  
        # <--- 核心升级 #5: 预测概率并计算AUC
        preds = model.predict_proba(X_val_scaled)[:, 1]
        score = roc_auc_score(y_val, preds)
        fold_scores.append(score)

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"    ✅ {group_name} 测试完成。平均AUC: {mean_score:.8f}")
    return mean_score, std_score

# ================= 3. 数据加载与特征“兵种”识别 =================
print("--- 步骤1：加载数据并准备分类任务 ---")
raw_df = pd.read_csv(RAW_DATA_FILE)
ae_features_df = pd.read_csv(AE_FEATURES_FILE)
with open(PARAMS_FILE, 'r') as f: best_params = json.load(f)
df = pd.merge(raw_df, ae_features_df, on='date_id', suffixes=('_orig', None))
if 'forward_returns_orig' in df.columns: df.drop(columns=['forward_returns_orig'], inplace=True)
modern_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)

# <--- 核心升级 #6: 准备分类目标y和样本权重sample_weight
y = (modern_df['forward_returns'] > 0).astype(int)
sample_weight = modern_df['forward_returns'].abs()
print("    -> 分类目标(y)和样本权重(sample_weight)准备完毕。")

# 自动化识别三大兵种
all_feature_names = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
ai_features = [f for f in all_feature_names if f.startswith(AI_PREFIX)]
handmade_features = [f for f in all_feature_names if any(suffix in f for suffix in HANDMADE_SUFFIXES)]
original_features = [f for f in all_feature_names if f not in ai_features and f not in handmade_features]

print(f"兵种识别完毕：{len(original_features)}个原始特征, {len(ai_features)}个AI特征, {len(handmade_features)}个手工特征。")

# ================= 4. 定义对照组与实验组，开始A/B测试 =================
print("\n--- 步骤2：开始AI特征价值的终极对决 ---")
final_params = best_params.copy()
# 确保模型配置与分类任务匹配
final_params.update({'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_jobs': -1})

results = {}
# 对照组: 纯原始部队
X_control = modern_df[original_features]
results['对照组 (纯原始)'] = run_validation(X_control, y, sample_weight, final_params, "对照组: 纯原始特征")

# 实验组: 混合部队 (原始 + AI)
X_test = modern_df[original_features + ai_features]
results['实验组 (原始+AI)'] = run_validation(X_test, y, sample_weight, final_params, "实验组: 原始+AI特征")

# ================= 5. 生成最终“战力排行榜”与审判 =================
print(f"\n\n{'='*25} AI特征A/B测试最终战报 {'='*25}")
print(f"{'测试组':<25} | {'特征数':<10} | {'平均AUC':<20} | {'AUC标准差':<20}")
print("-" * 85)

# 对结果按AUC降序排序
sorted_results = sorted(results.items(), key=lambda item: item[1][0] if not np.isnan(item[1][0]) else -np.inf, reverse=True)

feature_count_map = {
    '对照组 (纯原始)': len(X_control.columns),
    '实验组 (原始+AI)': len(X_test.columns),
}
    
for name, (auc, std) in sorted_results:
    feature_count = feature_count_map.get(name, 0)
    
    if not np.isnan(auc):
        print(f"{name:<25} | {feature_count:<10} | {auc:<20.8f} | {std:<20.8f}")
    else:
        print(f"{name:<25} | {feature_count:<10} | {'N/A':<20} | {'N/A':<20}")
print("=" * 85)

# --- 最终审判 ---
control_auc = results.get('对照组 (纯原始)', (np.nan,))[0]
test_auc = results.get('实验组 (原始+AI)', (np.nan,))[0]

print("\n--- 最终审判 ---")
if np.isnan(control_auc) or np.isnan(test_auc):
    print("审判无法进行，至少有一组成绩无效。")
elif test_auc > control_auc:
    improvement = ((test_auc - control_auc) / control_auc) * 100
    print(f"🏆 实验组胜出！")
    print(f"   AI特征的加入，使得模型的平均AUC从 {control_auc:.6f} 提升至 {test_auc:.6f}。")
    print(f"   这是一个 {improvement:+.2f}% 的相对性能提升！")
    print(f"   结论：AI特征在新战场规则下，是有效的“空中支援”！")
else:
    print(f"⚖️ 对照组胜出或持平。")
    print(f"   AI特征的加入未能带来明确的AUC提升。")
    print(f"   对照组AUC: {control_auc:.6f} vs 实验组AUC: {test_auc:.6f}")
    print(f"   结论：AI特征的价值尚未在此次对决中得到证明。")
print("=" * 85)