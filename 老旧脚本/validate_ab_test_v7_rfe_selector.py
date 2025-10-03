import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE
import warnings
import json
import time

# =================================================================
# validate_ab_test_v8_final_showdown.py (终局之战版)
# 目的: 使用交叉验证版的RFE筛选最强特征，并与全特征基准进行最终对决。
# =================================================================

warnings.filterwarnings('ignore')

# ================= 1. 全局配置 =================
# --- 实验配置 ---
FEATURE_COUNTS_TO_TEST = [30, 50, 80] 

# --- 文件与参数路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
AE_FEATURES_FILE = 'train_v11_autotune_clf_ae_features.csv' 
PARAMS_FILE = 'best_params_ORIGINAL_PLUS_AI.json' 

# --- 其他配置 ---
N_SPLITS = 5; PURGE_SIZE = 1; EMBARGO_SIZE = 40 
ANALYSIS_START_DATE_ID = 1055; MISSING_THRESHOLD = 0.30 
AI_PREFIX = 'AE_'; HANDMADE_SUFFIXES = ['_rol_', '_diff', '_rank', '_x_']

# ================= 2. 核心函数 (保持不变) =================
def preprocess_data_fine_grained(X_train, X_val):
    X_train_filled = X_train.ffill(limit=3)
    median_filler = X_train_filled.median()
    X_train_filled.fillna(median_filler, inplace=True); X_train_filled.fillna(0, inplace=True)
    X_val_filled = X_val.ffill(limit=3)
    X_val_filled.fillna(median_filler, inplace=True); X_val_filled.fillna(0, inplace=True)
    return X_train_filled, X_val_filled

def run_validation(X, y, sample_weight, params, group_name):
    print(f"\n--- 开始测试: {group_name} ({X.shape[1]}个特征) ---")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        purged_train_idx = train_idx[:-PURGE_SIZE]
        X_train_raw, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
        X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
        sw_train = sample_weight.iloc[purged_train_idx]
        X_train_filled, X_val_filled = preprocess_data_fine_grained(X_train_raw, X_val_raw)
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_filled, y_train, sample_weight=sw_train,
                  eval_set=[(X_val_filled, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
        preds = model.predict_proba(X_val_filled)[:, 1]
        score = roc_auc_score(y_val, preds)
        fold_scores.append(score)
    mean_score = np.mean(fold_scores); std_score = np.std(fold_scores)
    print(f"    ✅ 测试完成。平均AUC: {mean_score:.8f}")
    return mean_score, std_score

# ================= 3. 主执行模块 (终极升级版) =================
if __name__ == '__main__':
    print("--- 步骤1：加载数据并准备 ---")
    raw_df = pd.read_csv(RAW_DATA_FILE); ae_features_df = pd.read_csv(AE_FEATURES_FILE)
    with open(PARAMS_FILE, 'r') as f: best_params = json.load(f)
    df = pd.merge(raw_df, ae_features_df, on='date_id', suffixes=('_orig', None))
    modern_df = df[df['date_id'] > ANALYSIS_START_DATE_ID].copy().reset_index(drop=True)
    y = (modern_df['forward_returns'] > 0).astype(int)
    sample_weight = modern_df['forward_returns'].abs()
    all_feature_names = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
    missing_ratios = modern_df[all_feature_names].isnull().mean()
    features_to_keep = missing_ratios[missing_ratios <= MISSING_THRESHOLD].index.tolist()
    X_full = modern_df[features_to_keep]

    print(f"--- 步骤2：执行“交叉验证版RFE”筛选特征 ---")
    start_time = time.time()
    
    tscv_for_rfe = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    estimator = lgb.LGBMClassifier(**best_params)
    all_rankings = []
    
    for fold, (train_idx, _) in enumerate(tscv_for_rfe.split(X_full)):
        print(f"  > 正在第 {fold + 1}/{N_SPLITS} 折上运行RFE...")
        X_train_rfe, y_train_rfe = X_full.iloc[train_idx], y.iloc[train_idx]
        X_train_rfe_filled, _ = preprocess_data_fine_grained(X_train_rfe, X_train_rfe.head(1))
        
        # 我们只筛选到最小的目标数量，以节省时间
        selector = RFE(estimator=estimator, n_features_to_select=min(FEATURE_COUNTS_TO_TEST), step=0.1)
        selector.fit(X_train_rfe_filled, y_train_rfe)
        all_rankings.append(pd.Series(selector.ranking_, index=X_full.columns))

    # 计算平均排名
    avg_ranking = pd.concat(all_rankings, axis=1).mean(axis=1).sort_values()
    total_time = time.time() - start_time
    print(f"✅ RFE交叉验证排名完成！总耗时: {total_time:.2f} 秒。")

    print("\n--- 步骤3：开始最终对决 ---")
    results = {}
    
    # 基准组: 全特征大军
    results['全特征大军'] = run_validation(X_full, y, sample_weight, best_params, f"基准组: 全特征大军")
    
    # 实验组: 不同规模的RFE精英部队
    for count in FEATURE_COUNTS_TO_TEST:
        top_n_features = avg_ranking.head(count).index.tolist()
        X_test = modern_df[top_n_features]
        group_name = f"RFE精英部队 (Top {count})"
        results[group_name] = run_validation(X_test, y, sample_weight, best_params, group_name)

    print(f"\n\n{'='*25} 最终决战报告 {'='*25}")
    print(f"{'测试部队':<25} | {'特征数':<10} | {'平均AUC':<20} | {'AUC标准差':<20}")
    print("-" * 80)
    # 动态生成feature_count_map
    feature_count_map = {name: len(modern_df[avg_ranking.head(int(name.split(' ')[-1][:-1])).index.tolist()].columns) if 'Top' in name else len(X_full.columns) for name in results.keys()}
    sorted_results = sorted(results.items(), key=lambda item: item[1][0] if not np.isnan(item[1][0]) else -np.inf, reverse=True)
    for name, (auc, std) in sorted_results:
        feature_count = feature_count_map.get(name, 0)
        if not np.isnan(auc): print(f"{name:<25} | {feature_count:<10} | {auc:<20.8f} | {std:<20.8f}")
    print("=" * 80)
    print(f"\n--- 最终审判 ---")
    print(f"🏆 表现最佳的部队是: {sorted_results[0][0]}")
    print("=" * 80)