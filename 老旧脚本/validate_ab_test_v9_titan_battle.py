import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import warnings
import json

# =================================================================
# validate_v9_reproducibility_test.py (最高分复现版)
# 目的: 严格复现“Top 16 原始/手工 vs Top 16 AI”的巅峰对决，
#       验证我们 0.529 AUC 成绩的可靠性。
# =================================================================

warnings.filterwarnings('ignore')

# ================= 1. 全局配置 =================
# --- 对决规模 ---
N_FEATURES_PER_TEAM = 16

# --- 文件与参数路径 (与取得高分的v9版本完全一致) ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
AE_FEATURES_FILE = 'train_v11_autotune_clf_ae_features.csv' 
PARAMS_FILE = 'best_params_ORIGINAL_PLUS_AI.json' 
RANKING_FILE = 'feature_ranking_cv_rfe.csv'

# --- 其他配置 ---
N_SPLITS = 5; PURGE_SIZE = 1; EMBARGO_SIZE = 40 
ANALYSIS_START_DATE_ID = 1055
AI_PREFIX = 'AE_'

# ================= 2. 核心函数 =================
# --- 注意：这里使用的是“精细填充”函数，因为这是v9取得高分时所用的策略 ---
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
        
        # 确保使用与v9一致的填充策略
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

# ================= 3. 主执行模块 (与v9版本完全一致) =================
if __name__ == '__main__':
    print("--- 步骤1：加载数据并准备对决双方 ---")
    raw_df = pd.read_csv(RAW_DATA_FILE); ae_features_df = pd.read_csv(AE_FEATURES_FILE)
    with open(PARAMS_FILE, 'r') as f: best_params = json.load(f)
    df = pd.merge(raw_df, ae_features_df, on='date_id', suffixes=('_orig', None))
    modern_df = df[df['date_id'] > ANALYSIS_START_DATE_ID].copy().reset_index(drop=True)
    y = (modern_df['forward_returns'] > 0).astype(int)
    sample_weight = modern_df['forward_returns'].abs()
    
    try:
        ranking_df = pd.read_csv(RANKING_FILE, index_col=0, header=None)
        ranking_df.columns = ['avg_rank']
        ranking_df = ranking_df[ranking_df.index.notna()]
        ranking_df = ranking_df[ranking_df.index.map(type) == str]
    except FileNotFoundError:
        print(f"❌ 错误: 找不到排名文件 '{RANKING_FILE}'！")
        exit()

    is_ai_feature = pd.Series(ranking_df.index.str.startswith(AI_PREFIX), index=ranking_df.index).fillna(False)

    non_ai_features = ranking_df[~is_ai_feature]
    top_non_ai = non_ai_features.head(N_FEATURES_PER_TEAM).index.tolist()
    X_control = modern_df[top_non_ai]

    ai_only_features = ranking_df[is_ai_feature]
    top_ai = ai_only_features.head(N_FEATURES_PER_TEAM).index.tolist()
    X_test = modern_df[top_ai]
    print(f"✅ 两支 {N_FEATURES_PER_TEAM} 人的全明星队已组建完毕！")

    print("\n--- 步骤2：启动“巅峰对决”复现测试 ---")
    final_params = best_params.copy()
    final_params.update({'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_jobs': -1})
    results = {}
    results['Top 16 原始/手工特征'] = run_validation(X_control, y, sample_weight, final_params, "A队: Top 16 原始/手工")
    results['Top 16 AI特征'] = run_validation(X_test, y, sample_weight, final_params, "B队: Top 16 AI")

    print(f"\n\n{'='*25} 复现测试最终战报 {'='*25}")
    print(f"{'参赛队伍':<25} | {'平均AUC':<20} | {'AUC标准差':<20}")
    print("-" * 70)
    sorted_results = sorted(results.items(), key=lambda item: item[1][0] if not np.isnan(item[1][0]) else -np.inf, reverse=True)
    for name, (auc, std) in sorted_results:
        if not np.isnan(auc):
            print(f"{name:<25} | {auc:<20.8f} | {std:<20.8f}")
    print("=" * 70)