import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import warnings
import json

# =================================================================
# validate_ab_test_v6_final.py (预处理策略A/B测试平台)
# 目的: 科学地对比“不填充”与“精细填充”两种缺失值处理策略的优劣。
# =================================================================

warnings.filterwarnings('ignore')

# ================= 1. 全局配置 =================
# --- 文件路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
AE_FEATURES_FILE = 'train_v11_autotune_clf_ae_features.csv' 
PARAMS_FILE = 'best_params_ORIGINAL_PLUS_AI.json' 

# --- 验证与特征配置 ---
N_SPLITS = 5; PURGE_SIZE = 1; EMBARGO_SIZE = 40 
ANALYSIS_START_DATE_ID = 1055
MISSING_THRESHOLD = 0.30 
AI_PREFIX = 'AE_'; HANDMADE_SUFFIXES = ['_rol_', '_diff', '_rank', '_x_']

# ================= 2. “精细填充”预处理函数 =================
def preprocess_data_fine_grained(X_train, X_val):
    """
    我们最可靠的“精细填充”预处理函数。
    """
    X_train_filled = X_train.ffill(limit=3)
    median_filler = X_train_filled.median()
    X_train_filled.fillna(median_filler, inplace=True); X_train_filled.fillna(0, inplace=True)
    X_val_filled = X_val.ffill(limit=3)
    X_val_filled.fillna(median_filler, inplace=True); X_val_filled.fillna(0, inplace=True)
    return X_train_filled, X_val_filled

# ================= 3. 核心验证函数 (裁判) - 已完成 =================
def run_validation(X, y, sample_weight, params, group_name, preprocessing_strategy):
    """
    对给定的特征集，使用指定的预处理策略，执行一次完整的交叉验证。
    """
    print(f"\n{'='*20} 开始测试: {group_name} {'='*20}")
    print(f"    预处理策略: '{preprocessing_strategy}'")
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        purged_train_idx = train_idx[:-PURGE_SIZE]
        X_train_raw, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
        X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
        sw_train = sample_weight.iloc[purged_train_idx]
        
        X_train_processed = None
        X_val_processed = None

        # --- 预处理逻辑分支 ---
        if preprocessing_strategy == 'NO_FILLING':
            X_train_processed = X_train_raw
            X_val_processed = X_val_raw
        
        elif preprocessing_strategy == 'FINE_GRAINED_FILLING':
            # 调用我们最可靠的填充函数
            X_train_processed, X_val_processed = preprocess_data_fine_grained(X_train_raw, X_val_raw)

        else:
            raise ValueError(f"未知的预处理策略: {preprocessing_strategy}")
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_processed, y_train, 
                  sample_weight=sw_train,
                  eval_set=[(X_val_processed, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
                  
        preds = model.predict_proba(X_val_processed)[:, 1]
        score = roc_auc_score(y_val, preds)
        fold_scores.append(score)

    mean_score = np.mean(fold_scores); std_score = np.std(fold_scores)
    print(f"    ✅ {group_name} 测试完成。平均AUC: {mean_score:.8f}")
    return mean_score, std_score

# ================= 4. 主执行模块 - 已完成 =================
if __name__ == '__main__':
    print("--- 步骤1：加载数据并准备实验环境 ---")
    raw_df = pd.read_csv(RAW_DATA_FILE)
    ae_features_df = pd.read_csv(AE_FEATURES_FILE)
    with open(PARAMS_FILE, 'r') as f: best_params = json.load(f)
    df = pd.merge(raw_df, ae_features_df, on='date_id', suffixes=('_orig', None))
    modern_df = df[df['date_id'] > ANALYSIS_START_DATE_ID].copy().reset_index(drop=True)
    y = (modern_df['forward_returns'] > 0).astype(int)
    sample_weight = modern_df['forward_returns'].abs()
    
    # --- 特征筛选 ---
    all_feature_names = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
    missing_ratios = modern_df[all_feature_names].isnull().mean()
    features_to_keep = missing_ratios[missing_ratios <= MISSING_THRESHOLD].index.tolist()
    print(f"    -> 已根据 {MISSING_THRESHOLD:.0%} 的门槛，筛选出 {len(features_to_keep)} 个高质量特征用于对决。")
    X_battle = modern_df[features_to_keep]
    
    # --- 开始对决 ---
    print("\n--- 步骤2：启动预处理策略的终极对决 ---")
    final_params = best_params.copy()
    final_params.update({'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_jobs': -1})
    results = {}

    # 对照组: 不进行任何填充
    results['不填充 (No Filling)'] = run_validation(X_battle, y, sample_weight, final_params, "对照组", "NO_FILLING")

    # 实验组: 使用我们最可靠的精细填充策略
    results['精细填充 (Fine-grained)'] = run_validation(X_battle, y, sample_weight, final_params, "实验组", "FINE_GRAINED_FILLING")

    # --- 步骤3：生成最终战报 ---
    print(f"\n\n{'='*25} 预处理策略最终战报 {'='*25}")
    print(f"{'测试策略':<25} | {'平均AUC':<20} | {'AUC标准差':<20}")
    print("-" * 70)
    sorted_results = sorted(results.items(), key=lambda item: item[1][0] if not np.isnan(item[1][0]) else -np.inf, reverse=True)
    for name, (auc, std) in sorted_results:
        if not np.isnan(auc):
            print(f"{name:<25} | {auc:<20.8f} | {std:<20.8f}")
    print("=" * 70)

    # --- 最终审判 ---
    control_auc = results.get('不填充 (No Filling)', (np.nan,))[0]
    test_auc = results.get('精细填充 (Fine-grained)', (np.nan,))[0]
    print("\n--- 最终审判 ---")
    if np.isnan(control_auc) or np.isnan(test_auc):
        print("审判无法进行，至少有一组成绩无效。")
    elif test_auc > control_auc:
        print(f"🏆 ‘精细填充’策略胜出！")
    elif control_auc > test_auc:
        print(f"🏆 ‘不填充’策略胜出！")
    else:
        print(f"⚖️ 两种策略表现持平。")
    print("=" * 70)