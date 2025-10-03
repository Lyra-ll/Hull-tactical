import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFE
import warnings
import json
import time

# =================================================================
# feature_selector_v1_cv_rfe.py (交叉验证RFE特征筛选器)
# 目的: 使用最严谨的交叉验证版RFE，为所有特征生成一个稳健的排名。
# =================================================================

warnings.filterwarnings('ignore')

# ================= 1. 全局配置 =================
# --- 文件与参数路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
AE_FEATURES_FILE = 'train_v11_autotune_clf_ae_features.csv' 
PARAMS_FILE = 'best_params_ORIGINAL_PLUS_AI.json' 
OUTPUT_RANKING_FILE = 'feature_ranking_cv_rfe.csv' # 输出的排名文件

# --- RFE 配置 ---
# RFE最终要筛选到多少个特征，这个值应该小于等于我们后续想测试的最小规模
# 例如，如果我们想测试Top 30, 50, 80，这里就设为30
N_FEATURES_TO_SELECT_RFE = 50

# --- 其他配置 ---
N_SPLITS = 5; PURGE_SIZE = 1; EMBARGO_SIZE = 40 
ANALYSIS_START_DATE_ID = 1055; MISSING_THRESHOLD = 0.30 
AI_PREFIX = 'AE_'; HANDMADE_SUFFIXES = ['_rol_', '_diff', '_rank', '_x_']

# ================= 2. 预处理函数 =================
def preprocess_data_fine_grained(X_train):
    """
    注意：此版本的预处理只处理训练集，因为它只用于RFE的fit。
    """
    X_train_filled = X_train.ffill(limit=3)
    median_filler = X_train_filled.median()
    X_train_filled.fillna(median_filler, inplace=True); X_train_filled.fillna(0, inplace=True)
    return X_train_filled

# ================= 3. 主执行模块 =================
if __name__ == '__main__':
    print("--- 步骤1：加载数据并准备 ---")
    raw_df = pd.read_csv(RAW_DATA_FILE); ae_features_df = pd.read_csv(AE_FEATURES_FILE)
    with open(PARAMS_FILE, 'r') as f: best_params = json.load(f)
    df = pd.merge(raw_df, ae_features_df, on='date_id', suffixes=('_orig', None))
    modern_df = df[df['date_id'] > ANALYSIS_START_DATE_ID].copy().reset_index(drop=True)
    y = (modern_df['forward_returns'] > 0).astype(int)
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
        X_train_rfe_filled = preprocess_data_fine_grained(X_train_rfe)
        
        selector = RFE(estimator=estimator, n_features_to_select=N_FEATURES_TO_SELECT_RFE, step=0.1)
        selector.fit(X_train_rfe_filled, y_train_rfe)
        all_rankings.append(pd.Series(selector.ranking_, index=X_full.columns))

    # 计算平均排名
    avg_ranking = pd.concat(all_rankings, axis=1).mean(axis=1).sort_values()
    
    # 保存最终排名
    avg_ranking.to_csv(OUTPUT_RANKING_FILE)
    total_time = time.time() - start_time
    print(f"\n✅ RFE交叉验证排名完成！总耗时: {total_time:.2f} 秒。")
    print(f"   -> 最可靠的特征排名已保存至 '{OUTPUT_RANKING_FILE}'")