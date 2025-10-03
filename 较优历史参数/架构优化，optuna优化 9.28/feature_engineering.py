# feature_engineering.py
# =================================================================
# 特征挖掘战役 - 自动化特征生产脚本 V2.2 (纯净版)
# =================================================================

import pandas as pd
import time
import warnings

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# --- 1. 配置文件 ---
SOURCE_FILE = 'train.csv'
RANKING_FILE = 'feature_ranking_v11.csv'
OUTPUT_FILE = 'train_v4_engineered.csv'
N_BASE_FEATURES = 30
DIFF_PERIODS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [5, 10, 20, 30, 60]
import config

# --- 质量检测中心 ---
def quality_report(df, title=""):
    """计算并打印DataFrame中特征的缺失值情况分级报告"""
    print("\n" + "="*30)
    print(f" 特征质量分级报告: {title}")
    print("="*30)
    
    missing_ratios = df.isnull().mean()
    total_features = len(missing_ratios)
    
    bins = {
        "0% (完美)": (missing_ratios == 0).sum(),
        "0% - 20% (优秀)": ((missing_ratios > 0) & (missing_ratios <= 0.2)).sum(),
        "20% - 50% (及格)": ((missing_ratios > 0.2) & (missing_ratios <= 0.5)).sum(),
        "50% - 90% (差)": ((missing_ratios > 0.5) & (missing_ratios <= 0.9)).sum(),
        "> 90% (垃圾)": (missing_ratios > 0.9).sum(),
    }
    
    print(f"总特征数量: {total_features}")
    for level, count in bins.items():
        percentage = (count / total_features) * 100 if total_features > 0 else 0
        print(f"  > 缺失率 {level:<15}: {count:<5} 个 ({percentage:.2f}%)")
    print("="*30)

def create_derived_features(df, base_features):
    """在一个给定的DataFrame上，系统性地创造衍生特征"""
    print(f"\n--- 开始为 {len(base_features)} 个基础特征创造衍生特征 ---")
    df_engineered = df.copy()
    
    for i, feature in enumerate(base_features):
        print(f"  > 正在处理第 {i+1}/{len(base_features)} 个特征: {feature} ...")
        
        for period in DIFF_PERIODS:
            new_col_name = f'{feature}_diff{period}'
            df_engineered[new_col_name] = df_engineered[feature].diff(period)
        for window in ROLLING_WINDOWS:
            new_col_name_mean = f'{feature}_rol_mean_{window}'
            df_engineered[new_col_name_mean] = df_engineered[feature].rolling(window).mean()
        for window in ROLLING_WINDOWS:
            new_col_name_std = f'{feature}_rol_std_{window}'
            df_engineered[new_col_name_std] = df_engineered[feature].rolling(window).std()

    return df_engineered

if __name__ == '__main__':
    start_time = time.time()
    print("="*30)
    print("特征挖掘战役启动 (V2.2 纯净生产模式)！")
    print("="*30)

    print(f"1. 正在加载原始数据: '{SOURCE_FILE}'...")
    raw_df = pd.read_csv(SOURCE_FILE)
    
    print(f"2. 正在根据 ANALYSIS_START_DATE_ID ({config.ANALYSIS_START_DATE_ID}) 切分数据...")
    analysis_df = raw_df[raw_df['date_id'] >= config.ANALYSIS_START_DATE_ID].copy()
    
    try:
        print(f"3. 正在加载特征排名文件: '{RANKING_FILE}'...")
        ranking = pd.read_csv(RANKING_FILE, index_col=0).squeeze("columns")
        original_features_in_ranking = [feat for feat in ranking.index if '_diff' not in feat and '_rol_' not in feat]
        base_features_to_use = original_features_in_ranking[:N_BASE_FEATURES]
        print(f"   > 成功识别出 Top {len(base_features_to_use)} 的原始特征作为“原材料”。")
    except FileNotFoundError:
        print(f"   > 错误: 未找到排名文件 '{RANKING_FILE}'。")
        base_features_to_use = []

    if base_features_to_use:
        final_df = create_derived_features(analysis_df, base_features_to_use)
        
        # --- [质检环节] 检查“纯净”特征的质量 ---
        quality_report(final_df, title="最终纯净版")
        
        print(f"\n--- 正在保存最终成果到 '{OUTPUT_FILE}' ---")
        final_df.to_csv(OUTPUT_FILE, index=False)
        
        print("\n" + "="*30)
        print("✅ 特征挖掘战役胜利！")
        print(f"最终总特征数量: {len(final_df.columns)}")
        total_time = time.time() - start_time
        print(f"总耗时: {total_time:.2f} 秒。")
        print("="*30)
    else:
        print("\n未能确定用于工程的基础特征，程序中止。")