import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import os

# --- 配置 ---
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')
# 修复中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# --- 创建输出文件夹 ---
if not os.path.exists('eda_reports'):
    os.makedirs('eda_reports')

# --- 复用前序步骤的函数 ---
def load_data(file_path):
    """加载CSV文件。"""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print("✅ 步骤 1: 原始数据加载成功！")
        return df
    except FileNotFoundError:
        print(f"❌ 错误：无法找到文件 '{file_path}'。")
        return None

def preprocess_for_eda(df, start_coverage_threshold=0.80):
    """
    专为EDA设计的预处理：只截取，不填充。
    """
    print(f"\n--- 步骤 2: 开始为EDA进行数据预处理 (覆盖率阈值={start_coverage_threshold*100}%) ---")
    high_info_cols = [col for col in df.columns if not col.startswith('D') and col != 'date_id' and 'returns' not in col and 'rate' not in col]
    start_indices = [df[col].first_valid_index() for col in high_info_cols if df[col].first_valid_index() is not None]
    start_indices.sort()
    threshold_index = int(len(start_indices) * start_coverage_threshold)
    smart_start_index = start_indices[threshold_index]
    
    df_modern = df.loc[smart_start_index:].reset_index(drop=True)
    print(f"      已从索引 {smart_start_index} 开始截取 {len(df_modern)} 行高质量数据。")
    print("      策略：为保证EDA的真实性，不进行任何填充。")
    return df_modern

# --- EDA 核心函数 (在原始数据上运行) ---

def analyze_feature_distributions_to_csv(df, feature_cols):
    """计算特征的描述性统计信息并保存到CSV。"""
    print("\n--- EDA任务1: 分析特征分布 (输出至表格) ---")
    
    # .describe() 会自动忽略NaN值进行计算
    stats_df = df[feature_cols].describe().transpose()
    # .skew() 和 .kurtosis() 同样会自动处理NaN
    stats_df['skew'] = df[feature_cols].skew()
    stats_df['kurtosis'] = df[feature_cols].kurtosis()
    
    save_path = 'eda_reports/feature_distributions_honest.csv'
    stats_df.to_csv(save_path)
    print(f"      ✅ “诚实”的特征分布统计报告已保存至: {save_path}")

def analyze_correlation_to_csv(df, feature_cols, target_col='market_forward_excess_returns'):
    """计算特征与目标的相关性并保存到CSV。"""
    print("\n--- EDA任务2: 分析特征与目标的相关性 (输出至表格) ---")
    
    # .corr() 在计算时默认会忽略掉存在NaN的行对
    correlations = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)
    corr_df = correlations.to_frame(name='correlation').sort_values('correlation', ascending=False)

    save_path = 'eda_reports/feature_target_correlation_honest.csv'
    corr_df.to_csv(save_path)
    print(f"      ✅ “诚实”的特征相关性报告已保存至: {save_path}")

def plot_features_over_time(df, feature_cols, target_col='market_forward_excess_returns', n_samples=4):
    """绘制特征和目标随时间变化的趋势图。"""
    print("\n--- EDA任务3: 观察特征和目标的时序变化 (输出至图片) ---")
    
    sample_features = random.sample(feature_cols, min(n_samples, len(feature_cols)))
    
    fig, axes = plt.subplots(len(sample_features) + 1, 1, figsize=(15, 10), sharex=True)
    
    # cumsum() 在计算前会自动把NaN当作0处理，这里结果影响不大
    axes[0].plot(df['date_id'], df[target_col].cumsum())
    axes[0].set_title('目标变量的累计值 (模拟股市走向)')
    axes[0].grid(True)

    for i, col in enumerate(sample_features, 1):
        # plot() 会在有NaN的地方自动产生断点
        axes[i].plot(df['date_id'], df[col])
        axes[i].set_title(f'特征 "{col}" 随时间的变化')
        axes[i].grid(True)
        
    plt.xlabel('Date ID (时间)')
    plt.tight_layout()
    save_path = 'eda_reports/features_over_time_honest.png'
    plt.savefig(save_path)
    plt.close()
    print(f"      ✅ “诚实”的时序趋势图已保存至: {save_path}")

# --- 执行入口 ---
if __name__ == "__main__":
    TRAIN_FILE_PATH = 'train.csv'
    
    raw_data = load_data(TRAIN_FILE_PATH)
    
    if raw_data is not None:
        # 使用专为EDA设计的、不填充的预处理函数
        processed_data_for_eda = preprocess_for_eda(raw_data.copy(), start_coverage_threshold=0.80)
        
        TARGET_COL = 'market_forward_excess_returns'
        feature_cols = [col for col in processed_data_for_eda.columns if col not in ['date_id', 'forward_returns', 'risk_free_rate', TARGET_COL]]

        # 执行所有“诚实”的EDA任务
        analyze_feature_distributions_to_csv(processed_data_for_eda, feature_cols)
        analyze_correlation_to_csv(processed_data_for_eda, feature_cols, target_col=TARGET_COL)
        plot_features_over_time(processed_data_for_eda, feature_cols, target_col=TARGET_COL)
        
        print("\n🎉 作战地图V6.0 - 行动1.1：“诚实”的战场测绘完成！")
