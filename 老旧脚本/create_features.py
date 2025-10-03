import pandas as pd
import numpy as np
import time
import os

def feature_factory_v1(
    input_file: str, 
    output_file: str,
    top_features: dict,
    rolling_windows: list
):
    """
    特征工厂V1.0：基于Top 5特征的洞察，创造第一批合成特征。

    :param input_file: 原始 train.csv 的路径。
    :param output_file: 增强后数据集的保存路径。
    :param top_features: 一个包含需要处理的特征及其类型的字典。
    :param rolling_windows: 用于计算滚动统计量的窗口大小列表。
    """
    
    print("--- 欢迎来到特征工厂 V1.0 ---")
    print("任务: 基于精锐侦察的洞察，组建第一支合成特征部队。")
    start_time = time.time()

    # --- 1. 加载数据 ---
    try:
        print(f"\n[1/4] 正在加载原始数据 '{input_file}'...")
        # 为了计算滞后和滚动特征，我们需要一次性加载数据
        # 如果内存不足，未来可以优化为分块处理，但这会更复杂
        df = pd.read_csv(input_file)
        print("      数据加载成功！")
    except FileNotFoundError:
        print(f"      ❌ 错误: 文件未找到 -> '{input_-file}'")
        return
    
    # --- 2. 开始生成新特征 ---
    print("\n[2/4] 开始大规模生产新特征...")
    
    # 按date_id排序，确保时序计算的准确性
    df = df.sort_values('date_id').reset_index(drop=True)
    
    # 获取所有D*特征用于交互
    d_features = [col for col in df.columns if col.startswith('D')]

    # 循环处理每个王牌特征
    for feature, f_type in top_features.items():
        print(f"  > 正在为特征 '{feature}' (类型: {f_type}) 生产衍生特征...")
        
        # --- 假设 #1 (M4) & 通用: 计算变化率 (差分) ---
        df[f'{feature}_diff1'] = df[feature].diff(1)

        # --- 假设 #2 (E19) & 通用: 计算滚动统计量 ---
        for window in rolling_windows:
            df[f'{feature}_rol_mean_{window}'] = df[feature].rolling(window=window).mean()
            df[f'{feature}_rol_std_{window}'] = df[feature].rolling(window=window).std()
            
        # --- 假设 #3 (P4) & 通用: 与D*特征进行交互 ---
        # 我们选择最重要的D5, D8, D9作为代表进行交互
        for d_feat in ['D5', 'D8', 'D9']:
             df[f'{feature}_x_{d_feat}'] = df[feature] * df[d_feat]

        # --- 假设 #4 (S2) & 通用: 计算滚动窗口内的百分位排名 ---
        # rolling.rank(pct=True) 计算每个值在其滚动窗口内的百分位数
        df[f'{feature}_rol_rank_30'] = df[feature].rolling(window=30).rank(pct=True)

    print("      所有新特征已生产完毕！")
    #不处理缺失值！

    # --- 4. 保存增强后的数据集 ---
    try:
        print(f"\n[4/4] 正在将增强后的数据集保存至 '{output_file}'...")
        df.to_csv(output_file, index=False)
        total_time = time.time() - start_time
        print(f"      ✅ 作战成功！增强数据集已保存。总耗时: {total_time:.2f} 秒。")
    except Exception as e:
        print(f"      ❌ 保存文件时发生错误: {e}")


# ==============================================================================
# 主执行流程
# ==============================================================================
if __name__ == '__main__':
    # --- 配置区域 ---
    
    # 1. 定义输入和输出文件名
    INPUT_CSV_PATH = 'train.csv'
    OUTPUT_CSV_PATH = 'train_v3_featured_raw.csv'
    
    # 2. 定义我们的王牌特征和它们的类型（用于指导生产）
    TOP_FEATURES_TO_PROCESS = {
        'M4': '均值回归型',
        'P4': '开关型',
        'S2': '情绪震荡型',
        'E19': '危机爆发型',
        'P7': '稳健节拍器型'
    }
    
    # 3. 定义滚动窗口的大小
    ROLLING_WINDOWS = [10, 20, 50]

    # --- 执行特征工厂 ---
    feature_factory_v1(
        input_file=INPUT_CSV_PATH,
        output_file=OUTPUT_CSV_PATH,
        top_features=TOP_FEATURES_TO_PROCESS,
        rolling_windows=ROLLING_WINDOWS
    )