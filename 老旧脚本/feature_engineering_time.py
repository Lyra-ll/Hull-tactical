# feature_engineering.py
# =================================================================
# 特征挖掘战役 - 自动化特征生产脚本 V3.4 (持有期安全版)
# =================================================================

import pandas as pd
import time
import warnings
import argparse # <--- 引入命令行参数库

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# --- 1. 配置文件 ---
TARGET_HORIZONS = [1, 3, 5] 

def create_multi_horizon_targets(df, source_file_name):
    """为数据集创造多个时间尺度的未来收益目标和原始收益列"""
    print(f"\n--- [核心任务] 开始创造多个时间尺度的目标列 ---")
    df_with_targets = df.copy()
    
    base_return_col = 'forward_returns'
    if base_return_col not in df_with_targets.columns:
        print(f"    > 警告: 在'{source_file_name}'中未找到 '{base_return_col}'。正在从 'train.csv' 中加载...")
        raw_train_df = pd.read_csv('train.csv', usecols=['date_id', base_return_col])
        df_with_targets = pd.merge(df_with_targets, raw_train_df, on='date_id', how='left')

    for horizon in TARGET_HORIZONS:
        print(f"  > -----------------------------------------")
        print(f"  > 正在计算 {horizon} 日未来收益和决策...")
        
        # 计算未来N天的累计收益率
        multi_day_return = df_with_targets[base_return_col].shift(-horizon).rolling(window=horizon).sum()
        
        resp_col_name = f'resp_{horizon}d'
        df_with_targets[resp_col_name] = multi_day_return
        
        action_col_name = f'action_{horizon}d'
        df_with_targets[action_col_name] = (multi_day_return > 0).astype(int)
        
    return df_with_targets

if __name__ == '__main__':
    # <--- [核心升级] 接受命令行输入 ---
    parser = argparse.ArgumentParser(description="特征工程脚本 V3.4")
    parser.add_argument('--input', type=str, required=True, help="输入CSV文件名 (例如: train.csv)")
    parser.add_argument('--output', type=str, required=True, help="输出CSV文件名 (例如: train_engineered.csv)")
    args = parser.parse_args()
    #python feature_engineering_time.py --input train.csv --output train_engineered.csv

    start_time = time.time()
    print("="*30)
    print(f"特征挖掘战役启动 (V3.4) -> 处理文件: {args.input}")
    print("="*30)

    print(f"1. 正在加载源数据: '{args.input}'...")
    try:
        source_df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"❌ 错误: 未找到输入文件 '{args.input}'。")
        exit()
    
    final_df = create_multi_horizon_targets(source_df, args.input)
    
    # <--- [核心修复] 彻底杜绝未来填充泄露 ---
    print("\n--- [安全处理] 正在移除无法计算未来目标的行 ---")
    max_horizon = max(TARGET_HORIZONS)
    original_rows = len(final_df)
    # 通过判断最后一个目标列是否为NaN来找到需要删除的行
    # shift(-h)会在末尾产生h个NaN, rolling(h)会把前面h-1个也变成NaN, 总共是 2*h-1 个NaN
    # 为了保险，我们直接dropna基于所有目标列
    target_cols_to_check = [f'resp_{h}d' for h in TARGET_HORIZONS]
    final_df.dropna(subset=target_cols_to_check, inplace=True)
    
    new_rows = len(final_df)
    print(f"  > 已移除 {original_rows - new_rows} 行数据，确保没有未来信息泄露。")

    # 现在可以安全地对特征列中可能存在的NaN进行填充
    print("\n--- 正在对特征列进行常规填充 ---")
    final_df.ffill(inplace=True)
    final_df.fillna(0, inplace=True)
    
    print(f"\n--- 正在保存最终成果到 '{args.output}' ---")
    final_df.to_csv(args.output, index=False)
    
    print("\n" + "="*30)
    print("✅ 特征工程处理完毕！")
    total_time = time.time() - start_time
    print(f"总耗时: {total_time:.2f} 秒。")
    print("="*30)