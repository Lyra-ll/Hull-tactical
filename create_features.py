# create_features.py
# =================================================================
# 终极特征工程脚本 V2.0 (统一时空修复版)
# =================================================================
import pandas as pd
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Time-decay parameters
USE_TIME_DECAY = True
TIME_DECAY_HALF_LIFE = 365  # approximately one year
TIME_DECAY_COLUMN = 'date_id'

# --- 手工特征生成逻辑 (您的“特征兵工厂”) ---
def create_manual_features(df, top_n_base_features):
    #接收一个dataframe和之前决定的最优秀的基础特征，来生产手工特征
    """
    根据给定的基础特征列表，批量生产衍生特征。
    """
    print(f"\n--- [兵工厂] 开始基于 {len(top_n_base_features)} 个核心特征生产手工特征 ---")
    
    df_eng = df.copy()
    #创造副本避免修改原数据
    
    # 核心参数
    horizons = [1, 3, 5]
    #定义了用来生产日期范围差异的参数
    windows = [5, 10]
    #定义了用来生产滚动窗口范围的参数

    for col in top_n_base_features:
        # 1. 动量 (Momentum) 特征
        for h in horizons:
            df_eng[f'{col}_diff{h}'] = df_eng[col].diff(h)
            #计算与过去h天的差值
        # 2. 波动/趋势 (Volatility/Trend) 特征
        for w in windows:
            df_eng[f'{col}_rol_mean_{w}'] = df_eng[col].rolling(window=w).mean()
            #计算w时间窗口里的均值
            df_eng[f'{col}_rol_std_{w}'] = df_eng[col].rolling(window=w).std()
            #计算w时间窗口里的标准差
            #标准差越大，市场情绪越不稳定
    print(f"  > ✅ 手工特征生产完成。")
    return df_eng

def apply_time_decay_factor(df, weight_col):
    """Apply exponential time-decay on sample weights based on a date column."""
    if not USE_TIME_DECAY:
        return df
    if TIME_DECAY_COLUMN not in df.columns:
        print(f"    > Warning: column '{TIME_DECAY_COLUMN}' not found; skip time decay.")
        return df
    if TIME_DECAY_HALF_LIFE <= 0:
        print("    > Warning: TIME_DECAY_HALF_LIFE <= 0; skip time decay.")
        return df

    date_series = df[TIME_DECAY_COLUMN]
    distance = None

    if pd.api.types.is_datetime64_any_dtype(date_series):
        max_point = date_series.max()
        distance = (max_point - date_series).dt.days
    else:
        parsed = pd.to_datetime(date_series, errors='coerce')
        if parsed.notna().any():
            max_point = parsed.max()
            distance = (max_point - parsed).dt.days
        else:
            numeric = pd.to_numeric(date_series, errors='coerce')
            if numeric.notna().any():
                max_point = numeric.max()
                distance = max_point - numeric
            else:
                print("    > Warning: unable to parse time column; skip time decay.")
                return df

    distance = pd.Series(distance).fillna(distance.median()).astype(float)
    decay = np.power(0.5, distance / TIME_DECAY_HALF_LIFE)

    df[weight_col] = df[weight_col] * decay.values
    df[f"{weight_col}_time_decay"] = decay.values
    print(f"    > Applied time decay (half-life={TIME_DECAY_HALF_LIFE} days); recent rows receive higher weight.")
    return df





def create_safe_sample_weights(df, resp_columns):
    """
    [冠军策略修复版] 基于未来多个原始收益列的绝对值之和，创建样本权重。
    resp_columns就是resp_1d, resp_3d等等
    """
    print("\n--- [权重工厂] 开始生成冠军策略的样本权重 ---")
    df_with_weights = df.copy()

    if not all(col in df_with_weights.columns for col in resp_columns):
        #检查是否所有的例如resp_1d等存在于df_with_weights.
        print(f"    > ❌ 错误: 权重生成需要所有 resp 列: {resp_columns}。")
        df_with_weights['sample_weight'] = 1.0
        return df_with_weights

    df_with_weights['sample_weight'] = df_with_weights[resp_columns].abs().sum(axis=1)
    #从df里面找出resp columns那几行的绝对值加起来，赋值给
    # --- [警告修复] ---
    # 旧的写法: df_with_weights['sample_weight'].fillna(0, inplace=True)
    # 新的、更推荐的写法:
    df_with_weights['sample_weight'] = df_with_weights['sample_weight'].fillna(0)
    # --- [修复结束] ---
    df_with_weights = apply_time_decay_factor(df_with_weights, 'sample_weight')

    print("  > ✅ 已生成 'sample_weight' 列 (基于冠军策略)。")
    return df_with_weights


# --- 多目标列生成逻辑 (您的“弹药组装线”) ---
# [V3.0 - 终极正确性修复版]
def create_multi_horizon_targets(df, processing_mode):
    """
    [V5.0 - 终极兼容修复版] + [V6.1 DLS关键修复版]
    为数据集创造多个时间尺度的未来收益目标。
    新增了“动态平滑标签 (DLS)”作为新的监督信号，并修复了action列丢失的BUG。
    """
    print(f"\n--- [组装线 V6.1] 开始添加多尺度目标列 (DLS修复版) ---")
    df_with_targets = df.copy()
    
    TARGET_HORIZONS = [1, 3, 5]

    source_return_col = None
    if processing_mode == 'train':
        if 'forward_returns' in df_with_targets.columns:
            source_return_col = 'forward_returns'
            print(f"  > [训练集模式] 使用 '{source_return_col}' 作为源。")
        else:
            print(f"  > ❌ 致命错误: 训练模式下未找到 'forward_returns' 列。")
            return df
    elif processing_mode == 'test':
        if 'lagged_forward_returns' in df_with_targets.columns:
            source_return_col = 'lagged_forward_returns'
            print(f"  > [测试集模式] 使用 '{source_return_col}' 作为源。")
        else:
            if 'forward_returns' in df_with_targets.columns:
                 source_return_col = 'forward_returns'
                 processing_mode = 'train' 
                 print(f"  > [兼容模式] 在测试流程中发现 '{source_return_col}'，将按训练集逻辑计算。")
            else:
                print(f"  > ❌ 致命错误: 测试模式下未找到任何可用的收益列。")
                return df
            
    for horizon in TARGET_HORIZONS:
        shifted_returns = []

        # [关键修复] 根据收益列类型调整起点：
        #  - forward_returns 已与当前行对齐，应从 i=0 开始累加；
        #  - lagged_forward_returns 落后一天，需要从 i=1 开始推进才能回到当前行的未来收益。
        if source_return_col == 'lagged_forward_returns':
            shift_start = 1
        else:
            shift_start = 0

        for i in range(shift_start, shift_start + horizon):
            shifted_returns.append(df_with_targets[source_return_col].shift(-i))

        multi_day_matrix = pd.concat(shifted_returns, axis=1)

        # 当未来天数不足 horizon 时强制输出 NaN，避免被截断的收益混入标签
        multi_day_return = multi_day_matrix.sum(axis=1, min_count=horizon)

        # 1. 创建原始收益列(resp)
        resp_col_name = f'resp_{horizon}d'
        df_with_targets[resp_col_name] = multi_day_return
        
        # 2. 创建硬标签分类目标 (action)
        action_col_name = f'action_{horizon}d'
        conditions = [
            df_with_targets[resp_col_name] > 0,
            df_with_targets[resp_col_name] <= 0
        ]
        choices = [1, 0]

        # --- [!!! 关键修复 !!!] ---
        # 补上这一行，来真正地创建 action_*d 列，用于最终评估
        df_with_targets[action_col_name] = np.select(conditions, choices, default=np.nan)
        # --- [修复结束] ---
        
        # 3. 创建动态平滑标签 (dls_target)
        ALPHA = 200 
        dls_target_col_name = f'dls_target_{horizon}d'
        df_with_targets[dls_target_col_name] = 1 / (1 + np.exp(-ALPHA * df_with_targets[resp_col_name]))
        
    print(f"  > ✅ 多目标列生成完成 (已包含 action_*d 和 dls_target_*d)。")
    return df_with_targets

# --- 主程序入口 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="终极特征工程脚本 V4.0 (精英模式版)")
    parser.add_argument('--train_input', type=str, required=True, help="输入原始训练CSV (e.g., train.csv)")
    parser.add_argument('--test_input', type=str, required=True, help="输入原始测试CSV (e.g., test.csv)")
    parser.add_argument('--train_output', type=str, required=True, help="输出最终训练特征CSV")
    parser.add_argument('--test_output', type=str, required=True, help="输出最终测试特征CSV")
    # --- 新增模式开关 ---
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'lite'], help="选择模式: 'full' (全部潜力股) 或 'lite' (Top 12精英)")
    args = parser.parse_args()

    start_time = time.time()
    print("="*50 + f"\n终极特征工程流程启动 (模式: {args.mode.upper()})\n" + "="*50)

    # 1. 加载数据
    print("\n1. 正在加载源数据...")
    try:
        train_df_raw = pd.read_csv(args.train_input)
        test_df_raw = pd.read_csv(args.test_input)
        print(f"  > {args.train_input}: {len(train_df_raw)} 行 | {args.test_input}: {len(test_df_raw)} 行")
    except FileNotFoundError as e:
        print(f"❌ 错误: 未找到输入文件 {e}。"); exit()
    
    # 2. 定义特征工程参数
    # --- 【核心修改】根据模式选择不同的“种子”特征 ---
    if args.mode == 'lite':
        print("\n🔥 已启动 [精英模式]: 只为Top 12原始特征生成衍生特征。")
        # !!! 请将下面列表替换为您在第一步中找到的真实Top 12特征 !!!
        BASE_FEATURES_FOR_ENGINEERING = [
            'M4', 'P4', 'P3', 'V3', 'E19', 'P7', 'S2', 'P13', 'M3', 'P12', 'S5', 'S6' 
        ]
    else: # full 模式
        print("\n💣 已启动 [常规模式]: 为所有潜力股生成衍生特征。")
        BASE_FEATURES_FOR_ENGINEERING = [
            'E19', 'P12', 'P4', 'P3', 'P7', 'P13', 'M15', 'M4', 'V3', 'S2', 'S5', 
            'M3', 'S6', 'M9', 'P1', 'I6', 'P6', 'V13', 'P5', 'M12', 'V2', 'I3', 
            'M7', 'I8', 'V5', 'E14', 'E4', 'I4', 'E11', 'I7'
        ]

    MAX_LOOKBACK_WINDOW = 60
    resp_cols = [f'resp_{h}d' for h in [1, 3, 5]]

    target_cols_to_check = resp_cols

    # --- 后续所有流程 (流程A, 流程B) 保持完全不变 ---
    # ... (从 “流程A: 处理训练集” 开始的剩余代码，无需任何改动) ...
    # --- 流程A: 处理训练集 (完全独立) ---
    print("\n" + "-"*20 + " 正在处理【训练集】 " + "-"*20)
    print("2A. 正在为训练集生成手工特征...")
    train_df_eng = create_manual_features(train_df_raw, BASE_FEATURES_FOR_ENGINEERING)
    print("3A. 正在为训练集添加目标列...")
    train_df_targets = create_multi_horizon_targets(train_df_eng, processing_mode='train')
    print("4A. 正在为训练集添加样本权重...")
    train_df_weights = create_safe_sample_weights(train_df_targets, resp_cols)
    print("5A. 正在对训练集进行最终安全处理...")
    original_rows = len(train_df_weights)
    train_df_final = train_df_weights.dropna(subset=target_cols_to_check).copy()
    print(f"  > 已从训练集移除 {original_rows - len(train_df_final)} 行。")

    # --- 流程B: 处理测试集 (使用训练集的“尾巴”作为历史) ---
    print("\n" + "-"*20 + " 正在处理【测试集】 " + "-"*20)
    print(f"2B. 正在准备 {MAX_LOOKBACK_WINDOW} 天的历史数据作为上下文...")
    train_lookback = train_df_raw.tail(MAX_LOOKBACK_WINDOW)
    test_with_lookback = pd.concat([train_lookback, test_df_raw], ignore_index=True)
    print("3B. 正在为测试集生成手工特征 (带历史上下文)...")
    test_df_eng_full = create_manual_features(test_with_lookback, BASE_FEATURES_FOR_ENGINEERING)
    print("4B. 正在从生成结果中剥离出纯净的测试集特征...")
    test_df_final = test_df_eng_full.tail(len(test_df_raw)).copy()
    print("5B. 正在为【测试集】添加目标列（用于评估）...")
    test_df_targets = create_multi_horizon_targets(test_df_final, processing_mode='test')
    print("6B. 正在为【测试集】添加样本权重（用于流程统一）...")
    test_df_final_with_weights = create_safe_sample_weights(test_df_targets, resp_cols)
    print("7B. 正在对测试集进行最终安全处理...")
    original_rows_test = len(test_df_final_with_weights)
    test_df_final_clean = test_df_final_with_weights.dropna(subset=resp_cols).copy()
    print(f"  > 已从测试集移除 {original_rows_test - len(test_df_final_clean)} 行。")

    # 6. 保存最终成果
    print(f"\n6. 正在保存最终成果...")
    train_df_final.to_csv(args.train_output, index=False)
    test_df_final_clean.to_csv(args.test_output, index=False) 
    print(f"  > 训练集 -> '{args.train_output}'")
    print(f"  > 测试集 -> '{args.test_output}'")
    
    print("\n" + "="*50 + "\n✅ 终极特征工程流程胜利完成！\n" + f"总耗时: {time.time() - start_time:.2f} 秒。\n" + "="*50)
