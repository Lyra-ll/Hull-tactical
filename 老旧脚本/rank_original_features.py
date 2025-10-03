# rank_original_features.py
# =================================================================
# 原始特征“火力侦察”脚本 V1.0
# 目的：在最原始的数据上，为所有原生特征提供一个可靠的重要性排名。
# =================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import argparse
import time

def rank_features(train_file, output_file, start_date_id=1055):
    """
    对原始CSV文件中的原生特征进行稳健的排序。
    """
    print("="*50 + "\n🚀 启动原始特征火力侦察...\n" + "="*50)
    
    # 1. 加载并筛选数据
    print(f"\n1. 正在加载原始数据: '{train_file}'...")
    try:
        df = pd.read_csv(train_file)
        df = df[df['date_id'] >= start_date_id].copy()
        print(f"  > 已筛选 date_id >= {start_date_id} 的数据，剩余 {len(df)} 行。")
    except FileNotFoundError:
        print(f"❌ 错误: 未找到输入文件 '{train_file}'。"); return

    # 2. 创建简单的目标和特征集
    print("\n2. 正在定义目标和特征集...")
    # 定义目标：未来收益是正是负
    y = (df['forward_returns'] > 0).astype(int)
    
    # 定义特征：所有非ID、非未来信息的列
    feature_cols = [c for c in df.columns if c.startswith(('M', 'E', 'I', 'P', 'V', 'S', 'D'))]
    X = df[feature_cols]
    print(f"  > 已定义目标列和 {len(feature_cols)} 个原生特征。")

    # 3. 稳健的交叉验证排序
    print("\n3. 正在通过时间序列交叉验证进行特征排序...")
    tscv = TimeSeriesSplit(n_splits=5)
    all_importances = []
    
    # 初始化LGBM分类器
    model = lgb.LGBMClassifier(random_state=42, verbosity=-1)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"  > 正在处理第 {fold + 1}/5 折...")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        
        # 简单的、无未来信息泄露的填充，仅为让模型能运行
        X_train_filled = X_train.ffill().fillna(X_train.median()).fillna(0)
        
        model.fit(X_train_filled, y_train)
        
        # 记录本折的特征重要性
        fold_importance = pd.Series(model.feature_importances_, index=feature_cols)
        all_importances.append(fold_importance)

    # 4. 计算平均重要性并保存
    print("\n4. 正在计算平均重要性并保存结果...")
    # 将所有折的重要性合并并计算平均值
    avg_importance = pd.concat(all_importances, axis=1).mean(axis=1)
    # 按重要性从高到低排序
    final_ranking = avg_importance.sort_values(ascending=False)
    
    final_ranking.to_csv(output_file, header=False)
    print(f"\n✅ 火力侦察完成！原始特征排名已保存至: '{output_file}'")
    print("\n--- 排名前15的原始特征 ---")
    print(final_ranking.head(15))
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="原始特征重要性排序脚本")
    parser.add_argument('--input', type=str, default='train.csv', help="输入的原始训练CSV文件")
    parser.add_argument('--output', type=str, default='original_feature_ranking.csv', help="输出的特征排名CSV文件")
    args = parser.parse_args()
    
    start_time = time.time()
    rank_features(args.input, args.output)
    print(f"总耗时: {time.time() - start_time:.2f} 秒。")