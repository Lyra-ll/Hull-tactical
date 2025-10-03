# split_data.py
# =================================================================
# 黄金标准验证策略：创建时间上隔离的开发集与持有集
# =================================================================
import pandas as pd
import argparse
import time

def create_holdout_split(input_file, dev_output_file, holdout_output_file, holdout_size):
    """
    读取一个CSV文件，按时间顺序将其分割为开发集和持有集。
    """
    start_time = time.time()
    print("="*50)
    print("黄金标准验证策略：数据分割脚本启动")
    print("="*50)

    # 1. 加载源数据
    print(f"\n1. 正在加载源数据: '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
        print(f"  > 成功加载 {len(df)} 行数据。")
    except FileNotFoundError:
        print(f"❌ 错误: 未找到输入文件 '{input_file}'。请确保文件名正确。")
        return

    # 2. 确保数据按时间排序（关键步骤）
    print("\n2. 正在根据 'date_id' 对数据进行排序...")
    df.sort_values(by='date_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("  > ✅ 数据已按时间顺序排列。")

    # 3. 执行分割
    if len(df) <= holdout_size:
        print(f"❌ 错误: 数据总行数 ({len(df)}) 小于或等于持有集大小 ({holdout_size})，无法分割。")
        return

    split_point = len(df) - holdout_size
    dev_df = df.iloc[:split_point]
    holdout_df = df.iloc[split_point:]
    
    print(f"\n3. 正在将数据分割为开发集和持有集...")
    print(f"  > 开发集 (前 {len(dev_df)} 行)")
    print(f"  > 持有集 (后 {holdout_size} 行)")

    # 4. 保存文件
    print("\n4. 正在保存文件...")
    try:
        dev_df.to_csv(dev_output_file, index=False)
        print(f"  > ✅ 开发集已保存至: '{dev_output_file}'")
        holdout_df.to_csv(holdout_output_file, index=False)
        print(f"  > ✅ 持有集已保存至: '{holdout_output_file}'")
    except Exception as e:
        print(f"❌ 错误: 文件保存失败: {e}")
        return
        
    print("\n" + "="*50)
    print(f"✅ 数据分割成功完成！")
    print(f"总耗时: {time.time() - start_time:.2f} 秒。")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从源CSV创建时间隔离的开发集和持有集。")
    parser.add_argument('--input', type=str, default='train.csv', help="输入的原始CSV文件。")
    parser.add_argument('--dev_output', type=str, default='train_for_development.csv', help="输出的开发集文件名。")
    parser.add_argument('--holdout_output', type=str, default='holdout_for_testing.csv', help="输出的持有集文件名。")
    parser.add_argument('--holdout_size', type=int, default=180, help="持有集的大小（从末尾截取的行数）。")
    args = parser.parse_args()

    create_holdout_split(args.input, args.dev_output, args.holdout_output, args.holdout_size)