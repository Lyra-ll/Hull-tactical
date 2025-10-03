import pandas as pd
import time

def feature_factory_final_assault(
    input_file: str, 
    output_file: str,
    core_features: list
):
    """
    特征工厂-最终冲锋版：只为核心特征创造高阶差分特征。
    """
    print("--- 特征工厂 V2.0: 最终地面冲锋 ---")
    print("任务: 创造高阶动量特征，榨干手工工程的最后潜力。")
    start_time = time.time()

    # --- 1. 加载我们最强的“纯净”数据 ---
    try:
        print(f"\n[1/3] 正在加载 '{input_file}'...")
        df = pd.read_csv(input_file)
        print("      数据加载成功！")
    except FileNotFoundError:
        print(f"      ❌ 错误: 文件未找到 -> '{input_file}'")
        return
    
    # --- 2. 为核心特征生成高阶差分 ---
    print("\n[2/3] 正在为核心部队配备“高阶动量”武器...")
    
    for feature in core_features:
        print(f"  > 正在为王牌特征 '{feature}' 生产 diff(2) 和 diff(3)...")
        df[f'{feature}_diff2'] = df[feature].diff(2)
        df[f'{feature}_diff3'] = df[feature].diff(3)

    print("      所有新特征已生产完毕！")
    
    # --- 3. 保存最终增强后的数据集 ---
    # 注意：我们不再进行任何填充！
    try:
        print(f"\n[3/3] 正在将最终增强的数据集保存至 '{output_file}'...")
        df.to_csv(output_file, index=False)
        total_time = time.time() - start_time
        print(f"      ✅ 作战成功！最终数据集已保存。总耗时: {total_time:.2f} 秒。")
    except Exception as e:
        print(f"      ❌ 保存文件时发生错误: {e}")

if __name__ == '__main__':
    # --- 配置区域 ---
    INPUT_CSV_PATH = 'train_v3_featured_raw.csv'
    OUTPUT_CSV_PATH = 'train_v4_featured_final_manual.csv'
    
    # 我们只对已被反复证明的最强特征进行操作
    CORE_FEATURES_FOR_DIFFS = ['M4', 'S2', 'M2', 'P4', 'V5', 'M3', 'S5', 'E19', 'P5', 'I1']
    
    feature_factory_final_assault(
        input_file=INPUT_CSV_PATH,
        output_file=OUTPUT_CSV_PATH,
        core_features=CORE_FEATURES_FOR_DIFFS
    )