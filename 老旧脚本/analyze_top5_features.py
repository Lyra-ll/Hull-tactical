import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# --- 配置 ---
# 设置绘图风格
sns.set_style('whitegrid')
# 修复中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def analyze_top_features(file_path: str, feature_list: list, start_date_id_threshold: int = 1006):
    """
    对给定的特征列表进行深度可视化分析。
    
    对于每个特征，此函数会生成并保存两张图：
    1. 分布分析图 (直方图 + 箱形图)
    2. 时序行为图 (折线图)

    :param file_path: train.csv 文件的路径。
    :param feature_list: 需要分析的特征名称列表。
    :param start_date_id_threshold: 在时序图上标记的“现代史”起点。
    """
    
    print("--- 作战任务: 精锐侦察 (Top 5 特征深度分析) ---")
    start_time = time.time()

    # --- 1. 创建输出文件夹 ---
    output_dir = 'top5_feature_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出文件夹: '{output_dir}'")

    # --- 2. 加载数据 ---
    try:
        print(f"正在加载数据 '{file_path}'...")
        # 考虑到EDA需要全貌，这里一次性加载。如果内存不足，再考虑分块加载绘图。
        df = pd.read_csv(file_path, usecols=['date_id'] + feature_list)
        print("数据加载成功！")
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 -> '{file_path}'")
        return
    except Exception as e:
        print(f"❌ 加载数据时发生错误: {e}")
        return

    # --- 3. 循环分析每个特征 ---
    print("\n--- 开始逐个分析王牌特征 ---")
    for feature in feature_list:
        print(f"  正在分析特征: {feature} ...")
        
        # 移除缺失值以进行准确的可视化
        feature_data = df[['date_id', feature]].dropna()

        # --- a) 分布分析 ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"特征 '{feature}' 分布分析", fontsize=18, y=1.02)
        
        # 直方图 (Histogram)
        sns.histplot(feature_data[feature], kde=True, ax=axes[0])
        axes[0].set_title('数值分布直方图', fontsize=14)
        axes[0].set_xlabel(f'{feature} 的值')
        axes[0].set_ylabel('频数')
        
        # 箱形图 (Box Plot)
        sns.boxplot(x=feature_data[feature], ax=axes[1])
        axes[1].set_title('箱形图 (用于观察异常值)', fontsize=14)
        axes[1].set_xlabel(f'{feature} 的值')
        
        # 保存图像
        dist_save_path = os.path.join(output_dir, f'{feature}_1_distribution.png')
        plt.tight_layout()
        plt.savefig(dist_save_path)
        plt.close()

        # --- b) 时序行为分析 ---
        plt.figure(figsize=(16, 6))
        
        sns.lineplot(data=feature_data, x='date_id', y=feature, linewidth=0.8)
        
        # 在“现代史”起点处画一条垂直线，以观察结构性变化
        plt.axvline(x=start_date_id_threshold, color='red', linestyle='--', linewidth=1.5, label=f'现代史起点 (date_id={start_date_id_threshold})')
        
        plt.title(f"特征 '{feature}' 时序行为分析", fontsize=18)
        plt.xlabel('时间 (date_id)')
        plt.ylabel(f'{feature} 的值')
        plt.legend()
        
        # 保存图像
        ts_save_path = os.path.join(output_dir, f'{feature}_2_timeseries.png')
        plt.tight_layout()
        plt.savefig(ts_save_path)
        plt.close()

        print(f"    ✅ '{feature}' 的分析图表已保存至 '{output_dir}' 文件夹。")

    total_time = time.time() - start_time
    print(f"\n--- 所有王牌特征分析完毕！总耗时: {total_time:.2f} 秒 ---")

# ==============================================================================
# 主执行流程
# ==============================================================================
if __name__ == '__main__':
    # 定义要分析的文件和王牌特征列表
    TRAIN_CSV_PATH = 'train.csv'
    TOP_5_FEATURES = ['M4', 'P4', 'S2', 'E19', 'P7']
    
    analyze_top_features(file_path=TRAIN_CSV_PATH, feature_list=TOP_5_FEATURES)