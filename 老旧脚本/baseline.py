import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# ==============================================================================
# 新增功能 1: 动态计算分析起点
# ==============================================================================
def find_robust_start_date(df: pd.DataFrame, completeness_threshold: float = 0.75) -> int:
    """
    动态计算并返回第一个特征齐全率超过阈值的date_id。
    """
    print(f"--- 任务 1: 动态计算分析起点 (特征齐全率 > {completeness_threshold * 100}%) ---")
    
    feature_cols = [col for col in df.columns if col.startswith(('D', 'E', 'I', 'M', 'P', 'S', 'V'))]
    df['feature_completeness'] = df[feature_cols].notna().sum(axis=1) / len(feature_cols)
    
    robust_start_date = df.loc[df['feature_completeness'] >= completeness_threshold, 'date_id'].iloc[0]
    
    print(f"计算完成！稳健的分析起点 date_id 为: {robust_start_date}")
    return robust_start_date

# ==============================================================================
# 新增功能 2: 特征相关性分析
# ==============================================================================
def analyze_feature_correlation(df: pd.DataFrame, start_date_id: int, threshold: float = 0.6):
    """
    分析并可视化给定数据段的特征相关性。
    """
    print(f"\n--- 任务 2: 特征相关性分析 (筛选条件: date_id > {start_date_id}) ---")
    
    df_filtered = df[df['date_id'] > start_date_id].copy()
    feature_cols = [col for col in df.columns if col.startswith(('D', 'E', 'I', 'M', 'P', 'S', 'V'))]
    features_df = df_filtered[feature_cols]

    print("正在计算相关性矩阵并生成热力图...")
    corr_matrix = features_df.corr()

    plt.figure(figsize=(20, 18))
    sns.heatmap(corr_matrix, cmap='viridis', annot=False)
    plt.title(f'特征相关性矩阵热力图 (date_id > {start_date_id})', fontsize=16)
    plt.savefig('feature_correlation_heatmap.png')
    print("热力图已保存为 'feature_correlation_heatmap.png'")

    print(f"\n--- 正在识别相关性绝对值 > {threshold} 的特征对 ---")
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_stack = upper_tri[upper_tri.abs() > threshold].stack().reset_index()
    to_stack.columns = ['Feature 1', 'Feature 2', 'Correlation']

    if not to_stack.empty:
        print(f"找到了 {len(to_stack)} 对高相关性特征：")
        to_stack = to_stack.sort_values(by='Correlation', ascending=False, key=abs)
        print(to_stack.to_string(index=False))
    else:
        print(f"未找到相关性绝对值超过 {threshold} 的特征对。")

# ==============================================================================
# 原始功能 3: 基线模型训练 (已升级)
# ==============================================================================
def train_baseline_model(df: pd.DataFrame, start_date_id: int):
    """
    使用动态计算的起点训练LightGBM基线模型。
    """
    print(f"\n--- 任务 3: 基线模型训练 (筛选条件: date_id > {start_date_id}) ---")
    
    # 筛选数据，并使用前向填充处理缺失值
    train_filtered = df[df['date_id'] > start_date_id].copy()
    train_filled = train_filtered.ffill()

    # 定义特征和目标
    features = [col for col in train_filled.columns if col.startswith(('D', 'E', 'I', 'M', 'P', 'S', 'V'))]
    target = 'forward_returns'
    
    X = train_filled[features]
    y = train_filled[target]

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False) # 时序数据不应打乱

    # 训练LightGBM模型
    print("正在训练LightGBM模型...")
    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=True)])

    # 评估模型
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"模型验证集上的RMSE: {rmse}")

    # 绘制特征重要性
    lgb.plot_importance(model, max_num_features=20, figsize=(10, 8), importance_type='gain')
    plt.title('Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("特征重要性图已保存为 'feature_importance.png'")

# ==============================================================================
# 主执行流程
# ==============================================================================
if __name__ == '__main__':
    try:
        # 加载数据
        print("正在加载 train.csv...")
        train_df = pd.read_csv('train_v2_featured.csv')

        # 步骤 1: 动态计算起点
        robust_start_date = find_robust_start_date(train_df)
        
        # 步骤 2: 执行相关性分析
        analyze_feature_correlation(train_df, start_date_id=robust_start_date)

        # 步骤 3: 训练基线模型
        train_baseline_model(train_df, start_date_id=robust_start_date)

    except FileNotFoundError:
        print("错误: train.csv 文件未找到。请确保文件与脚本在同一目录下。")
    except Exception as e:
        print(f"执行过程中发生错误: {e}")