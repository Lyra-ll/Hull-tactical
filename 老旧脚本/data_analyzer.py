import pandas as pd

# 设置Pandas以显示所有列和行
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print("--- 开始进行全面的数据探索与分析 ---")

# 1. 加载数据
try:
    train_df = pd.read_csv('train.csv')
    print("成功加载 train.csv 文件。\n")
except FileNotFoundError:
    print("错误: 未找到 train.csv 文件。请确保文件在当前目录下。")
    exit()

# --- Section 1: 原始 Data Analyzer 的功能 ---

print("="*50)
print("1. 数据基本信息 (.info())")
print("="*50)
train_df.info()
print("\n")

print("="*50)
print("2. 数值特征统计描述 (.describe())")
print("="*50)
print(train_df.describe())
print("\n")

print("="*50)
print("3. 各列缺失值统计 (.isnull().sum())")
print("="*50)
print(train_df.isnull().sum())
print("\n")


# --- Section 2: 融合“寻找分类特征候选者”的功能 ---

print("="*50)
print("4. 特征唯一值数量分析 (用于寻找分类特征)")
print("="*50)

# 定义特征列
features = [col for col in train_df.columns if col not in ['id', 'target']]

# 计算每个特征的唯一值数量
feature_stats = []
for feature in features:
    nunique = train_df[feature].nunique()
    feature_stats.append({'feature': feature, 'dtype': train_df[feature].dtype, 'nunique': nunique})

# 将结果转为DataFrame并按唯一值数量升序排序
stats_df = pd.DataFrame(feature_stats).sort_values(by='nunique', ascending=True)

print("各特征的数据类型及唯一值（nunique）数量 (从低到高排序):")
print(stats_df)
print("\n提示: 请重点关注那些 'nunique' 值非常低的特征 (例如, 小于 50)。")
print("它们是分类特征的最有力候选者。")
print("="*50)

print("\n--- 全面数据分析完成 ---")
