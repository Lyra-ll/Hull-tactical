# ================= 引入所需工具 =================
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

# ================= 解决中文乱码问题 =================
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
    print("中文字体设置成功！")
except:
    print("警告：未能找到中文字体'SimHei'。")
# ==========================================================

print("\n--- 开始运行“特征全家桶”质量仪表盘 (V5-终极真相版) ---")

# ================= 加载并合并所有特征 =================
try:
    raw_df = pd.read_csv('train_v3_featured_raw.csv')
    ae_features_df = pd.read_csv('train_v6_kfold_leakfree_ae_features.csv')
    print("成功加载原始特征和AI特征文件。")
except FileNotFoundError as e:
    print(f"错误: 加载文件失败 - {e}")
    exit()

df = pd.merge(raw_df, ae_features_df, on='date_id', suffixes=('_orig', None))
if 'forward_returns_orig' in df.columns:
    df.drop(columns=['forward_returns_orig'], inplace=True)

# ================= 定义特征与目标 =================
feature_prefixes = ['D', 'E', 'I', 'M', 'P', 'S', 'V', 'AE_']
features = [c for c in df.columns if any(c.startswith(prefix) for prefix in feature_prefixes)]
target = 'forward_returns'
X = df[features]
y = df[target].fillna(0)
print(f"特征合并完成！发现 {len(features)} 个总特征。")

# ================= K-Fold交叉验证评估 =================
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
feature_importances = pd.DataFrame(index=features)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- <b>正在处理第 {fold + 1}/{N_SPLITS} 折** ---")
    
    X_train_raw, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    mean = np.nanmean(X_train_raw.values, axis=0)
    std = np.nanstd(X_train_raw.values, axis=0)
    std[std == 0] = 1.0
    
    X_train_scaled_with_nan = (X_train_raw.values - mean) / std
    X_val_scaled_with_nan = (X_val_raw.values - mean) / std
    
    X_train = pd.DataFrame(np.nan_to_num(X_train_scaled_with_nan, nan=0.0), columns=features, index=X_train_raw.index)
    X_val = pd.DataFrame(np.nan_to_num(X_val_scaled_with_nan, nan=0.0), columns=features, index=X_val_raw.index)
    
    # ================= 核心修复：明确指定 importance_type = 'gain' =================
    model = lgb.LGBMRegressor(
        importance_type='gain', # <--- 关键修复！
        random_state=42 + fold, 
        n_estimators=1000, 
        learning_rate=0.05, 
        n_jobs=-1
    )
    # =========================================================================
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    feature_importances[f'fold_{fold+1}'] = model.feature_importances_

# ================= 计算并可视化最终结果 =================
feature_importances['average'] = feature_importances.mean(axis=1)
top_30_features = feature_importances.sort_values(by='average', ascending=False).head(30)

print("\n--- “特征全家桶”Top 30重要性排名 (终极真相版) ---")
print(top_30_features[['average']])

plt.figure(figsize=(12, 10))
sns.barplot(x='average', y=top_30_features.index, data=top_30_features, palette='rocket')
plt.title('特征大比拼-最终Top 30 (终极真相版)', fontsize=18)
plt.xlabel('平均重要性得分 (Gain)', fontsize=14)
plt.ylabel('特征名称', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
output_filename = 'all_features_importance_top30_final_gain.png'
plt.savefig(output_filename)

print(f"\n✅ 特征重要性图表已生成并保存至 '{output_filename}'")