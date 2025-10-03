# =================================================================
# feature_quality_dashboard_v2_causal.py (V1.0 - Causal Ranking)
# 目的: 使用带净化和禁运的严格时序交叉验证，获得最真实的特征重要性排名。
# =================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit # <--- 核心升级
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

print("\n--- 开始运行“特征全家桶”质量仪表盘 (V6 - 因果验证版) ---")

# ================= 1. 加载并合并所有特征 (输入端改造) =================
try:
    raw_df = pd.read_csv('train_v3_featured_raw.csv')
    # 读取我们全新的、因果性的AI特征
    ae_features_df = pd.read_csv('train_v11_autotune_clf_ae_features.csv') 
    print("成功加载原始特征和全新的“因果性”AI特征文件。")
except FileNotFoundError as e:
    print(f"错误: 加载文件失败 - {e}")
    exit()

df = pd.merge(raw_df, ae_features_df, on='date_id', suffixes=('_orig', None))
if 'forward_returns_orig' in df.columns:
    df.drop(columns=['forward_returns_orig'], inplace=True)

modern_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)

# ================= 2. 定义特征与目标 =================
feature_prefixes = ['D', 'E', 'I', 'M', 'P', 'S', 'V', 'AE_']
features = [c for c in modern_df.columns if any(c.startswith(prefix) for prefix in feature_prefixes)]
target = 'forward_returns'
X = modern_df[features]
y = modern_df[target].fillna(0)
print(f"特征合并完成！发现 {len(features)} 个总特征。")

# ================= 3. “终极武器”交叉验证评估 (核心改造) =================
N_SPLITS = 5
PURGE_SIZE = 1
EMBARGO_SIZE = 40
feature_importances = pd.DataFrame(index=features)

tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"--- 正在处理第 {fold + 1}/{N_SPLITS} 折 ---")
    
    purged_train_idx = train_idx[:-PURGE_SIZE]

    X_train_raw, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
    #dataframe是默认根据标签寻找，比如a['A1']而不是普通数组的a[0]根据物理地址。
    #只有a.iloc[10]才是根据物理地址寻找。
    X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    mean = np.nanmean(X_train_raw.values, axis=0)
    std = np.nanstd(X_train_raw.values, axis=0)
    std[std == 0] = 1.0
    
    X_train = pd.DataFrame(np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0), columns=features)
    X_val = pd.DataFrame(np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0), columns=features)
    
    model = lgb.LGBMRegressor(importance_type='gain', random_state=42 + fold, n_estimators=1000, learning_rate=0.05, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    # 收集当前折的特征重要性
    feature_importances[f'fold_{fold+1}'] = model.feature_importances_

# ================= 4. 计算并可视化最终“诚实”排名 =================
feature_importances['average'] = feature_importances.mean(axis=1)
top_30_features = feature_importances.sort_values(by='average', ascending=False).head(30)

# ================= 5. 保存结果 (输出端改造) =================
ranked_output_filename = 'ranked_features_v2_causal.csv'
feature_importances.sort_values(by='average', ascending=False).to_csv(ranked_output_filename)
print(f"\n--- “特征全家桶”Top 30重要性“诚实”排名 ---")
print(top_30_features[['average']])

plt.figure(figsize=(12, 10))
sns.barplot(x='average', y=top_30_features.index, data=top_30_features, palette='rocket')
plt.title('特征大比拼-最终Top 30 (因果验证版)', fontsize=18)
plt.xlabel('平均重要性得分 (Gain)', fontsize=14)
plt.ylabel('特征名称', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plot_output_filename = 'all_features_importance_top30_v2_causal.png'
plt.savefig(plot_output_filename)

print(f"\n✅ “诚实”的特征重要性排名已保存至 '{ranked_output_filename}'")
print(f"✅ “诚实”的特征重要性图表已保存至 '{plot_output_filename}'")