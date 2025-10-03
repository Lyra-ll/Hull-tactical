import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score # 确保已正确引入
import warnings
import optuna
import shap

# --- 全局设置 ---
OPTUNA_ENABLED = True
N_TRIALS = 50
MISSING_THRESHOLD = 0.30
FEATURE_COVERAGE_THRESHOLD = 0.75
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
AE_FEATURES_FILE = 'train_v11_autotune_clf_ae_features.csv' 
RANKED_OUTPUT_FILENAME = 'ranked_features_v6.1_final.csv'
PLOT_OUTPUT_FILENAME = 'feature_importance_top50_v6.1_final.png'

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

print("\n--- 最高情报机构启动 (V6.1 - 最终修正版) ---")

# =================================================================
# 新增：统一的、无泄露的数据预处理函数
# =================================================================
def preprocess_data(X_train, X_val):
    """
    使用统一的、无数据泄露的逻辑来处理训练集和验证集。
    """
    # 步骤 1: 对训练集进行前向填充
    X_train_filled = X_train.ffill(limit=3)
    
    # 步骤 2: 只从（部分填充后的）训练集中计算填充值
    median_filler = X_train_filled.median()
    
    # 步骤 3: 应用填充，最后的fillna(0)是终极保险
    X_train_filled.fillna(median_filler, inplace=True)
    X_train_filled.fillna(0, inplace=True)
    
    # 步骤 4: 将从训练集学到的填充值应用到验证集
    X_val_filled = X_val.ffill(limit=3)
    X_val_filled.fillna(median_filler, inplace=True)
    X_val_filled.fillna(0, inplace=True)
    
    return X_train_filled, X_val_filled

# --- 阶段1-4 (数据加载、起点探测、质量门禁) ---
print("\n--- 阶段1-4: 执行数据加载、起点探测与质量门禁 ---")
# ... (这部分代码无需改变) ...
try:
    raw_df = pd.read_csv(RAW_DATA_FILE); ae_features_df = pd.read_csv(AE_FEATURES_FILE)
    df = pd.merge(raw_df, ae_features_df, on='date_id', how='left', suffixes=('_orig', None))
    if 'forward_returns_orig' in df.columns: df.drop(columns=['forward_returns_orig'], inplace=True)
except FileNotFoundError as e:
    print(f"❌ 错误: 关键文件缺失 - {e}"); exit()
base_start_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)
all_cols = [c for c in base_start_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
coverage_series = base_start_df[all_cols].notna().mean(axis=1)
first_valid_index = coverage_series[coverage_series >= FEATURE_COVERAGE_THRESHOLD].first_valid_index()
if first_valid_index is None:
    print(f"❌ 错误：在 date_id > 1055 后，没有任何时间点的特征就位率达到 {FEATURE_COVERAGE_THRESHOLD:.0%}！"); exit()
modern_df = base_start_df.iloc[first_valid_index:].copy().reset_index(drop=True)
all_feature_names = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
missing_ratios = modern_df[all_feature_names].isnull().mean()
features_to_keep = missing_ratios[missing_ratios <= MISSING_THRESHOLD].index.tolist()
print(f"✅ 数据准备与质量筛选完毕，剩余 {len(features_to_keep)} 个高质量特征。")
X = modern_df[features_to_keep]; y = (modern_df['forward_returns'] > 0).astype(int)
sample_weight = modern_df['forward_returns'].abs().fillna(0)

best_params = {}
# --- 阶段 5: Optuna 寻找最佳超参数 ---
if OPTUNA_ENABLED:
    print("\n--- 阶段5: 启动 Optuna 寻找最佳超参数 ---")
    tscv_tune = TimeSeriesSplit(n_splits=5, gap=40)
    all_indices = list(tscv_tune.split(X)); train_idx, val_idx = all_indices[-1]
    X_train_tune, y_train_tune = X.iloc[train_idx], y.iloc[train_idx]
    X_val_tune, y_val_tune = X.iloc[val_idx], y.iloc[val_idx]
    sw_train_tune = sample_weight.iloc[train_idx]

    # 使用统一的预处理函数
    X_train_tune_filled, X_val_tune_filled = preprocess_data(X_train_tune, X_val_tune)

    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 2500),
            'num_leaves': trial.suggest_int('num_leaves', 10, 80),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_tune_filled, y_train_tune, sample_weight=sw_train_tune,
                  eval_set=[(X_val_tune_filled, y_val_tune)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict_proba(X_val_tune_filled)[:, 1]
        return roc_auc_score(y_val_tune, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    print(f"✅ Optuna 搜索完成！最佳验证AUC: {study.best_value:.6f}")
else:
    print("\n--- 阶段5: 跳过 Optuna, 使用默认参数 ---"); best_params = {}

# --- 阶段 6: 使用最优参数进行最终的SHAP分析 ---
print("\n--- 阶段6: 使用最优参数进行最终SHAP交叉验证 ---")
final_params = best_params.copy()
final_params.update({'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_jobs': -1})
N_SPLITS = 5; PURGE_SIZE = 1; EMBARGO_SIZE = 40; all_fold_importances = []
tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"--- 正在进行第 {fold + 1}/{N_SPLITS} 折的SHAP深度分析 ---")
    purged_train_idx = train_idx[:-PURGE_SIZE]
    X_train, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    sw_train = sample_weight.iloc[purged_train_idx]
    
    # 再次使用完全相同的预处理函数
    X_train_filled, X_val_filled = preprocess_data(X_train, X_val)

    model = lgb.LGBMClassifier(**final_params)
    model.fit(X_train_filled, y_train, sample_weight=sw_train,
              eval_set=[(X_val_filled, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val_filled)
    fold_importance_values = np.abs(shap_values[1]).mean(axis=0)
    fold_importance_series = pd.Series(fold_importance_values, index=X.columns)
    all_fold_importances.append(fold_importance_series)

# --- 阶段 7: 汇总情报并生成最终报告 ---
print("\n--- 阶段7: 汇总情报并生成最终报告 ---")
# ... (这部分代码无需改变) ...
shap_importances = pd.concat(all_fold_importances, axis=1).fillna(0)
shap_importances.columns = [f'fold_{i+1}' for i in range(N_SPLITS)]
shap_importances['average'] = shap_importances.mean(axis=1)
top_50_features = shap_importances.sort_values(by='average', ascending=False).head(50)
shap_importances.sort_values(by='average', ascending=False).to_csv(RANKED_OUTPUT_FILENAME)
print(f"✅ 终极特征蓝图已保存至 '{RANKED_OUTPUT_FILENAME}'")
print(f"\n--- 神之眼：Top 50 特征SHAP重要性最终排名 (V6.1 Final) ---")
print(top_50_features[['average']])
plt.figure(figsize=(12, 16))
sns.barplot(x='average', y=top_50_features.index, data=top_50_features, palette='plasma')
plt.title('神之眼：Top 50 特征SHAP重要性最终排名 (V6.1 Final)', fontsize=18, fontweight='bold')
plt.xlabel('平均SHAP绝对值', fontsize=14); plt.ylabel('特征名称', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7); plt.tight_layout()
plt.savefig(PLOT_OUTPUT_FILENAME)
print(f"✅ 全新SHAP重要性图表已保存至 '{PLOT_OUTPUT_FILENAME}'")