# =================================================================
# final_battle_tuning.py (V5 - 绝对诚实最终版)
# =================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
import json
import os

warnings.filterwarnings('ignore')

# ================= 1. 全局配置与“双模式”开关 =================
RUN_OPTUNA_TUNING = True 
PARAMS_FILE = 'best_params.json'
N_TOP_FEATURES = 30 
N_TRIALS = 100

# --- 2. 数据加载与准备 ---
print("--- 步骤1：加载并准备数据 ---")
try:
    raw_df = pd.read_csv('train_v3_featured_raw.csv')
    ae_features_df = pd.read_csv('train_v6_kfold_leakfree_ae_features.csv')
except FileNotFoundError as e:
    print(f"错误: 加载文件失败 - {e}")
    exit()

df = pd.merge(raw_df, ae_features_df, on='date_id', suffixes=('_orig', None))
if 'forward_returns_orig' in df.columns:
    df.drop(columns=['forward_returns_orig'], inplace=True)
target = 'forward_returns'
y = df[target].fillna(0)

# --- 3. 特征选择 ---
print("--- 步骤2：动态特征选择 ---")
try:
    ranked_features_df = pd.read_csv('ranked_features.csv', index_col=0)
    DREAM_TEAM_FEATURES = ranked_features_df.head(N_TOP_FEATURES).index.tolist()
    print(f"已从“蓝图”中自动选取Top {N_TOP_FEATURES} 名特征。")
except FileNotFoundError:
    print("错误：未找到特征“蓝图”文件 'ranked_features.csv'。")
    exit()
X = df[DREAM_TEAM_FEATURES]

# --- 4. 定义“目标函数” ---
def objective(trial):
    params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'random_state': 42, 'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 800, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_predictions = np.zeros(X.shape[0])
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_raw, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # ================= 核心修复：应用正确的缩放与填充顺序 =================
        mean = np.nanmean(X_train_raw.values, axis=0)
        std = np.nanstd(X_train_raw.values, axis=0)
        std[std == 0] = 1.0
        X_train_scaled_with_nan = (X_train_raw.values - mean) / std
        X_val_scaled_with_nan = (X_val_raw.values - mean) / std
        X_train = np.nan_to_num(X_train_scaled_with_nan, nan=0.0)
        X_val = np.nan_to_num(X_val_scaled_with_nan, nan=0.0)
        # =================================================================

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_predictions[val_idx] = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    return rmse

# --- 5. 执行搜索 或 加载参数 ---
best_params = {}
if RUN_OPTUNA_TUNING:
    print(f"\n--- 步骤3：启动“搜索模式”，开始Optuna超参数搜索 ({N_TRIALS}次尝试) ---")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    print(f"\n{'='*25} Optuna搜索结束 {'='*25}")
    print(f"✅ 找到了最佳超参数！最佳OOF RMSE分数: {study.best_value:.8f}")
    with open(PARAMS_FILE, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"   -> 最佳参数已保存至 '{PARAMS_FILE}'")
else:
    print(f"\n--- 步骤3：启动“应用模式”，加载已保存的最优参数 ---")
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            best_params = json.load(f)
        print(f"   -> 成功从 '{PARAMS_FILE}' 加载参数！")
    else:
        print(f"错误: 未找到参数文件 '{PARAMS_FILE}'。")
        exit()

# --- 6. 使用最优参数进行最终的K-Fold训练和评估 ---
print("\n--- 步骤4：使用最优参数进行最终的K-Fold训练和评估 ---")
final_params = best_params.copy()
final_params.update({'objective': 'regression_l1', 'metric': 'rmse', 'random_state': 42, 'n_jobs': -1})
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_predictions = np.zeros(X.shape[0])

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- 正在处理第 {fold + 1}/{N_SPLITS} 折 ---")
    X_train_raw, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # ================= 核心修复：在最终评估中也应用正确的顺序 =================
    mean = np.nanmean(X_train_raw.values, axis=0)
    std = np.nanstd(X_train_raw.values, axis=0)
    std[std == 0] = 1.0
    X_train_scaled_with_nan = (X_train_raw.values - mean) / std
    X_val_scaled_with_nan = (X_val_raw.values - mean) / std
    X_train = np.nan_to_num(X_train_scaled_with_nan, nan=0.0)
    X_val = np.nan_to_num(X_val_scaled_with_nan, nan=0.0)
    # ====================================================================
    
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_predictions[val_idx] = model.predict(X_val)

final_oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
print(f"\n{'='*25} 最终决战结束 {'='*25}")
print(f"✅ 使用最优参数，在5折交叉验证中取得了最终的 OOF RMSE: {final_oof_rmse:.8f}")

oof_df = pd.DataFrame({'date_id': df['date_id'], 'target': y, 'oof_prediction': oof_predictions})
oof_filename = 'final_oof_predictions_tuned.csv'
oof_df.to_csv(oof_filename, index=False)
print(f"   OOF预测结果已保存至 '{oof_filename}'")