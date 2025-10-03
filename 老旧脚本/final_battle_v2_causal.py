# =================================================================
# final_battle_v3_flexible.py (V3.0 - Flexible Causal Tuning)
# 目的: 提供一个灵活的、多模式的自动化调参平台，为不同的核心特征集
#       寻找专属的“黄金参数”。
# =================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
import json
import os

warnings.filterwarnings('ignore')

# ================= 1. 全局配置与模式切换 =================
# --- 核心改造：模式切换开关 ---
# 只需修改这里，即可切换不同的调参目标
# 可选: 'TOP_30_MIXED' (我们之前的最优组合), 'ORIGINAL_ONLY' (我们最新的潜力股)
TUNING_MODE = 'ORIGINAL_ONLY' 

RUN_OPTUNA_TUNING = True 
N_TRIALS = 100 # Optuna 搜索次数

# --- 特征“兵种”识别指纹 ---
HANDMADE_SUFFIXES = ['_rol_', '_diff', '_rank']
AI_PREFIX = 'AE_'

# --- 文件路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
AE_FEATURES_FILE = 'train_v7_causal_ae_features.csv'

# --- 自动化命名与配置 ---
if TUNING_MODE == 'TOP_30_MIXED':
    RANKED_FEATURES_BLUEPRINT = 'ranked_features_v2_causal.csv'
    N_TOP_FEATURES = 30 
    PARAMS_FILE = 'best_params_v2_causal_mixed.json' # 为混合模式指定清晰的文件名
elif TUNING_MODE == 'ORIGINAL_ONLY':
    PARAMS_FILE = 'best_params_v3_original_only.json' # 为原始模式指定清晰的文件名
else:
    raise ValueError(f"未知的 TUNING_MODE: {TUNING_MODE}")

# --- 验证策略配置 (保持不变) ---
N_SPLITS = 5
PURGE_SIZE = 1
EMBARGO_SIZE = 40

# ================= 2. 数据加载与特征准备 =================
print("--- 步骤1：加载数据 ---")
try:
    raw_df = pd.read_csv(RAW_DATA_FILE)
    ae_features_df = pd.read_csv(AE_FEATURES_FILE)
except FileNotFoundError as e:
    print(f"错误: 加载文件失败 - {e}")
    exit()

df = pd.merge(raw_df, ae_features_df, on='date_id', suffixes=('_orig', None))
if 'forward_returns_orig' in df.columns: df.drop(columns=['forward_returns_orig'], inplace=True)
modern_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)

# --- 自动化特征选择 ---
print(f"--- 步骤2：根据模式 '{TUNING_MODE}' 自动选择特征集 ---")
all_feature_names = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
ai_features = [f for f in all_feature_names if f.startswith(AI_PREFIX)]
handmade_features = [f for f in all_feature_names if any(suffix in f for suffix in HANDMADE_SUFFIXES)]
original_features = [f for f in all_feature_names if f not in ai_features and f not in handmade_features]

if TUNING_MODE == 'TOP_30_MIXED':
    print(f"    -> 模式: Top {N_TOP_FEATURES} 混合特征")
    try:
        ranked_features_df = pd.read_csv(RANKED_FEATURES_BLUEPRINT, index_col=0)
    except FileNotFoundError as e:
        print(f"错误: 找不到特征排名文件 '{RANKED_FEATURES_BLUEPRINT}' - {e}")
        exit()
    TARGET_FEATURES = ranked_features_df.head(N_TOP_FEATURES).index.tolist()
elif TUNING_MODE == 'ORIGINAL_ONLY':
    print("    -> 模式: 纯原始特征")
    TARGET_FEATURES = original_features

X = modern_df[TARGET_FEATURES]
y = modern_df['forward_returns'].fillna(0)
print(f"    特征集准备完毕，共 {X.shape[1]} 个特征。")

# ================= 3. 定义“终极武器”版目标函数 (保持不变) =================
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
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    oof_predictions = np.zeros(X.shape[0])
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        purged_train_idx = train_idx[:-PURGE_SIZE]
        X_train_raw, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
        X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        mean = np.nanmean(X_train_raw.values, axis=0)
        std = np.nanstd(X_train_raw.values, axis=0)
        std[std == 0] = 1.0
        X_train = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
        X_val = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_predictions[val_idx] = model.predict(X_val)

    valid_indices = oof_predictions != 0
    rmse = np.sqrt(mean_squared_error(y[valid_indices], oof_predictions[valid_indices]))
    return rmse

# ================= 4. 执行搜索 或 加载参数 (保持不变) =================
# ... (此部分与V2版完全相同)
if RUN_OPTUNA_TUNING:
    print(f"\n--- 步骤3：为 '{TUNING_MODE}' 模式启动因果搜索 ({N_TRIALS}次尝试) ---")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    print(f"\n{'='*25} Optuna因果搜索结束 {'='*25}")
    print(f"✅ 找到了 '{TUNING_MODE}' 模式的最优参数！最佳诚实OOF RMSE: {study.best_value:.8f}")
    with open(PARAMS_FILE, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"   -> 最佳参数已保存至 '{PARAMS_FILE}'")
else:
    print(f"\n--- 步骤3：启动“应用模式”，加载 '{PARAMS_FILE}' 的参数 ---")
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            best_params = json.load(f)
        print(f"   -> 成功从 '{PARAMS_FILE}' 加载参数！")
    else:
        print(f"错误: 未找到参数文件 '{PARAMS_FILE}'。")
        exit()
# =======================================================================
