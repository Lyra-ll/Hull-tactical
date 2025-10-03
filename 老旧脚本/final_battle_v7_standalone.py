import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import warnings
import json
import os

# =================================================================
# final_battle_v8_specialist.py (专家模式版)
# 目的: 为不同的特征部队，分别寻找其专属的最优参数，并生成OOF预测文件。
# =================================================================

warnings.filterwarnings('ignore')

# ================= 1. 全局配置与模式切换 =================
# --- 作战部队选择开关 ---
# 每次运行时，请选择一支部队。
# 可选: 'ORIGINAL_ONLY' (纯原始部队) 或 'ORIGINAL_PLUS_AI' (混合部队)
FEATURE_MODE = 'ORIGINAL_PLUS_AI' 

# --- Optuna 配置 ---
# 在此脚本中，我们总是为选定的部队寻找最优参数
RUN_OPTUNA_TUNING = True 
N_TRIALS = 100 

# --- 文件路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
AE_FEATURES_FILE = 'train_v11_autotune_clf_ae_features.csv' 
# 参数和输出文件将根据模式自动命名，确保结果不会混淆
PARAMS_FILE = f'best_params_{FEATURE_MODE}.json' 
OOF_OUTPUT_FILE = f'oof_predictions_{FEATURE_MODE}.csv' 

# --- 特征“兵种”识别指纹 ---
AI_PREFIX = 'AE_'
HANDMADE_SUFFIXES = ['_rol_', '_diff', '_rank', '_x_']

# --- 其他配置 ---
N_SPLITS = 5; PURGE_SIZE = 1; EMBARGO_SIZE = 40; ANALYSIS_START_DATE_ID = 1055 

# ================= 2. 预处理函数 =================
def preprocess_data(X_train, X_val):
    X_train_filled = X_train.ffill(limit=3)
    median_filler = X_train_filled.median()
    X_train_filled.fillna(median_filler, inplace=True); X_train_filled.fillna(0, inplace=True)
    X_val_filled = X_val.ffill(limit=3)
    X_val_filled.fillna(median_filler, inplace=True); X_val_filled.fillna(0, inplace=True)
    return X_train_filled, X_val_filled

# ================= 3. 数据加载与特征准备 =================
print(f"--- 步骤1: 加载数据，当前作战模式: '{FEATURE_MODE}' ---")
try:
    raw_df = pd.read_csv(RAW_DATA_FILE)
    ae_features_df = pd.read_csv(AE_FEATURES_FILE)
except FileNotFoundError as e:
    print(f"❌ 错误: 加载文件失败 - {e}"); exit()

df = pd.merge(raw_df, ae_features_df, on='date_id', how='left')
modern_df = df[df['date_id'] > ANALYSIS_START_DATE_ID].copy().reset_index(drop=True)

all_feature_names = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
ai_features = [f for f in all_feature_names if f.startswith(AI_PREFIX)]
handmade_features = [f for f in all_feature_names if any(suffix in f for suffix in HANDMADE_SUFFIXES)]
original_features = [f for f in all_feature_names if f not in ai_features and f not in handmade_features]

TARGET_FEATURES = []
if FEATURE_MODE == 'ORIGINAL_ONLY':
    TARGET_FEATURES = original_features
elif FEATURE_MODE == 'ORIGINAL_PLUS_AI':
    TARGET_FEATURES = original_features + ai_features
else:
    raise ValueError(f"未知的 FEATURE_MODE: {FEATURE_MODE}")
print(f"✅ 已根据模式选择 {len(TARGET_FEATURES)} 个特征参战。")

X = modern_df[TARGET_FEATURES]
y = (modern_df['forward_returns'] > 0).astype(int)
sample_weight = modern_df['forward_returns'].abs().fillna(0)

# ================= 4. Optuna 目标函数 =================
def objective(trial):
    params = {
        'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 800, 4000),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 50.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 50.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
    }
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    oof_predictions = np.zeros(len(X))
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        purged_train_idx = train_idx[:-PURGE_SIZE]
        X_train, y_train_fold = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
        X_val, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
        sw_train = sample_weight.iloc[purged_train_idx]
        X_train_filled, X_val_filled = preprocess_data(X_train, X_val)
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_filled, y_train_fold, sample_weight=sw_train,
                  eval_set=[(X_val_filled, y_val_fold)], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_predictions[val_idx] = model.predict_proba(X_val_filled)[:, 1]
    valid_indices = np.where(oof_predictions != 0)[0]
    score = roc_auc_score(y.iloc[valid_indices], oof_predictions[valid_indices])
    return score

# ================= 5. 执行主流程 =================
print(f"\n--- 步骤2: 为 '{FEATURE_MODE}' 部队启动Optuna因果搜索 ({N_TRIALS}次尝试) ---")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
best_params = study.best_params
print(f"\n✅ 找到了 '{FEATURE_MODE}' 部队的最优参数！最佳诚实OOF AUC: {study.best_value:.8f}")
with open(PARAMS_FILE, 'w') as f: json.dump(best_params, f, indent=4)
print(f"   -> 最佳参数已保存至 '{PARAMS_FILE}'")

# ================= 6. 最终验证与保存OOF预测 =================
print("\n--- 步骤3: 使用最优参数进行最终验证并生成OOF文件 ---")
final_params = best_params.copy()
tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
oof_predictions = np.zeros(len(X))
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    purged_train_idx = train_idx[:-PURGE_SIZE]
    X_train, y_train_fold = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
    X_val, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
    sw_train = sample_weight.iloc[purged_train_idx]
    X_train_filled, X_val_filled = preprocess_data(X_train, X_val)
    model = lgb.LGBMClassifier(**final_params)
    model.fit(X_train_filled, y_train_fold, sample_weight=sw_train,
              eval_set=[(X_val_filled, y_val_fold)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_predictions[val_idx] = model.predict_proba(X_val_filled)[:, 1]

valid_indices = np.where(oof_predictions != 0)[0]
final_oof_score = roc_auc_score(y.iloc[valid_indices], oof_predictions[valid_indices])
print(f"✅ 最终 OOF AUC: {final_oof_score:.8f}")
oof_df = pd.DataFrame({
    'date_id': modern_df['date_id'].iloc[valid_indices],
    'target': y.iloc[valid_indices],
    'oof_prediction': oof_predictions[valid_indices]
})
oof_df.to_csv(OOF_OUTPUT_FILE, index=False)
print(f"✅ OOF预测结果已为 '{FEATURE_MODE}' 部队保存至 '{OOF_OUTPUT_FILE}'。")