# =================================================================
# final_battle_v5_fully_upgraded.py (V5.1 - The Ultimate Auto-Selector)
# 目的: 提供一个能同时处理“回归”和“加权分类”两种任务的、
#       端到端的、绝对诚实的自动化调参与评估平台。
#       V5.1 新增：终极调参模式，让模型在全特征集上自我学习与筛选。
# =================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, roc_auc_score
import warnings
import json
import os

warnings.filterwarnings('ignore')

# ================= 1. 全局配置与模式切换 =================
# <--- 核心升级 #1: 设置为我们全新的终极模式 ---
TUNING_MODE = 'WEIGHTED_CLASSIFICATION' 
RUN_OPTUNA_TUNING = True # 我们要执行全新的搜索
N_TRIALS = 150 # 或者更多，比如150-200轮，因为搜索空间更大了

# --- 特征“兵种”识别指纹 ---
HANDMADE_SUFFIXES = ['_rol_', '_diff', '_rank']
AI_PREFIX = 'AE_'

# --- 文件路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
# <--- 核心升级 #2: 确保加载我们最新的、最强的AI特征 ---
AE_FEATURES_FILE = 'train_v11_autotune_clf_ae_features.csv' 

# --- 自动化命名与配置 (已升级) ---
if TUNING_MODE == 'TOP_30_MIXED':
    # ... (旧模式保留)
    RANKED_FEATURES_BLUEPRINT = 'ranked_features_v4_shap.csv'
    N_TOP_FEATURES = 30 
    PARAMS_FILE = 'best_params_v2_causal_mixed.json'
elif TUNING_MODE == 'ORIGINAL_ONLY':
    # ... (旧模式保留)
    PARAMS_FILE = 'best_params_v3_original_only.json'
elif TUNING_MODE == 'WEIGHTED_CLASSIFICATION':
    # ... (旧模式保留)
    RANKED_FEATURES_BLUEPRINT = 'ranked_features_v4_shap.csv' # 使用最新的排行榜
    N_TOP_FEATURES = 50 # 可以尝试Top 50
    PARAMS_FILE = 'best_params_v4_weighted_clf.json'
# <--- 核心升级 #3: 为我们的终极模式添加新的逻辑分支 ---
elif TUNING_MODE == 'CHAMPION_FULL_SET_TUNING':
    # 这个模式不需要特征蓝图，因为它将使用所有特征
    PARAMS_FILE = 'best_params_v5_champion_full_set.json' # 定义新的参数输出文件
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
    print(f"成功加载AI特征文件: {AE_FEATURES_FILE}")
except FileNotFoundError as e:
    print(f"错误: 加载文件失败 - {e}")
    exit()

df = pd.merge(raw_df, ae_features_df, on='date_id', how='left', suffixes=('_orig', None))
if 'forward_returns_orig' in df.columns: df.drop(columns=['forward_returns_orig'], inplace=True)
modern_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)

print(f"--- 步骤2：根据模式 '{TUNING_MODE}' 自动选择特征集 ---")
all_feature_names = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
# ... (兵种识别不变)
ai_features = [f for f in all_feature_names if f.startswith(AI_PREFIX)]
handmade_features = [f for f in all_feature_names if any(suffix in f for suffix in HANDMADE_SUFFIXES)]
original_features = [f for f in all_feature_names if f not in ai_features and f not in handmade_features]

TARGET_FEATURES = [] 
if TUNING_MODE in ['TOP_30_MIXED', 'WEIGHTED_CLASSIFICATION']:
    # ... (旧逻辑不变)
    mode_name = "Top 30 混合回归" if TUNING_MODE == 'TOP_30_MIXED' else f"Top {N_TOP_FEATURES} 加权分类"
    print(f"    -> 模式: {mode_name}")
    try:
        ranked_features_df = pd.read_csv(RANKED_FEATURES_BLUEPRINT, index_col=0)
        TARGET_FEATURES = ranked_features_df.head(N_TOP_FEATURES).index.tolist()
    except FileNotFoundError as e:
        print(f"错误: 找不到特征排名文件 '{RANKED_FEATURES_BLUEPRINT}' - {e}")
        exit()
elif TUNING_MODE == 'ORIGINAL_ONLY':
    print("    -> 模式: 纯原始特征")
    TARGET_FEATURES = original_features
# <--- 核心升级 #4: 为终极模式选择所有特征 ---
elif TUNING_MODE == 'CHAMPION_FULL_SET_TUNING':
    print("    -> 模式: 终极模式 - 全特征集自动学习与筛选")
    TARGET_FEATURES = all_feature_names

X = modern_df[TARGET_FEATURES]

# <--- 核心升级 #5: 确保终极模式使用加权分类的目标 ---
if 'CLASSIFICATION' in TUNING_MODE or TUNING_MODE == 'CHAMPION_FULL_SET_TUNING':
    print("    -> 正在为分类模式准备目标(y)和样本权重(sample_weight)...")
    y = (modern_df['forward_returns'] > 0).astype(int)
    sample_weight = modern_df['forward_returns'].abs()
else: 
    print("    -> 正在为回归模式准备目标(y)...")
    y = modern_df['forward_returns'].fillna(0)
    sample_weight = None
print(f"    特征集和目标集准备完毕，共 {X.shape[1]} 个特征。")


# ================= 3. 定义“终极武器”版目标函数 (已修复) =================
def objective(trial):
    # <--- 核心升级 #6: 确保终极模式使用分类的参数空间，并扩大搜索范围 ---
    is_classification_mode = 'CLASSIFICATION' in TUNING_MODE or TUNING_MODE == 'CHAMPION_FULL_SET_TUNING'
    
    if is_classification_mode:
        params = {
            'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 1000, 5000), # 允许更多树
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True), # 稍低的L-rate
            'num_leaves': trial.suggest_int('num_leaves', 20, 100), # 限制复杂度防止过拟合
            
            # --- 授权模型进行自我特征筛选的关键参数 ---
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 50.0, log=True), # L1正则化，范围扩大
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50.0, log=True),# L2正则化，范围扩大
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9), # 特征采样，不总是用100%
            
            'subsample': trial.suggest_float('subsample', 0.5, 1.0), # 数据采样
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False])
        }
        ModelClass = lgb.LGBMClassifier
        eval_metric_func = roc_auc_score
    else: # 回归模式 (保持不变)
        # ... (回归参数)
        params = {
            'objective': 'regression_l1', 'metric': 'rmse', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 800, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        ModelClass = lgb.LGBMRegressor
        eval_metric_func = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))

    # ... (交叉验证循环部分完全不变)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    oof_predictions = np.zeros(X.shape[0])
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        purged_train_idx = train_idx[:-PURGE_SIZE] if PURGE_SIZE > 0 else train_idx
        X_train_raw, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
        X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        mean = np.nanmean(X_train_raw.values, axis=0); std = np.nanstd(X_train_raw.values, axis=0); std[std == 0] = 1.0
        X_train = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
        X_val = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)

        model = ModelClass(**params)
        
        fit_params = {'eval_set': [(X_val, y_val)], 'callbacks': [lgb.early_stopping(100, verbose=False)]}
        if is_classification_mode and sample_weight is not None:
            fit_params['sample_weight'] = sample_weight.iloc[purged_train_idx]
            fit_params['eval_sample_weight'] = [sample_weight.iloc[val_idx]]

        model.fit(X_train, y_train, **fit_params)
        
        if is_classification_mode:
            oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
        else:
            oof_predictions[val_idx] = model.predict(X_val)

    valid_indices = np.where(oof_predictions != 0)[0]
    score = eval_metric_func(y.iloc[valid_indices], oof_predictions[valid_indices])
    
    return score

# ================= 4. 执行搜索 或 加载参数 =================
# ... (此部分逻辑不变，会自动使用新模式的配置)
best_params = {}
if RUN_OPTUNA_TUNING:
    print(f"\n--- 步骤3：为 '{TUNING_MODE}' 模式启动因果搜索 ({N_TRIALS}次尝试) ---")
    
    direction = 'maximize' if 'CLASSIFICATION' in TUNING_MODE or TUNING_MODE == 'CHAMPION_FULL_SET_TUNING' else 'minimize'
    study = optuna.create_study(direction=direction)
    
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    
    print(f"\n{'='*25} Optuna因果搜索结束 {'='*25}")
    metric_name = "AUC" if direction == 'maximize' else "RMSE"
    print(f"✅ 找到了 '{TUNING_MODE}' 模式的最优参数！最佳诚实OOF {metric_name}: {study.best_value:.8f}")
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

# ================= 5. 使用最优参数进行最终的交叉验证评估 =================
# ... (此部分逻辑不变，会自动使用新模式的配置)
print("\n--- 步骤4：使用最优参数进行最终的因果交叉验证评估 ---")

is_classification_mode_final = 'CLASSIFICATION' in TUNING_MODE or TUNING_MODE == 'CHAMPION_FULL_SET_TUNING'
if is_classification_mode_final:
    # ... (分类逻辑)
    final_params = best_params.copy()
    final_params.update({'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1})
    FinalModelClass = lgb.LGBMClassifier
    final_eval_metric_func = roc_auc_score
else:
    # ... (回归逻辑)
    final_params = best_params.copy()
    final_params.update({'objective': 'regression_l1', 'metric': 'rmse', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1})
    FinalModelClass = lgb.LGBMRegressor
    final_eval_metric_func = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))

# ... (交叉验证循环不变)
tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
oof_predictions = np.zeros(X.shape[0])

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"--- 正在处理第 {fold + 1}/{N_SPLITS} 折 ---")
    purged_train_idx = train_idx[:-PURGE_SIZE] if PURGE_SIZE > 0 else train_idx
    X_train_raw, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
    X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    mean = np.nanmean(X_train_raw.values, axis=0); std = np.nanstd(X_train_raw.values, axis=0); std[std == 0] = 1.0
    X_train = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
    X_val = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)
    
    model = FinalModelClass(**final_params)
    
    fit_params = {'eval_set': [(X_val, y_val)], 'callbacks': [lgb.early_stopping(100, verbose=False)]}
    if is_classification_mode_final and sample_weight is not None:
        fit_params['sample_weight'] = sample_weight.iloc[purged_train_idx]
        fit_params['eval_sample_weight'] = [sample_weight.iloc[val_idx]]
        
    model.fit(X_train, y_train, **fit_params)
    
    if is_classification_mode_final:
        oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
    else:
        oof_predictions[val_idx] = model.predict(X_val)

valid_indices = np.where(oof_predictions != 0)[0]
final_oof_score = final_eval_metric_func(y.iloc[valid_indices], oof_predictions[valid_indices])
metric_name = "AUC" if is_classification_mode_final else "RMSE"

print(f"\n{'='*25} 最终决战结束 {'='*25}")
print(f"✅ 使用最优参数，在{N_SPLITS}折因果交叉验证中取得了最终的 OOF {metric_name}: {final_oof_score:.8f}")

# oof_df = pd.DataFrame({'date_id': modern_df['date_id'][valid_indices], 'target': y[valid_indices], 'oof_prediction': oof_predictions[valid_indices]})
# oof_filename = f'final_oof_predictions_{TUNING_MODE}.csv'
# oof_df.to_csv(oof_filename, index=False)
# print(f"   OOF预测结果已保存至 '{oof_filename}'")