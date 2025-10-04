# main.py
# =================================================================
# 自动化指挥中心 V2.9 (终极统一修复版)
# =================================================================
import argparse
import json
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
import torch
import copy
import importlib 
import sys

# 兼容Windows控制台编码，强制UTF-8以避免UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Optuna相关导入
import optuna.integration
from optuna.integration import LightGBMPruningCallback

# -----------------------------------------------------------------
# 函数定义区
# -----------------------------------------------------------------

class MultiFoldPruningCallback:
    """
    一个能在多折交叉验证中实现“N振出局”逻辑的智能剪枝回调。
    """
    def __init__(self, trial: optuna.trial.Trial, n_strikes: int = 3, metric: str = 'auc', step_base: int = 0):
        self.trial = trial
        self.n_strikes = n_strikes
        self.metric = metric
        self.current_strikes = 0
        self._step = 0
        self.step_base = step_base

    def __call__(self, env: lgb.callback.CallbackEnv) -> None:
        # 每次被调用时，都从 LightGBM 的环境中获取当前的分数
        current_score = env.evaluation_result_list[0][2]
        
        # 使用全局唯一的 step：step_base + 当前迭代数（若不可用则使用内部计数器）
        iteration = getattr(env, 'iteration', None)
        if iteration is None:
            step = self.step_base + self._step
            self._step += 1
        else:
            step = self.step_base + int(iteration)
        
        # 将分数汇报给 Optuna
        self.trial.report(current_score, step)

        # 检查是否应该剪枝
        if self.trial.should_prune():
            # Optuna 认为基于历史记录，这个 trial 已经没有希望了
            self.current_strikes += 1 # 记一次“警告”
            if self.current_strikes >= self.n_strikes:
                # 如果警告次数达到了上限，则抛出异常，真正地“枪毙”这个trial
                message = f"Trial was pruned at step {step} after {self.current_strikes} strikes."
                raise optuna.exceptions.TrialPruned(message)

def run_rfe(X, y, config, utils):
    print(f"\n--- 启动 'RFE' 模式: 正在基于主目标 '{config.PRIMARY_TARGET_COLUMN}' 进行特征排序 ---")
    
    # [核心修复] RFE的评估器也必须从分类器升级为回归器
    estimator = lgb.LGBMRegressor(random_state=42, verbosity=-1)

    tscv = utils.get_cv_splitter(config.N_SPLITS, config.EMBARGO_SIZE, config.PURGE_SIZE)
    all_rankings = []
    y_primary = y[config.PRIMARY_TARGET_COLUMN]

    for fold, (train_idx, _) in enumerate(tscv.split(X)):
        print(f"  > 正在第 {fold + 1}/{config.N_SPLITS} 折上运行RFE...")
        purged_train_idx = train_idx[:-config.PURGE_SIZE] if config.PURGE_SIZE > 0 else train_idx
        
        X_train_rfe = X.iloc[purged_train_idx]
        y_train_rfe = y_primary.iloc[purged_train_idx]
        
        preprocessor_params = utils.get_preprocessor_params(X_train_rfe)
        X_train_rfe_filled, _ = utils.apply_preprocessor(X_train_rfe, preprocessor_params)
        
        selector = RFE(estimator=estimator, n_features_to_select=config.N_FEATURES_TO_SELECT_RFE, step=0.1)
        selector.fit(X_train_rfe_filled, y_train_rfe)
        
        all_rankings.append(pd.Series(selector.ranking_, index=X.columns))

    avg_ranking = pd.concat(all_rankings, axis=1).mean(axis=1).sort_values()
    avg_ranking.to_csv(config.RANKING_FILE, header=False)
    print(f"\n✅ RFE排名完成！排名已保存至 '{config.RANKING_FILE}'")

# [核心修复] 重构折内逻辑，使用新的统一预处理器
def _run_fold_logic(X_train, y_train, sw_train, X_val, y_val, sw_val, ae_params, lgbm_params, config, utils):
    device = torch.device("cpu")  # 强制使用CPU，避免CUDA设备不一致

    # 1. 从训练集学习预处理参数
    preprocessor_params = utils.get_preprocessor_params(X_train)

    # 2. 应用参数到训练集和验证集
    X_train_proc, X_train_scaled = utils.apply_preprocessor(X_train, preprocessor_params)
    X_val_proc, X_val_scaled = utils.apply_preprocessor(X_val, preprocessor_params)

    # --- [!!! 关键修复 !!!] ---
    # 调用 train_fold_ae 时，y_train 和 y_val 必须明确只选择用于监督AE的软目标列。
    # y_train 和 y_val 在这里是包含了6个目标列的DataFrame。
    # 我们通过 .values 将其转换为Numpy数组，以匹配函数签名。
    ae_models = utils.train_fold_ae(
        ae_params,
        X_train_scaled.values,
        y_train[config.TARGET_COLUMNS].values, # 之前是 y_train.values (错误)
        sw_train.values,
        X_val_scaled.values,
        y_val[config.TARGET_COLUMNS].values,   # 之前是 y_val.values (错误)
        sw_val.values,
        X_train.isnull().values,
        X_val.isnull().values,
        seeds=[42, 2024]
    )
    # --- [修复结束] ---
    
    if not ae_models: 
        return np.full((len(X_val), len(config.TARGET_COLUMNS)), 0.5)
        
    # 4. 生成AI特征
    with torch.no_grad():
        train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32).to(device)
        val_tensor = torch.tensor(X_val_scaled.values, dtype=torch.float32).to(device)
        X_train_ai = np.mean([m.encoder(train_tensor).cpu().numpy() for m in ae_models], axis=0)
        X_val_ai = np.mean([m.encoder(val_tensor).cpu().numpy() for m in ae_models], axis=0)
        
    ai_cols = [f'{config.AI_PREFIX}{i}' for i in range(X_train_ai.shape[1])]
    X_train_ai_df = pd.DataFrame(X_train_ai, columns=ai_cols, index=X_train_proc.index)
    X_val_ai_df = pd.DataFrame(X_val_ai, columns=ai_cols, index=X_val_proc.index)

    X_train_final = pd.concat([X_train_scaled, X_train_ai_df], axis=1)
    X_val_final = pd.concat([X_val_scaled, X_val_ai_df], axis=1)

    # 5. 训练并预测LGBM
    fold_preds = []
    # LGBM的训练目标依然是软标签，这部分是正确的
    for target_col in config.TARGET_COLUMNS:
        model = lgb.LGBMRegressor(**lgbm_params).fit(X_train_final, y_train[target_col], sample_weight=sw_train)
        fold_preds.append(model.predict(X_val_final))
        
    return np.vstack(fold_preds).T


# [V3.1 终极性能优化修复版]
def run_tuning(X, y, y_action, sample_weight, tune_target, config, utils):
    print(f"\n--- 启动 '{tune_target.upper()}' 调优模式 (V3.1 终极性能优化修复版) ---")
    tscv_fortuning = utils.get_cv_splitter(config.N_SPLITS, config.EMBARGO_SIZE, config.PURGE_SIZE)
    folds_to_use = list(tscv_fortuning.split(X))
    print(f"    > [稳健性] Optuna将使用 {len(folds_to_use)} 折CV数据进行评估...")
    
    y_all_targets = pd.concat([y, y_action], axis=1)
    # 冒烟配置下TARGET与ACTION同名，拼接会出现重复列，这里去重以避免n_targets错误翻倍
    y_all_targets = y_all_targets.loc[:, ~y_all_targets.columns.duplicated()]

    def objective(trial):
        if tune_target == 'lgbm':
            try:
                with open(config.AE_PARAMS_FILE, 'r') as f: ae_params = json.load(f)
            except FileNotFoundError:
                ae_params = {'hidden_dim': 128, 'encoding_dim': 32, 'n_hidden_layers': 2, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'recon_weight': 0.5, 'bn':True}
            
            lgbm_params = { 
                'random_state': 42, 'n_jobs': -1, 'verbosity': -1, 
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True), 
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000), 
                'num_leaves': trial.suggest_int('num_leaves', 10, 80), 
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 50.0, log=True), 
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 50.0, log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            }
        elif tune_target == 'ae':
            with open(config.LGBM_PARAMS_FILE, 'r') as f: lgbm_params = json.load(f)
            lgbm_params.update({'random_state': 42, 'n_jobs': -1, 'verbosity': -1})
            ae_params = { 'hidden_dim': trial.suggest_int('hidden_dim', 64, 256, step=32), 'encoding_dim': trial.suggest_int('encoding_dim', 16, 64, step=8), 'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 3), 'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5), 'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True), 'recon_weight': trial.suggest_float('recon_weight', 0.2, 0.8), 'bn': True }
        
        oof_predictions_trial = np.zeros((len(X), len(config.TARGET_COLUMNS)))
        oof_valid_mask = np.zeros(len(X), dtype=bool)

        for fold, (train_idx, val_idx) in enumerate(folds_to_use):
            # 为每一折提供一个不同的 step_base，避免不同折之间的 step 冲突
            # 这里按每个目标最多迭代数近似估计：使用n_estimators作为每折的步长基数
            step_base = fold * trial.params.get('n_estimators', 1000)
            pruning_callback_fold = MultiFoldPruningCallback(trial, n_strikes=3, metric='auc', step_base=step_base)
            purged_train_idx = train_idx[:-config.PURGE_SIZE] if config.PURGE_SIZE > 0 else train_idx
            
            X_train, sw_train = X.iloc[purged_train_idx], sample_weight.iloc[purged_train_idx]
            y_train_all = y_all_targets.iloc[purged_train_idx]
            X_val, y_val_all, sw_val = X.iloc[val_idx], y_all_targets.iloc[val_idx], sample_weight.iloc[val_idx]

            missing_ratios_fold = X_train.isnull().mean()
            features_to_keep_fold = missing_ratios_fold[missing_ratios_fold < config.MISSING_THRESHOLD].index.tolist()
            X_train_fold_filtered = X_train[features_to_keep_fold]
            X_val_fold_filtered = X_val[features_to_keep_fold]

            if tune_target == 'lgbm':
                # --- [!!! 终极逻辑修复 !!!] ---
                # 1. 预处理
                preprocessor_params = utils.get_preprocessor_params(X_train_fold_filtered)
                _, X_train_scaled = utils.apply_preprocessor(X_train_fold_filtered, preprocessor_params)
                _, X_val_scaled = utils.apply_preprocessor(X_val_fold_filtered, preprocessor_params)
                
                # 2. 【关键】在LGBM调优的每一折中，也必须正确地、完整地训练AE模型
                ae_models = utils.train_fold_ae(
                    ae_params, X_train_scaled.values, y_train_all[config.TARGET_COLUMNS].values, sw_train.values,
                    X_val_scaled.values, y_val_all[config.TARGET_COLUMNS].values, sw_val.values,
                    X_train_fold_filtered.isnull().values, X_val_fold_filtered.isnull().values
                )
                if not ae_models: continue

                # 3. 从【训练好】的AE中生成AI特征
                device = torch.device("cpu")  # 强制使用CPU，避免CUDA设备不一致
                with torch.no_grad():
                    train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32).to(device)
                    val_tensor = torch.tensor(X_val_scaled.values, dtype=torch.float32).to(device)
                    X_train_ai = np.mean([m.encoder(train_tensor).cpu().numpy() for m in ae_models], axis=0)
                    X_val_ai = np.mean([m.encoder(val_tensor).cpu().numpy() for m in ae_models], axis=0)
                
                ai_cols = [f'{config.AI_PREFIX}{i}' for i in range(X_train_ai.shape[1])]
                X_train_ai_df = pd.DataFrame(X_train_ai, columns=ai_cols, index=X_train_scaled.index)
                X_val_ai_df = pd.DataFrame(X_val_ai, columns=ai_cols, index=X_val_scaled.index)
                X_train_final = pd.concat([X_train_scaled, X_train_ai_df], axis=1)
                X_val_final = pd.concat([X_val_scaled, X_val_ai_df], axis=1)
                
                pruning_callback = LightGBMPruningCallback(trial, "auc")

                # 4. 使用高质量的特征，训练LGBM并应用“智能剪枝”
                fold_preds = []
                for i, target_col in enumerate(config.TARGET_COLUMNS):
                    action_col = config.ACTION_COLUMNS[i]
                    
                    # 2. 我们只对最重要的那个模型 (i==0) 应用我们新的剪枝策略
                    active_callbacks = [pruning_callback_fold] if i == 0 else []
                    
                    model = lgb.LGBMRegressor(**lgbm_params)
                    # 可选：若配置允许且环境支持GPU版LightGBM，可添加如下参数：
                    # if getattr(config, 'LGBM_USE_GPU', False):
                    #     model.set_params(device_type='gpu')
                    model.fit(
                        X_train_final, y_train_all[target_col], sample_weight=sw_train,
                        eval_set=[(X_val_final, y_val_all[action_col])],
                        eval_sample_weight=[sw_val],
                        eval_metric="auc",
                        callbacks=active_callbacks
                    )
                    fold_preds.append(model.predict(X_val_final))
                
                oof_predictions_trial[val_idx] = np.vstack(fold_preds).T
                # --- [修复结束] ---
            else: # tune_target == 'ae', 逻辑保持不变，调用_run_fold_logic
                fold_predictions = _run_fold_logic(
                    X_train_fold_filtered, y_train_all, sw_train, 
                    X_val_fold_filtered, y_val_all, sw_val, 
                    ae_params, lgbm_params, config, utils
                )
                oof_predictions_trial[val_idx] = fold_predictions
            
            oof_valid_mask[val_idx] = True

        # 评估逻辑保持不变
        y_true_for_scoring = y_action.iloc[oof_valid_mask].values
        y_pred = oof_predictions_trial[oof_valid_mask]
        weight_for_scoring = np.nan_to_num(sample_weight.iloc[oof_valid_mask].values, nan=0.0)
        valid_weight_mask = weight_for_scoring > 0

        if y_true_for_scoring.shape[0] == 0 or not valid_weight_mask.any(): return 0.5
        y_true_filtered = y_true_for_scoring[valid_weight_mask]
        y_pred_filtered = y_pred[valid_weight_mask]

        scores = []
        for i in range(y_true_filtered.shape[1]):
            y_col = y_true_filtered[:, i]
            if np.unique(y_col).size < 2:
                scores.append(0.5)
            else:
                scores.append(roc_auc_score(y_col, y_pred_filtered[:, i]))

        return float(np.mean(scores)) if scores else 0.5
    
    # ... (函数剩余的study.optimize部分无需修改) ...
    if tune_target == 'lgbm':
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=config.N_TRIALS_LGBM)
        params_file = config.LGBM_PARAMS_FILE
    elif tune_target == 'ae':
        try:
            with open(config.LGBM_PARAMS_FILE, 'r') as f: pass
        except FileNotFoundError:
            print("❌ 错误: 必须先运行 'tune_lgbm' 模式。"); return
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=config.N_TRIALS_AE)
        params_file = config.AE_PARAMS_FILE
        
    print(f"\n{'='*25} Optuna搜索结束 {'='*25}")
    print(f"✅ 最佳 {tune_target.upper()} 参数 (AUC: {study.best_value:.8f}):")
    print(json.dumps(study.best_params, indent=4))
    with open(params_file, 'w') as f: json.dump(study.best_params, f, indent=4)
    print(f"-> 已保存至 '{params_file}'")

def run_validation(X, y, sample_weight, date_ids, config, utils):
    print("\n--- 启动 'VALIDATE' 模式 (V2.9.2 DLS最终修复版) ---")
    try:
        with open(config.LGBM_PARAMS_FILE, 'r') as f: lgbm_params = json.load(f)
        
        # --- [!!! 关键修复 !!!] ---
        # 移除了所有分类任务专属的参数('objective': 'binary', 'metric': 'auc')
        # LGBMRegressor会自动使用默认的回归目标(如'l2')
        lgbm_params.update({'random_state': 42, 'n_jobs': -1, 'verbosity': -1})
        # --- [修复结束] ---

        with open(config.AE_PARAMS_FILE, 'r') as f: ae_params = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ 错误: 缺少参数文件: {e}。请先运行所有tune模式。"); return

    print("✅ 参数加载成功。开始交叉验证...")
    tscv = utils.get_cv_splitter(config.N_SPLITS, config.EMBARGO_SIZE, config.PURGE_SIZE)
    oof_predictions = np.zeros((len(X), len(config.TARGET_COLUMNS)))
    oof_valid_mask = np.zeros(len(X), dtype=bool)
    fold_val_indices = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n{'='*20} 正在处理第 {fold + 1}/{config.N_SPLITS} 折 {'='*20}")
        purged_train_idx = train_idx[:-config.PURGE_SIZE] if config.PURGE_SIZE > 0 else train_idx
        X_train, y_train, sw_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx], sample_weight.iloc[purged_train_idx]
        X_val, y_val, sw_val = X.iloc[val_idx], y.iloc[val_idx], sample_weight.iloc[val_idx]

        print(f"  > [Fold {fold + 1}] 正在基于当前训练集进行动态缺失值筛选 (阈值: < {config.MISSING_THRESHOLD})...")
        missing_ratios_fold = X_train.isnull().mean()
        features_to_keep_fold = missing_ratios_fold[missing_ratios_fold < config.MISSING_THRESHOLD].index.tolist()
        
        print(f"    - 筛选前特征数: {X_train.shape[1]}")
        print(f"    - 筛选后特征数: {len(features_to_keep_fold)}")
        if X_train.shape[1] != len(features_to_keep_fold):
             print(f"    - 已剔除 {X_train.shape[1] - len(features_to_keep_fold)} 个特征。")

        X_train_fold_filtered = X_train[features_to_keep_fold]
        X_val_fold_filtered = X_val[features_to_keep_fold]

        fold_predictions = _run_fold_logic(
            X_train_fold_filtered, y_train, sw_train, 
            X_val_fold_filtered, y_val, sw_val, 
            ae_params, lgbm_params, config, utils
        )

        oof_predictions[val_idx] = fold_predictions
        oof_valid_mask[val_idx] = True
        fold_val_indices.append(val_idx)

    print(f"\n\n{'='*25} 最终交叉验证结束 {'='*25}")
    print(f"  > [冠军策略] 最终评估将只使用最后 {config.N_LAST_FOLDS_TO_USE_INFERENCE} 折的预测结果。")

    inference_indices = np.concatenate(fold_val_indices[-config.N_LAST_FOLDS_TO_USE_INFERENCE:])

    y_true_final = y[config.ACTION_COLUMNS].iloc[inference_indices]
    preds_final_df = pd.DataFrame(oof_predictions[inference_indices], index=y_true_final.index, columns=[f'pred_{c}' for c in config.TARGET_COLUMNS])
    weights_final = np.nan_to_num(sample_weight.iloc[inference_indices].values, nan=0.0)
    valid_weight_mask = weights_final > 0
    if not valid_weight_mask.any():
        print("  > Warning: no positive-weight samples available for OOF scoring; using all rows.")
        y_true_eval = y_true_final
        preds_eval_df = preds_final_df
    else:
        y_true_eval = y_true_final.iloc[valid_weight_mask]
        preds_eval_df = preds_final_df.iloc[valid_weight_mask]

    scores = {}
    for i, action_col in enumerate(config.ACTION_COLUMNS):
        pred_col = f'pred_{config.TARGET_COLUMNS[i]}'
        y_true_col = y_true_eval[action_col]
        y_pred_col = preds_eval_df[pred_col]
        if y_true_col.nunique(dropna=True) < 2:
            scores[action_col] = 0.5
        else:
            scores[action_col] = roc_auc_score(y_true_col, y_pred_col)

    for col, score in scores.items(): print(f"🎯 OOF AUC for '{col}': {score:.8f}")
    print("-" * 60 + f"\n🏆 最终OOF平均AUC: {np.mean(list(scores.values())):.8f}\n" + "-" * 60)

    oof_df_to_save = pd.concat([date_ids.iloc[oof_valid_mask].reset_index(drop=True), y.iloc[oof_valid_mask].reset_index(drop=True), pd.DataFrame(oof_predictions[oof_valid_mask], columns=[f'pred_{c}' for c in config.TARGET_COLUMNS]).reset_index(drop=True)], axis=1)
    oof_df_to_save.to_csv(config.OOF_OUTPUT_FILE, index=False)
    print(f"✅ 完整的OOF预测已保存至 '{config.OOF_OUTPUT_FILE}'。")

# [核心修复] 彻底重构Holdout模式，确保数据处理流程100%统一
# [V3 - 诊断优先版] 重构Holdout模式，实现最终的内部验证与外部持有对比
# [V2.9 DLS修复版]
def run_holdout_validation(X_train_full, y_train_full, sw_train_full, X_holdout, y_holdout, sw_holdout, config, utils):
    device = torch.device("cpu")  # 强制使用CPU，避免CUDA设备不一致
    print("\n" + "="*20 + " 启动 'HOLDOUT' 最终审判模式 (V2.9 DLS修复版) " + "="*20)
    try:
        with open(config.LGBM_PARAMS_FILE, 'r') as f: lgbm_params = json.load(f)
        lgbm_params.update({'random_state': 42, 'n_jobs': -1, 'verbosity': -1})
        with open(config.AE_PARAMS_FILE, 'r') as f: ae_params = json.load(f)
        print("✅ 参数加载成功。")
    except FileNotFoundError as e:
        print(f"❌ 错误: 缺少参数文件: {e}。"); return

    final_val_split_idx = int(len(X_train_full) * 0.9)
    print(f"\n--- 1. 正在创建最终训练集 (前90%) 和内部验证集 (后10%) ---")
    
    X_final_train = X_train_full.iloc[:final_val_split_idx]
    y_final_train = y_train_full.iloc[:final_val_split_idx] # 包含所有目标列
    sw_final_train = sw_train_full.iloc[:final_val_split_idx]
    
    X_final_val = X_train_full.iloc[final_val_split_idx:]
    y_final_val = y_train_full.iloc[final_val_split_idx:] # 包含所有目标列
    sw_final_val = sw_train_full.iloc[final_val_split_idx:]
    print(f"  > 最终训练集: {len(X_final_train)} 行 | 内部验证集: {len(X_final_val)} 行")

    print("\n--- 2. 正在从最终训练集学习预处理参数 ---")
    preprocessor_params = utils.get_preprocessor_params(X_final_train)
    
    print("--- 3. 正在应用参数转换所有数据集 ---")
    _, X_final_train_scaled = utils.apply_preprocessor(X_final_train, preprocessor_params)
    _, X_final_val_scaled = utils.apply_preprocessor(X_final_val, preprocessor_params)
    _, X_holdout_scaled = utils.apply_preprocessor(X_holdout, preprocessor_params)
    
    print("\n--- 4. 正在训练最终AE模型 (使用早停策略) ---")
    ae_models = utils.train_fold_ae(
        ae_params, 
        X_final_train_scaled.values, y_final_train[config.TARGET_COLUMNS].values, sw_final_train.values,
        X_final_val_scaled.values, y_final_val[config.TARGET_COLUMNS].values, sw_final_val.values,
        X_final_train.isnull().values,
        X_final_val.isnull().values
    )
    if not ae_models: print("❌ AE模型训练失败。"); return
    print("✅ 最终AE模型训练完成。")

    print("\n--- 5. 正在为所有数据集生成AI特征 ---")
    with torch.no_grad():
        train_tensor = torch.tensor(X_final_train_scaled.values, dtype=torch.float32).to(device)
        val_tensor = torch.tensor(X_final_val_scaled.values, dtype=torch.float32).to(device)
        holdout_tensor = torch.tensor(X_holdout_scaled.values, dtype=torch.float32).to(device)
        
        X_train_ai = np.mean([m.encoder(train_tensor).cpu().numpy() for m in ae_models], axis=0)
        X_val_ai = np.mean([m.encoder(val_tensor).cpu().numpy() for m in ae_models], axis=0)
        X_holdout_ai = np.mean([m.encoder(holdout_tensor).cpu().numpy() for m in ae_models], axis=0)

    ai_cols = [f'{config.AI_PREFIX}{i}' for i in range(X_train_ai.shape[1])]
    X_train_ai_df = pd.DataFrame(X_train_ai, columns=ai_cols, index=X_final_train.index)
    X_val_ai_df = pd.DataFrame(X_val_ai, columns=ai_cols, index=X_final_val.index)
    X_holdout_ai_df = pd.DataFrame(X_holdout_ai, columns=ai_cols, index=X_holdout.index)
    
    X_train_final = pd.concat([X_final_train_scaled, X_train_ai_df], axis=1)
    X_val_final = pd.concat([X_final_val_scaled, X_val_ai_df], axis=1)
    X_holdout_final = pd.concat([X_holdout_scaled, X_holdout_ai_df], axis=1)

    print("\n--- 6. 正在训练最终LGBM回归模型并进行两场评估 ---")
    val_scores, holdout_scores = {}, {}
    val_weights = np.nan_to_num(sw_final_val.values, nan=0.0)
    val_valid_mask = val_weights > 0
    holdout_weights = np.nan_to_num(sw_holdout.values, nan=0.0)
    holdout_valid_mask = holdout_weights > 0
    if not val_valid_mask.any(): print('  > Warning: no positive-weight samples in internal validation; using all rows.')
    if not holdout_valid_mask.any(): print('  > Warning: no positive-weight samples in holdout; using all rows.')
    for i, target_col in enumerate(config.TARGET_COLUMNS):
        action_col = config.ACTION_COLUMNS[i]
        print(f"  > 正在训练 '{target_col}' 并用 '{action_col}' 评估...")
        
        # [核心修复] 使用 LGBMRegressor
        model = lgb.LGBMRegressor(**lgbm_params).fit(X_train_final, y_final_train[target_col], sample_weight=sw_final_train)
        
        # [核心修复] 使用 .predict()
        val_preds = model.predict(X_val_final)
        holdout_preds = model.predict(X_holdout_final)
        
        # [核心修复] 评估时使用正确的 action 列
        val_truth = y_final_val[action_col].values[val_valid_mask] if val_valid_mask.any() else y_final_val[action_col].values
        val_pred = val_preds[val_valid_mask] if val_valid_mask.any() else val_preds
        holdout_truth = y_holdout[action_col].values[holdout_valid_mask] if holdout_valid_mask.any() else y_holdout[action_col].values
        holdout_pred = holdout_preds[holdout_valid_mask] if holdout_valid_mask.any() else holdout_preds
        val_scores[action_col] = 0.5 if np.unique(val_truth).size < 2 else roc_auc_score(val_truth, val_pred)
        holdout_scores[action_col] = 0.5 if np.unique(holdout_truth).size < 2 else roc_auc_score(holdout_truth, holdout_pred)

    print(f"\n\n{'='*25} 最终审判日结果对比 {'='*25}")
    print("\n--- 内部验证集 (考前模拟) 成绩 ---")
    for col, score in val_scores.items(): print(f"  🎯 内部验证 AUC for '{col}': {score:.8f}")
    avg_val_score = np.mean(list(val_scores.values()))
    print(f"  ----------------------------------------\n  🏆 内部验证平均AUC: {avg_val_score:.8f}\n")
    
    print("--- 外部持有集 (正式高考) 成绩 ---")
    for col, score in holdout_scores.items(): print(f"  🎯 外部持有 AUC for '{col}': {score:.8f}")
    avg_holdout_score = np.mean(list(holdout_scores.values()))
    print(f"  ----------------------------------------\n  🏆 外部持有平均AUC: {avg_holdout_score:.8f}\n")

    score_diff = avg_val_score - avg_holdout_score
    print("=" * 65)
    print(f"诊断分析: 内部验证与外部持有分数差距: {score_diff:.8f}")
    print("=" * 65)


# -----------------------------------------------------------------
# 主程序入口 (V2.9.1 最终校准版)
# -----------------------------------------------------------------
if __name__ == '__main__':
     # --- [!!! 关键修复 !!!] ---
    # 导入多进程库
    import torch.multiprocessing as mp
    
    # 在Windows上使用多核DataLoader时，必须添加这行代码。
    # 它告诉PyTorch以一种更安全、干净的方式来启动新的“帮厨”进程，
    # 从而避免CUDA和进程初始化之间的冲突，防止死锁。
    # 这必须是主程序入口的第一行可执行代码。
    try:
        mp.set_start_method('spawn', force=True)
        print("\n--- [多进程模式] 已成功设置为 'spawn' ---")
    except RuntimeError:
        pass
    # --- [修复结束] ---
    parser = argparse.ArgumentParser(description="自动化机器学习作战平台 V2.9.1 (DLS最终校准版)")
    parser.add_argument('--mode', type=str, required=True, choices=['rfe', 'tune_lgbm', 'tune_ae', 'validate', 'holdout'])
    parser.add_argument('--config', type=str, default='config', help="配置文件 (e.g., 'config', 'smoke')")
    args = parser.parse_args()

    try:
        config_module = importlib.import_module(args.config)
        print(f"--- 配置文件 '{args.config}.py' 加载成功 ---")
    except ImportError:
        print(f"❌ 错误: 无法找到配置文件 '{args.config}.py'。"); exit()

    import utils
    start_time = time.time()
    
    print("\n--- 启动统一数据加载 ---")
    dev_df = utils.load_data(config_module.RAW_DATA_FILE, config_module.ANALYSIS_START_DATE_ID)
    
    print("--- 启动统一特征筛选 ---")
    
    all_available_features = [c for c in dev_df.columns if c.startswith(('M', 'E', 'I', 'P', 'V', 'S', 'D')) or '_diff' in c or '_rol_' in c]

    if args.mode == 'rfe':
        top_features = all_available_features
        print(f"  > [RFE模式] 已激活。将使用全部 {len(top_features)} 个可用特征进行排序。")
    else:
        try:
            ranking = pd.read_csv(config_module.RANKING_FILE, index_col=0, header=None).squeeze("columns")
            top_features_from_file = ranking.nsmallest(config_module.N_TOP_FEATURES_TO_USE).index.tolist()
            top_features = [feat for feat in top_features_from_file if feat in all_available_features]
            print(f"  > 已加载排名，将使用 {len(top_features)} 个在当前数据中可用的Top特征。")
        except FileNotFoundError:
            top_features = all_available_features
            print(f"  > 警告: 未找到特征排名文件 '{config_module.RANKING_FILE}'。")
            print(f"  > 将使用全部 {len(top_features)} 个可用特征。")

    # --- [!!! 关键修复 !!!] ---
    # 分别准备 X, y(软目标), y_action(硬目标), 和样本权重
    X_dev = dev_df[top_features]
    y_dev_dls = dev_df[config_module.TARGET_COLUMNS] # 用于训练的软目标
    y_dev_action = dev_df[config_module.ACTION_COLUMNS] # 用于评估的硬目标
    y_dev_all = dev_df[config_module.TARGET_COLUMNS + config_module.ACTION_COLUMNS] # 包含所有目标的版本

    date_ids_dev = dev_df[['date_id']]
    
    print("--- 正在加载预计算的安全样本权重 ---")
    if 'sample_weight' in dev_df.columns:
        sample_weight_dev = dev_df['sample_weight']
        print("  > ✅ 'sample_weight' 列加载成功。")
    else:
        print("  > ❌ 错误: 未在数据文件中找到 'sample_weight' 列！请先重新运行 create_features.py。")
        exit()

    # 根据不同的模式，调用对应的函数，并传递正确的参数
    if args.mode == 'holdout':
        print(f"  > [Holdout模式] 正在加载持有集: '{config_module.HOLDOUT_DATA_FILE}'")
        holdout_df = utils.load_data(config_module.HOLDOUT_DATA_FILE, -1) # holdout不筛选date_id
        if 'sample_weight' in holdout_df.columns:
            sample_weight_holdout = holdout_df['sample_weight'].fillna(0.0)
        else:
            sample_weight_holdout = holdout_df[config_module.RESP_COLUMNS].abs().sum(axis=1).fillna(0.0)
        
        X_holdout = holdout_df[top_features]
        # holdout的y也需要包含所有目标列
        y_holdout_all = holdout_df[config_module.TARGET_COLUMNS + config_module.ACTION_COLUMNS]
        
        # 调用已升级的 run_holdout_validation 函数
        run_holdout_validation(X_dev, y_dev_all, sample_weight_dev, X_holdout, y_holdout_all, sample_weight_holdout, config_module, utils)
    
    elif args.mode == 'rfe':
        # rfe 只需要软目标或硬目标之一即可，这里用软目标保持一致
        run_rfe(X_dev, y_dev_dls, config_module, utils)

    elif args.mode == 'tune_lgbm' or args.mode == 'tune_ae':
        # 调用已升级的 run_tuning 函数，分别传入软目标和硬目标
        run_tuning(X_dev, y_dev_dls, y_dev_action, sample_weight_dev, args.mode.split('_')[1], config_module, utils)

    elif args.mode == 'validate':
        # 调用已升级的 run_validation 函数，传入包含所有目标的y
        run_validation(X_dev, y_dev_all, sample_weight_dev, date_ids_dev, config_module, utils)
    # --- [修复结束] ---

    print(f"\n任务 '{args.mode}' 完成！总耗时: {time.time() - start_time:.2f} 秒。")
