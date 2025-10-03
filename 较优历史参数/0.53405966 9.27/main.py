# main.py
# =================================================================
# 自动化指挥中心 V1.5.1 (最终修正和完整版)
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

# 导入我们的自定义模块
import config
import utils

# --- 1. 作战模式: 特征筛选 (RFE) ---
def run_rfe(X, y):
    print("\n--- 启动 'RFE' 模式: 正在进行交叉验证特征排序 ---")
    estimator = lgb.LGBMClassifier(random_state=42, verbosity=-1)
    tscv = utils.get_cv_splitter(config.N_SPLITS, config.EMBARGO_SIZE, config.PURGE_SIZE)
    all_rankings = []
    
    for fold, (train_idx, _) in enumerate(tscv.split(X)):
        print(f"  > 正在第 {fold + 1}/{config.N_SPLITS} 折上运行RFE...")
        purged_train_idx = train_idx[:-config.PURGE_SIZE] if config.PURGE_SIZE > 0 else train_idx
        X_train_rfe, y_train_rfe = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
        X_train_rfe_filled = X_train_rfe.ffill(limit=3).fillna(X_train_rfe.median()).fillna(0)
        
        selector = RFE(estimator=estimator, n_features_to_select=config.N_FEATURES_TO_SELECT_RFE, step=0.1)
        selector.fit(X_train_rfe_filled, y_train_rfe)
        all_rankings.append(pd.Series(selector.ranking_, index=X.columns))

    avg_ranking = pd.concat(all_rankings, axis=1).mean(axis=1).sort_values()
    avg_ranking.to_csv(config.RANKING_FILE)
    print(f"\n✅ RFE排名完成！排名已保存至 '{config.RANKING_FILE}'")

# --- 2. 作战模式: 超参数搜索 (Optuna) ---
def run_tuning(X, y, sample_weight, tune_target):
    
    tscv_fortuning = utils.get_cv_splitter(config.N_SPLITS, config.EMBARGO_SIZE, config.PURGE_SIZE)
    folds_to_use = list(tscv_fortuning.split(X))

    print(f"    > [终极稳健性] Optuna将使用全部 {len(folds_to_use)} 折C/V数据进行评估...")

    def objective(trial):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if tune_target == 'lgbm':
            try:
                with open(config.AE_PARAMS_FILE, 'r') as f:
                    ae_params = json.load(f)
            except FileNotFoundError:
                print(f"    > 警告: 未找到AE参数文件 '{config.AE_PARAMS_FILE}'。将使用一套默认的简单参数。")
                ae_params = {'hidden_dim': 128, 'encoding_dim': 32, 'n_hidden_layers': 2, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'recon_weight': 0.5, 'bn':True}
            lgbm_params = {
                'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1,
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
                'num_leaves': trial.suggest_int('num_leaves', 10, 80),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 50.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 50.0, log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            }
        
        elif tune_target == 'ae':
            with open(config.LGBM_PARAMS_FILE, 'r') as f: lgbm_params = json.load(f)
            lgbm_params.update({'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1})
            ae_params = {
                'hidden_dim': trial.suggest_int('hidden_dim', 64, 256, step=32),
                'encoding_dim': trial.suggest_int('encoding_dim', 16, 64, step=8),
                'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 3),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'recon_weight': trial.suggest_float('recon_weight', 0.2, 0.8),
                'bn': trial.suggest_categorical('bn', [True, False]),
            }

        oof_predictions_trial = np.zeros(len(X))
        
        for fold, (train_idx, val_idx) in enumerate(folds_to_use):
            print(f"\n[调试] ======== FOLD {fold + 1} 开始 ========")
            fold_start_time = time.time()

            purged_train_idx = train_idx[:-config.PURGE_SIZE] if config.PURGE_SIZE > 0 else train_idx
            X_train, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
            sw_train = sample_weight.iloc[purged_train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            train_mask_np = X_train.isnull().values

            X_train_processed, X_val_processed = utils.preprocess_for_lgbm(X_train, X_val)
            print(f"[调试] FOLD {fold + 1} | 步骤1: 数据预处理 | 耗时: {time.time() - fold_start_time:.2f} 秒")

            X_train_np, y_train_np = X_train_processed.values, y_train.values
            sw_train_np, X_val_np = sw_train.values, X_val_processed.values
            y_val_np = y_val.values
            
            step1_time = time.time()
            ae_models = utils.train_fold_ae(ae_params, X_train_np, y_train_np, sw_train_np, X_val_np, y_val_np, train_mask_np)
            print(f"[调试] FOLD {fold + 1} | 步骤2: AE模型训练 | 耗时: {time.time() - step1_time:.2f} 秒")
            step2_time = time.time()
            
            if not ae_models:
                oof_predictions_trial[val_idx] = 0.5
                continue

            with torch.no_grad():
                mean, std = np.mean(X_train_np, axis=0), np.std(X_train_np, axis=0)
                std[std==0] = 1.0
                
                train_scaled = torch.tensor((X_train_np - mean) / std, dtype=torch.float32).to(device)
                val_scaled = torch.tensor((X_val_np - mean) / std, dtype=torch.float32).to(device)
                
                all_train_ai, all_val_ai = [], []
                for ae_model in ae_models:
                    all_train_ai.append(ae_model.encoder(train_scaled).cpu().numpy())
                    all_val_ai.append(ae_model.encoder(val_scaled).cpu().numpy())
                
                X_train_ai, X_val_ai = np.mean(all_train_ai, axis=0), np.mean(all_val_ai, axis=0)

            ai_feature_names = [f'{config.AI_PREFIX}{i}' for i in range(X_train_ai.shape[1])]
            X_train_ai_df = pd.DataFrame(X_train_ai, columns=ai_feature_names, index=X_train_processed.index)
            X_val_ai_df = pd.DataFrame(X_val_ai, columns=ai_feature_names, index=X_val_processed.index)
            
            X_train_combined = pd.concat([X_train_processed, X_train_ai_df], axis=1)
            X_val_combined = pd.concat([X_val_processed, X_val_ai_df], axis=1)
            
            X_train_final, X_val_final = utils.preprocess_for_lgbm(X_train_combined, X_val_combined)
            
            model_lgbm = lgb.LGBMClassifier(**lgbm_params)
            model_lgbm.fit(X_train_final, y_train, sample_weight=sw_train,
                           eval_set=[(X_val_final, y_val)],
                           callbacks=[lgb.early_stopping(100, verbose=False)])

            preds = model_lgbm.predict_proba(X_val_final)[:, 1]
            oof_predictions_trial[val_idx] = preds
            print(f"[调试] FOLD {fold + 1} | 步骤3: AI特征提取与LGBM训练 | 耗时: {time.time() - step2_time:.2f} 秒")
            print(f"[调试] ======== FOLD {fold + 1} 总耗时: {time.time() - fold_start_time:.2f} 秒 ========")
        
        valid_indices = np.where(oof_predictions_trial != 0)[0]
        if len(valid_indices) == 0: return 0.5
        final_score = roc_auc_score(y.iloc[valid_indices], oof_predictions_trial[valid_indices])
        return final_score

    if tune_target == 'lgbm':
        print(f"\n--- 启动 'TUNE_LGBM' 模式 ({config.N_TRIALS_LGBM}次尝试) ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=config.N_TRIALS_LGBM)
        params_file = config.LGBM_PARAMS_FILE
    elif tune_target == 'ae':
        print(f"\n--- 启动 'TUNE_AE' 模式 ({config.N_TRIALS_AE}次尝试) ---")
        try:
            with open(config.LGBM_PARAMS_FILE, 'r') as f: pass
        except FileNotFoundError:
            print(f"❌ 错误: 无法启动AE调优, '{config.LGBM_PARAMS_FILE}' 不存在。请先运行 'tune_lgbm'。")
            return
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=config.N_TRIALS_AE)
        params_file = config.AE_PARAMS_FILE

    print(f"\n{'='*25} Optuna搜索结束 {'='*25}")
    print(f"✅ 找到了 {tune_target.upper()} 的最优参数！最佳全局 OOF AUC: {study.best_value:.8f}")
    print("最优参数组合:")
    print(json.dumps(study.best_params, indent=4))
    
    with open(params_file, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\n-> 最佳参数已保存至 '{params_file}'")

# --- 3. 作战模式: 最终验证 ---
def run_validation(X, y, sample_weight, date_ids):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n--- 启动 'VALIDATE' 模式: 使用最优参数进行最终性能评估 ---")
    try:
        with open(config.LGBM_PARAMS_FILE, 'r') as f: lgbm_params = json.load(f)
        lgbm_params.update({'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1})
        with open(config.AE_PARAMS_FILE, 'r') as f: ae_params = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ 错误: 缺少参数文件: {e}。请确保已成功运行 'tune_lgbm' 和 'tune_ae'。")
        return

    print("✅ 所有最优参数加载成功。开始最终的5折交叉验证...")
    tscv = utils.get_cv_splitter(config.N_SPLITS, config.EMBARGO_SIZE, config.PURGE_SIZE)
    oof_predictions = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n{'='*20} 正在处理第 {fold + 1}/{config.N_SPLITS} 折 {'='*20}")
        purged_train_idx = train_idx[:-config.PURGE_SIZE] if config.PURGE_SIZE > 0 else train_idx
        X_train, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
        sw_train = sample_weight.iloc[purged_train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        train_mask_np = X_train.isnull().values
        X_train_processed, X_val_processed = utils.preprocess_for_lgbm(X_train, X_val)
        
        X_train_np, y_train_np = X_train_processed.values, y_train.values
        sw_train_np, X_val_np = sw_train.values, X_val_processed.values
        y_val_np = y_val.values

        ae_models = utils.train_fold_ae(ae_params, X_train_np, y_train_np, sw_train_np, X_val_np, y_val_np, train_mask_np)
        
        if not ae_models:
            print(f"  > 警告: 第 {fold + 1} 折未能成功训练任何AE模型，将跳过此折的预测。")
            continue

        with torch.no_grad():
            mean, std = np.mean(X_train_np, axis=0), np.std(X_train_np, axis=0)
            std[std==0] = 1.0
            
            train_scaled = torch.tensor((X_train_np - mean) / std, dtype=torch.float32).to(device)
            val_scaled = torch.tensor((X_val_np - mean) / std, dtype=torch.float32).to(device)
            
            all_train_ai, all_val_ai = [], []
            for ae_model in ae_models:
                all_train_ai.append(ae_model.encoder(train_scaled).cpu().numpy())
                all_val_ai.append(ae_model.encoder(val_scaled).cpu().numpy())
            
            X_train_ai, X_val_ai = np.mean(all_train_ai, axis=0), np.mean(all_val_ai, axis=0)

        ai_feature_names = [f'{config.AI_PREFIX}{i}' for i in range(X_train_ai.shape[1])]
        X_train_ai_df = pd.DataFrame(X_train_ai, columns=ai_feature_names, index=X_train_processed.index)
        X_val_ai_df = pd.DataFrame(X_val_ai, columns=ai_feature_names, index=X_val_processed.index)
        X_train_combined = pd.concat([X_train_processed, X_train_ai_df], axis=1)
        X_val_combined = pd.concat([X_val_processed, X_val_ai_df], axis=1)
        
        X_train_final, X_val_final = utils.preprocess_for_lgbm(X_train_combined, X_val_combined)

        model_lgbm = lgb.LGBMClassifier(**lgbm_params)
        model_lgbm.fit(X_train_final, y_train, sample_weight=sw_train,
                       eval_set=[(X_val_final, y_val)],
                       callbacks=[lgb.early_stopping(100, verbose=False)])

        oof_predictions[val_idx] = model_lgbm.predict_proba(X_val_final)[:, 1]

    valid_indices = np.where(oof_predictions != 0)[0]
    final_score = roc_auc_score(y.iloc[valid_indices], oof_predictions[valid_indices])
    
    print(f"\n\n{'='*25} 最终验证结束 {'='*25}")
    print(f"🏆 最终的、可信的 OOF AUC: {final_score:.8f}")
    
    oof_df = pd.DataFrame({
        'date_id': date_ids['date_id'].iloc[valid_indices],
        'target': y.iloc[valid_indices],
        'oof_prediction': oof_predictions[valid_indices]
    })
    oof_df.to_csv(config.OOF_OUTPUT_FILE, index=False)
    print(f"✅ OOF预测结果已保存至 '{config.OOF_OUTPUT_FILE}'。")

# --- 主程序入口 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="自动化机器学习作战平台 V1.5 (最终性能优化版)")
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['rfe', 'tune_lgbm', 'tune_ae', 'validate'],
                        help="选择要执行的作战模式")
    args = parser.parse_args()

    start_time = time.time()
    
    X, y, sample_weight, date_ids = utils.load_data(
        config.RAW_DATA_FILE, config.ANALYSIS_START_DATE_ID, config.MISSING_THRESHOLD
    )
    
    if args.mode == 'rfe':
        run_rfe(X, y)
    elif args.mode == 'tune_lgbm':
        run_tuning(X, y, sample_weight, 'lgbm')
    elif args.mode == 'tune_ae':
        run_tuning(X, y, sample_weight, 'ae')
    elif args.mode == 'validate':
        run_validation(X, y, sample_weight, date_ids)

    total_time = time.time() - start_time
    print(f"\n任务 '{args.mode}' 完成！总耗时: {total_time:.2f} 秒。")