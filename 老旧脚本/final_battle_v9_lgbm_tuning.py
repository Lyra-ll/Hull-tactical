import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import warnings
import json
import time
import copy
import optuna # <--- 新增Optuna

# =================================================================
# final_battle_v9_lgbm_tuning.py (集成化作战平台 V1.1 - LGBM调优版)
# 目的: 在V8的诚实流水线上，集成Optuna，为LightGBM寻找最优超参数。
# =================================================================

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. 全局配置 (新增模式切换) =================
# --- [新] 模式切换 ---
MODE = 'TUNE'  # 'TUNE' 或 'VALIDATE'
N_TRIALS = 50 # Optuna 搜索次数

# --- 文件路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
# [新] 参数文件将根据模式自动命名
LGBM_PARAMS_FILE = 'best_params_v9_integrated_lgbm.json' 
OOF_OUTPUT_FILE = 'oof_predictions_v9_tuned.csv' 

# --- 数据质量门禁 ---
MISSING_THRESHOLD = 0.20

# --- 验证策略 ---
N_SPLITS = 5; PURGE_SIZE = 1; EMBARGO_SIZE = 40 
ANALYSIS_START_DATE_ID = 1055

# ================= 2. AI兵工厂：监督式AE模块 (无变化) =================
class SupervisedAE(nn.Module):
    # (网络结构暂时保持不变)
    def __init__(self, input_dim, hidden_dim=128, encoding_dim=32, n_hidden_layers=2, dropout_rate=0.2):
        super(SupervisedAE, self).__init__()
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dim)); encoder_layers.append(nn.ReLU()); encoder_layers.append(nn.Dropout(dropout_rate))
        for _ in range(n_hidden_layers - 1):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim)); encoder_layers.append(nn.ReLU()); encoder_layers.append(nn.Dropout(dropout_rate))
        encoder_layers.append(nn.Linear(hidden_dim, encoding_dim)); encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        decoder_layers.append(nn.Linear(encoding_dim, hidden_dim)); decoder_layers.append(nn.ReLU()); decoder_layers.append(nn.Dropout(dropout_rate))
        for _ in range(n_hidden_layers - 1):
            decoder_layers.append(nn.Linear(hidden_dim, hidden_dim)); decoder_layers.append(nn.ReLU()); decoder_layers.append(nn.Dropout(dropout_rate))
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        self.predictor = nn.Sequential(nn.Linear(encoding_dim, 1), nn.Sigmoid())
    def forward(self, x):
        encoded = self.encoder(x); decoded = self.decoder(encoded); prediction = self.predictor(encoded)
        return decoded, prediction

# ================= 3. 统一的预处理与AE训练函数 (无变化) =================
def train_fold_ae(X_train_raw, y_train_raw, sw_train_raw, X_val_raw, y_val_raw):
    # (此函数内部逻辑与V8完全一致)
    X_train_filled = X_train_raw.ffill(limit=3); median_filler = X_train_filled.median()
    X_train_filled.fillna(median_filler, inplace=True); X_train_filled.fillna(0, inplace=True)
    X_val_filled = X_val_raw.ffill(limit=3)
    X_val_filled.fillna(median_filler, inplace=True); X_val_filled.fillna(0, inplace=True)
    mean = X_train_filled.mean(axis=0).values; std = X_train_filled.std(axis=0).values; std[std == 0] = 1.0
    X_train_scaled = (X_train_filled.values - mean) / std
    X_val_scaled = (X_val_filled.values - mean) / std
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(X_train_raw.isnull().values, dtype=torch.bool), torch.tensor(y_train_raw.values, dtype=torch.float32).unsqueeze(1), torch.tensor(sw_train_raw.values, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    input_dim = X_train_raw.shape[1]
    model = SupervisedAE(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    recon_loss_func, pred_loss_func = nn.MSELoss(), nn.BCELoss(reduction='none')
    epochs, patience, epochs_no_improve, best_model_state, min_val_loss = 100, 7, 0, None, np.inf
    for epoch in range(epochs):
        model.train()
        for batch_features, batch_mask, batch_target, batch_weights in train_loader:
            batch_features, batch_mask, batch_target, batch_weights = batch_features.to(device), batch_mask.to(device), batch_target.to(device), batch_weights.to(device)
            recon_outputs, pred_outputs = model(batch_features)
            r_loss = recon_loss_func(recon_outputs[~batch_mask], batch_features[~batch_mask])
            p_loss = (pred_loss_func(pred_outputs, batch_target) * batch_weights).mean()
            loss = 0.5 * r_loss + 0.5 * p_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            _, val_preds = model(val_tensor)
            val_bce_loss = nn.BCELoss()(val_preds, torch.tensor(y_val_raw.values, dtype=torch.float32).unsqueeze(1).to(device)).item()
        if val_bce_loss < min_val_loss:
            min_val_loss, epochs_no_improve, best_model_state = val_bce_loss, 0, copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break
    if best_model_state: model.load_state_dict(best_model_state)
    return model.eval()

# ================= 4. [新] 主执行模块被改造为Optuna的目标函数 =================
def objective(trial, X_full, y, sample_weight):
    # --- 步骤4.1: 定义LGBM的搜索空间 ---
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

    # --- 步骤4.2: 执行与V8完全一致的集成化验证流程 ---
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    oof_predictions = np.zeros(len(X_full))
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
        purged_train_idx = train_idx[:-PURGE_SIZE] if PURGE_SIZE > 0 else train_idx
        X_train_fold, y_train_fold = X_full.iloc[purged_train_idx], y.iloc[purged_train_idx]
        sw_train_fold = sample_weight.iloc[purged_train_idx]
        X_val_fold, y_val_fold = X_full.iloc[val_idx], y.iloc[val_idx]

        # --- AE特征生成 (保持不变) ---
        ae_model = train_fold_ae(X_train_fold, y_train_fold, sw_train_fold, X_val_fold, y_val_fold)
        with torch.no_grad():
            train_filled = X_train_fold.ffill(limit=3).fillna(X_train_fold.ffill(limit=3).median()).fillna(0)
            val_filled = X_val_fold.ffill(limit=3).fillna(train_filled.median()).fillna(0)
            mean = train_filled.mean().values; std = train_filled.std().values; std[std==0] = 1.0
            train_scaled = torch.tensor((train_filled.values - mean) / std, dtype=torch.float32).to(device)
            val_scaled = torch.tensor((val_filled.values - mean) / std, dtype=torch.float32).to(device)
            X_train_ai_features = ae_model.encoder(train_scaled).cpu().numpy()
            X_val_ai_features = ae_model.encoder(val_scaled).cpu().numpy()

        X_train_final = np.concatenate([X_train_fold.values, X_train_ai_features], axis=1)
        X_val_final = np.concatenate([X_val_fold.values, X_val_ai_features], axis=1)
        
        # --- 使用Optuna建议的参数来训练LGBM ---
        model_lgbm = lgb.LGBMClassifier(**lgbm_params)
        model_lgbm.fit(X_train_final, y_train_fold, 
                       sample_weight=sw_train_fold,
                       eval_set=[(X_val_final, y_val_fold)],
                       callbacks=[lgb.early_stopping(100, verbose=False)])

        oof_predictions[val_idx] = model_lgbm.predict_proba(X_val_final)[:, 1]

    # --- 步骤4.3: 返回最终分数给Optuna ---
    valid_indices = np.where(oof_predictions != 0)[0]
    final_oof_score = roc_auc_score(y.iloc[valid_indices], oof_predictions[valid_indices])
    
    return final_oof_score

# ================= 5. 主程序入口 (新增模式切换逻辑) =================
if __name__ == '__main__':
    start_time = time.time()
    print("--- 步骤1：加载数据并执行“数据质量门禁” ---")
    raw_df = pd.read_csv(RAW_DATA_FILE)
    modern_df = raw_df[raw_df['date_id'] > ANALYSIS_START_DATE_ID].copy().reset_index(drop=True)
    y = (modern_df['forward_returns'] > 0).astype(int)
    sample_weight = modern_df['forward_returns'].abs().fillna(0)
    feature_cols = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
    missing_ratios = modern_df[feature_cols].isnull().mean()
    features_to_keep = missing_ratios[missing_ratios <= MISSING_THRESHOLD].index.tolist()
    X_full = modern_df[features_to_keep]
    print(f"✅ 数据质量门禁完成：保留了 {len(features_to_keep)} 个特征。")

    if MODE == 'TUNE':
        print(f"\n--- 启动 'TUNE' 模式，开始为LGBM寻找最优超参数 ({N_TRIALS}次尝试) ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_full, y, sample_weight), n_trials=N_TRIALS)
        
        print(f"\n{'='*25} Optuna搜索结束 {'='*25}")
        print(f"✅ 找到了LGBM的最优参数！最佳诚实OOF AUC: {study.best_value:.8f}")
        print("最优参数组合:")
        print(json.dumps(study.best_params, indent=4))
        
        with open(LGBM_PARAMS_FILE, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        print(f"\n-> 最佳参数已保存至 '{LGBM_PARAMS_FILE}'")

    elif MODE == 'VALIDATE':
        print(f"\n--- 启动 'VALIDATE' 模式，使用最优参数进行最终验证 ---")
        try:
            with open(LGBM_PARAMS_FILE, 'r') as f: lgbm_params = json.load(f)
            print("加载的最优LGBM参数组合:")
            print(json.dumps(lgbm_params, indent=4))
        except FileNotFoundError:
            print(f"❌ 错误: 未找到参数文件 '{LGBM_PARAMS_FILE}'。请先在 'TUNE' 模式下运行。")
            exit()
            
        # (VALIDATE模式的代码与objective函数内的流程几乎一样, 此处为简化可复用objective)
        # 为了清晰，我们重写一遍验证流程
        tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
        oof_predictions = np.zeros(len(X_full))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
            print(f"\n{'='*20} 正在处理第 {fold + 1}/{N_SPLITS} 折 {'='*20}")
            purged_train_idx = train_idx[:-PURGE_SIZE] if PURGE_SIZE > 0 else train_idx
            X_train_fold, y_train_fold = X_full.iloc[purged_train_idx], y.iloc[purged_train_idx]
            sw_train_fold = sample_weight.iloc[purged_train_idx]
            X_val_fold, y_val_fold = X_full.iloc[val_idx], y.iloc[val_idx]

            ae_model = train_fold_ae(X_train_fold, y_train_fold, sw_train_fold, X_val_fold, y_val_fold)
            with torch.no_grad():
                train_filled = X_train_fold.ffill(limit=3).fillna(X_train_fold.ffill(limit=3).median()).fillna(0)
                val_filled = X_val_fold.ffill(limit=3).fillna(train_filled.median()).fillna(0)
                mean = train_filled.mean().values; std = train_filled.std().values; std[std==0] = 1.0
                train_scaled = torch.tensor((train_filled.values - mean) / std, dtype=torch.float32).to(device)
                val_scaled = torch.tensor((val_filled.values - mean) / std, dtype=torch.float32).to(device)
                X_train_ai_features = ae_model.encoder(train_scaled).cpu().numpy()
                X_val_ai_features = ae_model.encoder(val_scaled).cpu().numpy()

            X_train_final = np.concatenate([X_train_fold.values, X_train_ai_features], axis=1)
            X_val_final = np.concatenate([X_val_fold.values, X_val_ai_features], axis=1)
            
            model_lgbm = lgb.LGBMClassifier(**lgbm_params)
            model_lgbm.fit(X_train_final, y_train_fold, sample_weight=sw_train_fold, eval_set=[(X_val_final, y_val_fold)], callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_predictions[val_idx] = model_lgbm.predict_proba(X_val_final)[:, 1]

        valid_indices = np.where(oof_predictions != 0)[0]
        final_oof_score = roc_auc_score(y.iloc[valid_indices], oof_predictions[valid_indices])
        print(f"\n\n{'='*25} 最终验证结束 {'='*25}")
        print(f"✅ 使用最优参数得到的最终诚实OOF AUC: {final_oof_score:.8f}")
        
        oof_df = pd.DataFrame({'date_id': modern_df['date_id'].iloc[valid_indices], 'target': y.iloc[valid_indices], 'oof_prediction': oof_predictions[valid_indices]})
        oof_df.to_csv(OOF_OUTPUT_FILE, index=False)
        print(f"✅ OOF预测结果已保存至 '{OOF_OUTPUT_FILE}'。")

    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.2f} 秒。")