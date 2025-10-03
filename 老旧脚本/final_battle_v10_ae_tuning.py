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
import optuna

# =================================================================
# final_battle_v10_ae_tuning.py (集成化作战平台 V1.2 - AE调优版)
# 目的: 在LGBM参数固定后，为监督式AE寻找最优超参数，以提升AI特征质量。
# =================================================================

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. 全局配置 (模式切换) =================
MODE = 'TUNE_AE'  # 'TUNE_AE' 或 'VALIDATE'
N_TRIALS_AE = 50 # AE的Optuna搜索次数

# --- 文件路径 (更新) ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
# [锁定] 加载已找到的最佳LGBM参数
LGBM_PARAMS_FILE = 'best_params_v9_integrated_lgbm.json' 
# [新] 为AE创建新的参数文件
AE_PARAMS_FILE = 'best_params_v10_integrated_ae.json'
OOF_OUTPUT_FILE = 'oof_predictions_v10_ae_tuned.csv' 

# --- 其他配置 (无变化) ---
MISSING_THRESHOLD = 0.20
N_SPLITS = 5; PURGE_SIZE = 1; EMBARGO_SIZE = 40 
ANALYSIS_START_DATE_ID = 1055

# ================= 2. AI兵工厂：监督式AE模块 (升级为可配置) =================
class SupervisedAE(nn.Module):
    # [升级] 构造函数现在接收所有可调参数
    def __init__(self, input_dim, hidden_dim, encoding_dim, n_hidden_layers, dropout_rate):
        super(SupervisedAE, self).__init__()
        # (网络结构与之前一致，但尺寸和深度是动态的)
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

# ================= 3. AE训练函数 (升级为可配置) =================
# [升级] 函数签名增加了ae_params参数
def train_fold_ae(ae_params, X_train_raw, y_train_raw, sw_train_raw, X_val_raw, y_val_raw):
    # (数据预处理部分无变化)
    X_train_filled = X_train_raw.ffill(limit=3); median_filler = X_train_filled.median()
    X_train_filled.fillna(median_filler, inplace=True); X_train_filled.fillna(0, inplace=True)
    X_val_filled = X_val_raw.ffill(limit=3)
    X_val_filled.fillna(median_filler, inplace=True); X_val_filled.fillna(0, inplace=True)
    mean = X_train_filled.mean(axis=0).values; std = X_train_filled.std(axis=0).values; std[std == 0] = 1.0
    X_train_scaled = (X_train_filled.values - mean) / std
    X_val_scaled = (X_val_filled.values - mean) / std
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(X_train_raw.isnull().values, dtype=torch.bool), torch.tensor(y_train_raw.values, dtype=torch.float32).unsqueeze(1), torch.tensor(sw_train_raw.values, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    

    # ---> [代码修复开始] <---
    # 1. 将ae_params拆分为“模型结构参数”和“训练过程参数”
    model_architecture_params = {
        'hidden_dim': ae_params['hidden_dim'],
        'encoding_dim': ae_params['encoding_dim'],
        'n_hidden_layers': ae_params['n_hidden_layers'],
        'dropout_rate': ae_params['dropout_rate']
    }
    
    input_dim = X_train_raw.shape[1]
    # 2. 只把“模型结构参数”传递给SupervisedAE
    model = SupervisedAE(input_dim, **model_architecture_params).to(device)
    
    # 3. 在后续的训练中，从主ae_params字典中正确引用训练参数
    optimizer = torch.optim.Adam(model.parameters(), lr=ae_params['learning_rate'])
    # ---> [代码修复结束] <---


    recon_loss_func, pred_loss_func = nn.MSELoss(), nn.BCELoss(reduction='none')
    
    epochs, patience, epochs_no_improve, best_model_state, min_val_loss = 100, 7, 0, None, np.inf
    for epoch in range(epochs):
        model.train()
        for batch_features, batch_mask, batch_target, batch_weights in train_loader:
            batch_features, batch_mask, batch_target, batch_weights = batch_features.to(device), batch_mask.to(device), batch_target.to(device), batch_weights.to(device)
            recon_outputs, pred_outputs = model(batch_features)
            r_loss = recon_loss_func(recon_outputs[~batch_mask], batch_features[~batch_mask])
            p_loss = (pred_loss_func(pred_outputs, batch_target) * batch_weights).mean()
            loss = ae_params['recon_weight'] * r_loss + (1 - ae_params['recon_weight']) * p_loss # 使用动态权重
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

# ================= 4. Optuna目标函数 (升级为AE调优) =================
# [升级] 函数现在接收固定的lgbm_params
def objective(trial, X_full, y, sample_weight, lgbm_params):
    # --- 步骤4.1: 定义AE的搜索空间 ---
    ae_params = {
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 256, step=32),
        'encoding_dim': trial.suggest_int('encoding_dim', 16, 64, step=8),
        'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 3),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'recon_weight': trial.suggest_float('recon_weight', 0.2, 0.8) # 重建损失的权重
    }

    # --- 步骤4.2: 执行集成化验证流程 ---
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    oof_predictions = np.zeros(len(X_full))
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
        purged_train_idx = train_idx[:-PURGE_SIZE] if PURGE_SIZE > 0 else train_idx
        X_train_fold, y_train_fold = X_full.iloc[purged_train_idx], y.iloc[purged_train_idx]
        sw_train_fold = sample_weight.iloc[purged_train_idx]
        X_val_fold, y_val_fold = X_full.iloc[val_idx], y.iloc[val_idx]

        # [升级] 使用Optuna建议的参数来训练AE
        ae_model = train_fold_ae(ae_params, X_train_fold, y_train_fold, sw_train_fold, X_val_fold, y_val_fold)
        
        # (后续流程与之前一致)
        with torch.no_grad():
            encoding_dim = ae_params['encoding_dim']
            train_filled = X_train_fold.ffill(limit=3).fillna(X_train_fold.ffill(limit=3).median()).fillna(0)
            val_filled = X_val_fold.ffill(limit=3).fillna(train_filled.median()).fillna(0)
            mean = train_filled.mean().values; std = train_filled.std().values; std[std==0] = 1.0
            train_scaled = torch.tensor((train_filled.values - mean) / std, dtype=torch.float32).to(device)
            val_scaled = torch.tensor((val_filled.values - mean) / std, dtype=torch.float32).to(device)
            X_train_ai_features = ae_model.encoder(train_scaled).cpu().numpy()
            X_val_ai_features = ae_model.encoder(val_scaled).cpu().numpy()

        X_train_final = np.concatenate([X_train_fold.values, X_train_ai_features], axis=1)
        X_val_final = np.concatenate([X_val_fold.values, X_val_ai_features], axis=1)
        
        # [锁定] 使用固定的最佳LGBM参数
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

# ================= 5. 主程序入口 (模式更新) =================
if __name__ == '__main__':
    start_time = time.time()
    print("--- 步骤1：加载数据并执行“数据质量门禁” ---")
    # (数据加载部分无变化)
    raw_df = pd.read_csv(RAW_DATA_FILE)
    modern_df = raw_df[raw_df['date_id'] > ANALYSIS_START_DATE_ID].copy().reset_index(drop=True)
    y = (modern_df['forward_returns'] > 0).astype(int)
    sample_weight = modern_df['forward_returns'].abs().fillna(0)
    feature_cols = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
    missing_ratios = modern_df[feature_cols].isnull().mean()
    features_to_keep = missing_ratios[missing_ratios <= MISSING_THRESHOLD].index.tolist()
    X_full = modern_df[features_to_keep]
    print(f"✅ 数据质量门禁完成：保留了 {len(features_to_keep)} 个特征。")

    # [锁定] 加载最佳LGBM参数，为所有模式做准备
    try:
        with open(LGBM_PARAMS_FILE, 'r') as f: 
            lgbm_params = json.load(f)
            # 确保一些固定参数存在
            lgbm_params.update({'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1})
    except FileNotFoundError:
        print(f"❌ 关键错误: 未找到LGBM参数文件 '{LGBM_PARAMS_FILE}'！无法继续。")
        exit()

    if MODE == 'TUNE_AE':
        print(f"\n--- 启动 'TUNE_AE' 模式，开始为AE寻找最优超参数 ({N_TRIALS_AE}次尝试) ---")
        print("    > 使用的LGBM参数已锁定。")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_full, y, sample_weight, lgbm_params), n_trials=N_TRIALS_AE)
        
        print(f"\n{'='*25} Optuna搜索结束 {'='*25}")
        print(f"✅ 找到了AE的最优参数！最佳诚实OOF AUC: {study.best_value:.8f}")
        print("最优AE参数组合:")
        print(json.dumps(study.best_params, indent=4))
        
        with open(AE_PARAMS_FILE, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        print(f"\n-> 最佳AE参数已保存至 '{AE_PARAMS_FILE}'")

    # (VALIDATE模式暂时省略，逻辑与V9类似，但需要加载AE参数，后续再完善)

    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.2f} 秒。")