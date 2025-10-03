# =================================================================
# supervised_ae_v7_autotune_clf.py (V1.1 - Bug Fixed)
# 目的: 一个集成了Optuna的、端到端的AI特征锻造平台。
# =================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import optuna
import copy
import json
import os
import warnings

warnings.filterwarnings('ignore')

# ================= 1. 全局配置与模式切换 =================
MODE = 'GENERATE'  # 'SEARCH' 或 'GENERATE'
N_TRIALS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 将使用设备: {device} ---")

# --- 文件路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
PARAMS_FILE = 'best_ae_params_v7_autotune_clf.json'
OUTPUT_FILE = 'train_v11_autotune_clf_ae_features.csv'

# --- 验证策略 ---
N_SPLITS = 5
PURGE_SIZE = 1
EMBARGO_SIZE = 40 

# ================= 2. 神经网络蓝图 (无变化) =================
class SupervisedAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoding_dim, n_hidden_layers, dropout_rate):
        super(SupervisedAE, self).__init__()
        # ... (此处代码无变化)
        # --- Encoder ---
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dim)); encoder_layers.append(nn.ReLU()); encoder_layers.append(nn.Dropout(dropout_rate))
        for _ in range(n_hidden_layers - 1):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim)); encoder_layers.append(nn.ReLU()); encoder_layers.append(nn.Dropout(dropout_rate))
        encoder_layers.append(nn.Linear(hidden_dim, encoding_dim)); encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        # --- Decoder ---
        decoder_layers = []
        decoder_layers.append(nn.Linear(encoding_dim, hidden_dim)); decoder_layers.append(nn.ReLU()); decoder_layers.append(nn.Dropout(dropout_rate))
        for _ in range(n_hidden_layers - 1):
            decoder_layers.append(nn.Linear(hidden_dim, hidden_dim)); decoder_layers.append(nn.ReLU()); decoder_layers.append(nn.Dropout(dropout_rate))
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        # --- Predictor ---
        self.predictor = nn.Sequential(nn.Linear(encoding_dim, 1), nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        prediction = self.predictor(encoded)
        return decoded, prediction

# ================= 3. 数据加载 (无变化) =================
def load_and_prepare_data():
    print("--- 正在加载数据并准备分类任务... ---")
    df = pd.read_csv(RAW_DATA_FILE)
    modern_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)
    features_to_exclude = ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']
    feature_columns = [col for col in modern_df.columns if col not in features_to_exclude]
    features_df = modern_df[feature_columns]
    target_df = (modern_df['forward_returns'] > 0).astype(float)
    sample_weight_df = modern_df['forward_returns'].abs()
    print("    -> 分类目标和样本权重准备完毕。")
    return features_df, target_df, sample_weight_df, modern_df

# ================= 4. Optuna目标函数 `objective` (已修复) =================
def objective(trial, features_df, target_df, sample_weight_df):
    params = {
        'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 3),
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 256),
        'encoding_dim': trial.suggest_int('encoding_dim', 16, 64),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'recon_weight': trial.suggest_float('recon_weight', 0.1, 0.7)
    }
    
    # <--- BUG FIX #1: 分离出只给模型构造函数用的参数 ---
    model_architecture_params = {
        'hidden_dim': params['hidden_dim'],
        'encoding_dim': params['encoding_dim'],
        'n_hidden_layers': params['n_hidden_layers'],
        'dropout_rate': params['dropout_rate']
    }
    input_dim = features_df.shape[1]
    
    fold_aucs = []
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    for fold, (train_indices, val_indices) in enumerate(tscv.split(features_df)):
        # ... (数据准备部分无变化)
        purged_train_idx = train_indices[:-PURGE_SIZE]
        X_train_raw, y_train_raw = features_df.iloc[purged_train_idx], target_df.iloc[purged_train_idx]
        X_val_raw, y_val_raw = features_df.iloc[val_indices], target_df.iloc[val_indices]
        sw_train_raw = sample_weight_df.iloc[purged_train_idx]
        mean = np.nanmean(X_train_raw.values, axis=0); std = np.nanstd(X_train_raw.values, axis=0); std[std == 0] = 1.0
        X_train_scaled = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
        X_val_scaled = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(X_train_raw.isnull().values, dtype=torch.bool),
            torch.tensor(y_train_raw.values, dtype=torch.float32).unsqueeze(1),
            torch.tensor(sw_train_raw.values, dtype=torch.float32).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_features_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

        # <--- BUG FIX #2: 使用分离出的参数来创建模型 ---
        model = SupervisedAE(input_dim, **model_architecture_params)
        model.to(device)
        
        # <--- BUG FIX #3: 从主params字典中正确引用训练参数 ---
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        recon_loss_func = nn.MSELoss()
        pred_loss_func = nn.BCELoss(reduction='none')
        epochs, patience, epochs_no_improve, best_model_state, min_val_loss = 100, 7, 0, None, np.inf

        for epoch in range(epochs):
            model.train()
            for batch_features, batch_mask, batch_target, batch_weights in train_loader:
                batch_features, batch_mask, batch_target, batch_weights = batch_features.to(device), batch_mask.to(device), batch_target.to(device), batch_weights.to(device)
                recon_outputs, pred_outputs = model(batch_features)
                r_loss = recon_loss_func(recon_outputs[~batch_mask], batch_features[~batch_mask])
                p_loss = (pred_loss_func(pred_outputs, batch_target) * batch_weights).mean()
                
                # <--- BUG FIX #4: 从主params字典中正确引用训练参数 ---
                loss = params['recon_weight'] * r_loss + (1 - params['recon_weight']) * p_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model.predictor(model.encoder(val_features_tensor.to(device)))
                val_loss = nn.BCELoss()(val_preds, torch.tensor(y_val_raw.values, dtype=torch.float32).unsqueeze(1).to(device)).item()
            
            if val_loss < min_val_loss:
                min_val_loss, epochs_no_improve, best_model_state = val_loss, 0, copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
        
        if best_model_state:
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                val_preds = model.predictor(model.encoder(val_features_tensor.to(device))).cpu().numpy()
            fold_aucs.append(roc_auc_score(y_val_raw, val_preds))
        else:
            fold_aucs.append(0.5)

    return np.mean(fold_aucs)

# ================= 5. 最终特征生成函数 (同样已修复) =================
def generate_final_features(best_params, features_df, target_df, sample_weight_df, modern_df):
    print("\n--- 使用最优参数，启动“加权分类”版AI特征生成流程 ---")
    
    # <--- BUG FIX #5: 在生成函数中也应用同样的逻辑 ---
    model_architecture_params = {
        'hidden_dim': best_params['hidden_dim'],
        'encoding_dim': best_params['encoding_dim'],
        'n_hidden_layers': best_params['n_hidden_layers'],
        'dropout_rate': best_params['dropout_rate']
    }
    input_dim = features_df.shape[1]
    out_of_fold_ae_features = np.zeros((len(features_df), best_params['encoding_dim']))
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    for fold, (train_indices, val_indices) in enumerate(tscv.split(features_df)):
        print(f"\n{'='*25} 开始处理第 {fold + 1}/{N_SPLITS} 折 {'='*25}")
        # ... (数据准备无变化)
        purged_train_idx = train_indices[:-PURGE_SIZE]
        X_train_raw, y_train_raw = features_df.iloc[purged_train_idx], target_df.iloc[purged_train_idx]
        X_val_raw, y_val_raw = features_df.iloc[val_indices], target_df.iloc[val_indices]
        sw_train_raw = sample_weight_df.iloc[purged_train_idx]
        mean = np.nanmean(X_train_raw.values, axis=0); std = np.nanstd(X_train_raw.values, axis=0); std[std == 0] = 1.0
        X_train_scaled = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
        X_val_scaled = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)
        train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(X_train_raw.isnull().values, dtype=torch.bool), torch.tensor(y_train_raw.values, dtype=torch.float32).unsqueeze(1), torch.tensor(sw_train_raw.values, dtype=torch.float32).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_features_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        
        model = SupervisedAE(input_dim, **model_architecture_params) # <-- 使用修复后的参数
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate']) # <-- 正确引用
        recon_loss_func, pred_loss_func = nn.MSELoss(), nn.BCELoss(reduction='none')
        epochs, patience, epochs_no_improve, best_model_state, min_val_loss = 200, 10, 0, None, np.inf
        for epoch in range(epochs):
            model.train()
            for batch_features, batch_mask, batch_target, batch_weights in train_loader:
                # ... (训练循环内部逻辑无变化)
                batch_features, batch_mask, batch_target, batch_weights = batch_features.to(device), batch_mask.to(device), batch_target.to(device), batch_weights.to(device)
                recon_outputs, pred_outputs = model(batch_features)
                r_loss = recon_loss_func(recon_outputs[~batch_mask], batch_features[~batch_mask])
                p_loss = (pred_loss_func(pred_outputs, batch_target) * batch_weights).mean()
                loss = best_params['recon_weight'] * r_loss + (1 - best_params['recon_weight']) * p_loss # <-- 正确引用
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                val_preds = model.predictor(model.encoder(val_features_tensor.to(device)))
                val_loss = nn.BCELoss()(val_preds, torch.tensor(y_val_raw.values, dtype=torch.float32).unsqueeze(1).to(device)).item()
            if (epoch + 1) % 50 == 0: print(f'    周期 [{epoch+1}/{epochs}], 验证损失: {val_loss:.6f}')
            if val_loss < min_val_loss:
                min_val_loss, epochs_no_improve, best_model_state = val_loss, 0, copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'    -> 早停触发！最佳验证损失: {min_val_loss:.6f}'); break
        if best_model_state: model.load_state_dict(best_model_state)
        print(f'    -> 正在为第 {fold + 1} 折的验证集生成特征...')
        model.eval()
        with torch.no_grad():
            encoded_features = model.encoder(val_features_tensor.to(device))
            out_of_fold_ae_features[val_indices] = encoded_features.cpu().numpy()
            
    print(f"\n{'='*25} 所有折处理完毕 {'='*25}")
    ae_feature_names = [f'AE_{i}' for i in range(best_params['encoding_dim'])]
    ae_features_df = pd.DataFrame(out_of_fold_ae_features, columns=ae_feature_names)
    final_df = pd.concat([modern_df[['date_id']].reset_index(drop=True), ae_features_df], axis=1)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ 全新的、经过自动调优的“AI特种部队”已锻造完毕！")
    print(f"   新特征集已保存至 '{OUTPUT_FILE}'！")

# ================= 6. 主程序入口 (无变化) =================
if __name__ == "__main__":
    features_df, target_df, sample_weight_df, modern_df = load_and_prepare_data()
    if MODE == 'SEARCH':
        print(f"\n--- 启动 'SEARCH' 模式，开始为“加权分类AE”寻找最优超参数 ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, features_df, target_df, sample_weight_df), n_trials=N_TRIALS)
        print(f"\n{'='*25} Optuna搜索结束 {'='*25}")
        print(f"✅ 找到了分类AE的最优参数！最佳平均验证AUC: {study.best_value:.6f}")
        print("最优参数组合:")
        print(json.dumps(study.best_params, indent=4))
        with open(PARAMS_FILE, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        print(f"\n-> 最佳参数已保存至 '{PARAMS_FILE}'")
    elif MODE == 'GENERATE':
        if not os.path.exists(PARAMS_FILE):
            print(f"错误: 未找到参数文件 '{PARAMS_FILE}'。请先在 'SEARCH' 模式下运行。")
        else:
            print(f"\n--- 启动 'GENERATE' 模式，加载最优参数 ---")
            with open(PARAMS_FILE, 'r') as f:
                best_params = json.load(f)
            print("加载的最优参数组合:")
            print(json.dumps(best_params, indent=4))
            generate_final_features(best_params, features_df, target_df, sample_weight_df, modern_df)