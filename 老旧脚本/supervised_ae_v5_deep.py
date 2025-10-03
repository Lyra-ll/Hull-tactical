# =================================================================
# supervised_ae_v5_deep.py (V1.0 - Deep AE Tuning with GPU)
# 目的: 引入网络深度作为超参数，让Optuna自动寻找最佳的网络结构，
#       以期通过更深层次的抽象能力，创造出超越原始特征的AI神髓。
# =================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
import optuna
import copy
import time
import json
import os
import warnings

warnings.filterwarnings('ignore')

# ================= 1. 全局配置与模式切换 =================
MODE = 'GENERATE'
N_TRIALS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 将使用设备: {device} ---")

# --- 文件路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
PARAMS_FILE = 'best_ae_params_v5_deep.json'
OUTPUT_FILE = 'train_v9_deep_ae_features.csv'

# --- 验证策略 ---
N_SPLITS = 5
PURGE_SIZE = 1
EMBARGO_SIZE = 40 

# ================= 2. 动态神经网络蓝图 (核心升级) =================
class SupervisedAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoding_dim, n_hidden_layers, dropout_rate):
        super(SupervisedAE, self).__init__()
        
        # --- Encoder ---
        encoder_layers = []
        # First layer
        encoder_layers.append(nn.Linear(input_dim, hidden_dim))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Dropout(dropout_rate))
        # Additional hidden layers
        for _ in range(n_hidden_layers - 1):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
        # Bottleneck layer
        encoder_layers.append(nn.Linear(hidden_dim, encoding_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder (symmetrical to encoder) ---
        decoder_layers = []
        # First layer (from bottleneck)
        decoder_layers.append(nn.Linear(encoding_dim, hidden_dim))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Dropout(dropout_rate))
        # Additional hidden layers
        for _ in range(n_hidden_layers - 1):
            decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
        # Output layer
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # --- Predictor ---
        self.predictor = nn.Linear(encoding_dim, 1)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        prediction = self.predictor(encoded)
        return decoded, prediction

# ================= 3. 数据加载 (封装为函数) =================
def load_data():
    # ... (与V4版完全相同, 此处省略以保持简洁)
    print("--- 正在加载数据... ---")
    df = pd.read_csv(RAW_DATA_FILE)
    modern_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)
    features_to_exclude = ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']
    feature_columns = [col for col in modern_df.columns if col not in features_to_exclude]
    features_df = modern_df[feature_columns]
    target_df = modern_df['forward_returns']
    return features_df, target_df, modern_df
    
# ================= 4. Optuna目标函数 `objective` (核心升级) =================
def objective(trial, features_df, target_df):
    params = {
        # --- 新增: 网络深度超参数 ---
        'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 3), # 让Optuna在1到3个隐藏层之间选择
        # ---------------------------
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 256),
        'encoding_dim': trial.suggest_int('encoding_dim', 16, 64),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'recon_weight': trial.suggest_float('recon_weight', 0.05, 0.5) # 根据上次经验，将重心放在预测上
    }
    input_dim = features_df.shape[1]
    
    # ... (完整的5折交叉验证循环与V4版完全相同, 此处省略以保持简洁)
    fold_losses = []
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    for fold, (train_indices, val_indices) in enumerate(tscv.split(features_df)):
        purged_train_idx = train_indices[:-PURGE_SIZE]
        X_train_raw, y_train_raw = features_df.iloc[purged_train_idx], target_df.iloc[purged_train_idx]
        X_val_raw, y_val_raw = features_df.iloc[val_indices], target_df.iloc[val_indices]
        mean = np.nanmean(X_train_raw.values, axis=0); std = np.nanstd(X_train_raw.values, axis=0); std[std == 0] = 1.0
        X_train_scaled = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
        X_val_scaled = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)
        train_mask = torch.tensor(X_train_raw.isnull().values, dtype=torch.bool); val_mask = torch.tensor(X_val_raw.isnull().values, dtype=torch.bool)
        train_features = torch.tensor(X_train_scaled, dtype=torch.float32); val_features = torch.tensor(X_val_scaled, dtype=torch.float32)
        train_target = torch.tensor(y_train_raw.fillna(0).values, dtype=torch.float32).unsqueeze(1); val_target = torch.tensor(y_val_raw.fillna(0).values, dtype=torch.float32).unsqueeze(1)
        train_dataset = TensorDataset(train_features, train_mask, train_target); val_dataset = TensorDataset(val_features, val_mask, val_target)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True); val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        model = SupervisedAE(input_dim, params['hidden_dim'], params['encoding_dim'], params['n_hidden_layers'], params['dropout_rate'])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        epochs, patience, epochs_no_improve, min_val_loss = 200, 10, 0, np.inf
        for epoch in range(epochs):
            model.train()
            for batch_features, batch_mask, batch_target in train_loader:
                batch_features, batch_mask, batch_target = batch_features.to(device), batch_mask.to(device), batch_target.to(device)
                recon_outputs, pred_outputs = model(batch_features)
                non_nan_mask = ~batch_mask
                r_loss = nn.MSELoss()(recon_outputs[non_nan_mask], batch_features[non_nan_mask])
                p_loss = nn.MSELoss()(pred_outputs, batch_target)
                loss = params['recon_weight'] * r_loss + (1 - params['recon_weight']) * p_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_features, batch_mask, batch_target in val_loader:
                    batch_features, batch_mask, batch_target = batch_features.to(device), batch_mask.to(device), batch_target.to(device)
                    recon_outputs, pred_outputs = model(batch_features)
                    non_nan_mask = ~batch_mask
                    r_loss = nn.MSELoss()(recon_outputs[non_nan_mask], batch_features[non_nan_mask])
                    p_loss = nn.MSELoss()(pred_outputs, batch_target)
                    loss = params['recon_weight'] * r_loss + (1 - params['recon_weight']) * p_loss
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve == patience:
                break
        fold_losses.append(min_val_loss)
    return np.mean(fold_losses)

# ================= 5. 最终特征生成函数 (与V4修复版逻辑相同) =================
def generate_final_features(best_params, features_df, target_df, modern_df):
    # ... (与V4修复版完全相同, 此处省略以保持简洁)
    print("\n--- 使用最优参数，启动最终特征生成流程 ---")
    input_dim = features_df.shape[1]
    out_of_fold_ae_features = np.zeros((len(features_df), best_params['encoding_dim']))
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    for fold, (train_indices, val_indices) in enumerate(tscv.split(features_df)):
        print(f"\n{'='*25} 开始处理第 {fold + 1}/{N_SPLITS} 折 {'='*25}")
        purged_train_idx = train_indices[:-PURGE_SIZE]
        X_train_raw, y_train_raw = features_df.iloc[purged_train_idx], target_df.iloc[purged_train_idx]
        X_val_raw, y_val_raw = features_df.iloc[val_indices], target_df.iloc[val_indices]
        print(f"    Train period: index {purged_train_idx[0]} to {purged_train_idx[-1]} (Size: {len(purged_train_idx)})")
        print(f"    Validation period: index {val_indices[0]} to {val_indices[-1]} (Size: {len(val_indices)})")
        mean = np.nanmean(X_train_raw.values, axis=0); std = np.nanstd(X_train_raw.values, axis=0); std[std == 0] = 1.0
        X_train_scaled = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
        X_val_scaled = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)
        train_mask = torch.tensor(X_train_raw.isnull().values, dtype=torch.bool); val_mask = torch.tensor(X_val_raw.isnull().values, dtype=torch.bool)
        train_features = torch.tensor(X_train_scaled, dtype=torch.float32); val_features = torch.tensor(X_val_scaled, dtype=torch.float32)
        train_target = torch.tensor(y_train_raw.fillna(0).values, dtype=torch.float32).unsqueeze(1); val_target = torch.tensor(y_val_raw.fillna(0).values, dtype=torch.float32).unsqueeze(1)
        train_dataset = TensorDataset(train_features, train_mask, train_target); val_dataset = TensorDataset(val_features, val_mask, val_target)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True); val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        model = SupervisedAE(input_dim, best_params['hidden_dim'], best_params['encoding_dim'], best_params['n_hidden_layers'], best_params['dropout_rate'])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        epochs, patience, epochs_no_improve, min_val_loss, best_model_state = 200, 10, 0, np.inf, None
        for epoch in range(epochs):
            model.train()
            for batch_features, batch_mask, batch_target in train_loader:
                batch_features, batch_mask, batch_target = batch_features.to(device), batch_mask.to(device), batch_target.to(device)
                recon_outputs, pred_outputs = model(batch_features)
                non_nan_mask = ~batch_mask
                r_loss = nn.MSELoss()(recon_outputs[non_nan_mask], batch_features[non_nan_mask])
                p_loss = nn.MSELoss()(pred_outputs, batch_target)
                loss = best_params['recon_weight'] * r_loss + (1 - best_params['recon_weight']) * p_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_features, batch_mask, batch_target in val_loader:
                    batch_features, batch_mask, batch_target = batch_features.to(device), batch_mask.to(device), batch_target.to(device)
                    recon_outputs, pred_outputs = model(batch_features)
                    non_nan_mask = ~batch_mask
                    r_loss = nn.MSELoss()(recon_outputs[non_nan_mask], batch_features[non_nan_mask])
                    p_loss = nn.MSELoss()(pred_outputs, batch_target)
                    loss = best_params['recon_weight'] * r_loss + (1 - best_params['recon_weight']) * p_loss
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            if (epoch + 1) % 50 == 0: print(f'    周期 [{epoch+1}/{epochs}], 验证损失: {avg_val_loss:.6f}')
            if avg_val_loss < min_val_loss:
                min_val_loss, epochs_no_improve, best_model_state = avg_val_loss, 0, copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'    -> 早停触发！最佳验证损失: {min_val_loss:.6f}')
                break
        if best_model_state: model.load_state_dict(best_model_state)
        print(f'    -> 正在为第 {fold + 1} 折的验证集生成特征...')
        model.eval()
        with torch.no_grad():
            val_features = val_features.to(device)
            encoded_features = model.encoder(val_features)
            out_of_fold_ae_features[val_indices] = encoded_features.cpu().numpy()
    print(f"\n{'='*25} 所有折处理完毕 {'='*25}")
    ae_feature_names = [f'AE_{i}' for i in range(best_params['encoding_dim'])]
    ae_features_df = pd.DataFrame(out_of_fold_ae_features, columns=ae_feature_names)
    final_df = pd.concat([modern_df[['date_id', 'forward_returns']].reset_index(drop=True), ae_features_df], axis=1)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ 全新的、经过调优的“深度AI武器”已锻造完毕！")
    print(f"   AI精华特征已成功提取并保存至 '{OUTPUT_FILE}'！")
    print(f"   新特征集的形状: {final_df.shape}")

# ================= 6. 主程序入口 (与V4版完全相同) =================
if __name__ == "__main__":
    features_df, target_df, modern_df = load_data()

    if MODE == 'SEARCH':
        print(f"\n--- 启动 'SEARCH' 模式 (深度探索版)，开始Optuna超参数搜索 ---")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, features_df, target_df), n_trials=N_TRIALS)
        
        print(f"\n{'='*25} Optuna深度搜索结束 {'='*25}")
        print(f"✅ 找到了深度AE的最优参数！最佳平均验证损失: {study.best_value:.6f}")
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
            generate_final_features(best_params, features_df, target_df, modern_df)
