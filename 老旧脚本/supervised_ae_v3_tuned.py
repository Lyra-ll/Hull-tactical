# =================================================================
# supervised_ae_v4_gpu.py (V1.1 - Rigorous Causal AE Tuning with GPU - Fixed)
# 目的: 利用GPU加速，使用Optuna和完整的交叉验证，自动寻找SupervisedAE的
#       最佳超参数组合，并用其生成更高质量的、严格遵守因果律的AI特征。
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
# 可选: 'SEARCH' (运行Optuna搜索), 'GENERATE' (使用最优参数生成最终特征)
MODE = 'GENERATE' 
N_TRIALS = 100 # 在SEARCH模式下，Optuna的尝试次数

# --- 自动设备选择 (GPU / CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 将使用设备: {device} ---")

# --- 文件路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
PARAMS_FILE = 'best_ae_params_rigorous_gpu.json' 
OUTPUT_FILE = 'train_v8_tuned_ae_features_rigorous_gpu.csv'

# --- 验证策略 ---
N_SPLITS = 5
PURGE_SIZE = 1
EMBARGO_SIZE = 40 

# ================= 2. 神经网络蓝图 (保持不变) =================
class SupervisedAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoding_dim, dropout_rate):
        super(SupervisedAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
        self.predictor = nn.Linear(encoding_dim, 1)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        prediction = self.predictor(encoded)
        return decoded, prediction

# ================= 3. 数据加载 (封装为函数) =================
def load_data():
    print("--- 正在加载数据... ---")
    df = pd.read_csv(RAW_DATA_FILE)
    modern_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)
    features_to_exclude = ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']
    feature_columns = [col for col in modern_df.columns if col not in features_to_exclude]
    features_df = modern_df[feature_columns]
    target_df = modern_df['forward_returns']
    return features_df, target_df, modern_df

# ================= 4. Optuna目标函数 `objective` (保持不变) =================
def objective(trial, features_df, target_df):
    params = {
        'hidden_dim': trial.suggest_int('hidden_dim', 32, 128),
        'encoding_dim': trial.suggest_int('encoding_dim', 8, 32),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'recon_weight': trial.suggest_float('recon_weight', 0.1, 0.9)
    }
    input_dim = features_df.shape[1]
    
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
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True); val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        model = SupervisedAE(input_dim, params['hidden_dim'], params['encoding_dim'], params['dropout_rate'])
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        epochs, patience, epochs_no_improve, min_val_loss = 100, 7, 0, np.inf

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

# ================= 5. 最终特征生成函数 (已修复) =================
def generate_final_features(best_params, features_df, target_df, modern_df):
    print("\n--- 使用最优参数，启动最终特征生成流程 ---")
    input_dim = features_df.shape[1]
    
    # 初始化一个空的Numpy数组来存储我们最终的、OOF（折外）的AI特征
    out_of_fold_ae_features = np.zeros((len(features_df), best_params['encoding_dim']))
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    
    for fold, (train_indices, val_indices) in enumerate(tscv.split(features_df)):
        print(f"\n{'='*25} 开始处理第 {fold + 1}/{N_SPLITS} 折 {'='*25}")
        
        # 净化与数据分割
        purged_train_idx = train_indices[:-PURGE_SIZE]
        X_train_raw, y_train_raw = features_df.iloc[purged_train_idx], target_df.iloc[purged_train_idx]
        X_val_raw, y_val_raw = features_df.iloc[val_indices], target_df.iloc[val_indices]
        
        print(f"    Train period: index {purged_train_idx[0]} to {purged_train_idx[-1]} (Size: {len(purged_train_idx)})")
        print(f"    Validation period: index {val_indices[0]} to {val_indices[-1]} (Size: {len(val_indices)})")

        # 数据准备 (与objective函数中完全一致)
        mean = np.nanmean(X_train_raw.values, axis=0); std = np.nanstd(X_train_raw.values, axis=0); std[std == 0] = 1.0
        X_train_scaled = np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0)
        X_val_scaled = np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0)
        train_mask = torch.tensor(X_train_raw.isnull().values, dtype=torch.bool); val_mask = torch.tensor(X_val_raw.isnull().values, dtype=torch.bool)
        train_features = torch.tensor(X_train_scaled, dtype=torch.float32); val_features = torch.tensor(X_val_scaled, dtype=torch.float32)
        train_target = torch.tensor(y_train_raw.fillna(0).values, dtype=torch.float32).unsqueeze(1); val_target = torch.tensor(y_val_raw.fillna(0).values, dtype=torch.float32).unsqueeze(1)
        train_dataset = TensorDataset(train_features, train_mask, train_target); val_dataset = TensorDataset(val_features, val_mask, val_target)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True); val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # 使用最优参数训练模型
        model = SupervisedAE(input_dim, best_params['hidden_dim'], best_params['encoding_dim'], best_params['dropout_rate'])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        epochs, patience, epochs_no_improve, min_val_loss, best_model_state = 200, 10, 0, np.inf, None

        # 完整的训练循环
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

        # 为当前折的验证集生成特征
        print(f'    -> 正在为第 {fold + 1} 折的验证集生成特征...')
        model.eval()
        with torch.no_grad():
            # 将验证集数据也移动到GPU
            val_features = val_features.to(device)
            encoded_features = model.encoder(val_features)
            out_of_fold_ae_features[val_indices] = encoded_features.cpu().numpy()

    # ================= 封装并储存战利品 =================
    print(f"\n{'='*25} 所有折处理完毕 {'='*25}")
    ae_feature_names = [f'AE_{i}' for i in range(best_params['encoding_dim'])]
    ae_features_df = pd.DataFrame(out_of_fold_ae_features, columns=ae_feature_names)

    final_df = pd.concat([
        modern_df[['date_id', 'forward_returns']].reset_index(drop=True),
        ae_features_df
    ], axis=1)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ 全新的、经过调优的“诚实”AI武器已锻造完毕！")
    print(f"   AI精华特征已成功提取并保存至 '{OUTPUT_FILE}'！")
    print(f"   新特征集的形状: {final_df.shape}")


# ================= 6. 主程序入口 =================
if __name__ == "__main__":
    features_df, target_df, modern_df = load_data()

    if MODE == 'SEARCH':
        print(f"\n--- 启动 'SEARCH' 模式 (GPU加速版)，开始Optuna超参数搜索 ---")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, features_df, target_df), n_trials=N_TRIALS)
        
        print(f"\n{'='*25} Optuna搜索结束 {'='*25}")
        print(f"✅ 找到了AE的最优参数！最佳平均验证损失: {study.best_value:.6f}")
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

