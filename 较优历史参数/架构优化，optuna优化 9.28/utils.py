# utils.py
# =================================================================
# 项目通用工具函数库 (V1.5.1 - 生产版)
# =================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
import copy
import random
import config

def set_global_seeds(seed):
    """设置所有相关的随机种子以确保实验的可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 1. 数据加载与准备 ---
def load_data(raw_file, start_date_id, missing_threshold):
    """
    [V1.5.1] 加载原始数据，执行时间筛选和基于RFE的Top N特征选择。
    """
    print("--- 正在加载数据并执行“数据质量门禁” ---")
    raw_df = pd.read_csv(raw_file)
    modern_df = raw_df[raw_df['date_id'] >= start_date_id].copy().reset_index(drop=True)
    
    y = (modern_df['forward_returns'] > 0).astype(int)
    sample_weight = modern_df['forward_returns'].abs().fillna(0)
    
    feature_cols = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
    
    # 缺失值门禁
    missing_ratios = modern_df[feature_cols].isnull().mean()
    features_after_missing_check = missing_ratios[missing_ratios <= missing_threshold].index.tolist()
    
    # 根据RFE排名和配置参数，选择顶级特征
    if config.N_TOP_FEATURES_TO_USE > 0:
        print(f"--- 正在根据RFE排名筛选 Top {config.N_TOP_FEATURES_TO_USE} 特征 ---")
        try:
            ranking = pd.read_csv(config.RANKING_FILE, index_col=0).squeeze("columns")
            top_features_from_rfe = ranking.head(config.N_TOP_FEATURES_TO_USE).index.tolist()
            final_features_to_use = sorted(list(set(features_after_missing_check) & set(top_features_from_rfe)))
            print(f"    > RFE排名与缺失值门禁的交集为 {len(final_features_to_use)} 个。")
        except FileNotFoundError:
            print(f"    > 警告: RFE排名文件 '{config.RANKING_FILE}' 未找到。将使用所有通过缺失值门禁的特征。")
            final_features_to_use = features_after_missing_check
    else:
        print("--- N_TOP_FEATURES_TO_USE 设置为-1或0，将使用所有通过缺失值门禁的特征 ---")
        final_features_to_use = features_after_missing_check

    # 只保留最终需要的列，包括目标和ID
    final_cols_to_keep = final_features_to_use + ['date_id', 'forward_returns']
    final_df = modern_df[final_cols_to_keep]
    X = final_df[final_features_to_use]
    
    print(f"✅ 数据加载与最终筛选完成：模型将使用 {len(final_features_to_use)} 个特征。")
    return X, y, sample_weight, final_df[['date_id']]

# --- 2. 交叉验证 ---
def get_cv_splitter(n_splits, embargo_size, purge_size):
    """[V1.5.1] 返回一个配置好的TimeSeriesSplit对象, 参数名更清晰"""
    # purge_size 在主循环中手动实现，这里不需要
    return TimeSeriesSplit(n_splits=n_splits, gap=embargo_size)

# --- 3. 监督式AE模块 (V1.2 - ResNet架构) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate, bn=True):
        super(ResidualBlock, self).__init__()
        self.main_path = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim) if bn else nn.Identity(),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.skip_connection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.main_path(x) + self.skip_connection(x)

class SupervisedAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoding_dim, n_hidden_layers, dropout_rate, bn=True):
        super(SupervisedAE, self).__init__()
        encoder_layers = []
        encoder_layers.append(ResidualBlock(input_dim, hidden_dim, dropout_rate, bn))
        for _ in range(n_hidden_layers - 1):
            encoder_layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout_rate, bn))
        encoder_layers.append(nn.Linear(hidden_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        decoder_layers.append(ResidualBlock(encoding_dim, hidden_dim, dropout_rate, bn))
        for _ in range(n_hidden_layers - 1):
            decoder_layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout_rate, bn))
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.predictor = nn.Sequential(
            ResidualBlock(encoding_dim, hidden_dim // 2, dropout_rate, bn),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        prediction = self.predictor(encoded)
        return decoded, prediction

def train_fold_ae(ae_params, X_train_np, y_train_np, sw_train_np, X_val_np, y_val_np, train_mask_np, seeds=[42, 2024, 777]):
    """[V1.5.1] 直接接收NumPy数组，在内部完成Tensor转换和训练"""
    mean = np.mean(X_train_np, axis=0)
    std = np.std(X_train_np, axis=0)
    std[std == 0] = 1.0
    
    X_train_scaled = (X_train_np - mean) / std
    X_val_scaled = (X_val_np - mean) / std

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), 
                                torch.tensor(train_mask_np, dtype=torch.bool),
                                torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1), 
                                torch.tensor(sw_train_np, dtype=torch.float32).unsqueeze(1))
    
    trained_models = []
    for seed in seeds:
        set_global_seeds(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=4096, shuffle=True,
            num_workers=0, pin_memory=True
        )
        model_architecture_params = {
            'hidden_dim': ae_params['hidden_dim'], 
            'encoding_dim': ae_params['encoding_dim'], 
            'n_hidden_layers': ae_params['n_hidden_layers'], 
            'dropout_rate': ae_params['dropout_rate']
        }
        if 'bn' in ae_params: model_architecture_params['bn'] = ae_params['bn']
        
        input_dim = X_train_np.shape[1]
        model = SupervisedAE(input_dim, **model_architecture_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=ae_params['learning_rate'])
        recon_loss_func, pred_loss_func = nn.MSELoss(), nn.BCELoss(reduction='none')
        epochs, patience, epochs_no_improve, best_seed_state, min_seed_loss = 100, 7, 0, None, np.inf
        
        for epoch in range(epochs):
            model.train()
            for batch_features, batch_mask, batch_target, batch_weights in train_loader:
                batch_features, batch_mask, batch_target, batch_weights = batch_features.to(device), batch_mask.to(device), batch_target.to(device), batch_weights.to(device)
                recon_outputs, pred_outputs = model(batch_features)
                r_loss = recon_loss_func(recon_outputs[~batch_mask], batch_features[~batch_mask])
                p_loss = (pred_loss_func(pred_outputs, batch_target) * batch_weights).mean()
                loss = ae_params['recon_weight'] * r_loss + (1 - ae_params['recon_weight']) * p_loss 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
                _, val_preds = model(val_tensor)
                y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1).to(device)
                val_bce_loss = nn.BCELoss()(val_preds, y_val_tensor).item()
                
            if val_bce_loss < min_seed_loss:
                min_seed_loss, epochs_no_improve, best_seed_state = val_bce_loss, 0, copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
        
        if best_seed_state:
            final_model_for_seed = SupervisedAE(input_dim, **model_architecture_params).to(device)
            final_model_for_seed.load_state_dict(best_seed_state)
            trained_models.append(final_model_for_seed.eval())
            
    return trained_models

# --- 4. 最终模型预处理 ---
def preprocess_for_lgbm(X_train, X_val):
    """为LightGBM进行稳健的、无数据泄露的预处理。"""
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    X_train_processed = X_train_processed.ffill(limit=3)
    median_filler = X_train_processed.median()
    X_train_processed.fillna(median_filler, inplace=True)
    X_train_processed.fillna(0, inplace=True)
    X_val_processed = X_val_processed.ffill(limit=3)
    X_val_processed.fillna(median_filler, inplace=True)
    X_val_processed.fillna(0, inplace=True)
    
    return X_train_processed, X_val_processed