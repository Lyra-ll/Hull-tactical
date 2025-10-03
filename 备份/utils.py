# utils.py
# =================================================================
# 项目通用工具函数库 V2.6 (终极统一修复版)
# =================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
import copy
import random

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cv_splitter(n_splits, embargo_size, purge_size):
    return TimeSeriesSplit(n_splits=n_splits, gap=embargo_size)

def load_data(raw_file, start_date_id):
    print(f"--- 正在加载数据: {raw_file} ---")
    raw_df = pd.read_csv(raw_file)
    print(f"  > 原始数据 {len(raw_df)} 行。")
    modern_df = raw_df[raw_df['date_id'] >= start_date_id].copy().reset_index(drop=True)
    print(f"  > 筛选 date_id >= {start_date_id} 后，剩余 {len(modern_df)} 行。")
    return modern_df

# [核心修复] 引入职责分离的预处理器函数
def get_preprocessor_params(X_train):
    """
    [V5 - 稳健版] 从训练数据学习填充和缩放所需的核心参数。
    简化了填充逻辑，使其更透明，风险更低。
    """
    print("    > [预处理器V5] 正在从训练集学习参数...")
    params = {}
    
    # 1. 学习中位数，用于后续的单步填充
    params['median_filler'] = X_train.median()
    print("      - 已学习中位数 (median_filler)")
    
    # 2. 【关键简化】在一个仅用中位数填充的、更接近原始分布的数据上学习缩放参数
    X_train_for_scaling = X_train.fillna(params['median_filler'])
    
    params['mean'] = X_train_for_scaling.mean()
    params['std'] = X_train_for_scaling.std()
    print("      - 已学习均值 (mean) 和标准差 (std)")

    # 3. 依然保留至关重要的“除零保护”
    params['std'][params['std'] == 0] = 1.0
    print("      - 已执行“除零保护”")
    
    return params

def apply_preprocessor(df, params):
    """
    [V5 - 稳健版] 应用预处理参数。
    填充逻辑与参数学习过程完全对齐。
    """
    df_proc = df.copy()
    
    # 1. 【关键简化】执行与参数学习时完全一致的“单步中位数填充”
    df_proc.fillna(params['median_filler'], inplace=True)
    
    # 2. 标准化
    df_scaled = (df_proc - params['mean']) / params['std']
    
    # 3. 返回值 (注意：此版本df_proc是已填充但未标准化的)
    #    为了与你现有代码兼容，我们依然返回两个DataFrame
    #    但df_proc现在仅经过了中位数填充
    return df_proc, df_scaled

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    '''
    残差块
    '''
    def __init__(self, input_dim, output_dim, dropout_rate, bn=True):
        super(ResidualBlock, self).__init__()
        #调用父类nn module初始化
        self.main_path = nn.Sequential(
            #容器放入元素
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim) if bn else nn.Identity(),
            #如果bn打开的时候启用，在每次训练迭代的时候将上一层的输出值重新缩放到均值为0方差为1的分布
            #稳定模型训练过程，防止过拟合  nn.identity()表示啥也不干
            nn.SiLU(),
            #(x * sigmoid(x))，代替relu
            nn.Dropout(dropout_rate),
        )
        self.skip_connection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        #让原始输入跳过主路径的复杂变换
        #如果输入和输出维度不同则需要一个简单的线性层将输入维度调整到output_dim
    def forward(self, x):
        return self.main_path(x) + self.skip_connection(x)

class SupervisedAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoding_dim, n_hidden_layers, dropout_rate, bn=True, n_targets=1):
        super(SupervisedAE, self).__init__()
        encoder_layers = [ResidualBlock(input_dim, hidden_dim, dropout_rate, bn)]
        for _ in range(n_hidden_layers - 1):
            encoder_layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout_rate, bn))
        encoder_layers.append(nn.Linear(hidden_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = [ResidualBlock(encoding_dim, hidden_dim, dropout_rate, bn)]
        for _ in range(n_hidden_layers - 1):
            decoder_layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout_rate, bn))
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.predictor = nn.Sequential(
            ResidualBlock(encoding_dim, hidden_dim // 2, dropout_rate, bn),
            nn.Linear(hidden_dim // 2, n_targets),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        prediction = self.predictor(encoded)
        return decoded, prediction

def train_fold_ae(ae_params, X_train_scaled_np, y_train_np, sw_train_np, X_val_scaled_np, y_val_np, sw_val_np, train_mask_np, X_val_mask_np, seeds=[42, 2024]):
    """接收已经标准化处理过的NumPy数组，并执行模型训练"""
    train_dataset = TensorDataset(torch.tensor(X_train_scaled_np, dtype=torch.float32), 
                                torch.tensor(train_mask_np, dtype=torch.bool),
                                torch.tensor(y_train_np, dtype=torch.float32), 
                                torch.tensor(sw_train_np, dtype=torch.float32).unsqueeze(1))
    
    trained_models = []
    if device.type == 'cuda':
        print(f"    > [AE] 使用CUDA设备: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    for seed in seeds:
        set_global_seeds(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
        
        model_params = { 'hidden_dim': ae_params['hidden_dim'], 'encoding_dim': ae_params['encoding_dim'], 'n_hidden_layers': ae_params['n_hidden_layers'], 'dropout_rate': ae_params['dropout_rate'], 'bn': ae_params.get('bn', True) }
        
        input_dim = X_train_scaled_np.shape[1]
        n_targets = y_train_np.shape[1] if y_train_np.ndim > 1 else 1
        model = SupervisedAE(input_dim, **model_params, n_targets=n_targets).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=ae_params['learning_rate'])
        
        # --- [核心修改] ---
        # 旧的评分标准: BCELoss 用于比较 概率 vs 0/1硬标签
        # recon_loss_fn, pred_loss_fn = nn.MSELoss(), nn.BCELoss(reduction='none')
        
        # 新的评分标准: MSELoss 用于比较 概率 vs 0-1软标签
        # 我们现在比较的是两个连续值之间的差距，所以使用均方误差损失。
        recon_loss_fn, pred_loss_fn = nn.MSELoss(), nn.MSELoss(reduction='none')
        # --- [修改结束] ---
        
        epochs, patience, epochs_no_improve, best_state, min_loss = 100, 7, 0, None, np.inf
        # [健壮性修复] 从参数中读取损失权重，避免硬编码
        loss_weights = torch.tensor(ae_params.get('loss_weights', [0.5, 0.3, 0.2]), device=device) 
        
        for epoch in range(epochs):
            model.train()
            for features, mask, target, weights in train_loader:
                features = features.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                recon, pred = model(features)
                r_loss = recon_loss_fn(recon[~mask], features[~mask])
                p_loss = ( (pred_loss_fn(pred, target) * loss_weights) * weights ).mean()
                loss = ae_params['recon_weight'] * r_loss + (1 - ae_params['recon_weight']) * p_loss 
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            model.eval()
            with torch.no_grad():
                val_tensor = torch.tensor(X_val_scaled_np, dtype=torch.float32).to(device, non_blocking=True)
                y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).to(device, non_blocking=True)
                sw_val_tensor = torch.tensor(sw_val_np, dtype=torch.float32).unsqueeze(1).to(device, non_blocking=True)
                val_mask_tensor = torch.tensor(X_val_mask_np, dtype=torch.bool).to(device, non_blocking=True)

                val_recon, val_preds = model(val_tensor)
                val_r_loss = recon_loss_fn(val_recon[~val_mask_tensor], val_tensor[~val_mask_tensor])
                # [修复] 使用与主损失函数完全相同的加权逻辑来计算验证损失
                val_p_loss = ( (pred_loss_fn(val_preds, y_val_tensor) * loss_weights) * sw_val_tensor ).mean()
                val_loss = ae_params['recon_weight'] * val_r_loss.item() + (1 - ae_params['recon_weight']) * val_p_loss.item()
                
            if val_loss < min_loss:
                min_loss, epochs_no_improve, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience: break
        
        if best_state:
            final_model = SupervisedAE(input_dim, **model_params, n_targets=n_targets).to(device)
            final_model.load_state_dict(best_state)
            trained_models.append(final_model.eval())
            
    return trained_models