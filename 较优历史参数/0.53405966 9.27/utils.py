# utils.py
# =================================================================
# 项目通用工具函数库 (可复用的战术模块)
# =================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
import copy
import config
import random

def set_global_seeds(seed):
    """设置所有相关的随机种子以确保实验的可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 下面这两行对于保证CUDA操作的可复现性至关重要
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 1. 数据加载与准备 ---
def load_data(raw_file, start_date_id, missing_threshold):
    """
    加载原始数据，执行时间筛选和数据质量门禁。
    返回特征DataFrame, 目标Series, 和样本权重Series。
    # 定义一个函数，名为 load_data。
    # 它接收三个输入参数：
    # 1. raw_file: 原始数据文件的路径 (一个字符串)。
    # 2. start_date_id: 我们分析的起始日期ID (一个整数)。
    # 3. missing_threshold: 特征缺失率的容忍上限 (一个浮点数, 如0.2)。
    """
    print("--- 正在加载数据并执行“数据质量门禁” ---")
    raw_df = pd.read_csv(raw_file)
    #读取数据
    modern_df = raw_df[raw_df['date_id'] > start_date_id].copy().reset_index(drop=True)
    #截取start_date id之后的数据，并且这里.copy()的作用是与原数据脱钩，创造独立的副本。
    #reset_index作用是重置行索引，比如直接截取完毕后行索引就是从原数据里面的1055开始。
    y = (modern_df['forward_returns'] > 0).astype(int)
    #modern_df['forward_returns']会选取这一列的数据同时进行判断，如果大于0则结果返回True或者false
    #然后生成一列仅含True或者false的列表。
    #astype（int）会将其转换为int。
    sample_weight = modern_df['forward_returns'].abs().fillna(0)
    #定义样本权重，其值为未来收益的绝对值，然后把Na填为0
    feature_cols = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
    #定义一个列表包含所有候选人特征，排除几个没用的和具有未来数据的。
    missing_ratios = modern_df[feature_cols].isnull().mean()
    #定义缺失比例，计算方式是使用isnull，他会返回一个形状同样的dataframe但是仅含true和false
    #然后对1和0的列表求平均值，正好是缺失的比例。
    #然后返回一个series，含有每列的缺失比例。
    features_after_missing_check = missing_ratios[missing_ratios <= missing_threshold].index.tolist()
    
    # --- [战略升级] 根据RFE排名和配置参数，选择顶级特征 ---
    if config.N_TOP_FEATURES_TO_USE > 0:
        print(f"--- 正在根据RFE排名筛选 Top {config.N_TOP_FEATURES_TO_USE} 特征 ---")
        try:
            ranking = pd.read_csv(config.RANKING_FILE, index_col=0).squeeze("columns")
            # 1. pd.read_csv(...): 读取 config.py 中指定的 RANKING_FILE 文件。
            # 2. index_col=0: 告诉pandas，CSV文件的第一列应该作为新生成DataFrame的索引（Index）。因为我们的排名文件中第一列是特征名，所以这样做非常正确。
            # 3. .squeeze("columns"): 这是一个小技巧。因为读取后的DataFrame只有一个数据列（排名值），.squeeze("columns") 会把它“挤压”成一个Pandas Series，数据处理起来更方便。
            top_features_from_rfe = ranking.head(config.N_TOP_FEATURES_TO_USE).index.tolist()
            # 1. ranking.head(...): .head(N) 方法会选取Series最上面的N行。因为我们的排名文件是按重要性降序排列的，所以这正好是Top N特征。
            # 2. .index.tolist(): 和我们之前讨论过的一样，这会提取出这N个特征的索引（也就是特征名），并转换成一个标准的Python列表。
            # 取交集：确保选出的特征既是Top N，也通过了缺失值门禁
            final_features_to_use = sorted(list(set(features_after_missing_check) & set(top_features_from_rfe)))            
            # 这是非常关键和严谨的一步。一个特征最终被选中，必须同时满足两个条件：
            # 1. 它必须在 `top_features_from_rfe` 列表中（RFE认为它很重要）。
            # 2. 它必须在 `features_after_missing_check` 列表中（它的缺失值比例很低）。
            # set(...) & set(...) 操作正是用来计算这两个列表的“交集”。
            # sorted(list(...)) 则是我们为了保证100%可复现性加入的最终保障。

            print(f"    > RFE排名Top {config.N_TOP_FEATURES_TO_USE} 与 缺失值筛选后特征 的交集为 {len(final_features_to_use)} 个。")
        except FileNotFoundError:
            print(f"    > 警告: RFE排名文件 '{config.RANKING_FILE}' 未找到。将使用所有通过缺失值门禁的特征。")
            final_features_to_use = features_after_missing_check
    else:
        # 如果N_TOP_FEATURES_TO_USE设置为-1或0，则使用所有通过缺失值门禁的特征
        print("--- N_TOP_FEATURES_TO_USE 设置为-1或0，将使用所有通过缺失值门禁的特征 ---")
        final_features_to_use = features_after_missing_check

    X = modern_df[final_features_to_use]
    
    print(f"✅ 数据加载与最终筛选完成：模型将使用 {len(final_features_to_use)} 个特征。")
    return X, y, sample_weight, modern_df[['date_id']]
    # return 语句一次性将四个变量作为结果打包返回给调用它的地方（也就是 main.py）。
    # 这四个变量分别是：
    # 1. X: 最终的特征DataFrame。
    # 2. y: 目标Series (0或1)。
    # 3. sample_weight: 样本权重Series。
    # 4. modern_df[['date_id']]: date_id列。注意这里用了两层方括号 [['...']]，
    #    这能确保返回的是一个DataFrame（带列标题），而不是一个Series，通常在后续合并数据时更方便。
# --- 2. 交叉验证 ---
def get_cv_splitter(n_splits, embargo, gap):
    """返回一个配置好的TimeSeriesSplit对象。"""
    #接收三个参数，要分的折数，以及禁运的天数,每折的gap，
    return TimeSeriesSplit(n_splits=n_splits, gap=embargo) # 注意：sklearn的gap参数对应我们的embargo

# --- 3. 监督式AE模块 ---
# --- 3. 监督式AE模块 (V1.1 - 稳健性升级版) ---
# --- 3. 监督式AE模块 (V1.2 - ResNet架构修正版) ---
# [新] 定义一个可复用的残差块 (Residual Block)
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
        
        # --- Encoder (ResNet架构) ---
        encoder_layers = []
        encoder_layers.append(ResidualBlock(input_dim, hidden_dim, dropout_rate, bn))
        for _ in range(n_hidden_layers - 1):
            encoder_layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout_rate, bn))
        encoder_layers.append(nn.Linear(hidden_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # --- Decoder (ResNet架构) ---
        decoder_layers = []
        decoder_layers.append(ResidualBlock(encoding_dim, hidden_dim, dropout_rate, bn))
        for _ in range(n_hidden_layers - 1):
            decoder_layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout_rate, bn))
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # --- Predictor (V2 - Deeper ResNet架构) ---
        self.predictor = nn.Sequential(
            ResidualBlock(encoding_dim, hidden_dim // 2, dropout_rate, bn),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    # --- [核心修复] 添加缺失的 forward 方法 ---
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        prediction = self.predictor(encoded)
        return decoded, prediction
# [升级] train_fold_ae 函数现在包含多种子训练逻辑
def train_fold_ae(ae_params, X_train_np, y_train_np, sw_train_np, X_val_np, y_val_np, train_mask_np, seeds=[42, 2024, 777]):
    """
    [终极版 V1.5.1] 直接接收NumPy数组，在内部完成最后的Tensor转换和训练 (修正版)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            train_dataset, 
            batch_size=4096, 
            shuffle=True,
            num_workers=0, 
            pin_memory=True
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
                # --- [最终修正] 使用 y_val_np, 而不是不存在的 y_val.values ---
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

# --- 4. 预处理 (暂时留空，等待第三步决策) ---
# --- 4. 最终模型预处理 ---
def preprocess_for_lgbm(X_train, X_val):
    """
    为LightGBM进行稳健的、无数据泄露的预处理。
    该策略（ffill + median）已被证明是有效的。
    """
    # 复制以避免修改原始DataFrame
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    
    # 使用训练集信息来填充训练集
    X_train_processed = X_train_processed.ffill(limit=3)
    median_filler = X_train_processed.median() # [无泄露] 只从训练集计算中位数
    X_train_processed.fillna(median_filler, inplace=True)
    X_train_processed.fillna(0, inplace=True) # 最后的保障
    
    # [关键] 使用从训练集学到的信息来填充验证集
    X_val_processed = X_val_processed.ffill(limit=3)
    X_val_processed.fillna(median_filler, inplace=True)
    X_val_processed.fillna(0, inplace=True)
    
    return X_train_processed, X_val_processed