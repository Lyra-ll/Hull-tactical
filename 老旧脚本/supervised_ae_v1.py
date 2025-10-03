# ================= 核心：专业化训练 + 最终特征提取 (无泄露最终版 V4) =================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import copy
import time
from sklearn.model_selection import TimeSeriesSplit #添加时序交叉验证

# ================= 1. 全局配置与神经网络蓝图 =================
ENCODING_DIM = 16
BATCH_SIZE = 64
N_SPLITS = 5

class SupervisedAE(nn.Module):
    def __init__(self, input_dim, encoding_dim=ENCODING_DIM, dropout_rate=0.2):
        super(SupervisedAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, input_dim)
        )
        self.predictor = nn.Linear(encoding_dim, 1)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        prediction = self.predictor(encoded)
        return decoded, prediction

# ================= 2. 数据加载与初步准备 =================
print("--- 步骤1：加载“现代史”完整数据 ---")
df = pd.read_csv('train_v3_featured_raw.csv')
modern_history_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)

features_to_exclude = ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']
feature_columns = [col for col in modern_history_df.columns if col not in features_to_exclude]
features_df = modern_history_df[feature_columns]
target_df = modern_history_df['forward_returns'] # 修复隐患4: 此处暂时保留NaN

print(f"数据加载完毕，共 {len(modern_history_df)} 条记录。")
input_dim = len(feature_columns)

# ================= 3. K-Fold 交叉验证流程 =================
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
loss_reconstruction = nn.MSELoss()
loss_prediction = nn.MSELoss()
out_of_fold_ae_features = np.zeros((len(modern_history_df), ENCODING_DIM))
#先创建一个numpy数组，
print(f"--- 步骤2：K-Fold已准备就绪，将进行 {N_SPLITS} 折交叉验证 ---")

for fold, (train_indices, val_indices) in enumerate(kf.split(features_df)):
    start_time = time.time()
    print(f"\n{'='*25} 开始处理第 {fold + 1}/{N_SPLITS} 折 {'='*25}")

    # a. 获取当前折的、未经处理的训练集和验证集 (Pandas格式)
    X_train_raw, y_train_raw = features_df.iloc[train_indices], target_df.iloc[train_indices]
    X_val_raw, y_val_raw = features_df.iloc[val_indices], target_df.iloc[val_indices]

    # b. “诚实”缩放：只在当前折的“训练集”上计算均值和标准差
    print("    > 正在为当前折计算“诚实”的缩放标准...")
    mean = np.nanmean(X_train_raw.values, axis=0)
    std = np.nanstd(X_train_raw.values, axis=0)
    std[std == 0] = 1.0

    # c. “无泄露”地应用缩放，并进行“真实均值”填充
    print("    > 正在应用“无泄露”缩放并进行均值填充...")
    X_train_scaled_with_nan = (X_train_raw.values - mean) / std
    X_val_scaled_with_nan = (X_val_raw.values - mean) / std
    X_train_scaled = np.nan_to_num(X_train_scaled_with_nan, nan=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled_with_nan, nan=0.0)
    
    # d. 为当前折创建专属的 Tensors 和 DataLoaders
    train_mask = torch.tensor(X_train_raw.isnull().values, dtype=torch.bool)
    train_features = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_target = torch.tensor(y_train_raw.fillna(0).values, dtype=torch.float32).unsqueeze(1) # 修复隐患4: 在这里填充
    
    val_mask = torch.tensor(X_val_raw.isnull().values, dtype=torch.bool)
    val_features = torch.tensor(X_val_scaled, dtype=torch.float32)
    val_target = torch.tensor(y_val_raw.fillna(0).values, dtype=torch.float32).unsqueeze(1) # 修复隐患4: 在这里填充

    train_dataset_fold = TensorDataset(train_features, train_mask, train_target)
    val_dataset_fold = TensorDataset(val_features, val_mask, val_target)
    
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, shuffle=False)
    
    # e. 训练当前折的模型
    model_fold = SupervisedAE(input_dim)
    optimizer = torch.optim.Adam(model_fold.parameters(), lr=0.001)
    epochs, recon_weight, patience, epochs_no_improve, min_val_loss, best_model_state = 200, 0.5, 10, 0, np.inf, None

    for epoch in range(epochs):
        model_fold.train()
        # ... (内部训练循环无变化)
        total_train_loss = 0
        for batch_features, batch_mask, batch_target in train_loader_fold:
            recon_outputs, pred_outputs = model_fold(batch_features)
            non_nan_mask = ~batch_mask
            r_loss = loss_reconstruction(recon_outputs[non_nan_mask], batch_features[non_nan_mask])
            p_loss = loss_prediction(pred_outputs, batch_target)
            loss = recon_weight * r_loss + (1 - recon_weight) * p_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_train_loss += loss.item()
        
        model_fold.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_features, batch_mask, batch_target in val_loader_fold:
                recon_outputs, pred_outputs = model_fold(batch_features)
                non_nan_mask = ~batch_mask
                r_loss = loss_reconstruction(recon_outputs[non_nan_mask], batch_features[non_nan_mask])
                p_loss = loss_prediction(pred_outputs, batch_target)
                loss = recon_weight * r_loss + (1 - recon_weight) * p_loss
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader_fold)
        if (epoch + 1) % 50 == 0: print(f'    周期 [{epoch+1}/{epochs}], 验证损失: {avg_val_loss:.6f}')
        if avg_val_loss < min_val_loss:
            min_val_loss, epochs_no_improve, best_model_state = avg_val_loss, 0, copy.deepcopy(model_fold.state_dict())
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f'    -> 早停触发！最佳验证损失: {min_val_loss:.6f}')
            break
    
    if best_model_state: model_fold.load_state_dict(best_model_state)
    
    # f. 生成并存储“无泄露”特征
    print(f'    -> 正在为第 {fold + 1} 折的验证集生成特征...')
    model_fold.eval()
    with torch.no_grad():
        encoded_features = model_fold.encoder(val_features)
        out_of_fold_ae_features[val_indices] = encoded_features.cpu().numpy()

    fold_time = time.time() - start_time
    print(f"    -> 第 {fold + 1} 折处理完毕，耗时: {fold_time:.2f} 秒")

# ================= 4. 封装并储存最终战利品 =================
print(f"\n{'='*25} 所有折处理完毕 {'='*25}")
print("--- 步骤3：开始封装并储存我们的最终战利品 ---")
ae_feature_names = [f'AE_{i}' for i in range(ENCODING_DIM)]
ae_features_df = pd.DataFrame(out_of_fold_ae_features, columns=ae_feature_names)

# 修复隐患3: 使用 reset_index 确保合并的绝对安全
final_df = pd.concat([
    modern_history_df[['date_id', 'forward_returns']].reset_index(drop=True),
    ae_features_df
], axis=1)

# 修复隐患2: 使用新的、更能体现版本迭代的文件名
output_filename = 'train_v6_kfold_leakfree_ae_features.csv'
final_df.to_csv(output_filename, index=False)

print(f"\n✅ 我们的“绝对诚实”的次世代武器已锻造完毕！")
print(f"   AI精华特征已成功提取并保存至 '{output_filename}'！")
print(f"   新特征集的形状: {final_df.shape}")