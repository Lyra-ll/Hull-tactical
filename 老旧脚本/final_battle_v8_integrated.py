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

# =================================================================
# final_battle_v8_integrated.py (集成化作战平台 V1.0)
# 目的: 彻底根除方法论泄露，建立一个因果一致的、集成了
#       AE特征工程和LightGBM训练的单一验证流水线。
# =================================================================

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. 全局配置 (已合并与更新) =================
# --- 文件路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
# 使用我们之前最好的LGBM参数作为此次的基准参数
PARAMS_FILE = 'best_params_ORIGINAL_PLUS_AI.json' 
OOF_OUTPUT_FILE = 'oof_predictions_v8_integrated.csv' 

# --- [新] 数据质量门禁 ---
MISSING_THRESHOLD = 0.20 # 剔除缺失率超过20%的特征

# --- 验证策略 (保持不变) ---
N_SPLITS = 5; PURGE_SIZE = 1; EMBARGO_SIZE = 40 
ANALYSIS_START_DATE_ID = 1055

# ================= 2. AI兵工厂：监督式AE模块 (从 supervised_ae 中移入) =================
class SupervisedAE(nn.Module):
    # (网络结构暂时保持不变，后续第二阶段再进行升级)
    def __init__(self, input_dim, hidden_dim=128, encoding_dim=32, n_hidden_layers=2, dropout_rate=0.2):
        super(SupervisedAE, self).__init__()
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
        # --- Predictor (我们的“捷径”) ---
        self.predictor = nn.Sequential(nn.Linear(encoding_dim, 1), nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        prediction = self.predictor(encoded)
        return decoded, prediction

# ================= 3. [新] 统一的预处理与AE训练函数 =================
def train_fold_ae(X_train_raw, y_train_raw, sw_train_raw, X_val_raw, y_val_raw):
    """
    在当前CV折内部，完成从数据预处理到AE模型训练的所有步骤。
    返回一个训练好的、可用于特征生成的AE模型。
    """
    # --- 步骤1: 稳健的填充与缩放 (只在Fold内部进行) ---
    X_train_filled = X_train_raw.ffill(limit=3)
    median_filler = X_train_filled.median() # 只从训练集计算中位数
    X_train_filled.fillna(median_filler, inplace=True); X_train_filled.fillna(0, inplace=True)
    
    X_val_filled = X_val_raw.ffill(limit=3)
    X_val_filled.fillna(median_filler, inplace=True); X_val_filled.fillna(0, inplace=True)

    mean = X_train_filled.mean(axis=0).values # 只从训练集计算均值
    std = X_train_filled.std(axis=0).values   # 只从训练集计算标准差
    std[std == 0] = 1.0

    X_train_scaled = (X_train_filled.values - mean) / std
    X_val_scaled = (X_val_filled.values - mean) / std

    # --- 步骤2: 准备DataLoader ---
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(X_train_filled.isnull().values, dtype=torch.bool), # 使用填充前的数据来创建mask
        torch.tensor(y_train_raw.values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(sw_train_raw.values, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    # --- 步骤3: 模型训练 ---
    input_dim = X_train_raw.shape[1]
    # (暂时硬编码AE参数，后续可加入Optuna)
    model = SupervisedAE(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
            loss = 0.5 * r_loss + 0.5 * p_loss # 暂时使用均等权重
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        # --- 智能早停：只监控BCE损失 ---
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
            #print(f'    -> AE早停触发！最佳验证BCE损失: {min_val_loss:.6f}')
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model.eval() # 返回训练好的最佳模型

# ================= 4. 主执行模块 (全新集成化流程) =================
if __name__ == '__main__':
    start_time = time.time()
    print("--- 步骤1：加载数据并执行“数据质量门禁” ---")
    raw_df = pd.read_csv(RAW_DATA_FILE)
    with open(PARAMS_FILE, 'r') as f: lgbm_params = json.load(f)
    
    modern_df = raw_df[raw_df['date_id'] > ANALYSIS_START_DATE_ID].copy().reset_index(drop=True)
    y = (modern_df['forward_returns'] > 0).astype(int)
    sample_weight = modern_df['forward_returns'].abs().fillna(0)
    
    feature_cols = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
    
    missing_ratios = modern_df[feature_cols].isnull().mean()
    features_to_keep = missing_ratios[missing_ratios <= MISSING_THRESHOLD].index.tolist()
    X_full = modern_df[features_to_keep]
    
    print(f"✅ 数据质量门禁完成：从 {len(feature_cols)} 个特征中，保留了 {len(features_to_keep)} 个 (缺失率 <= {MISSING_THRESHOLD*100}%)。")

    # --- 准备工作 ---
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)
    oof_predictions = np.zeros(len(X_full))
    
    print(f"\n--- 步骤2：启动集成化、因果一致的交叉验证流程 ---")
    
    # ========================= [核心] 集成化训练循环 =========================
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
        print(f"\n{'='*20} 正在处理第 {fold + 1}/{N_SPLITS} 折 {'='*20}")
        
        # --- 步骤2.1: 准备当前折的数据 ---
        purged_train_idx = train_idx[:-PURGE_SIZE] if PURGE_SIZE > 0 else train_idx
        X_train_fold, y_train_fold = X_full.iloc[purged_train_idx], y.iloc[purged_train_idx]
        sw_train_fold = sample_weight.iloc[purged_train_idx]
        X_val_fold, y_val_fold = X_full.iloc[val_idx], y.iloc[val_idx]

        # --- 步骤2.2: 在当前折内部，训练一个“临时AE” ---
        print(f"  > [AE] 正在为第 {fold+1} 折训练专属AE模型...")
        ae_model = train_fold_ae(X_train_fold, y_train_fold, sw_train_fold, X_val_fold, y_val_fold)
        print(f"  > [AE] 专属AE模型训练完毕。")

        # --- 步骤2.3: 用临时AE生成“纯净”的OOF特征 ---
        print(f"  > [AE] 正在生成AI特征...")
        with torch.no_grad():
            # (复用AE训练函数中的预处理逻辑，以保证一致性)
            train_filled = X_train_fold.ffill(limit=3).fillna(X_train_fold.ffill(limit=3).median()).fillna(0)
            val_filled = X_val_fold.ffill(limit=3).fillna(train_filled.median()).fillna(0)
            mean = train_filled.mean().values; std = train_filled.std().values; std[std==0] = 1.0
            
            train_scaled = torch.tensor((train_filled.values - mean) / std, dtype=torch.float32).to(device)
            val_scaled = torch.tensor((val_filled.values - mean) / std, dtype=torch.float32).to(device)

            X_train_ai_features = ae_model.encoder(train_scaled).cpu().numpy()
            X_val_ai_features = ae_model.encoder(val_scaled).cpu().numpy()

        # --- 步骤2.4: 合并特征，组建“混合部队” ---
        ai_feature_names = [f'AE_{i}' for i in range(X_train_ai_features.shape[1])]
        X_train_final = np.concatenate([X_train_fold.values, X_train_ai_features], axis=1)
        X_val_final = np.concatenate([X_val_fold.values, X_val_ai_features], axis=1)
        final_feature_names = X_train_fold.columns.tolist() + ai_feature_names
        
        # --- 步骤2.5: 在当前折内部，训练LightGBM ---
        print(f"  > [LGBM] 正在训练LightGBM分类器...")
        model_lgbm = lgb.LGBMClassifier(**lgbm_params)
        model_lgbm.fit(X_train_final, y_train_fold, 
                       sample_weight=sw_train_fold,
                       eval_set=[(X_val_final, y_val_fold)],
                       callbacks=[lgb.early_stopping(100, verbose=False)])

        # --- 步骤2.6: 做出OOF预测 ---
        oof_predictions[val_idx] = model_lgbm.predict_proba(X_val_final)[:, 1]
        print(f"  > 第 {fold+1} 折处理完毕。")
    # ========================= 循环结束 =========================

    # --- 步骤3: 计算并报告最终的“诚实”分数 ---
    valid_indices = np.where(oof_predictions != 0)[0]
    final_oof_score = roc_auc_score(y.iloc[valid_indices], oof_predictions[valid_indices])
    
    total_time = time.time() - start_time
    print(f"\n\n{'='*25} 集成化验证流程结束 {'='*25}")
    print(f"总耗时: {total_time:.2f} 秒。")
    print(f"✅ 最终的、诚实的 OOF AUC: {final_oof_score:.8f}")
    
    # --- 保存OOF预测文件，以备后续分析和融合 ---
    oof_df = pd.DataFrame({
        'date_id': modern_df['date_id'].iloc[valid_indices],
        'target': y.iloc[valid_indices],
        'oof_prediction': oof_predictions[valid_indices]
    })
    oof_df.to_csv(OOF_OUTPUT_FILE, index=False)
    print(f"✅ OOF预测结果已保存至 '{OOF_OUTPUT_FILE}'。")
