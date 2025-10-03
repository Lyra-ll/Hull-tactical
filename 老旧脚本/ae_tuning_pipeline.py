# ================= 核心：“AI武器自动化测试平台” =================
import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error 
import os
import time

# --- 字体修复 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 以下是完整的辅助函数定义
# (为了保持主流程清晰，将它们放在脚本末尾)

class SupervisedAE(nn.Module):
    def __init__(self, input_dim, encoding_dim=16):
        super(SupervisedAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, encoding_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(encoding_dim, 64), nn.ReLU(), nn.Linear(64, input_dim))
        self.predictor = nn.Linear(encoding_dim, 1)
    def forward(self, x):
        encoded = self.encoder(x); decoded = self.decoder(encoded); prediction = self.predictor(encoded)
        return decoded, prediction

def train_ae(features_tensor, target_tensor, experiment_params):
    dataset = TensorDataset(features_tensor, target_tensor)
    train_size = int(0.8 * len(dataset)); val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    input_dim = features_tensor.shape[1]
    
    model = SupervisedAE(input_dim, encoding_dim=experiment_params['encoding_dim'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_recon = nn.MSELoss(); loss_pred = nn.MSELoss()
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    epochs = 200; patience = 10; epochs_no_improve = 0
    min_val_loss = np.inf; best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch_features, batch_target in train_loader:
            recon, pred = model(batch_features)
            loss = experiment_params['recon_weight'] * loss_recon(recon, batch_features) + \
                   (1 - experiment_params['recon_weight']) * loss_pred(pred, batch_target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_features, batch_target in val_loader:
                recon, pred = model(batch_features)
                loss = experiment_params['recon_weight'] * loss_recon(recon, batch_features) + \
                       (1 - experiment_params['recon_weight']) * loss_pred(pred, batch_target)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience:
            break
            
    if best_model_state: model.load_state_dict(best_model_state)
    return model

def extract_ae_features(model, full_dataset, original_df):
    model.eval()
    loader = DataLoader(full_dataset, batch_size=64, shuffle=False)
    all_encoded = []
    with torch.no_grad():
        for batch_features, _ in loader:
            encoded = model.encoder(batch_features)
            all_encoded.append(encoded.cpu().numpy())
    
    encoded_np = np.concatenate(all_encoded, axis=0)
    ae_names = [f'AE_{i}' for i in range(encoded_np.shape[1])]
    ae_df = pd.DataFrame(encoded_np, columns=ae_names)
    final_df = pd.concat([original_df[['date_id', 'forward_returns']].reset_index(drop=True), ae_df], axis=1)
    return final_df

def evaluate_with_lightgbm(manual_features_df, ai_features_df):
    ai_to_merge = ai_features_df.drop(columns=['forward_returns'])
    full_df = pd.merge(manual_features_df, ai_to_merge, on='date_id')
    
    features_to_exclude = ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']
    all_features = [col for col in full_df.columns if col not in features_to_exclude]
    target = 'forward_returns'
    
    X = full_df[all_features]
    y = full_df[target].fillna(0)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    golden_params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 5000, 'verbose': -1,
        'n_jobs': -1, 'seed': 42, 'importance_type': 'gain', 'learning_rate': 0.04485,
        'num_leaves': 26, 'feature_fraction': 0.7057, 'bagging_fraction': 0.9293,
        'bagging_freq': 7, 'min_child_samples': 12, 'lambda_l1': 6.693e-08, 'lambda_l2': 0.1426
    }
    
    recon_model = lgb.LGBMRegressor(**golden_params).fit(X_train, y_train)
    importance_df = pd.DataFrame({'feature': all_features, 'gain': recon_model.feature_importances_}).sort_values('gain', ascending=False)
    top_30_hybrid = importance_df.head(30)['feature'].tolist()
    
    final_model = lgb.LGBMRegressor(**golden_params)
    final_model.fit(X_train[top_30_hybrid], y_train, eval_set=[(X_val[top_30_hybrid], y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    preds = final_model.predict(X_val[top_30_hybrid])
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse, importance_df.head(30)

def generate_dashboard(importance_df, manual_features_df, ai_features_df, experiment_id):
    output_dir = f'dashboard_{experiment_id}'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
        
    plt.figure(figsize=(10, 12)); sns.barplot(x='gain', y='feature', data=importance_df)
    plt.title(f'仪表盘A: 特征重要性 - {experiment_id}'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'A_feature_importance.png')); plt.close()
    
    # (t-SNE部分与之前相同，但需要重新准备数据)
    full_df = pd.merge(manual_features_df, ai_features_df.drop(columns=['forward_returns']), on='date_id')
    X_top30 = full_df[importance_df['feature'].tolist()]
    y = full_df['forward_returns']
    
    n_samples = 1000
    if len(X_top30) > n_samples:
        sample_indices = np.random.choice(X_top30.index, n_samples, replace=False)
        X_sample = X_top30.loc[sample_indices].fillna(0)
        y_sample = y.loc[sample_indices]
    else:
        X_sample = X_top30.fillna(0); y_sample = y

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_sample)
    
    df_tsne = pd.DataFrame({'tsne_1': X_tsne[:, 0], 'tsne_2': X_tsne[:, 1], '涨跌': ['涨' if val > 0 else '跌' for val in y_sample]})
    plt.figure(figsize=(12, 10)); sns.scatterplot(x='tsne_1', y='tsne_2', hue='涨跌', palette=sns.color_palette(["#d62728", "#1f77b4"]), data=df_tsne, alpha=0.6)
    plt.title(f'仪表盘B: t-SNE 可视化 - {experiment_id}'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'B_tsne_visualization.png')); plt.close()
    print(f"--- 可视化报告已保存至 '{output_dir}' 文件夹 ---")

# ==================== 主执行流程 ====================
def run_experiment(experiment_id, params, manual_features_path='train_v3_featured_raw.csv'):
    """
    执行一次完整的端到端实验。
    """
    print(f"\n{'='*80}\n  开始执行实验: {experiment_id} | 参数: {params}\n{'='*80}")
    start_time = time.time()
    
    # --- 1. 准备数据 ---
    df_manual = pd.read_csv(manual_features_path)
    modern_history_df = df_manual[df_manual['date_id'] > 1055].copy().reset_index(drop=True)
    features_df = modern_history_df.filter(regex=r'^(D|E|I|M|P|S|V)_*')
    target_df = modern_history_df['forward_returns']
    features_tensor = torch.tensor(features_df.fillna(0).values, dtype=torch.float32)
    target_tensor = torch.tensor(target_df.fillna(0).values, dtype=torch.float32).unsqueeze(1)
    
    # --- 2. 训练AE模型 ---
    print(f"\n--- [阶段1] 正在训练AE模型... ---")
    trained_ae_model = train_ae(features_tensor, target_tensor, params)
    
    # --- 3. 提取AI精华特征 ---
    print(f"\n--- [阶段2] 正在提取AI精华特征... ---")
    dataset = TensorDataset(features_tensor, target_tensor)
    ai_features_df = extract_ae_features(trained_ae_model, dataset, modern_history_df)
    
    # --- 4. LightGBM评估 ---
    print(f"\n--- [阶段3] 正在用LightGBM评估混合部队... ---")
    rmse, top_30_importance = evaluate_with_lightgbm(modern_history_df, ai_features_df)
    
    # --- 5. 生成可视化仪表盘 ---
    print(f"\n--- [阶段4] 正在生成可视化仪表盘... ---")
    generate_dashboard(top_30_importance, modern_history_df, ai_features_df, experiment_id)
    
    total_time = time.time() - start_time
    print(f"\n--- 实验 {experiment_id} 完成！总耗时: {total_time:.2f} 秒 ---")
    
    return rmse, top_30_importance

# ==================== 实验定义与执行 ====================
if __name__ == '__main__':
    # 定义我们的实验清单
    experiment_list = {
        "Exp-0_Baseline": {'encoding_dim': 16, 'recon_weight': 0.5},
        "Exp-1.1_Encode-8": {'encoding_dim': 8, 'recon_weight': 0.5},
        "Exp-1.2_Encode-32": {'encoding_dim': 32, 'recon_weight': 0.5},
        "Exp-2.1_Pred-Focused": {'encoding_dim': 16, 'recon_weight': 0.2},
        "Exp-2.2_Recon-Focused": {'encoding_dim': 16, 'recon_weight': 0.8},
    }
    
    results_summary = {}
    
    for exp_id, exp_params in experiment_list.items():
        exp_rmse, _ = run_experiment(exp_id, exp_params)
        results_summary[exp_id] = exp_rmse
        
    # --- 打印最终的实验结果汇总 ---
    print(f"\n{'='*80}\n  所有实验已完成！最终结果汇总：\n{'='*80}")
    
    best_exp = min(results_summary, key=results_summary.get)
    
    for exp_id, rmse in results_summary.items():
        print(f">>> 实验: {exp_id.ljust(20)} | 最终RMSE: {rmse:.10f}")
        
    print("\n--- [最终裁决] ---")
    print(f"🏆 最优AI武器设计方案为: {best_exp}！🏆")

