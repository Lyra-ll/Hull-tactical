import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# --- 全局配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

def prepare_data(df_path='train_v3_featured_raw.csv', start_date_id=1055, n_features=30):
    """一个统一的数据准备函数。"""
    print("--- 准备数据和精英特征集 ---")
    df = pd.read_csv(df_path)
    train_filtered = df[df['date_id'] > start_date_id].copy()
    train_processed = train_filtered

    all_features = [col for col in train_processed.columns if col.startswith(('D', 'E', 'I', 'M', 'P', 'S', 'V'))]
    target = 'forward_returns'
    
    X = train_processed[all_features]
    y = train_processed[target].fillna(0)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 筛选固定的Top 30特征
    temp_params = {'objective': 'regression_l1', 'seed': 42, 'n_jobs': -1, 'verbose': -1, 'importance_type': 'gain'}
    recon_model = lgb.LGBMRegressor(**temp_params).fit(X_train, y_train)
    feature_importance_df = pd.DataFrame({'feature': all_features, 'gain': recon_model.feature_importances_}).sort_values('gain', ascending=False)
    elite_features = feature_importance_df.head(n_features)['feature'].tolist()
    
    X_train_elite = X_train[elite_features]
    X_val_elite = X_val[elite_features]
    
    print("数据准备完毕！")
    return X_train_elite, X_val_elite, y_train, y_val

def main():
    # --- 步骤 1: 定义两个时代的模型参数 ---
    # “旧的笨模型”：只迭代几次就放弃
    dumb_params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 5000,
        'learning_rate': 0.05, 'num_leaves': 31, # 使用了较为激进的默认参数
        'verbose': -1, 'n_jobs': -1, 'seed': 42
    }

    # “新的聪明模型”：使用我们AI找到的黄金参数
    smart_params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 5000,
        'verbose': -1, 'n_jobs': -1, 'seed': 42, 'importance_type': 'gain',
        'learning_rate': 0.04485, 'num_leaves': 26, 'feature_fraction': 0.7057,
        'bagging_fraction': 0.9293, 'bagging_freq': 7, 'min_child_samples': 12,
        'lambda_l1': 6.693e-08, 'lambda_l2': 0.1426
    }

    # --- 步骤 2: 准备数据 ---
    X_train, X_val, y_train, y_val = prepare_data(n_features=30)

    # --- 步骤 3: 分别训练两个模型并获取预测 ---
    print("\n--- 正在训练'笨模型'... ---")
    dumb_model = lgb.LGBMRegressor(**dumb_params)
    dumb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    preds_dumb = dumb_model.predict(X_val)
    rmse_dumb = np.sqrt(mean_squared_error(y_val, preds_dumb))

    print("--- 正在训练'聪明模型'... ---")
    smart_model = lgb.LGBMRegressor(**smart_params)
    smart_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    preds_smart = smart_model.predict(X_val)
    rmse_smart = np.sqrt(mean_squared_error(y_val, preds_smart))

    print("\n--- 训练完毕 ---")
    print(f"笨模型 RMSE: {rmse_dumb} (迭代 {dumb_model.best_iteration_} 次)")
    print(f"聪明模型 RMSE: {rmse_smart} (迭代 {smart_model.best_iteration_} 次)")


    # --- 步骤 4: 开始可视化法医鉴定 ---
    print("\n--- 正在生成可视化法医报告... ---")
    output_dir = 'visual_forensics_reports'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 报告A: 预测一致性散点图
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=preds_dumb, y=preds_smart, alpha=0.5)
    plt.plot([min(preds_dumb), max(preds_dumb)], [min(preds_dumb), max(preds_dumb)], 'r--', linewidth=2)
    plt.title('法医报告A: 聪明模型 vs 笨模型 的预测一致性')
    plt.xlabel('笨模型的预测值')
    plt.ylabel('聪明模型的预测值')
    plt.savefig(os.path.join(output_dir, 'A_prediction_consistency.png'))
    plt.close()

    # 报告B: 预测值分布图
    plt.figure(figsize=(10, 6))
    sns.kdeplot(preds_dumb, label='笨模型', fill=True)
    sns.kdeplot(preds_smart, label='聪明模型', fill=True)
    plt.title('法医报告B: 两个模型预测值的分布')
    plt.xlabel('预测值')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'B_prediction_distribution.png'))
    plt.close()

    # 报告C: 预测误差分布图
    errors_dumb = y_val - preds_dumb
    errors_smart = y_val - preds_smart
    plt.figure(figsize=(10, 6))
    sns.kdeplot(errors_dumb, label='笨模型', fill=True)
    sns.kdeplot(errors_smart, label='聪明模型', fill=True)
    plt.title('法医报告C: 两个模型预测误差的分布')
    plt.xlabel('预测误差 (真实值 - 预测值)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'C_error_distribution.png'))
    plt.close()

    # 报告D: 时序预测对比图 (抽样最后200个点以便观察)
    df_plot = pd.DataFrame({'真实值': y_val, '笨模型预测': preds_dumb, '聪明模型预测': preds_smart}).tail(200)
    plt.figure(figsize=(16, 8))
    df_plot.plot(kind='line', figsize=(16, 8), alpha=0.8)
    plt.title('法医报告D: 时序预测对比 (最后200个交易日)')
    plt.xlabel('时间 (验证集中的顺序)')
    plt.ylabel('回报率')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'D_timeseries_comparison.png'))
    plt.close()
    
    print(f"✅ 所有报告已保存至 '{output_dir}' 文件夹。请立刻检查！")

if __name__ == '__main__':
    main()