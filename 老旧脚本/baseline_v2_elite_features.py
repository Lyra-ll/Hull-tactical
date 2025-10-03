import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 全局配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def find_robust_start_date(df: pd.DataFrame, completeness_threshold: float = 0.75) -> int:
    # ... (此函数无变化)
    print(f"--- 任务 1: 动态计算分析起点 (特征齐全率 > {completeness_threshold * 100}%) ---")
    feature_cols = [col for col in df.columns if col.startswith(('D', 'E', 'I', 'M', 'P', 'S', 'V'))]
    df['feature_completeness'] = df[feature_cols].notna().sum(axis=1) / len(feature_cols)
    robust_start_date = df.loc[df['feature_completeness'] >= completeness_threshold, 'date_id'].iloc[0]
    print(f"计算完成！稳健的分析起点 date_id 为: {robust_start_date}")
    return robust_start_date

def select_top_features_by_gain(X_train, y_train, all_features, lgb_params, n_top_features=50):
    # ... (此函数无变化)
    print(f"\n--- [筛选阶段] 正在训练侦察模型以进行特征排序... ---")
    recon_model = lgb.LGBMRegressor(**lgb_params)
    recon_model.fit(X_train, y_train, eval_metric='rmse')
    feature_importance_df = pd.DataFrame({
        'feature': all_features,
        'gain': recon_model.feature_importances_
    }).sort_values('gain', ascending=False)
    elite_features = feature_importance_df.head(n_top_features)['feature'].tolist()
    print(f"--- [筛选阶段] 特征筛选完成！已选出Top {n_top_features} 精英特征。---")
    print("排名前5的特征是:", elite_features[:5])
    return elite_features

def train_and_evaluate(df: pd.DataFrame, start_date_id: int, num_elite_features: int = 50, imputation_strategy: str = 'none'):
    """
    一个完整的训练和评估流程，返回最终的RMSE分数。
    """
    print(f"\n{'='*70}")
    print(f"  开始新战役 | 部队规模: Top {num_elite_features} | 缺失值策略: {imputation_strategy.upper()}")
    print(f"{'='*70}")
    
    train_filtered = df[df['date_id'] > start_date_id].copy()

    # 我们已经确定'none'是最佳策略，所以只保留这一个分支
    print("\n[数据处理] 采用 'none' (不填充) 策略。")
    train_processed = train_filtered
    
    all_features = [col for col in train_processed.columns if col.startswith(('D', 'E', 'I', 'M', 'P', 'S', 'V'))]
    target = 'forward_returns'
    
    X = train_processed[all_features]
    y = train_processed[target].fillna(0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    lgb_params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 5000,
        'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'verbose': -1, 'n_jobs': -1, 'seed': 42, 
        'importance_type': 'gain', 'num_leaves': 31
    }

    elite_features = select_top_features_by_gain(X_train, y_train, all_features, lgb_params, n_top_features=num_elite_features)

    print(f"\n--- [总攻阶段] 正在使用 {len(elite_features)} 个精英特征训练最终模型... ---")
    X_train_elite = X_train[elite_features]
    X_val_elite = X_val[elite_features]

    final_model = lgb.LGBMRegressor(**lgb_params)
    final_model.fit(X_train_elite, y_train,
                    eval_set=[(X_val_elite, y_val)],
                    eval_metric='rmse',
                    callbacks=[lgb.early_stopping(100, verbose=False)])

    preds = final_model.predict(X_val_elite)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    print(f"\n--- [战役结束] 部队规模 Top {num_elite_features} 的RMSE为: {rmse} ---")
    
    # 为每个战役保存独立的Gain图
    lgb.plot_importance(final_model, max_num_features=num_elite_features, figsize=(10, 12), importance_type='gain')
    plt.title(f'Top {num_elite_features} 精英特征重要性 (Gain)')
    plt.tight_layout()
    image_filename = f'feature_importance_top_{num_elite_features}.png'
    plt.savefig(image_filename)
    plt.close()
    
    return rmse

# ==============================================================================
# 主执行流程 (已升级为自动化战役循环)
# ==============================================================================
if __name__ == '__main__':
    try:
        print("正在加载 train_v3_featured_raw.csv...")
        train_df = pd.read_csv('train_v3_featured_raw.csv')

        robust_start_date = find_robust_start_date(train_df)
        
        # --- 核心修改：自动化战役循环 ---
        battle_plans = [30, 50, 80] # 定义我们要测试的三种部队规模
        results = {}

        for n_features in battle_plans:
            rmse_result = train_and_evaluate(
                train_df, 
                start_date_id=robust_start_date, 
                num_elite_features=n_features,
                imputation_strategy='none' # 锁定最优策略
            )
            results[f'Top_{n_features}_Features'] = rmse_result

        # --- 最终战报汇总 ---
        print("\n" + "="*70)
        print("!!!               最终战报汇总               !!!")
        print("="*70)
        
        best_plan = min(results, key=results.get)
        
        for plan, score in results.items():
            print(f">>> 部队规模: {plan.replace('_', ' ')} | 最终RMSE: {score}")
            
        print("\n--- [最终裁决] ---")
        print(f"🏆 最优部队规模为: {best_plan.replace('_', ' ')}，其RMSE最低！🏆")
        print("="*70)

    except FileNotFoundError:
        print("\n错误: train_v3_featured_raw.csv 文件未找到。请确保您已成功运行修改后的 create_features.py。")
    except Exception as e:
        print(f"\n执行过程中发生错误: {e}")