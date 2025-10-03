import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# --- 全局配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def find_robust_start_date(df: pd.DataFrame, completeness_threshold: float = 0.75) -> int:
    # ... (此函数无变化)
    print(f"--- 任务 1: 动态计算分析起点 ---")
    feature_cols = [col for col in df.columns if col.startswith(('D', 'E', 'I', 'M', 'P', 'S', 'V'))]
    df['feature_completeness'] = df[feature_cols].notna().sum(axis=1) / len(feature_cols)
    robust_start_date = df.loc[df['feature_completeness'] >= completeness_threshold, 'date_id'].iloc[0]
    print(f"计算完成！稳健的分析起点 date_id 为: {robust_start_date}")
    return robust_start_date

def train_and_evaluate(X_train, y_train, X_val, y_val, lgb_params):
    """一个精简的训练和评估函数。"""
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

def find_robust_start_date(df: pd.DataFrame, completeness_threshold: float = 0.75) -> int:
    # ...
    print(f"--- 任务 1: 动态计算分析起点 ---")
    feature_cols = [col for col in df.columns if col.startswith(('D', 'E', 'I', 'M', 'P', 'S', 'V'))]
    df['feature_completeness'] = df[feature_cols].notna().sum(axis=1) / len(feature_cols)
    robust_start_date = df.loc[df['feature_completeness'] >= completeness_threshold, 'date_id'].iloc[0]
    print(f"计算完成！稳健的分析起点 date_id 为: {robust_start_date}")
    return robust_start_date

def train_and_evaluate(X_train, y_train, X_val, y_val, lgb_params):
    # ...
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse


if __name__ == '__main__':
    try:
        # --- 步骤 1: 准备数据 ---
        print("train_v3_featured_raw.csv")
        train_df = pd.read_csv('train_v3_featured_raw.csv')
        robust_start_date = find_robust_start_date(train_df)
        
        train_filtered = train_df[train_df['date_id'] > robust_start_date].copy()
        train_processed = train_filtered
        
        all_features = [col for col in train_processed.columns if col.startswith(('D', 'E', 'I', 'M', 'P', 'S', 'V'))]
        target = 'forward_returns'
        
        # ==================== 终极诊断代码 ====================
        print("\n--- [终极诊断] 正在检查目标(y)的缺失情况... ---")
        y_original = train_processed[target]
        y_nan_count = y_original.isnull().sum()
        y_total_count = len(y_original)
        print(f"  > 在 date_id > {robust_start_date} 的数据范围内，目标(y)共有 {y_total_count} 行。")
        print(f"  > 其中，缺失的'标准答案'(NaN)共有: {y_nan_count} 行。")
        print(f"  > 缺失率: {y_nan_count / y_total_count:.4%}")
        print("--- 诊断完毕 ---")
        # =======================================================

        X = train_processed[all_features]
        y = train_processed[target].fillna(0) # 执行必要的技术性填充
        X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # ... (后续代码完全无变化) ...
        print("\n--- 任务 2: 加载“黄金参数组合” ---")
        golden_params = {
            'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 5000,
            'verbose': -1, 'n_jobs': -1, 'seed': 42, 'importance_type': 'gain',
            'learning_rate': 0.04485147098842579, 'num_leaves': 26,
            'feature_fraction': 0.70576707581891, 'bagging_fraction': 0.929368387238464,
            'bagging_freq': 7, 'min_child_samples': 12,
            'lambda_l1': 6.693101867960152e-08, 'lambda_l2': 0.1426934284068358
        }
        print("黄金参数已装载！")

        print("\n--- 任务 3: 使用黄金参数进行一次性特征侦察... ---")
        recon_model = lgb.LGBMRegressor(**golden_params).fit(X_train_full, y_train)
        feature_importance_df = pd.DataFrame({'feature': all_features, 'gain': recon_model.feature_importances_}).sort_values('gain', ascending=False)
        feature_pool = feature_importance_df.head(80)['feature'].tolist()
        print("Top 80 特征池已建立。")

        print("\n--- 任务 4: 在巅峰状态下，对不同规模的部队进行最终裁决... ---")
        battle_plans = [30, 50, 80]
        results = {}

        for n_features in battle_plans:
            print(f"\n  > 正在评估 Top {n_features} 部队...")
            current_features = feature_pool[:n_features]
            X_train_elite = X_train_full[current_features]
            X_val_elite = X_val_full[current_features]
            
            rmse_result = train_and_evaluate(X_train_elite, y_train, X_val_elite, y_val, golden_params)
            results[f'Top_{n_features}_Features'] = rmse_result
            print(f"    ...评估完成，RMSE: {rmse_result}")

        print("\n" + "="*70)
        print("!!!               最终验证战报汇总               !!!")
        print("="*70)
        best_plan = min(results, key=results.get)
        for plan, score in results.items():
            print(f">>> 部队规模: {plan.replace('_', ' ')} | 最终RMSE: {score}")
        print("\n--- [最终裁决] ---")
        print(f"🏆 在“黄金参数”加持下，最优部队规模为: {best_plan.replace('_', ' ')}！🏆")
        print(f"    其创造的最新RMSE记录为: {results[best_plan]}")
        print("="*70)

    except FileNotFoundError:
        print("\n错误: train_v3_featured_raw.csv 文件未找到。")
    except Exception as e:
        print(f"\n执行过程中发生错误: {e}")