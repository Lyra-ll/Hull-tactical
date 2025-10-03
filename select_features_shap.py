# select_features_shap.py
# =================================================================
# 终极特征筛选脚本 V1.2 (终极防御版)
# =================================================================
import argparse
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import importlib
import warnings

warnings.filterwarnings('ignore')

def run_shap_selection(config, n_features_to_select):
    """
    执行基于SHAP的特征筛选流程
    """
    print("--- 1. 加载依赖模块 (utils.py) ---")
    import utils

    print("\n--- 2. 加载数据 ---")
    dev_df = utils.load_data(config.RAW_DATA_FILE, config.ANALYSIS_START_DATE_ID)
    
    all_features = [c for c in dev_df.columns if c.startswith(('M', 'E', 'I', 'P', 'V', 'S', 'D')) or '_diff' in c or '_rol_' in c]
    print(f"  > 将对全部 {len(all_features)} 个特征进行SHAP重要性评估。")

    X = dev_df[all_features]
    y = dev_df[config.TARGET_COLUMNS]
    
    if 'sample_weight' in dev_df.columns:
        sample_weight = dev_df['sample_weight']
        print("  > ✅ 成功加载样本权重。")
    else:
        print("  > ❌ 错误: 未在数据中找到 'sample_weight' 列。")
        return

    y_primary = y[config.PRIMARY_TARGET_COLUMN]

    print("\n--- 3. 设置交叉验证 (CV) ---")
    tscv = utils.get_cv_splitter(config.N_SPLITS, config.EMBARGO_SIZE, config.PURGE_SIZE)
    print(f"  > ✅ CV分割器已设置: {config.N_SPLITS} 折...")

    print("\n--- 4. 开始交叉验证与SHAP值计算 ---")
    all_shap_df_list = [] 

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"  > 正在处理第 {fold + 1}/{config.N_SPLITS} 折...")
        
        purged_train_idx = train_idx[:-config.PURGE_SIZE] if config.PURGE_SIZE > 0 else train_idx
        X_train, y_train = X.iloc[purged_train_idx], y_primary.iloc[purged_train_idx]
        sw_train = sample_weight.iloc[purged_train_idx]
        X_val, y_val = X.iloc[val_idx], y_primary.iloc[val_idx]

        preprocessor_params = utils.get_preprocessor_params(X_train)
        _, X_train_scaled = utils.apply_preprocessor(X_train, preprocessor_params)
        _, X_val_scaled = utils.apply_preprocessor(X_val, preprocessor_params)

        model = lgb.LGBMClassifier(
            objective='binary', metric='auc', random_state=42, n_estimators=200,
            learning_rate=0.05, num_leaves=8, max_depth=3, reg_alpha=10,
            reg_lambda=10, colsample_bytree=0.5, n_jobs=-1, verbosity=-1
        )
        
        model.fit(X_train_scaled, y_train, sample_weight=sw_train)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_scaled)
        
        # --- [终极核心修复] ---
        # 增加防御性检查。当模型完全学不到东西时，shap_values可能不是一个list，或者list长度不为2
        shap_values_for_class_1 = None
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # 这是正常情况，我们取类别1的SHAP值
            shap_values_for_class_1 = shap_values[1]
        else:
            # 这是异常情况，可能模型变成了“哑巴”模型。
            # 此时 shap_values 可能直接就是我们需要的那个矩阵。
            # 我们做一个形状检查来确认。
            if shap_values.shape == X_val_scaled.shape:
                 shap_values_for_class_1 = shap_values
            else:
                print(f"    ⚠️ 警告: 第 {fold + 1} 折的SHAP值形状异常，已跳过。")
                print(f"       期望形状: {X_val_scaled.shape}, 实际形状: {shap_values.shape}")
                continue # 跳过这一折

        shap_df = pd.DataFrame(shap_values_for_class_1, columns=X_val_scaled.columns)
        all_shap_df_list.append(shap_df)
        # --- [修复结束] ---

    if not all_shap_df_list:
        print("\n❌ 严重错误: 所有折的SHAP值计算均失败。无法生成特征排名。")
        return

    print("\n--- 5. 汇总SHAP值并生成最终排名 ---")

    shap_importance_df = pd.concat(all_shap_df_list)
    feature_importance = shap_importance_df.abs().mean().sort_values(ascending=False)

    print(f"\n✅ SHAP特征重要性计算完成！")
    print("=" * 60)
    print("Top 10 最重要的特征:")
    print(feature_importance.head(10))
    print("=" * 60)
    print("Top 10 最不重要的特征:")
    print(feature_importance.tail(10))
    print("=" * 60)

    top_features = feature_importance.head(n_features_to_select).index.tolist()

    new_ranking_filename = f"feature_ranking_shap_top{n_features_to_select}.csv"
    pd.Series(top_features).to_csv(new_ranking_filename, index=False, header=False)

    print(f"\n🏆 已选出 Top {n_features_to_select} 个王牌特征！")
    print(f"   列表已保存至: '{new_ranking_filename}'")
    print("   你现在可以在 config.py 中将 RANKING_FILE 更新为此文件名，")
    print(f"   并将 N_TOP_FEATURES_TO_USE 更新为 {n_features_to_select}，然后重新运行你的诊断或完整流程。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="终极特征筛选脚本 V1.2 (终极防御版)")
    parser.add_argument('--config', type=str, default='config', help="配置文件")
    parser.add_argument('--n_select', type=int, default=50, help="最终选拔出的王牌特征数量")
    args = parser.parse_args()

    try:
        config_module = importlib.import_module(args.config)
        print(f"--- 配置文件 '{args.config}.py' 加载成功 ---")
    except ImportError:
        print(f"❌ 错误: 无法找到配置文件 '{args.config}.py'。"); exit()

    start_time = time.time()
    run_shap_selection(config_module, args.n_select)
    print(f"\n任务完成！总耗时: {time.time() - start_time:.2f} 秒。")