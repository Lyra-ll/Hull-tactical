# diagnose_pipeline_v2_strict.py
# =================================================================
# 自动化指挥中心 - 深度诊断脚本 V2.0 (严格筛选版)
# =================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import warnings

# 导入您项目中的配置文件和工具库
import config
import utils 

# 忽略一些常见的性能警告，让输出更清晰
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

print("="*80)
print("🚀 [启动] 自动化流水线深度诊断程序 (严格筛选版)...")
print("="*80)

# --- 步骤 1: 加载环境配置与工具 ---
# (这部分代码与之前版本相同)
print("✅ 步骤 1: 成功加载 'config.py' 和 'utils.py'。")

# --- 步骤 2: 加载原始数据并进行基础健康检查 ---
print("\n--- 步骤 2: 加载原始数据并进行基础健康检查 ---")
dev_df = utils.load_data(config.RAW_DATA_FILE, config.ANALYSIS_START_DATE_ID)
all_features_initial = [c for c in dev_df.columns if c.startswith(('M', 'E', 'I', 'P', 'V', 'S', 'D')) or '_diff' in c or '_rol_' in c]
X = dev_df[all_features_initial]
y = dev_df[config.TARGET_COLUMNS]
sample_weight = dev_df['sample_weight']
print("  > 数据加载与分离完成。")

# --- 步骤 3: 模拟单折数据切分、筛选与预处理 ---
print("\n--- 步骤 3: 模拟第一折 (Fold 0) 的数据切分、筛选与预处理 ---")
try:
    tscv = utils.get_cv_splitter(config.N_SPLITS, config.EMBARGO_SIZE, config.PURGE_SIZE)
    train_idx, val_idx = next(tscv.split(X))
    print(f"\n[信息] Fold 0 切分完成:")
    print(f"   - 训练集索引范围: {train_idx[0]} - {train_idx[-1]} (大小: {len(train_idx)})")
    
    X_train, y_train, sw_train = X.iloc[train_idx], y.iloc[train_idx], sample_weight.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # --- [核心逻辑] 加入动态折内特征筛选 ---
    print(f"\n[检查 3.1] 正在基于当前训练集进行动态缺失值筛选 (阈值: < {config.MISSING_THRESHOLD})...")
    missing_ratios_fold = X_train.isnull().mean()
    features_to_keep_fold = missing_ratios_fold[missing_ratios_fold < config.MISSING_THRESHOLD].index.tolist()
    
    print(f"   - 筛选前特征数: {X_train.shape[1]}")
    print(f"   - 筛选后特征数: {len(features_to_keep_fold)}")
    if X_train.shape[1] != len(features_to_keep_fold):
         print(f"   - ✅ 已成功剔除 {X_train.shape[1] - len(features_to_keep_fold)} 个高缺失率特征。")

    X_train_filtered = X_train[features_to_keep_fold]
    X_val_filtered = X_val[features_to_keep_fold]
    # --- [修改结束] ---

    print("\n[检查 3.2] 测试 'utils.get_preprocessor_params' (在筛选后的数据上)...")
    preprocessor_params = utils.get_preprocessor_params(X_train_filtered)
    if preprocessor_params['std'].isnull().any():
         print("   🚨 警告: 学习到的标准差中依然存在NaN！")
    else:
         print("   ✔️ 参数学习过程正常，无NaN标准差。")

    print("\n[检查 3.3] 测试 'utils.apply_preprocessor'...")
    _, X_train_scaled = utils.apply_preprocessor(X_train_filtered, preprocessor_params)
    _, X_val_scaled = utils.apply_preprocessor(X_val_filtered, preprocessor_params)

    train_nan_count = X_train_scaled.isnull().sum().sum()
    val_nan_count = X_val_scaled.isnull().sum().sum()
    print(f"   - 预处理后的训练集NaN数量: {train_nan_count}")
    if train_nan_count == 0:
        print("   ✔️ 预处理后的数据干净，无NaN。")
    else:
        print("   ❌ 致命缺陷: 预处理后仍有NaN！")

except Exception as e:
    print(f"❌ 在步骤3中发生错误: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 步骤 4: 简化模型训练与评估 ---
# 我们暂时跳过复杂的AE，直接用LGBM测试预处理后的特征是否有信号
print("\n--- 步骤 4: 简化LGBM模型训练与评估 (跳过AE) ---")
try:
    target_col = config.PRIMARY_TARGET_COLUMN
    print(f"[信息] 使用主目标 '{target_col}' 进行训练。")

    model = lgb.LGBMClassifier(random_state=42, n_estimators=100) # 使用一个简单的LGBM
    
    print("[信息] 正在训练模型...")
    model.fit(X_train_scaled, y_train[target_col], sample_weight=sw_train)
    
    print("[信息] 正在进行预测...")
    val_preds = model.predict_proba(X_val_scaled)[:, 1]

    score = roc_auc_score(y_val[target_col], val_preds)

    print("\n" + "*"*30)
    print(f"📊 简化模型诊断AUC: {score:.8f}")
    print("*"*30)

    if score > 0.51:
        print("\n[诊断结论] ✅ 好消息！预处理后的特征本身具有预测能力。")
        print("   问题很可能出在 'main.py' 的 '_run_fold_logic' 函数中，特别是与Autoencoder (AE) 相关的部分。")
        print("   请重点审查: AE的输入、AI特征的生成与拼接过程。")
    else:
        print("\n[诊断结论] ⚠️ 坏消息！即使跳过AE，模型也无法学习。")
        print("   问题根源很可能在更上游，请回顾步骤2和3的输出。")
        print("   如果预处理没问题，那可能是特征本身没有预测能力，或LGBM训练调用有问题。")

except Exception as e:
    print(f"❌ 在步骤4中发生错误: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 步骤 5: 黄金标准 - 完美特征测试 ---
print("\n--- 步骤 5: 黄金标准 - 完美特征测试 ---")
print("[信息] 我们将手动添加一个'完美特征'（即目标本身），看模型能否识别它。")
try:
    X_train_leaked = X_train_scaled.copy()
    X_val_leaked = X_val_scaled.copy()
    
    # 创造一个与目标完全相同的作弊特征
    leak_feature_name = 'perfect_feature_DO_NOT_USE'
    X_train_leaked[leak_feature_name] = y_train[target_col].values
    X_val_leaked[leak_feature_name] = y_val[target_col].values
    
    leaked_model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
    
    print("[信息] 正在训练带有'完美特征'的模型...")
    leaked_model.fit(X_train_leaked, y_train[target_col], sample_weight=sw_train)

    leaked_preds = leaked_model.predict_proba(X_val_leaked)[:, 1]
    leaked_score = roc_auc_score(y_val[target_col], leaked_preds)

    print("\n" + "*"*30)
    print(f"🏆 完美特征测试AUC: {leaked_score:.8f}")
    print("*"*30)

    if leaked_score > 0.99:
        print("\n[诊断结论] ✅ '完美特征'测试通过！LGBM的fit/predict流程本身是正常的。")
        print("   这进一步证明问题在于您现有特征集的信噪比过低。")
    else:
        print("\n[诊断结论] ❌ 致命缺陷！模型甚至无法利用一个完美的特征！")
        print(f"   这表明问题出在LGBM的调用方式上，或者 sample_weight 存在严重问题。")
        print("   请仔细检查 model.fit() 的参数，并回顾步骤 2.2 的样本权重检查。")

except Exception as e:
    print(f"❌ 在步骤5中发生错误: {e}")
    import traceback
    traceback.print_exc()
    exit()

print("\n" + "="*80)
print("🏁 [完成] 诊断程序运行结束。请根据以上输出分析问题。")
print("="*80)