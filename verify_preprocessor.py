# verify_preprocessor.py
# =================================================================
# “预处理器”专项审查脚本 V1.0
# 目的: 靶向检验 get_preprocessor_params 和 apply_preprocessor 函数
#       是否在处理过程中破坏了特征。
# =================================================================
import pandas as pd
import numpy as np
import warnings

# 导入您项目中的配置文件和工具库
import config
import utils

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

print("="*80)
print("🚀 [启动] 'utils.py' 预处理器专项审查程序...")
print("="*80)

# --- 步骤 1: 准备“案发现场”数据 ---
print("\n--- 步骤 1: 加载数据并提取第一折 (Fold 0) 的训练集 ---")
try:
    dev_df = utils.load_data(config.RAW_DATA_FILE, config.ANALYSIS_START_DATE_ID)
    all_features = [c for c in dev_df.columns if c.startswith(('M', 'E', 'I', 'P', 'V', 'S', 'D')) or '_diff' in c or '_rol_' in c]
    X = dev_df[all_features]
    
    tscv = utils.get_cv_splitter(config.N_SPLITS, config.EMBARGO_SIZE, config.PURGE_SIZE)
    train_idx, _ = next(tscv.split(X))
    # Purge a a final row for consistency with main.py
    purged_train_idx = train_idx[:-config.PURGE_SIZE] if config.PURGE_SIZE > 0 else train_idx
    X_train_fold0 = X.iloc[purged_train_idx]
    
    print(f"  > Fold 0 的训练数据准备完毕，共 {len(X_train_fold0)} 行。")

except Exception as e:
    print(f"❌ 在准备数据时发生错误: {e}")
    exit()

# --- 步骤 2: 选取“检验样本”并审查处理前状态 ---
# 我们选取一个之前表现正常的特征 M4 和一个已知在Fold 0有问题的特征 E7
features_to_inspect = ['M4', 'E7'] 
print(f"\n--- 步骤 2: 审查特征 {features_to_inspect} 在处理前的状态 ---")

try:
    pre_inspection_df = X_train_fold0[features_to_inspect].describe().transpose()
    print(pre_inspection_df)
    print("\n[观察提示]：请注意 'count' 列。如果 count 为 0，说明该特征在本折中完全是NaN。")
except KeyError:
    print(f"❌ 错误: 原始数据中找不到指定的特征 {features_to_inspect}。")
    exit()
    
# --- 步骤 3: 执行预处理函数 ---
print("\n--- 步骤 3: 正在调用 get_preprocessor_params 和 apply_preprocessor ---")
# 我们假设使用的是未修复的 V5 版本 utils.py，以重现问题
preprocessor_params = utils.get_preprocessor_params(X_train_fold0)
_, X_train_fold0_scaled = utils.apply_preprocessor(X_train_fold0, preprocessor_params)
print("  > 预处理执行完毕。")

# --- 步骤 4: 审查处理后状态 ---
print(f"\n--- 步骤 4: 审查特征 {features_to_inspect} 在处理后的状态 ---")
post_inspection_df = X_train_fold0_scaled[features_to_inspect].describe().transpose()
print(post_inspection_df)

# --- 步骤 5: 最终诊断 ---
print("\n--- 步骤 5: 最终诊断结论 ---")

# 检查 M4 是否被正常处理
m4_after = post_inspection_df.loc['M4']
if abs(m4_after['mean']) < 0.1 and abs(m4_after['std'] - 1) < 0.1:
    print("✅ [正常特征 'M4'] 诊断通过：特征被成功标准化 (均值≈0, 标准差≈1)。函数工作正常。")
else:
    print("❌ [正常特征 'M4'] 诊断失败：特征未被正确标准化。")

# 检查 E7 是否导致了问题
e7_after = post_inspection_df.loc['E7']
if e7_after['count'] == 0:
    print("❌ [问题特征 'E7'] 诊断确认：该特征在处理后仍然是完全的NaN。")
    print("   [根本原因]: 您当前的预处理函数，在面对一个整列都是NaN的输入时，无法正确处理，导致NaN“泄漏”。")
    print("   [结论]: 函数本身没有“破坏”正常特征，但它不够“健壮”，无法处理这种极端情况。")
else:
    print("✅ [问题特征 'E7'] 诊断通过：特征被成功处理，没有发现NaN泄漏。")

print("\n" + "="*80)
print("🏁 [完成] 审查结束。")
print("="*80)