# analyze_high_auc.py
# =================================================================
# “0.544高分专案组”专用调查脚本 V1.0
# 目的: 复现并分析导致意外高分的特征。
# =================================================================
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import warnings

# 导入您项目中的配置文件和工具库
import config
import utils 

warnings.filterwarnings('ignore', category=UserWarning)

print("="*80)
print("🚀 [启动] 0.544高分专案调查程序...")
print("="*80)

# --- 1. 加载数据 ---
print("\n--- 步骤 1: 加载数据 ---")
dev_df = utils.load_data(config.RAW_DATA_FILE, config.ANALYSIS_START_DATE_ID)
all_features = [c for c in dev_df.columns if c.startswith(('M', 'E', 'I', 'P', 'V', 'S', 'D')) or '_diff' in c or '_rol_' in c]
X = dev_df[all_features]
y = dev_df[config.TARGET_COLUMNS]
sample_weight = dev_df['sample_weight']
print("  > 数据加载完成。")

# --- 2. 精确模拟 Fold 0 的环境 ---
print("\n--- 步骤 2: 模拟 Fold 0 的数据切分 (无动态筛选) ---")
tscv = utils.get_cv_splitter(config.N_SPLITS, config.EMBARGO_SIZE, config.PURGE_SIZE)
train_idx, val_idx = next(tscv.split(X))
X_train, y_train, sw_train = X.iloc[train_idx], y.iloc[train_idx], sample_weight.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
print(f"  > Fold 0 数据切分完成，训练集大小: {len(X_train)}")

# --- 3. 使用有漏洞的 V5 预处理器处理数据 ---
print("\n--- 步骤 3: 使用旧版 V5 预处理器处理数据 ---")
# 在这个流程中，我们知道预处理器会产生大量NaN，这是“案发现场”的一部分
preprocessor_params = utils.get_preprocessor_params(X_train)
_, X_train_scaled = utils.apply_preprocessor(X_train, preprocessor_params)
_, X_val_scaled = utils.apply_preprocessor(X_val, preprocessor_params)
print(f"  > 数据已按旧方法处理，训练集NaN数量: {X_train_scaled.isnull().sum().sum()}")


# --- 4. 训练LGBM并复现分数 ---
print("\n--- 步骤 4: 训练LGBM模型并验证分数 ---")
target_col = config.PRIMARY_TARGET_COLUMN
model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train[target_col], sample_weight=sw_train)
val_preds = model.predict_proba(X_val_scaled)[:, 1]
score = roc_auc_score(y_val[target_col], val_preds)

print("\n" + "*"*35)
print(f"📊 复现的分数 AUC: {score:.8f}")
print("*"*35)
if abs(score - 0.5448) < 0.001:
    print("  > ✅ 成功！我们已稳定复现了高分现场！")
else:
    print("  > ⚠️ 警告: 复现的分数与目标有偏差，请检查环境。")


# --- 5. 核心调查：提取特征重要性 ---
print("\n--- 步骤 5: 提取并保存“功勋特征”列表 ---")
feature_importances = pd.Series(model.feature_importances_, index=X_train_scaled.columns)
# 过滤掉重要性为0的特征（包含那些全NaN的列）
feature_importances = feature_importances[feature_importances > 0]
feature_importances = feature_importances.sort_values(ascending=False)

output_filename = 'feature_importance_0.544_model.csv'
feature_importances.to_csv(output_filename)

print(f"  > ✅ 特征重要性已保存至 '{output_filename}'")
print("\n--- 请查看该文件，排在最前面的就是“功勋特征” ---")
print(feature_importances.head(20)) # 打印前20名

print("\n" + "="*80)
print("🏁 [完成] 调查结束。")
print("="*80)