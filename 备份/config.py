# config.py
# =================================================================
# 项目统一配置文件 (V2.2 - 冠军策略版)
# =================================================================

# --- 1. 文件与路径 ---
RAW_DATA_FILE = 'train_lite_features.csv' # 用于模型开发和CV
HOLDOUT_DATA_FILE = 'test_lite_features.csv' # 用于最终评估

RANKING_FILE = 'feature_ranking_v13.csv' 
LGBM_PARAMS_FILE = 'best_params_v13_lgbm.json'
AE_PARAMS_FILE = 'best_params_v13_ae.json'
OOF_OUTPUT_FILE = 'oof_predictions_v13_final.csv'

# --- 2. 验证策略 ---
N_SPLITS = 5
PURGE_SIZE = 1
EMBARGO_SIZE = 40 
ANALYSIS_START_DATE_ID = 1055
# [核心升级] 定义在最终验证/预测时，只使用最后几折训练出的模型
N_LAST_FOLDS_TO_USE_INFERENCE = 3 

# --- 3. 特征工程与筛选 ---
MISSING_THRESHOLD = 0.30 
N_FEATURES_TO_SELECT_RFE = 100
N_TOP_FEATURES_TO_USE = 30 
AI_PREFIX = 'AE_'

# --- 4. 目标定义 ---
# [核心修改] 将我们的主要训练目标，从硬标签切换到新的软标签
TARGET_COLUMNS = ['dls_target_1d', 'dls_target_3d', 'dls_target_5d']

# 我们仍然需要旧的action列，用于最终的AUC评估
ACTION_COLUMNS = ['action_1d', 'action_3d', 'action_5d']

# [核心升级] 定义与目标对应的原始收益列，用于计算样本权重
RESP_COLUMNS = ['resp_1d', 'resp_3d', 'resp_5d']
# 主目标也相应切换（虽然在回归任务中意义减弱，但保持一致性）
PRIMARY_TARGET_COLUMN = 'dls_target_1d' 

# --- 5. Optuna 配置 ---
N_TRIALS_LGBM = 5
N_TRIALS_AE = 5

# --- 6. GPU 配置 ---
# 是否尝试让LightGBM使用GPU（需要本机支持GPU版LightGBM编译）
LGBM_USE_GPU = False