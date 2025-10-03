
#=================================================================
#[冒烟测试专用] 项目配置文件 V2.1 (隔离版)
#=================================================================

# --- 1. 文件与路径 ---
# [核心修改] 更新为新的、统一的特征文件名
RAW_DATA_FILE = 'train_final_features.csv'
HOLDOUT_DATA_FILE = 'test_final_features.csv'

# 测试文件也更新
RANKING_FILE = 'feature_ranking_smoke_v13.csv'
LGBM_PARAMS_FILE = 'best_params_smoke_v13_lgbm.json'
AE_PARAMS_FILE = 'best_params_smoke_v13_ae.json'
OOF_OUTPUT_FILE = 'oof_predictions_smoke_v13.csv'
#--- 2. 验证策略 ---
#[冒烟测试修改] 只用2折CV，这是能跑通CV的最小值
N_SPLITS = 2
PURGE_SIZE = 1
EMBARGO_SIZE = 40
ANALYSIS_START_DATE_ID = 1055
N_LAST_FOLDS_TO_USE_INFERENCE = 2
#--- 3. 特征工程与筛选 ---
MISSING_THRESHOLD = 0.30
N_FEATURES_TO_SELECT_RFE = 60
N_TOP_FEATURES_TO_USE = 60
AI_PREFIX = 'AE_'
#--- 4. 目标定义 ---
# [关键修复] 冒烟测试也要使用软标签体系，与主配置保持一致
TARGET_COLUMNS = ['dls_target_1d', 'dls_target_3d', 'dls_target_5d']
ACTION_COLUMNS = ['action_1d', 'action_3d', 'action_5d']
RESP_COLUMNS = ['resp_1d', 'resp_3d', 'resp_5d']
PRIMARY_TARGET_COLUMN = 'dls_target_1d'
#--- 5. Optuna 配置 ---
#[冒烟测试修改] 只跑3次试验，足够验证代码逻辑是否正确
N_TRIALS_LGBM = 3
N_TRIALS_AE = 3