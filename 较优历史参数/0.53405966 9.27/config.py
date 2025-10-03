# config.py
# =================================================================
# 项目统一配置文件 (唯一的真理之源)
# =================================================================

# --- 1. 文件与路径 ---
RAW_DATA_FILE = 'train_v3_featured_raw.csv'
# AE特征文件将由集成化脚本在运行时生成，不再需要作为输入
# AE_FEATURES_FILE = '...' 

# 输出文件
RANKING_FILE = 'feature_ranking_v11.csv'
LGBM_PARAMS_FILE = 'best_params_v11_lgbm.json'
AE_PARAMS_FILE = 'best_params_v11_ae.json'
OOF_OUTPUT_FILE = 'oof_predictions_v11_final.csv'

# --- 2. 验证策略 ---
N_SPLITS = 5
PURGE_SIZE = 1
EMBARGO_SIZE = 40 
ANALYSIS_START_DATE_ID = 1055

# --- 3. 特征工程与筛选 ---
MISSING_THRESHOLD = 0.20 # 剔除缺失率超过20%的特征
N_FEATURES_TO_SELECT_RFE = 60 # RFE要筛选到多少个特征
N_TOP_FEATURES_TO_USE = 60 # <--- 新增参数：选择要使用的顶级特征数量。设置为-1则使用所有特征。
AI_PREFIX = 'AE_' # AI特征的前缀

# --- 4. Optuna 配置 ---
N_TRIALS_LGBM = 50
N_TRIALS_AE = 50