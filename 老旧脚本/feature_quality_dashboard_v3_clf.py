# =================================================================
# feature_quality_dashboard_v3_clf.py (V3.0 - Weighted Classification Ranking)
# 目的: A major upgrade to our ranking system. It now uses the EXACT same
#       methodology (Weighted Classification) and optimal parameters as our
#       champion model to produce the most honest and relevant feature ranking.
# =================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import warnings
import json

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

print("\n--- Initializing Feature Intelligence Agency (V3 - Weighted Classification Mode) ---")

# ================= 1. Load All Data and The Champion's Blueprint =================
try:
    raw_df = pd.read_csv('train_v3_featured_raw.csv')
    ae_features_df = pd.read_csv('train_v11_autotune_clf_ae_features.csv') 
    # <--- CORE UPGRADE #1: Load the optimal parameters for our champion LightGBM classifier
    with open('best_params_v4_weighted_clf.json', 'r') as f:
        best_lgbm_params = json.load(f)
    print("Successfully loaded raw features, AI features, and the champion's parameter blueprint.")
except FileNotFoundError as e:
    print(f"ERROR: A required file was not found - {e}")
    exit()

df = pd.merge(raw_df, ae_features_df, on='date_id', suffixes=('_orig', None))
if 'forward_returns_orig' in df.columns:
    df.drop(columns=['forward_returns_orig'], inplace=True)
modern_df = df[df['date_id'] > 1055].copy().reset_index(drop=True)

# ================= 2. Define Features and The Weighted Classification Target =================
all_features = [c for c in modern_df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']]
X = modern_df[all_features]

# <--- CORE UPGRADE #2: Prepare the target and weights exactly as in our winning model
y = (modern_df['forward_returns'] > 0).astype(int)
sample_weight = modern_df['forward_returns'].abs()

print(f"Feature set and weighted classification target prepared. Total features to be ranked: {len(all_features)}")

# ================= 3. Cross-Validation with the Champion's Mindset =================
N_SPLITS = 5
PURGE_SIZE = 1
EMBARGO_SIZE = 40
feature_importances = pd.DataFrame(index=all_features)

tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=EMBARGO_SIZE)

# <--- CORE UPGRADE #3: Use the optimal parameters, ensuring our "judge" is an expert
final_params = best_lgbm_params.copy()
final_params.update({
    'objective': 'binary', 
    'metric': 'auc', 
    'random_state': 42, 
    'n_jobs': -1,
    'importance_type': 'gain' # Use 'gain' for a robust measure of importance
})


for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"--- Interrogating features in Fold {fold + 1}/{N_SPLITS} ---")
    
    purged_train_idx = train_idx[:-PURGE_SIZE]

    X_train_raw, y_train = X.iloc[purged_train_idx], y.iloc[purged_train_idx]
    X_val_raw, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Get corresponding sample weights
    sw_train = sample_weight.iloc[purged_train_idx]
    sw_val = sample_weight.iloc[val_idx]
    
    # Standardize data
    mean = np.nanmean(X_train_raw.values, axis=0)
    std = np.nanstd(X_train_raw.values, axis=0)
    std[std == 0] = 1.0
    
    X_train = pd.DataFrame(np.nan_to_num((X_train_raw.values - mean) / std, nan=0.0), columns=all_features)
    X_val = pd.DataFrame(np.nan_to_num((X_val_raw.values - mean) / std, nan=0.0), columns=all_features)
    
    # <--- CORE UPGRADE #4: Use LGBMClassifier, not Regressor
    model = lgb.LGBMClassifier(**final_params)
    
    # <--- CORE UPGRADE #5: Fit the model with sample weights
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)], 
              sample_weight=sw_train,
              eval_sample_weight=[sw_val],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    
    feature_importances[f'fold_{fold+1}'] = model.feature_importances_

# ================= 4. Calculate and Visualize The "True" Ranking =================
feature_importances['average'] = feature_importances.mean(axis=1)
top_50_features = feature_importances.sort_values(by='average', ascending=False).head(50)

# ================= 5. Save the New, Honest Blueprint =================
ranked_output_filename = 'ranked_features_v3_clf.csv'
feature_importances.sort_values(by='average', ascending=False).to_csv(ranked_output_filename)
print(f"\n--- Top 50 Feature Importance Ranking (Weighted Classification Method) ---")
print(top_50_features[['average']])

plt.figure(figsize=(12, 16)) # Taller figure for more features
sns.barplot(x='average', y=top_50_features.index, data=top_50_features, palette='viridis')
plt.title('The True Top 50 Features (Judged by Weighted Classification)', fontsize=18, fontweight='bold')
plt.xlabel('Average Importance Score (Gain)', fontsize=14)
plt.ylabel('Feature Name', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plot_output_filename = 'all_features_importance_top50_v3_clf.png'
plt.savefig(plot_output_filename)

print(f"\n✅ The TRUE feature importance ranking has been saved to '{ranked_output_filename}'")
print(f"✅ The new importance chart has been saved to '{plot_output_filename}'")

