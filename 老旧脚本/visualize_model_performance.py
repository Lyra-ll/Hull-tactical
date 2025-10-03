# =================================================================
# visualize_model_performance.py (V1.0 - The Hall of Fame)
# 目的: To visualize and intuitively understand the real-world impact
#       of our models' performance improvements (e.g., AUC gains).
# =================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. Configuration =================
# --- Define the Out-of-Fold (OOF) prediction files for comparison ---
# NOTE: You will need to generate these files using final_battle_v4_clf.py
#       by setting the feature set and output filenames accordingly.
#       For now, let's assume we have them. I will show you how to generate them
#       if you don't have a separate one for the control group.

# Let's assume you have an OOF file for your old model (Control)
# and one for your new model (Test).
CONTROL_OOF_FILE = 'oof_predictions_CONTROL_original_only.csv'
TEST_OOF_FILE = 'oof_predictions_TEST_original_plus_ai.csv' 
# We will create these two files in the next step.

# --- Global Settings ---
N_DECILES = 10 # For the lift chart

plt.style.use('seaborn-v0_8-whitegrid')
print("--- Visualization Platform Initialized ---")

# ================= 2. Data Loading and Preparation Function =================
def load_and_prepare_oof(filepath, actual_returns_df):
    """Loads an OOF prediction file and merges it with actual returns."""
    try:
        oof_df = pd.read_csv(filepath)
        # Ensure date_id is the same type for merging
        actual_returns_df['date_id'] = actual_returns_df['date_id'].astype(int)
        oof_df['date_id'] = oof_df['date_id'].astype(int)
        
        # Merge to get the actual forward returns for each prediction
        merged_df = pd.merge(oof_df, actual_returns_df, on='date_id', how='inner')
        return merged_df
    except FileNotFoundError:
        print(f"ERROR: OOF file not found at '{filepath}'. Please generate it first.")
        return None

# ================= 3. Visualization Function 1: Lift Chart (Decile Analysis) =================
def plot_lift_chart(control_df, test_df, n_deciles=10):
    """Creates a decile analysis lift chart to compare model sorting power."""
    print("\n--- Generating Lift Chart (Decile Analysis) ---")
    
    # Calculate decile returns for both models
    control_df['decile'] = pd.qcut(control_df['oof_prediction'], n_deciles, labels=False, duplicates='drop')
    control_lift = control_df.groupby('decile')['actual_forward_returns'].mean() * 10000 # In basis points
    
    test_df['decile'] = pd.qcut(test_df['oof_prediction'], n_deciles, labels=False, duplicates='drop')
    test_lift = test_df.groupby('decile')['actual_forward_returns'].mean() * 10000 # In basis points

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(control_lift.index, control_lift.values, marker='o', linestyle='--', label='Control (Original Features)')
    plt.plot(test_lift.index, test_lift.values, marker='s', linestyle='-', label='Test (Original + AI Features)', color='green')
    
    plt.title('Model Lift Chart (Decile Analysis)', fontsize=18, fontweight='bold')
    plt.xlabel('Prediction Decile (0 = Lowest Confidence, 9 = Highest Confidence)', fontsize=12)
    plt.ylabel('Average Actual Forward Return (Basis Points)', fontsize=12)
    plt.xticks(range(n_deciles))
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    filename = 'lift_chart_comparison.png'
    plt.savefig(filename)
    print(f"✅ Lift chart saved to '{filename}'")
    plt.show()

# ================= 4. Visualization Function 2: Equity Curve =================
def plot_equity_curve(control_df, test_df):
    """Simulates a simple strategy and plots the cumulative returns (equity curve)."""
    print("\n--- Generating Equity Curve Simulation ---")

    # Simple strategy: position size = prediction - 0.5
    control_df['position'] = control_df['oof_prediction'] - 0.5
    control_df['strategy_return'] = control_df['position'] * control_df['actual_forward_returns']
    control_df['cumulative_return'] = (1 + control_df['strategy_return']).cumprod() - 1

    test_df['position'] = test_df['oof_prediction'] - 0.5
    test_df['strategy_return'] = test_df['position'] * test_df['actual_forward_returns']
    test_df['cumulative_return'] = (1 + test_df['strategy_return']).cumprod() - 1

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(control_df['date_id'], control_df['cumulative_return'] * 100, linestyle='--', label='Control (Original Features)')
    plt.plot(test_df['date_id'], test_df['cumulative_return'] * 100, linestyle='-', label='Test (Original + AI Features)', color='green', linewidth=2)

    plt.title('Simulated Strategy Equity Curve', fontsize=18, fontweight='bold')
    plt.xlabel('Date ID', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    filename = 'equity_curve_comparison.png'
    plt.savefig(filename)
    print(f"✅ Equity curve saved to '{filename}'")
    plt.show()

# ================= 5. Main Execution Block =================
if __name__ == "__main__":
    # First, we need the actual forward returns from the source
    raw_df = pd.read_csv('train_v3_featured_raw.csv')
    modern_df_returns = raw_df[raw_df['date_id'] > 1055][['date_id', 'forward_returns']].copy()
    modern_df_returns.rename(columns={'forward_returns': 'actual_forward_returns'}, inplace=True)

    # Load the OOF data for both models
    control_data = load_and_prepare_oof(CONTROL_OOF_FILE, modern_df_returns)
    test_data = load_and_prepare_oof(TEST_OOF_FILE, modern_df_returns)

    if control_data is not None and test_data is not None:
        # --- Generate the table ---
        print("\n" + "="*25 + " Performance Table " + "="*25)
        control_sharpe = (control_data['strategy_return'].mean() / control_data['strategy_return'].std()) * np.sqrt(252)
        test_sharpe = (test_data['strategy_return'].mean() / test_data['strategy_return'].std()) * np.sqrt(252)
        print(f"{'Metric':<20} | {'Control Model':<20} | {'Test Model':<20}")
        print("-" * 65)
        print(f"{'AUC':<20} | {pd.read_csv(CONTROL_OOF_FILE)['target'].mean():<20.6f} | {pd.read_csv(TEST_OOF_FILE)['target'].mean():<20.6f}")
        print(f"{'Final Return (%)':<20} | {control_data['cumulative_return'].iloc[-1]*100:<20.2f} | {test_data['cumulative_return'].iloc[-1]*100:<20.2f}")
        print(f"{'Sharpe Ratio':<20} | {control_sharpe:<20.2f} | {test_sharpe:<20.2f}")
        print("=" * 65)

        # --- Generate the plots ---
        plot_lift_chart(control_data, test_data)
        plot_equity_curve(control_data, test_data)
    else:
        print("\nExecution halted because one or both OOF files could not be loaded.")