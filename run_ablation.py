import warnings
import os
import sys
import datetime
import platform
from collections import defaultdict
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost as xgb
import glob
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from utils import set_seed

# 导入你的消融版 HeSCo 类
from hesco import HeSCo

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._discretization")
warnings.filterwarnings("ignore", message=".*Bins whose width are too small.*")
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# ================= Configuration =================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEEDS = [42, 123, 456, 789, 1011, 2024, 2025, 888, 999, 1111]   # 10个随机种子
TARGET_DATASETS = [
    "Pol.csv",              
    "Ames_Housing.csv",     
    "Diamonds.csv",        
    "California_Housing.csv",
    "Elevators.csv",
    "House_Sales.csv"
]
TEST_SIZE = 0.50           
LABELED_RATIO = 0.20       
VERBOSE = False            

# 显著性分析配置
ENABLE_SIGNIFICANCE_TEST = True 
SIGNIFICANCE_BASELINE = "HeSCo (Full)" # 基准模型名称 (必须与 configs 中的名称一致)
# ===============================================

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

from preprocess_utils import load_data, preprocess_data

def get_data_splits(X, y, test_size=0.5, labeled_ratio=0.1, random_state=42):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    n_total_train = len(X_train_full)
    n_labeled = int(n_total_train * labeled_ratio)
    
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_total_train)
    rng.shuffle(indices)
    
    labeled_idx = indices[:n_labeled]
    unlabeled_idx = indices[n_labeled:]
    
    if hasattr(X_train_full, 'iloc'):
        X_labeled = X_train_full.iloc[labeled_idx]
        X_unlabeled = X_train_full.iloc[unlabeled_idx]
    else:
        X_labeled = X_train_full[labeled_idx]
        X_unlabeled = X_train_full[unlabeled_idx]
        
    y_labeled = y_train_full[labeled_idx]
    
    return X_labeled, y_labeled, X_unlabeled, X_test, y_test

def holm_bonferroni_correction(p_values):
    """
    Apply Holm-Bonferroni correction to a list of p-values.
    Returns adjusted p-values.
    """
    if not p_values:
        return []
    
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values and keep original indices
    sorted_indices = np.argsort(p_values)
    sorted_p_vals = p_values[sorted_indices]
    
    adjusted_p_vals = np.zeros(n)
    
    # Calculate adjusted p-values (Step-down)
    # P_adj[i] = max(P_adj[i-1], min(1, (n - i) * P[i]))
    prev_adj_p = 0
    for i in range(n):
        m = n - i  # Number of tests remaining
        current_adj = sorted_p_vals[i] * m
        
        # Enforce monotonicity
        current_adj = max(current_adj, prev_adj_p)
        # Cap at 1.0
        current_adj = min(1.0, current_adj)
        
        adjusted_p_vals[i] = current_adj
        prev_adj_p = current_adj
        
    # Restore original order
    final_adjusted_p = np.zeros(n)
    final_adjusted_p[sorted_indices] = adjusted_p_vals
    
    return final_adjusted_p.tolist()

def print_repro_info():
    print("Reproducibility:")
    print(f"  - OS: {platform.platform()}")
    print(f"  - Python: {sys.version.split()[0]}")
    print(f"  - numpy: {np.__version__}")
    print(f"  - pandas: {pd.__version__}")
    print(f"  - sklearn: {sklearn.__version__}")
    print(f"  - xgboost: {xgb.__version__}")
    print(f"  - torch: {torch.__version__}")
    print(f"  - cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - cuda_version: {torch.version.cuda}")
        print(f"  - cudnn_version: {torch.backends.cudnn.version()}")
        print(f"  - cudnn_deterministic: {torch.backends.cudnn.deterministic}")
        print(f"  - cudnn_benchmark: {torch.backends.cudnn.benchmark}")

def run_ablation_incremental():
    # --- 1. Setup Logging ---
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Text Log (Full Output)
    log_filename = os.path.join(log_dir, f"hesco_ablation_results_{timestamp_str}.log")
    sys.stdout = Logger(log_filename)

    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Ablation Study Start Time: {start_time_str}")
    print("="*100)
    print_repro_info()
    print("="*100)
    print(f"Configuration:")
    print(f"  - Significance Test: {ENABLE_SIGNIFICANCE_TEST} (Baseline: {SIGNIFICANCE_BASELINE})")
    print(f"  - Seeds: {SEEDS}")
    print(f"  - Logs Directory: {log_dir}")
    print("="*100)
    
    dataset_paths = [os.path.join(DATA_DIR, name) for name in TARGET_DATASETS]
    dataset_paths = [p for p in dataset_paths if os.path.exists(p)]
    
    if not dataset_paths:
        print(f"No specified CSV files found in {DATA_DIR}")
        return
    
    # --- 2. Iterate Datasets ---
    all_summary_rows = []

    for data_path in dataset_paths:
        dataset_name = os.path.basename(data_path)
        print(f"\nProcessing Dataset: {dataset_name}")
        print("-" * 80)
        
        # Store results for this dataset: metrics[variant] = [ {RMSE:.., R2:..}, ... ]
        current_dataset_metrics = defaultdict(list)
        
        try:
            X, y = load_data(data_path)
            
            # --- 3. Iterate Seeds ---
            for seed in SEEDS:
                if VERBOSE:
                    print(f"\n  [Seed {seed}] Processing...")
                else:
                    sys.stdout.terminal.write(".")
                    sys.stdout.terminal.flush()
                
                set_seed(seed)
                
                X_lbl, y_lbl, X_unl, X_test, y_test = get_data_splits(
                    X, y, 
                    test_size=TEST_SIZE, 
                    labeled_ratio=LABELED_RATIO, 
                    random_state=seed
                )
                
                # --- Preprocess Data (Imputation, Encoding, Scaling) ---
                X_lbl_scaled, X_unl_scaled, X_test_scaled = preprocess_data(X_lbl, X_unl, X_test)
                
                xgb_params = {
                    'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 6,
                    'subsample': 0.8, 'tree_method': "hist", 'n_jobs': 4, 'random_state': seed
                }

                # Define Ablation Variants
                configs = [
                    ("Supervised XGBoost", 
                     xgb.XGBRegressor(**xgb_params), "supervised"),

                    ("HeSCo (Full)",
                     HeSCo(random_state=seed, verbose=VERBOSE,
                           use_mutual_update=True, use_ensemble=True), "semi"),

                    ("w/o Mutual Update",
                     HeSCo(random_state=seed, verbose=VERBOSE,
                           use_mutual_update=False, use_ensemble=True), "semi"),

                    ("w/o Ensemble (NN Only)",
                     HeSCo(random_state=seed, verbose=VERBOSE,
                           use_mutual_update=True, use_ensemble=False), "semi"),
                ]
                
                for name, model, mode in configs:
                    start_time = time.time()
                    try:
                        if mode == "supervised":
                            model.fit(X_lbl_scaled, y_lbl)
                        else:
                            model.fit(X_lbl_scaled, y_lbl, X_unl_scaled)
                            
                        elapsed = time.time() - start_time
                        y_pred = model.predict(X_test_scaled)
                        
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        # Store in memory for summary
                        current_dataset_metrics[name].append({
                            "RMSE": rmse, "R2": r2, "MAE": mae, "Time": elapsed
                        })
                        
                    except Exception as e:
                        print(f"\n    [Seed {seed}] {name} Failed: {str(e)}")
                        import traceback
                        traceback.print_exc()
                             
                        # Store NaN metric to indicate failure for this seed
                        current_dataset_metrics[name].append({
                            "RMSE": np.nan, "R2": np.nan, "MAE": np.nan, "Time": np.nan, "Seed": seed
                        })
                        
                    # Also store successful seeds
                    if name in current_dataset_metrics and not np.isnan(current_dataset_metrics[name][-1]["RMSE"]):
                         current_dataset_metrics[name][-1]["Seed"] = seed

            # --- 4. End of Dataset: Aggregate, Test Significance, Sort & Save ---
            print("\n" + "-"*100)
            print(f"Ablation Summary for {dataset_name} (sorted by RMSE):")
            
            # Prepare Baseline Samples for T-Test
            # Store as dict: {seed: rmse}
            baseline_seed_map = {}
            if ENABLE_SIGNIFICANCE_TEST and SIGNIFICANCE_BASELINE in current_dataset_metrics:
                baseline_metrics = current_dataset_metrics[SIGNIFICANCE_BASELINE]
                for m in baseline_metrics:
                    if "Seed" in m and not np.isnan(m["RMSE"]):
                        baseline_seed_map[m["Seed"]] = m["RMSE"]

            # --- Collect P-values for Multiple Comparison Control (Holm-Bonferroni) ---
            config_names = [c[0] for c in configs]
            raw_p_values = []
            variants_for_test = []
            
            if ENABLE_SIGNIFICANCE_TEST and baseline_seed_map:
                for algo in config_names:
                    if algo == SIGNIFICANCE_BASELINE:
                        continue
                        
                    metrics_list = current_dataset_metrics.get(algo, [])
                    
                    # Align seeds
                    current_seed_map = {}
                    for m in metrics_list:
                        if "Seed" in m and not np.isnan(m["RMSE"]):
                            current_seed_map[m["Seed"]] = m["RMSE"]
                            
                    common_seeds = sorted(list(set(baseline_seed_map.keys()) & set(current_seed_map.keys())))
                    
                    if len(common_seeds) > 1:
                        b_samples = [baseline_seed_map[s] for s in common_seeds]
                        v_samples = [current_seed_map[s] for s in common_seeds]
                        
                        try:
                            # Paired T-test (Two-sided)
                            _, p_val = stats.ttest_rel(b_samples, v_samples)
                            raw_p_values.append(p_val)
                            variants_for_test.append(algo)
                        except Exception:
                            pass # Skip failed tests

            # Apply Holm-Bonferroni Correction
            adjusted_p_values = holm_bonferroni_correction(raw_p_values)
            adj_p_map = dict(zip(variants_for_test, adjusted_p_values))

            summary_rows = []
            
            for algo in config_names:
                metrics_list = current_dataset_metrics.get(algo, [])
                
                rmses = [m["RMSE"] for m in metrics_list if not np.isnan(m["RMSE"])]
                r2s = [m["R2"] for m in metrics_list if not np.isnan(m["R2"])]
                maes = [m["MAE"] for m in metrics_list if not np.isnan(m["MAE"])]
                times = [m["Time"] for m in metrics_list if not np.isnan(m["Time"])]
                
                if not rmses:
                    summary_rows.append({
                        "Variant": algo, "RMSE": "N/A", "R2": "N/A", "MAE": "N/A", "Time (s)": "N/A", "P-Value": "-",
                        "_sort_key": float('inf') # Put failed runs at bottom
                    })
                    continue

                mean_rmse, std_rmse = np.mean(rmses), np.std(rmses)
                mean_r2, std_r2 = np.mean(r2s), np.std(r2s)
                mean_mae, std_mae = np.mean(maes), np.std(maes)
                mean_time, std_time = np.mean(times), np.std(times)
                
                # --- Significance Test Result (Adjusted) ---
                p_value_str = "-"
                
                if ENABLE_SIGNIFICANCE_TEST and baseline_seed_map:
                    if algo == SIGNIFICANCE_BASELINE:
                        p_value_str = "(Ref)"
                    elif algo in adj_p_map:
                        adj_p = adj_p_map[algo]
                        sig_mark = ""
                        if adj_p < 0.01: sig_mark = "**"
                        elif adj_p < 0.05: sig_mark = "*"
                        
                        p_value_str = f"{adj_p:.4f}{sig_mark}"
                    else:
                        p_value_str = "N/A"
                
                summary_rows.append({
                    "Variant": algo,
                    "RMSE": f"{mean_rmse:.4f} ± {std_rmse:.4f}",
                    "R2": f"{mean_r2:.4f} ± {std_r2:.4f}",
                    "MAE": f"{mean_mae:.4f} ± {std_mae:.4f}",
                    "Time (s)": f"{mean_time:.2f}s",
                    "P-Value": p_value_str,
                    "_sort_key": mean_rmse  # For sorting
                })

                # Add to global summary
                all_summary_rows.append({
                    "Dataset": dataset_name,
                    "Variant": algo,
                    "RMSE": f"{mean_rmse:.4f} ± {std_rmse:.4f}",
                    "R2": f"{mean_r2:.4f} ± {std_r2:.4f}",
                    "MAE": f"{mean_mae:.4f} ± {std_mae:.4f}",
                    "Time (s)": f"{mean_time:.2f}s",
                    "P-Value": p_value_str,
                    "_sort_key": (dataset_name, mean_rmse) # For sorting
                })

            # --- Sort and Display ---
            if summary_rows:
                # Sort by Mean RMSE (Ascending) -> Best models on top
                df_summary = pd.DataFrame(summary_rows).sort_values(by="_sort_key")
                
                # Drop helper column for display
                table_str = df_summary.drop(columns=["_sort_key"]).to_string(index=False)
                print(table_str)
            else:
                print("No successful runs.")
            
            print("-" * 100)

        except Exception as e:
            print(f"\nCRITICAL ERROR processing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # --- 5. Global Summary Table ---
    if all_summary_rows:
        print("\n" + "="*100)
        print(f"FINAL ABLATION RESULTS w/ SIGNIFICANCE TEST (Baseline: {SIGNIFICANCE_BASELINE})")
        print("="*100)
        
        df_global = pd.DataFrame(all_summary_rows)
        # Sort by Dataset then RMSE (using _sort_key which is a tuple)
        # However, _sort_key is not in column if we print directly unless we keep it
        # Let's rebuild _sort_key column for sorting
        
        # Sort
        df_global.sort_values(by=["Dataset", "Variant"], key=lambda x: x if x.name == "Dataset" else [r for r in range(len(x))], inplace=True)
        # Better: Sort by dataset name, then by RMSE score which was implicit in the local sort order?
        # Actually let's just use the appended order which is dataset alphabetical (from glob) -> variant order
        # But wait, glob order is not guaranteed alphabetical.
        # Let's sort explicitly by Dataset Name
        
        # We want to group by Dataset and sort within dataset by RMSE (best first)
        # But we don't have raw RMSE in all_summary_rows, only string.
        # Let's rely on _sort_key
        
        df_global = df_global.sort_values(by="_sort_key")
        
        # Drop helper column
        final_table_str = df_global.drop(columns=["_sort_key"]).to_string(index=False)
        print(final_table_str)

    print(f"Global Log saved to: {log_filename}")

if __name__ == "__main__":
    run_ablation_incremental()
