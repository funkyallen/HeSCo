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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils import set_seed

# Import your algorithm classes
from hesco import HeSCo
# Add src to path if needed (Assuming files are in same dir for simplicity based on previous context)
# sys.path.append(...) 
from vime import VIME
from drill import Drill
from ucvme import UCVME
from rankup import RankUp
from co_training import CoTrainingRegressor

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._discretization")
warnings.filterwarnings("ignore", message=".*Bins whose width are too small.*")
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")
os.environ['LOKY_MAX_CPU_COUNT'] = '4' 

# ================= Configuration =================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEEDS = [42, 123, 456, 789, 1011]   # Random seeds
TEST_SIZE = 0.50           
LABELED_RATIO = 0.20       
VERBOSE = False            
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

def run_benchmark():
    # --- 1. Setup Logging ---
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 主日志 (包含所有打印信息)
    log_filename = os.path.join(log_dir, f"hesco_benchmark_log_{timestamp_str}.log")
    sys.stdout = Logger(log_filename)

    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Benchmark Run Start Time: {start_time_str}")
    print("="*80)
    print_repro_info()
    print("="*80)
    
    dataset_paths = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not dataset_paths:
        print(f"No CSV files found in {DATA_DIR}")
        return

    print("HeSCo vs XGBoost Benchmark (Incremental Logging Enabled)")
    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {[os.path.basename(p) for p in dataset_paths]}")
    print("="*80)
    
    # --- 2. Iterate Datasets ---
    for data_path in dataset_paths:
        dataset_name = os.path.basename(data_path)
        print(f"\nProcessing Dataset: {dataset_name}")
        print("-" * 80)
        
        # 存储当前数据集的结果
        # current_dataset_results[algo] = [ {RMSE:.., R2:..}, ... ]
        current_dataset_results = defaultdict(list)
        
        try:
            X, y = load_data(data_path)
            
            # --- 3. Iterate Seeds ---
            for seed in SEEDS:
                if VERBOSE:
                    print(f"\n  [Seed {seed}] Processing...")
                else:
                    # 打印进度点，不换行
                    sys.stdout.terminal.write(f".") 
                    sys.stdout.terminal.flush()

                set_seed(seed)
                
                X_lbl, y_lbl, X_unl, X_test, y_test = get_data_splits(
                    X, y, 
                    test_size=TEST_SIZE, 
                    labeled_ratio=LABELED_RATIO, 
                    random_state=seed
                )
                
                scaler_y = StandardScaler()
                y_lbl_scaled = scaler_y.fit_transform(y_lbl.reshape(-1, 1)).flatten()
                
                # --- Preprocess Data (Imputation, Encoding, Scaling) ---
                X_lbl_scaled, X_unl_scaled, X_test_scaled = preprocess_data(X_lbl, X_unl, X_test)
                
                # Note: preprocess_data returns numpy arrays
                # X_lbl, X_unl, X_test are still original DataFrames/Arrays if we need raw
                # But typically for models we use the processed versions
                
                # Update "raw" X variables to be the processed ones for compatibility with models expecting numeric input
                # However, models might expect specific things. 
                # Our new preprocess_data returns scaled and encoded data.
                
                # Important: XGBoost can handle raw data, but Neural Nets need processed.
                # Since we are comparing them, using the SAME processed data is fairest 
                # (except tree models might prefer Ordinal vs OneHot, which preprocess_data handles adaptively)
                
                # The "use_scaled_x" flag in existing code controlled whether to use scaled X.
                # Now X_lbl_scaled IS the standard input for everyone.
                
                # Let's override the old X_lbl with processed versions for the models to use
                # But wait, the loop below selects based on `use_scaled_x`.
                # If we want to use the new robust preprocessing, we should use the output of preprocess_data
                # as the "Scaled" version. The "Unscaled" version (raw) is hard to use because of strings.
                # So we should probably force all models to use the processed data.
                
                # Actually, the user requirement is:
                # "Fix data leakage... separate fit on X_lbl... adaptive encoding"
                # So `X_lbl_scaled` from `preprocess_data` is the CORRECT data to use.
                
                # The old code had `X_scaled = X` in load_and_preprocess (it was already processed globally).
                # So `X_lbl` was already processed.
                # Now `X` is RAW. So `X_lbl` is RAW.
                # `X_lbl_scaled` is PROCESSED.
                
                # So we MUST use `X_lbl_scaled` for all models because `X_lbl` might contain strings/NaNs.
                
                # We need to ensure `input_dim` is correct AFTER encoding.
                input_dim = X_lbl_scaled.shape[1]
                
                # --- Define Models ---
                # Format: (Name, ModelObj, Mode, UseScaledY, UseScaledX)
                xgb_params = {
                    'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 6,
                    'subsample': 0.8, 'tree_method': "hist", 'n_jobs': 4, 'random_state': seed
                }

                models = [
                    ("Supervised XGBoost", 
                     xgb.XGBRegressor(**xgb_params), "supervised", False, False),
                    
                    ("VIME", 
                     VIME(input_dim=input_dim, epochs_pretext=20, epochs_sup=50, lr=1e-3, verbose=VERBOSE, random_state=seed), "semi", True, True),
                     
                    ("DRILL", 
                     Drill(input_dim=input_dim, n_bins=10, epochs_proxy=30, epochs_reg=50, lr=1e-3, verbose=VERBOSE, random_state=seed), "semi", True, True),
                     
                    ("UCVME", 
                     UCVME(input_dim=input_dim, epochs=100, lr=1e-3, verbose=VERBOSE, random_state=seed), "semi", True, True),
                     
                    ("RankUp",
                     RankUp(input_dim=input_dim, epochs=100, lr=1e-3, verbose=VERBOSE, random_state=seed), "semi", True, True),

                    ("Co-Training",
                     CoTrainingRegressor(xgb_params=xgb_params, standardize_input=True, random_state=seed, verbose=VERBOSE), "semi", False, False),

                    # ("Mean Teacher", 
                    #  MeanTeacher(input_dim=input_dim, epochs=100, lr=1e-3, verbose=VERBOSE, random_state=seed), "semi", True, True),

                    ("HeSCo (Ours)",
                     HeSCo(random_state=seed, verbose=VERBOSE, standardize_input=False), "semi", False, True)
                ]
                
                # --- Train & Evaluate ---
                for name, model, mode, use_scaled_y, use_scaled_x in models:
                    start_time = time.time()
                    
                    # Prepare Data based on Model Requirements
                    # NOTE: With new preprocessing, "Unscaled" (Raw) data might contain strings/NaNs which will crash models.
                    # So we FORCE using the processed data (X_lbl_scaled) which is leakage-free and robust.
                    # We ignore `use_scaled_x` flag and always use the processed data.
                    
                    curr_X_lbl = X_lbl_scaled 
                    curr_y_lbl = y_lbl_scaled if use_scaled_y else y_lbl
                    curr_X_unl = X_unl_scaled 
                    curr_X_test = X_test_scaled
                    
                    try:
                        if mode == "supervised":
                            model.fit(curr_X_lbl, curr_y_lbl)
                        else:
                            model.fit(curr_X_lbl, curr_y_lbl, curr_X_unl)
                            
                        elapsed = time.time() - start_time
                        y_pred_raw = model.predict(curr_X_test)
                        
                        if use_scaled_y:
                            y_pred = scaler_y.inverse_transform(y_pred_raw.reshape(-1, 1)).flatten()
                        else:
                            y_pred = y_pred_raw
                        
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        current_dataset_results[name].append({
                            "RMSE": rmse, "R2": r2, "MAE": mae, "Time": elapsed
                        })
                        
                    except Exception as e:
                        print(f"\n    [Seed {seed}] {name} Failed: {str(e)}")
                        import traceback
                        traceback.print_exc()

            # --- 4. End of Dataset: Aggregate & Save Summary ---
            print("\n" + "-"*40)
            print(f"Summary for {dataset_name} (over {len(SEEDS)} seeds):")
            
            summary_rows = []
            
            # 使用 models 列表的顺序来保持输出一致性
            algo_names = [m[0] for m in models]
            
            for algo in algo_names:
                metrics_list = current_dataset_results.get(algo, [])
                
                # Filter NaNs
                rmses = [m["RMSE"] for m in metrics_list if not np.isnan(m["RMSE"])]
                r2s = [m["R2"] for m in metrics_list if not np.isnan(m["R2"])]
                times = [m["Time"] for m in metrics_list if not np.isnan(m["Time"])]
                
                if rmses:
                    mean_rmse, std_rmse = np.mean(rmses), np.std(rmses)
                    mean_r2, std_r2 = np.mean(r2s), np.std(r2s)
                    
                    maes = [m["MAE"] for m in metrics_list if not np.isnan(m["MAE"])]
                    mean_mae, std_mae = np.mean(maes), np.std(maes)
                    
                    mean_time, std_time = np.mean(times), np.std(times)
                    
                    # Prepare for Console Display
                    summary_rows.append({
                        "Algorithm": algo,
                        "RMSE": f"{mean_rmse:.4f} ± {std_rmse:.4f}",
                        "R2": f"{mean_r2:.4f} ± {std_r2:.4f}",
                        "MAE": f"{mean_mae:.4f} ± {std_mae:.4f}",
                        "Time": f"{mean_time:.2f}s",
                        "_sort": mean_rmse
                    })
                else:
                    pass

            # Display Table for this dataset
            if summary_rows:
                df_summary = pd.DataFrame(summary_rows).sort_values(by="_sort")
                df_summary = df_summary.drop(columns=["_sort"])
                table_str = df_summary.to_string(index=False)
                print(table_str)
            else:
                print("No successful runs for this dataset.")
            print("-" * 40)

        except Exception as e:
            print(f"\nCRITICAL ERROR processing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    completion_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nBenchmark Completed Time: {completion_time_str}")
    print(f"Global Log saved to: {log_filename}")

if __name__ == "__main__":
    run_benchmark()
