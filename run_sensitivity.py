import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
import datetime
import platform
import sklearn
import torch
import xgboost as xgb # [新增] 用于计算纯监督的基线分数
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 1. 导入依赖与配置
# ============================================================

try:
    from preprocess_utils import load_data, preprocess_data
    from run_benchmark import DATA_DIR, get_data_splits
    from utils import set_seed
    from hesco import HeSCo
except ImportError as e:
    print(f"Error: 缺少必要文件。请确保 'hesco.py' 和 'run_benchmark.py' 在当前目录。\n详细错误: {e}")
    sys.exit(1)

# 全局配置
OUTPUT_CSV = "hesco_sensitivity_results.csv"
REPORT_FILE = "hesco_sensitivity_report.txt"
PLOT_DIR = "plots_hesco_sensitivity"
os.makedirs(PLOT_DIR, exist_ok=True)
VERBOSE = False

# ================= Configuration =================
TARGET_DATASETS = [
    "Pol.csv",              
    "Ames_Housing.csv",     
    "Elevators.csv",        
    "California_Housing.csv" 
]

SENSITIVITY_CONFIG = {
    # 证明伪标签权重的鲁棒性 (加入 0.0 与消融实验对齐)
    "unlabeled_weight_ratio": [0.0, 0.1, 0.3, 0.5, 0.8],
    
    # 证明平滑分位数 (你的核心创新) 不是一个脆弱的魔法数字
    "pinball_beta": [0.01, 0.05, 0.1, 0.5, 1.0], 
    
    # 证明质与量的权衡：过滤最自信的 10% 还是 70%？
    "conf_percentile": [10, 30, 50, 70, 90],
    
    # 互学习迭代的深浅
    "inc_trees": [10, 30, 50, 100],
    
    "batch_size": [128, 256, 512],
    
    # 标签稀疏度鲁棒性: 验证不同标签样本比例下的表现
    "labeled_ratio": [0.05, 0.1, 0.2, 0.3, 0.5]
}

DEFAULT_PARAMS = {
    "unlabeled_weight_ratio": 0.5,
    "adversarial_weight": 1.0,
    "unlabeled_batch_ratio": 0.5,
    "inc_trees": 30,
    "hidden_dim": 128,
    "batch_size": 256,
    "epochs": 100,
    "use_mutual_update": True,
    "use_ensemble": True,
    "pinball_beta": 0.1,
    "conf_percentile": 30
}

SEEDS = [42, 123, 456] 

# ================= 辅助函数 =================

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

def get_dataset_splits_with_preprocessing(dataset_name, seed, labeled_ratio=0.2):
    data_path = os.path.join(DATA_DIR, dataset_name)
    X, y = load_data(data_path)
    X_labeled, y_labeled, X_unlabeled, X_test, y_test = get_data_splits(
        X, y, test_size=0.5, labeled_ratio=labeled_ratio, random_state=seed
    )
    X_lbl_final, X_unl_final, X_test_final = preprocess_data(X_labeled, X_unlabeled, X_test)
    return X_lbl_final, y_labeled, X_unl_final, X_test_final, y_test

def compute_xgboost_baselines(datasets, seeds):
    """
    运行 Supervised XGBoost 获取基线分数，用于画图时的虚线参考。
    """
    print("\n--- Pre-computing Supervised XGBoost Baselines ---")
    baselines = {}
    xgb_params = {
        'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'tree_method': "hist", 'n_jobs': 4
    }
    
    for ds in datasets:
        r2_scores = []
        for seed in seeds:
            try:
                X_L, y_L, X_U, X_test, y_test = get_dataset_splits_with_preprocessing(ds, seed)
                model = xgb.XGBRegressor(**xgb_params, random_state=seed)
                model.fit(X_L, y_L)
                pred = model.predict(X_test)
                r2_scores.append(r2_score(y_test, pred))
            except Exception as e:
                print(f"Failed baseline for {ds} seed {seed}: {e}")
        if r2_scores:
            baselines[ds] = np.mean(r2_scores)
            print(f"[{ds}] XGBoost R2 Baseline: {baselines[ds]:.4f}")
        else:
            baselines[ds] = 0.0
    print("--------------------------------------------------\n")
    return baselines

def run_single_experiment(dataset, param_name, param_value, seed):
    current_labeled_ratio = DEFAULT_PARAMS.get("labeled_ratio", 0.2)
    if param_name == "labeled_ratio":
        current_labeled_ratio = param_value

    X_L, y_L, X_U, X_test, y_test = get_dataset_splits_with_preprocessing(dataset, seed, labeled_ratio=current_labeled_ratio)
    
    current_params = DEFAULT_PARAMS.copy()
    current_params[param_name] = param_value
    
    # 当权重参数为 0 时，等价于关闭互学习，节省开销
    if param_name in ["unlabeled_weight_ratio", "adversarial_weight"] and param_value == 0.0:
        current_params["use_mutual_update"] = False

    model_params = current_params.copy()
    if "labeled_ratio" in model_params:
        del model_params["labeled_ratio"]

    xgb_params = {
        'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'tree_method': "hist", 'n_jobs': 4, 'random_state': seed
    }

    model = HeSCo(xgb_params=xgb_params, standardize_input=False, random_state=seed, verbose=VERBOSE, **model_params)
    model.fit(X_L, y_L, X_U)
    preds = model.predict(X_test)
    
    return {
        "Dataset": dataset,
        "Parameter": param_name,
        "Value": param_value,
        "Seed": seed,
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds)
    }

# ================= 主流程 =================

def main():
    print(f"Starting Sensitivity Analysis at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_repro_info()
    
    # 计算基线时使用默认的 labeled_ratio (0.2)
    baselines = compute_xgboost_baselines(TARGET_DATASETS, SEEDS)

    results = []
    total_runs = sum(len(v) for v in SENSITIVITY_CONFIG.values()) * len(TARGET_DATASETS) * len(SEEDS)
    current_run = 0

    ratio_baselines = {} # {dataset: {ratio: mean_r2}}

    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
    else:
        existing_df = pd.DataFrame(columns=["Dataset", "Parameter", "Value", "Seed", "RMSE", "R2", "XGB_Baseline"])

    # Pre-calculate baselines for specific ratios if needed
    if "labeled_ratio" in SENSITIVITY_CONFIG:
        print("\n--- Pre-computing per-ratio XGBoost Baselines ---")
        for ds in TARGET_DATASETS:
            ratio_baselines[ds] = {}
            for ratio in SENSITIVITY_CONFIG["labeled_ratio"]:
                r2_scores = []
                for seed in SEEDS:
                    try:
                        X_L, y_L, X_U, X_test, y_test = get_dataset_splits_with_preprocessing(ds, seed, labeled_ratio=ratio)
                        model = xgb.XGBRegressor(
                            n_estimators=100, max_depth=6, learning_rate=0.05,
                            subsample=0.8, tree_method="hist", n_jobs=4, random_state=seed
                        )
                        model.fit(X_L, y_L)
                        pred = model.predict(X_test)
                        r2_scores.append(r2_score(y_test, pred))
                    except Exception as e:
                        print(f"Failed ratio baseline for {ds} ratio {ratio} seed {seed}: {e}")
                if r2_scores:
                    ratio_baselines[ds][ratio] = np.mean(r2_scores)
        print("--------------------------------------------------\n")

    new_results = []
    safe_values = pd.to_numeric(existing_df["Value"], errors='coerce')
    for param_name, values in SENSITIVITY_CONFIG.items():
        print(f"\n>> Testing Parameter: {param_name} over values {values}")
        for val in values:
            for ds in TARGET_DATASETS:
                for seed in SEEDS:
                    current_run += 1
                    
                    mask = (
                        (existing_df["Dataset"] == ds) &
                        (existing_df["Parameter"] == param_name) &
                        (np.isclose(safe_values, float(val), atol=1e-5)) &
                        (existing_df["Seed"] == seed)
                    )
                    is_done = mask.any()
                    
                    if is_done:
                        row = existing_df[mask].iloc[0].to_dict()
                        results.append(row)
                        continue
                        
                    print(f"[{current_run}/{total_runs}] Dataset: {ds} | {param_name}={val} | Seed={seed}")
                    try:
                        res = run_single_experiment(ds, param_name, val, seed)
                        # 添加基线信息到结果中，方便后续绘图
                        if param_name == "labeled_ratio":
                            # 如果有缓存的比例基线，直接使用；否则计算单次
                            res["XGB_Baseline"] = ratio_baselines[ds].get(val, 0.0)
                        else:
                            res["XGB_Baseline"] = baselines.get(ds, 0.0)
                            
                        results.append(res)
                        new_results.append(res)
                        pd.DataFrame([res]).to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
                    except Exception as e:
                        print(f"    ! Error running {ds} with {param_name}={val}, Seed {seed}: {e}")

    df = pd.DataFrame(results)
    # 填充缺失的 XGB_Baseline 列（针对旧缓存数据）
    if "XGB_Baseline" not in df.columns:
        df["XGB_Baseline"] = 0.0
    
    for idx, row in df.iterrows():
        if pd.isna(row["XGB_Baseline"]) or row["XGB_Baseline"] == 0.0:
            if row["Parameter"] == "labeled_ratio":
                df.at[idx, "XGB_Baseline"] = ratio_baselines.get(row["Dataset"], {}).get(row["Value"], 0.0)
            else:
                df.at[idx, "XGB_Baseline"] = baselines.get(row["Dataset"], 0.0)

    if len(new_results) > 0:
        print(f"\nNew experiments finished. Existing+New runs saved to {OUTPUT_CSV}")
    else:
        print(f"\nNo new experiments to run. Using cached results from {OUTPUT_CSV}")

    print("Generating Publication-Quality Plots...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False 
    
    palette = sns.color_palette("deep", len(TARGET_DATASETS))

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(f"HeSCo Sensitivity Analysis Report\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for param_name, values in SENSITIVITY_CONFIG.items():
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            axes_flat = axes.flatten()
            
            f.write(f"=== Parameter: {param_name} ===\n")
            
            for i, dataset in enumerate(TARGET_DATASETS):
                ax = axes_flat[i]
                data_ds = df[(df['Dataset'] == dataset) & (df['Parameter'] == param_name)]
                
                if data_ds.empty: continue
                
                # 画 HeSCo 折线
                sns.lineplot(
                    data=data_ds, x="Value", y="R2", 
                    marker="o", markersize=8, linewidth=2.5,
                    errorbar='sd', ax=ax, color=palette[i], label="HeSCo"
                )
                
                # 画基准
                if param_name == "labeled_ratio":
                    # 针对 labeled_ratio，XGBoost 的表现也会随比例变化，因此画成对应的折线
                    # 计算各个 Value 下的 XGB_Baseline 平均值
                    xgb_line = data_ds.groupby("Value")["XGB_Baseline"].mean().reset_index()
                    sns.lineplot(
                        data=xgb_line, x="Value", y="XGB_Baseline",
                        color='gray', linestyle='--', linewidth=2, marker="s", markersize=6,
                        ax=ax, label="XGB Baseline"
                    )
                else:
                    # 其他参数下，XGB 基线是固定的水平线
                    ax.axhline(y=baselines.get(dataset, 0.0), color='gray', linestyle='--', linewidth=2, label="XGB Baseline")
                
                if i >= 2:
                    ax.set_xlabel(param_name, fontsize=12) 
                else:
                    ax.set_xlabel("")
                    
                if i % 2 == 0:
                    ax.set_ylabel(r"Test $R^2$", fontsize=12)
                else:
                    ax.set_ylabel("")
                
                ax.legend(loc="best", fontsize=10, frameon=True)
                
                if param_name in ["inc_trees", "hidden_dim", "batch_size", "conf_percentile"] or \
                   (data_ds["Value"].nunique() < 10 and data_ds["Value"].dtype == 'int'):
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                
                ax.grid(True, linestyle='--', alpha=0.5)

                agg_data = data_ds.groupby("Value")["R2"].agg(['mean', 'std'])
                f.write(f"  Dataset: {dataset}\n")
                f.write(f"{agg_data.to_string()}\n")

            for j in range(len(TARGET_DATASETS), 4):
                 fig.delaxes(axes_flat[j])

            plt.tight_layout()
            filename = os.path.join(PLOT_DIR, f"sensitivity_{param_name}_R2.tif")
            plt.savefig(filename, dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
            try:
                # 只在能支持的系统上生成 PDF
                plt.savefig(filename.replace(".tif", ".pdf"), bbox_inches='tight') 
            except Exception:
                pass
            plt.close()
            f.write("\n")

    print(f"Plots saved to directory: {PLOT_DIR}/")
    print(f"Text report saved to: {REPORT_FILE}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
