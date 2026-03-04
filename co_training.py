
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from utils import set_seed
import warnings
import os
import copy

# 环境配置
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 1. 核心损失函数模块 (From nr_sgp.py)
# =============================================================================

class RobustLoss(nn.Module):
    """
    Log-Cosh Loss: 
    比 MSE 对异常值更鲁棒，比 MAE 在零点处更平滑（二阶可导）。
    Loss = log(cosh(y_pred - y_true)) ≈ x^2/2 (x small) or |x| - log(2) (x large)
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = torch.tensor(alpha).to(DEVICE)
            
    def forward(self, pred, target):
        diff = pred - target
        # 使用 log(1 + x^2) 近似 log(cosh(x)) 的效果，计算更稳定
        loss = torch.log(1 + (diff ** 2) / self.alpha)
        return loss.mean()

# =============================================================================
# 2. 神经网络架构 (From nr_sgp.py)
# =============================================================================

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout), 
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.net(x)
        return self.relu(x + out)

class RegressionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=2, dropout=0.1):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 残差骨干网络
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        # 预测头
        self.pred_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.pred_head(h)

# =============================================================================
# 3. Co-Training 核心算法类
# =============================================================================

class CoTrainingRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        # --- Co-Training Params ---
        max_iter=3,             # 最大迭代轮数
        samples_per_iter=500,      # 每轮每个视图选择的高置信度样本数
        pool_size=5000,          # 每轮从 Unlabeled Pool 中采样的候选集大小
        
        # --- XGBoost Params ---
        xgb_params=None,
        
        # --- NN Params ---
        hidden_dim=128,
        num_blocks=2,
        dropout=0.1,
        lr=1e-3,
        batch_size=256,
        epochs=50,               # 每轮 Co-Training 的训练轮数
        
        # --- Common Params ---
        standardize_input=True,
        feature_subsample_ratio=0.8,
        random_state=42,
        verbose=False
    ):
        # Co-Training 参数
        self.max_iter = max_iter
        self.samples_per_iter = samples_per_iter
        self.pool_size = pool_size
        
        # XGB 参数
        self.xgb_params = xgb_params if xgb_params else {
            'n_estimators': 100, 
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'n_jobs': 0,
            'tree_method': 'hist',
            'random_state': random_state
        }
        
        # NN 参数
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        
        # 通用参数
        self.standardize_input = standardize_input
        self.feature_subsample_ratio = feature_subsample_ratio
        self.random_state = random_state
        self.verbose = verbose
        
        # 内部状态
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.xgb_model = None
        self.nn_model = None
        
        self.view1_cols = None
        self.view2_cols = None
        
        self.L1_X = None # Labeled set for View 1 (XGB)
        self.L1_y = None
        self.L2_X = None # Labeled set for View 2 (NN)
        self.L2_y = None

    def _train_xgboost(self, X, y):
        """ 训练一个全新的 XGBoost """
        params = self.xgb_params.copy()
        params['random_state'] = self.random_state
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        return model

    def _train_nn(self, X, y, init_model=None):
        """ 训练神经网络 (支持 fine-tuning 或从头训练) """
        if init_model:
            model = copy.deepcopy(init_model)
        else:
            model = RegressionNetwork(
                input_dim=X.shape[1],
                hidden_dim=self.hidden_dim,
                num_blocks=self.num_blocks,
                dropout=self.dropout
            ).to(DEVICE)
            
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        # 较短的 scheduler，适应快速迭代
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = RobustLoss(alpha=1.0).to(DEVICE)
        
        t_X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        t_y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
        
        dataset = TensorDataset(t_X, t_y)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(t_X)), shuffle=True)
        
        model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(bx).squeeze()
                
                loss = criterion(pred, by)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            
        return model

    def _estimate_uncertainty(self, model, X, model_type='xgb', n_samples=5, noise_std=0.05):
        """ 
        通过输入扰动(Input Perturbation)估计不确定性 
        返回: (predictions, uncertainty)
        """
        preds_list = []
        
        # 原始预测
        if model_type == 'xgb':
            base_pred = model.predict(X)
        else: # nn
            model.eval()
            with torch.no_grad():
                t_X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
                base_pred = model(t_X).cpu().numpy().flatten()
        preds_list.append(base_pred)
        
        # 扰动预测
        for _ in range(n_samples):
            X_noise = X + np.random.normal(0, noise_std, size=X.shape)
            if model_type == 'xgb':
                p = model.predict(X_noise)
            else:
                model.eval()
                with torch.no_grad():
                    t_X_noise = torch.tensor(X_noise, dtype=torch.float32).to(DEVICE)
                    p = model(t_X_noise).cpu().numpy().flatten()
            preds_list.append(p)
            
        preds_stack = np.vstack(preds_list)
        # 不确定性 = 预测的标准差 (Std Dev across perturbations)
        uncertainty = np.std(preds_stack, axis=0)
        mean_pred = np.mean(preds_stack, axis=0)
        
        return mean_pred, uncertainty

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        set_seed(self.random_state)
        
        # --- Speed Optimization: Cap unlabeled samples to prevent excessive computation ---
        if X_unlabeled is not None and len(X_unlabeled) > 5000:
            idx = np.random.choice(len(X_unlabeled), 5000, replace=False)
            X_unlabeled = X_unlabeled[idx]
        
        # 1. 数据标准化
        X_all = np.vstack([X_labeled, X_unlabeled])
        if self.standardize_input:
            self.scaler_x.fit(X_all)
            X_L_sc = self.scaler_x.transform(X_labeled)
            X_U_sc = self.scaler_x.transform(X_unlabeled)
        else:
            X_L_sc = X_labeled
            X_U_sc = X_unlabeled
            
        y_L_sc = self.scaler_y.fit_transform(y_labeled.reshape(-1, 1)).flatten()
        
        # Generate View Masks (Feature Subsampling)
        n_features = X_all.shape[1]
        n_sub = max(1, int(n_features * self.feature_subsample_ratio))
        
        # Randomly select features for each view (with potential overlap)
        all_cols = np.arange(n_features)
        self.view1_cols = np.random.choice(all_cols, n_sub, replace=False)
        self.view2_cols = np.random.choice(all_cols, n_sub, replace=False) # Independent selection
        
        if self.verbose:
            print(f"[Co-Training] Feature Subsampling: {n_sub}/{n_features} features per view.")
            
        # 初始化 L1, L2 集合 (View 1 & View 2 的训练集)
        self.L1_X, self.L1_y = X_L_sc.copy(), y_L_sc.copy()
        self.L2_X, self.L2_y = X_L_sc.copy(), y_L_sc.copy()
        
        # current unlabeled pool
        U_X = X_U_sc.copy()
        # mask to track which samples are still unlabeled
        U_indices = np.arange(len(U_X))
        
        # 2. 初始训练 (Bootstrap)
        if self.verbose: print(f"[Co-Training] Initial Training...")
        # Train on View 1 Features
        self.xgb_model = self._train_xgboost(self.L1_X[:, self.view1_cols], self.L1_y)
        # Train on View 2 Features
        self.nn_model = self._train_nn(self.L2_X[:, self.view2_cols], self.L2_y)
        
        # 3. Co-Training Loop
        for iteration in range(self.max_iter):
            if len(U_indices) == 0:
                break
                
            if self.verbose: print(f"[Co-Training] Iteration {iteration+1}/{self.max_iter} | U remaining: {len(U_indices)}")
            
            # --- Step A: 从 U 中采样一个 Pool 候选集 (Pool-based selection) ---
            # 为了效率，不评估所有 U，只随机选 pool_size 个
            current_pool_size = min(self.pool_size, len(U_indices))
            pool_idx_in_U = np.random.choice(len(U_indices), current_pool_size, replace=False)
            pool_global_indices = U_indices[pool_idx_in_U] # 在原始 U 中的索引
            pool_X = U_X[pool_idx_in_U] # 实际特征数据
            
            # --- Step B: 两个模型分别预测并评估不确定性 (Uncertainty Estimation) ---
            # Model 1 (XGB) - Use View 1 Features
            pred_1, unc_1 = self._estimate_uncertainty(self.xgb_model, pool_X[:, self.view1_cols], model_type='xgb')
            
            # Model 2 (NN) - Use View 2 Features
            pred_2, unc_2 = self._estimate_uncertainty(self.nn_model, pool_X[:, self.view2_cols], model_type='nn')
            
            # --- Step C: 选择高置信度样本 (Selection) ---
            # Select top-k confident samples from XGB (Model 1) to add to NN (Model 2)'s training set
            n_select = min(self.samples_per_iter, len(pool_X))
            
            # 策略: 选择 unc 最小的
            idx_1_sorted = np.argsort(unc_1)
            selected_idx_1 = idx_1_sorted[:n_select]
            
            # Select top-k confident samples from NN (Model 2) to add to XGB (Model 1)'s training set
            idx_2_sorted = np.argsort(unc_2)
            selected_idx_2 = idx_2_sorted[:n_select]
            
            # --- Step D: 更新训练集 (Update Labeled Sets) ---
            # 添加到 L2 (NN 的训练集) 的样本来自 XGB 的预测
            X_add_to_2 = pool_X[selected_idx_1]
            y_add_to_2 = pred_1[selected_idx_1]
            self.L2_X = np.vstack([self.L2_X, X_add_to_2])
            self.L2_y = np.concatenate([self.L2_y, y_add_to_2])
            
            # 添加到 L1 (XGB 的训练集) 的样本来自 NN 的预测
            X_add_to_1 = pool_X[selected_idx_2]
            y_add_to_1 = pred_2[selected_idx_2]
            self.L1_X = np.vstack([self.L1_X, X_add_to_1])
            self.L1_y = np.concatenate([self.L1_y, y_add_to_1])
            
            # --- Step E: 从 U 中移除被选中的样本 (Remove from Pool) ---
            # 注意: 如果两个模型选中了同一个样本，它会被移除
            # 我们需要找出所有被选中的实际 global indices
            selected_indices_in_pool = np.union1d(selected_idx_1, selected_idx_2)
            removed_global_indices = pool_global_indices[selected_indices_in_pool]
            
            # 更新 U_indices, 移除这些 ID
            # 使用 set diff
            U_indices = np.setdiff1d(U_indices, removed_global_indices)
            
            # 更新 U_X... 其实不需要物理删除 U_X，只要维护 U_indices 即可
            # 但为了简单，我们只维护 U_indices 映射到原始 X_U_sc
            
            # --- Step F: 重新训练模型 (Retrain) ---
            # 重新训练比增量更稳定，尤其是在 Co-Training 中加入了伪标签后
            # Train on View 1 Features
            self.xgb_model = self._train_xgboost(self.L1_X[:, self.view1_cols], self.L1_y)
            # NN 选择从头训练 (Retrain) 以避免灾难性遗忘和累积的确认偏差
            # Train on View 2 Features
            self.nn_model = self._train_nn(self.L2_X[:, self.view2_cols], self.L2_y, init_model=None)
            
        return self

    def predict(self, X):
        if self.standardize_input:
            X_sc = self.scaler_x.transform(X)
        else:
            X_sc = X
            
        if self.xgb_model is None or self.nn_model is None:
            raise RuntimeError("Not fitted")
            
        # 1. XGB Prediction (View 1)
        xgb_pred = self.scaler_y.inverse_transform(self.xgb_model.predict(X_sc[:, self.view1_cols]).reshape(-1,1)).flatten()
        
        # 2. NN Prediction (View 2)
        # Note: NN expects subsampled features
        t_X = torch.tensor(X_sc[:, self.view2_cols], dtype=torch.float32).to(DEVICE)
        self.nn_model.eval()
        with torch.no_grad():
            nn_pred_sc = self.nn_model(t_X).cpu().numpy().flatten()
        nn_pred = self.scaler_y.inverse_transform(nn_pred_sc.reshape(-1,1)).flatten()
        
        # 3. Ensemble (动态置信度集成，参考 HeSCo)
        _, nn_instability = self._estimate_uncertainty(self.nn_model, X_sc[:, self.view2_cols], model_type='nn')

        min_inst = nn_instability.min()
        max_inst = nn_instability.max()
        if max_inst > min_inst:
            norm_instability = (nn_instability - min_inst) / (max_inst - min_inst)
        else:
            norm_instability = np.zeros_like(nn_instability)

        nn_weight = np.clip(np.exp(-norm_instability), 0.4, 0.6)

        return nn_weight * nn_pred + (1 - nn_weight) * xgb_pred

if __name__ == "__main__":
    print("Testing Co-Training SGP implementation...")
    # Generate dummy regression data
    X_train = np.random.rand(100, 10)
    y_train = X_train[:, 0] * 5 + X_train[:, 1] * 3 + np.random.normal(0, 0.1, 100)
    X_unlabeled = np.random.rand(500, 10)
    X_test = np.random.rand(20, 10)
    
    model = CoTrainingRegressor(
        max_iter=3,
        samples_per_iter=10,
        epochs=10,
        verbose=True
    )
    
    model.fit(X_train, y_train, X_unlabeled)
    preds = model.predict(X_test)
    print("Prediction shape:", preds.shape)
    print("Test passed!")
