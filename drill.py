# file: drill.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Tuple, Optional, List
from utils import set_seed

# ================= Configuration =================
DATA_PATH = "data/California_Housing.csv"
LABELED_RATIO = 0.1
EPOCHS_PROXY = 100
EPOCHS_REG = 100
N_BINS = 10 # Default fallback if needed, though usually adaptive
LR = 1e-3
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Model Architecture =================
class DrillNet(nn.Module):
    """
    DRILL Network: Shared encoder with dual heads (Classification and Regression).
    
    Architecture:
    Encoder: [Input -> 64 -> ReLU -> 32 -> ReLU]
    Cls Head: [32 -> n_bins]
    Reg Head: [32 -> 1]
    """
    def __init__(self, input_dim: int, n_bins: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),  # Added LayerNorm for stability
            nn.ReLU(),
            nn.Dropout(0.1),    # Added Dropout
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.cls_head = nn.Linear(64, n_bins)
        self.reg_head = nn.Linear(64, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Classification (Proxy Task)"""
        return self.cls_head(self.features(x))

    def forward_reg(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Regression (Target Task)"""
        return self.reg_head(self.features(x))

    def forward_all(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both tasks (Multi-task Learning)"""
        feat = self.features(x)
        return self.cls_head(feat), self.reg_head(feat)


# ================= Algorithm Implementation =================
class Drill:
    """
    DRILL: Discretized Regression Imposed on Learned Latents (Simplified).
    
    Strategy:
    1. Pre-train encoder via Classification (discretized targets).
    2. Fine-tune encoder + reg_head via Regression (continuous targets).
    
    Attributes:
        input_dim (int): Dimension of input features.
        n_bins (int): Number of discretization bins.
    """
    def __init__(self, 
                 input_dim: int, 
                 n_bins: Optional[int] = None, 
                 lr: float = LR, 
                 epochs_proxy: int = EPOCHS_PROXY, 
                 epochs_reg: int = EPOCHS_REG, 
                 batch_size: int = BATCH_SIZE,
                 unlabeled_batch_ratio: float = 0.5,
                 device: torch.device = DEVICE, 
                 random_state: int = 42,
                 verbose: bool = False):
        self.input_dim = input_dim
        self.n_bins = n_bins
        self.lr = lr
        self.epochs_proxy = epochs_proxy
        self.epochs_reg = epochs_reg
        self.batch_size = batch_size
        self.unlabeled_batch_ratio = unlabeled_batch_ratio
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None
        self.bins_discretizer = None

    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, X_unlabeled: Optional[np.ndarray] = None):
        """
        Fit DRILL model.
        
        Args:
            X_labeled: Labeled training features (N_L, D)
            y_labeled: Labeled training targets (N_L,)
            X_unlabeled: Unlabeled training features (N_U, D)
        """
        set_seed(self.random_state)
        
        # --- Speed Optimization: Cap unlabeled samples to prevent excessive computation ---
        if X_unlabeled is not None and len(X_unlabeled) > 5000:
            idx = np.random.choice(len(X_unlabeled), 5000, replace=False)
            X_unlabeled = X_unlabeled[idx]
                
        if self.n_bins is None:
            # Adaptive n_bins: min(20, max(5, N // 20))
            self.n_bins = int(min(20, max(5, len(y_labeled) // 20)))
            if self.verbose:
                print(f"  [Drill] Adaptive n_bins set to {self.n_bins} (Labels: {len(y_labeled)})")

        if self.model is None:
            self.model = DrillNet(self.input_dim, self.n_bins).to(self.device)
            self.bins_discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy="quantile")

        # 1. Prepare Data
        y_labeled_reshaped = y_labeled.reshape(-1, 1)
        self.bins_discretizer.fit(y_labeled_reshaped)
        y_binned = self.bins_discretizer.transform(y_labeled_reshaped).flatten().astype(int)

        # To Tensor
        X_lbl_t = torch.tensor(X_labeled, dtype=torch.float32).to(self.device)
        y_lbl_cls_t = torch.LongTensor(y_binned).to(self.device)
        y_lbl_reg_t = torch.tensor(y_labeled, dtype=torch.float32).reshape(-1, 1).to(self.device)

        if X_unlabeled is not None and len(X_unlabeled) > 0:
            # Unlabeled data on GPU for fast iteration
            X_unlbl_t = torch.tensor(X_unlabeled, dtype=torch.float32).to(self.device)
        else:
            X_unlbl_t = None

        # Create DataLoader for labeled data (CPU -> GPU)
        lbl_dataset_cls = TensorDataset(X_lbl_t, y_lbl_cls_t)
        lbl_loader_cls = DataLoader(
            lbl_dataset_cls, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # DataLoader for combined (X, y_cls, y_reg) for Phase 2 multi-task
        lbl_dataset_all = TensorDataset(X_lbl_t, y_lbl_cls_t, y_lbl_reg_t)
        lbl_loader_all = DataLoader(
            lbl_dataset_all, 
            batch_size=self.batch_size, 
            shuffle=True
        )

        if X_unlbl_t is not None:
            use_unlabeled = True
        else:
            use_unlabeled = False

        # === Phase 1: Classification + Pseudo-labeling (Proxy Task) ===
        optimizer_cls = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion_cls = nn.CrossEntropyLoss()
        
        if self.verbose:
            print(f"Phase 1: Classification Proxy Task ({self.epochs_proxy} epochs)")

        for epoch in range(self.epochs_proxy):
            self.model.train()
            epoch_loss = 0.0
            
            steps = len(lbl_loader_cls)
            
            for batch_idx, (X_batch_lbl, y_batch_cls) in enumerate(lbl_loader_cls):

                loss_sup = criterion_cls(self.model.forward_cls(X_batch_lbl), y_batch_cls)
                loss_unsup = torch.tensor(0.0, device=self.device)

                ul_interval = int(1.0 / self.unlabeled_batch_ratio) if self.unlabeled_batch_ratio > 0 else 0
                do_unlabeled = (ul_interval > 0) and ((batch_idx % ul_interval) == 0)

                if use_unlabeled and do_unlabeled:
                    idx_u = torch.randperm(len(X_unlbl_t))[:X_batch_lbl.size(0)].to(self.device)
                    X_batch_u = X_unlbl_t[idx_u]
                    
                    # Pseudo-labeling with dynamic thresholding
                    with torch.no_grad():
                        logits_u = self.model.forward_cls(X_batch_u)
                        probs_u = torch.softmax(logits_u, dim=1)
                        max_prob_u, pseudo_labels = torch.max(probs_u, dim=1)
                        current_thresh = 0.8 - 0.4 * (epoch / max(1, self.epochs_proxy - 1))
                        mask = max_prob_u > current_thresh
                    
                    if mask.sum() > 0:
                        logits_u_train = self.model.forward_cls(X_batch_u)
                        loss_unsup = criterion_cls(logits_u_train[mask], pseudo_labels[mask])

                loss = loss_sup + 0.5 * loss_unsup
                
                optimizer_cls.zero_grad()
                loss.backward()
                optimizer_cls.step()
                epoch_loss += loss.item()

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs_proxy} | Loss: {epoch_loss / max(1, steps):.4f}")

        # === Phase 2: Fine-tune Regression ===
        # Re-initialize optimizer (no weight reset, fine-tuning)
        optimizer_reg = optim.Adam(self.model.parameters(), lr=self.lr * 0.5) # Lower LR for fine-tuning
        criterion_reg = nn.MSELoss()

        if self.verbose:
            print(f"Phase 2: Regression Fine-tuning ({self.epochs_reg} epochs)")

        # Combine Labeled + Unlabeled (optional: in original DRILL, regression is on labeled only or with pseudo-reg)
        # Here we follow standardization: Regress on labeled set primarily
        
        for epoch in range(self.epochs_reg):
            self.model.train()
            epoch_loss = 0.0
            
            steps = len(lbl_loader_all)
            
            # Use combined loader to keep classification head active (anti-forgetting)
            for batch_idx, (X_batch_lbl, y_batch_cls, y_batch_reg) in enumerate(lbl_loader_all):
                
                # Multi-task Forward
                logits_cls, pred_reg = self.model.forward_all(X_batch_lbl)
                
                loss_reg = criterion_reg(pred_reg, y_batch_reg)
                loss_cls = criterion_cls(logits_cls, y_batch_cls)
                
                loss_unsup = torch.tensor(0.0, device=self.device)
                
                ul_interval = int(1.0 / self.unlabeled_batch_ratio) if self.unlabeled_batch_ratio > 0 else 0
                do_unlabeled = (ul_interval > 0) and ((batch_idx % ul_interval) == 0)

                if use_unlabeled and do_unlabeled:
                    idx_u = torch.randperm(len(X_unlbl_t))[:X_batch_lbl.size(0)].to(self.device)
                    X_batch_u = X_unlbl_t[idx_u]
                    
                    # Representation alignment via pseudo discrete labels + pseudo regression labels
                    with torch.no_grad():
                        logits_u_teacher, pred_reg_teacher = self.model.forward_all(X_batch_u)
                        probs_u = torch.softmax(logits_u_teacher, dim=1)
                        max_prob_u, pseudo_cls_labels = torch.max(probs_u, dim=1)
                        current_thresh = 0.8 - 0.4 * (epoch / max(1, self.epochs_reg - 1))
                        mask = max_prob_u > current_thresh # dynamic confidence threshold
                        
                    if mask.sum() > 0:
                        logits_u_student, pred_reg_student = self.model.forward_all(X_batch_u)
                        # Discrete distribution alignment
                        loss_cls_u = criterion_cls(logits_u_student[mask], pseudo_cls_labels[mask])
                        # Continuous alignment (optional regularizer)
                        loss_reg_u = criterion_reg(pred_reg_student[mask], pred_reg_teacher[mask].detach())
                        
                        loss_unsup = 0.5 * loss_cls_u + 0.5 * loss_reg_u
                        
                # Weighted Sum: Main task is Regression, but keep Cls as auxiliary
                loss = loss_reg + 0.1 * loss_cls + loss_unsup
                
                optimizer_reg.zero_grad()
                loss.backward()
                optimizer_reg.step()
                epoch_loss += loss.item()
                
            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs_reg} | Loss: {epoch_loss / max(1, steps):.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features (N, D)
            
        Returns:
            predictions (N,)
        """
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        
        # Batch inference for large datasets
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch_X in loader:
                pred_batch = self.model.forward_reg(batch_X[0])
                preds.append(pred_batch.cpu().numpy())
                
        return np.concatenate(preds).flatten()


def run_drill():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}")
        return

    data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)
    X, y = data[:,:-1], data[:,-1]

    # Split First
    Xtr, Xtest, ytr, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing (Fit on Train)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xtest = scaler.transform(Xtest)

    # Simulate labeled/unlabeled split
    N_lbl = int(len(Xtr) * LABELED_RATIO)
    idx = np.random.permutation(len(Xtr))

    X_labeled = Xtr[idx[:N_lbl]]
    y_labeled = ytr[idx[:N_lbl]]
    X_unlabeled = Xtr[idx[N_lbl:]]

    # Initialize and train model
    model = Drill(input_dim=X.shape[1], verbose=True)
    model.fit(X_labeled, y_labeled, X_unlabeled)

    # Evaluate
    pred = model.predict(Xtest)
    rmse = np.sqrt(mean_squared_error(ytest, pred))
    print(f'[DRILL] RMSE (10% label): {rmse:.4f}')


if __name__ == "__main__":
    run_drill()
