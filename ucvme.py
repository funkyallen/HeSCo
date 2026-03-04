# file: ucvme.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, List
from utils import set_seed

# ================= Configuration =================
DATA_PATH = "data/California_Housing.csv"
LABELED_RATIO = 0.1
EPOCHS = 200
LR = 1e-3
BATCH_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================= Model Architecture =================
class UncertaintyNet(nn.Module):
    """
    Network estimating both mean and variance (uncertainty).
    Output: (mean, var)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.mean_head = nn.Linear(64, 1)
        self.logvar_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        mean = self.mean_head(f)
        logvar = self.logvar_head(f)
        
        # Stability: Softplus for variance
        raw_var = self.logvar_head(f)
        var = F.softplus(raw_var) + 1e-6
        return mean, var


# ================= Loss Functions =================
def nll_loss(mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Negative Log Likelihood Loss for Gaussian distribution.
    L = 0.5 * log(var) + 0.5 * (y - mean)^2 / var
    
    Args:
        mean: Predicted mean (N, 1)
        var: Predicted variance (N, 1)
        target: True labels (N, 1) or (N,)
    """
    # Ensure target shape matches mean
    if target.shape != mean.shape:
        target = target.view_as(mean)
        
    # Stability: clamp var to avoid division by zero
    var = torch.clamp(var, min=1e-6)
    
    loss = 0.5 * torch.mean(torch.log(var) + (target - mean) ** 2 / var)
    return loss


# ================= Algorithm Implementation =================
class UCVME:
    """
    UCVME: Uncertainty-Constrained Variational Mean-Teacher Ensemble.
    Uses two networks and enforces consistency on both mean and variance predictions on unlabeled data.
    """
    def __init__(self, 
                 input_dim: int, 
                 lr: float = LR, 
                 epochs: int = EPOCHS, 
                 batch_size: int = BATCH_SIZE,
                 ema_decay: float = 0.99,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 50,
                 unlabeled_batch_ratio: float = 0.5,
                 device: torch.device = DEVICE, 
                 random_state: int = 42,
                 verbose: bool = False):
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.ema_decay = ema_decay
        self.consistency_weight = consistency_weight
        self.rampup_epochs = rampup_epochs
        self.unlabeled_batch_ratio = unlabeled_batch_ratio
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        set_seed(self.random_state)
        self.net1 = UncertaintyNet(input_dim).to(device)
        self.net2 = UncertaintyNet(input_dim).to(device)

    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, X_unlabeled: Optional[np.ndarray] = None):
        """Fit UCVME model."""
        set_seed(self.random_state)
        
        # --- Speed Optimization: Cap unlabeled samples to prevent excessive computation ---
        if X_unlabeled is not None and len(X_unlabeled) > 5000:
            idx = np.random.choice(len(X_unlabeled), 5000, replace=False)
            X_unlabeled = X_unlabeled[idx]
            
        # Prepare DataLoaders
        # Keep labeled data on CPU for DataLoader
        X_lbl_t = torch.tensor(X_labeled, dtype=torch.float32).to(self.device)
        y_lbl_t = torch.tensor(y_labeled, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        lbl_dataset = TensorDataset(X_lbl_t, y_lbl_t)
        lbl_loader = DataLoader(
            lbl_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        if X_unlabeled is not None and len(X_unlabeled) > 0:
            X_unlbl_t = torch.tensor(X_unlabeled, dtype=torch.float32).to(self.device)
            use_unlabeled = True
        else:
            X_unlbl_t = None
            use_unlabeled = False

        # Optimizer for Student (net1) only
        optim1 = optim.Adam(self.net1.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Initialize Teacher (net2) with Student weights and detach
        self.net2.load_state_dict(self.net1.state_dict())
        for param in self.net2.parameters():
            param.detach_()

        # Scheduler
        scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optim1, mode='min', patience=10, factor=0.5)

        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20

        for epoch in range(self.epochs):
            self.net1.train()
            self.net2.eval() # Teacher in eval mode (standard Mean Teacher)
            # Standard MT uses train mode for teacher if it has noise, but here we want stability. 
            # Often Teacher is in Eval mode or Train without dropout update. 
            # Given UncertaintyNet, let's keep both in train or eval? 
            # Let's keep net2 in train mode to match distribution if BN is used, 
            # but usually Teacher BN follows Student or is updated via EMA?
            # Creating simple EMA update:
            
            epoch_loss = 0.0
            
            # Calculate Consistency Weight Ramp-up
            current_cons_weight = self.consistency_weight * min(1.0, (epoch + 1) / self.rampup_epochs)

            steps = len(lbl_loader)

            for batch_idx, (X_batch_lbl, y_batch_lbl) in enumerate(lbl_loader):
                
                # 1. Supervised Loss (Student only)
                mu1, var1 = self.net1(X_batch_lbl)
                loss_sup = nll_loss(mu1, var1, y_batch_lbl)
                
                # 2. Unsupervised Consistency Loss (Student vs Teacher)
                loss_cons = torch.tensor(0.0, device=self.device)
                
                ul_interval = int(1.0 / self.unlabeled_batch_ratio) if self.unlabeled_batch_ratio > 0 else 0
                do_unlabeled = (ul_interval > 0) and ((batch_idx % ul_interval) == 0)

                if use_unlabeled and do_unlabeled and X_unlbl_t is not None:
                    idx_u = torch.randperm(len(X_unlbl_t))[:X_batch_lbl.size(0)].to(self.device)
                    X_batch_u = X_unlbl_t[idx_u]

                    # Add Gaussian noise to Student input for consistency
                    noise = torch.randn_like(X_batch_u) * 0.05
                    
                    # Student Forward (with noise)
                    mu1_u, var1_u = self.net1(X_batch_u + noise)
                    
                    # Teacher Forward (No Grad)
                    with torch.no_grad():
                        mu2_u, var2_u = self.net2(X_batch_u)

                    # Consistency: Mean MSE + Variance MSE
                    loss_mean = torch.mean((mu1_u - mu2_u) ** 2)
                    loss_var = torch.mean((var1_u - var2_u) ** 2)

                    loss_cons = loss_mean + loss_var

                # Total Loss
                loss = loss_sup + current_cons_weight * loss_cons
                
                # Update Student
                optim1.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net1.parameters(), max_norm=1.0)
                optim1.step()
                
                # Update Teacher via EMA
                with torch.no_grad():
                    for (name1, param1), (name2, param2) in zip(self.net1.named_parameters(), self.net2.named_parameters()):
                        param2.data = self.ema_decay * param2.data + (1.0 - self.ema_decay) * param1.data
                    for (name1, buf1), (name2, buf2) in zip(self.net1.named_buffers(), self.net2.named_buffers()):
                        buf2.data = self.ema_decay * buf2.data + (1.0 - self.ema_decay) * buf1.data
                
                epoch_loss += loss.item()

            # Scheduler update
            current_loss = epoch_loss / steps
            scheduler1.step(current_loss)

            # Early Stopping
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                if self.verbose:
                    print(f"UCVME: Early stopping at epoch {epoch + 1}")
                break
            
            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | Loss: {current_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction."""
        self.net1.eval()
        self.net2.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        
        # Batch inference
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch_x in loader:
                mu2, _ = self.net2(batch_x[0])
                preds.append(mu2.cpu().numpy())
                
        return np.concatenate(preds).flatten()


def run_ucvme():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}")
        return

    data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)
    X, y = data[:,:-1], data[:,-1]

    # Split first
    Xtr, Xtest, ytr, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing (Fit on train only)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xtest = scaler.transform(Xtest)

    # Simulate labeled/unlabeled split
    N_labeled = int(len(Xtr) * LABELED_RATIO)
    idx = np.random.permutation(len(Xtr))

    X_labeled = Xtr[idx[:N_labeled]]
    y_labeled = ytr[idx[:N_labeled]]
    X_unlabeled = Xtr[idx[N_labeled:]]

    # Initialize and train model
    model = UCVME(input_dim=X.shape[1], verbose=True)
    model.fit(X_labeled, y_labeled, X_unlabeled)

    # Evaluate
    pred = model.predict(Xtest)
    rmse = np.sqrt(mean_squared_error(ytest, pred))
    print(f'[UCVME] RMSE (10% label): {rmse:.4f}')


if __name__ == "__main__":
    run_ucvme()
