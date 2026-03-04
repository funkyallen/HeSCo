# file: vime.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, List
from utils import set_seed

# ================= Configuration =================
DATA_PATH = "data/California_Housing.csv"
LABELED_RATIO = 0.1
BATCH_SIZE = 256
EPOCHS_PRETEXT = 20
EPOCHS_SUP = 100
MASK_P = 0.3
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================= Model Architecture =================
class VIMENet(nn.Module):
    """
    VIME Network Architecture.
    
    Components:
        - encoder: Feature extractor
        - mask_head: Predicts which features were corrupted (Binary Classification)
        - recon_head: Reconstructs original features (Regression)
        - reg_head: Final regression task head
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mask_head = nn.Linear(128, input_dim)
        self.recon_head = nn.Linear(128, input_dim)
        self.reg_head = nn.Linear(128, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_pretext(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for self-supervised pretext task.
        Returns: (mask_probability, reconstructed_features)
        """
        h = self.encode(x)
        mask_logits = self.mask_head(h)
        recon = self.recon_head(h)
        return torch.sigmoid(mask_logits), recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for supervised regression."""
        h = self.encode(x)
        return self.reg_head(h)


# ================= Utilities =================
def vime_feature_corrupt(x: torch.Tensor, mask: torch.Tensor, empirical_dist: torch.Tensor) -> torch.Tensor:
    """
    Vectorized Swap Noise implementation using global empirical distribution.
    
    Args:
        x: Input tensor (B, D)
        mask: Binary mask where 1 indicates corruption (B, D)
        empirical_dist: Global feature pool (N, D)
        
    Returns:
        x_corrupt: Corrupted input tensor
    """
    batch_size, feat_dim = x.shape
    x_shuffled = torch.zeros_like(x)
    n_samples = empirical_dist.shape[0]
    
    for col in range(feat_dim):
        idx = torch.randint(0, n_samples, (batch_size,), device=x.device)
        x_shuffled[:, col] = empirical_dist[idx, col]
    
    x_corrupt = torch.where(mask > 0.5, x_shuffled, x)
    
    return x_corrupt


# ================= Algorithm Implementation =================
class VIME:
    """
    VIME: Value Imputation and Mask Estimation.
    Self-supervised learning framework for tabular data.
    """
    def __init__(self, 
                 input_dim: int, 
                 lr: float = LR, 
                 batch_size: int = BATCH_SIZE, 
                 epochs_pretext: int = EPOCHS_PRETEXT, 
                 epochs_sup: int = EPOCHS_SUP, 
                 mask_p: float = MASK_P, 
                 unlabeled_batch_ratio: float = 0.5,
                 device: torch.device = DEVICE, 
                 random_state: int = 42,
                 verbose: bool = False):
        self.input_dim = input_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs_pretext = epochs_pretext
        self.epochs_sup = epochs_sup
        self.mask_p = mask_p
        self.unlabeled_batch_ratio = unlabeled_batch_ratio
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        
        set_seed(self.random_state)
        self.model = VIMENet(input_dim).to(device)

    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, X_unlabeled: Optional[np.ndarray] = None):
        """Fit VIME model."""
        set_seed(self.random_state)
        
        # --- Speed Optimization: Cap unlabeled samples to prevent excessive computation ---
        if X_unlabeled is not None and len(X_unlabeled) > 5000:
            idx = np.random.choice(len(X_unlabeled), 5000, replace=False)
            X_unlabeled = X_unlabeled[idx]
            
        # 1. Prepare Data
        if X_unlabeled is not None and len(X_unlabeled) > 0:
            X_all = np.vstack([X_labeled, X_unlabeled])
        else:
            X_all = X_labeled

        self.empirical_dist = torch.tensor(X_all, dtype=torch.float32, device=self.device)

        # === Phase 1: Pretext (Self-supervised) ===
        optimizer_pretext = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion_mask = nn.BCELoss()
        criterion_feat = nn.MSELoss()

        if self.verbose:
            print(f"Phase 1: Self-Supervised Pretext Task ({self.epochs_pretext} epochs)")

        # Create DataLoader for Pretext
        pretext_dataset = TensorDataset(torch.tensor(X_all, dtype=torch.float32))
        pretext_loader = DataLoader(
            pretext_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True,
            num_workers=0
        )

        self.model.train()
        for epoch in range(self.epochs_pretext):
            epoch_loss = 0.0
            total_samples = 0
            
            for batch in pretext_loader:
                x_batch = batch[0].to(self.device, non_blocking=True)
                
                # Generate Mask
                mask = torch.bernoulli(torch.full(x_batch.shape, self.mask_p, device=self.device))
                
                # Corrupt Input
                x_corrupt = vime_feature_corrupt(x_batch, mask, self.empirical_dist)
                
                # Forward
                mask_pred, x_recon = self.model.forward_pretext(x_corrupt)
                
                # Loss
                loss_m = criterion_mask(mask_pred, mask)
                loss_r = criterion_feat(x_recon, x_batch)
                loss = loss_m + 2.0 * loss_r
                
                optimizer_pretext.zero_grad()
                loss.backward()
                optimizer_pretext.step()
                
                batch_size_curr = x_batch.size(0)
                epoch_loss += loss.item() * batch_size_curr
                total_samples += batch_size_curr
                
            avg_loss = epoch_loss / total_samples if total_samples > 0 else 0.0
                
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs_pretext} | Loss: {avg_loss:.4f}")

        # === Phase 2: Fine-tune (Supervised) ===
        if self.verbose:
            print(f"Phase 2: Supervised Fine-tuning ({self.epochs_sup} epochs)")
            
        X_lbl_t = torch.tensor(X_labeled, dtype=torch.float32).to(self.device)
        y_lbl_t = torch.tensor(y_labeled, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        sup_dataset = TensorDataset(X_lbl_t, y_lbl_t)
        sup_loader = DataLoader(
            sup_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )

        # Lower LR for fine-tuning encoder, normal LR for reg_head
        optimizer_sup = optim.Adam([
            {'params': self.model.encoder.parameters(), 'lr': self.lr * 0.1},
            {'params': self.model.reg_head.parameters(), 'lr': self.lr}
        ])
        criterion_reg = nn.MSELoss()
        
        # Prepare Unlabeled tensor for consistency regularization
        if X_unlabeled is not None and len(X_unlabeled) > 0:
            X_unlbl_t = torch.tensor(X_unlabeled, dtype=torch.float32).to(self.device)
            use_unlabeled = True
        else:
            X_unlbl_t = None
            use_unlabeled = False

        for epoch in range(self.epochs_sup):
            self.model.train()
            epoch_loss = 0.0
            
            steps = len(sup_loader)

            for batch_idx, (x_batch, y_batch) in enumerate(sup_loader):
                
                # Labeled Supervised Loss
                pred_l = self.model(x_batch)
                loss_sup = criterion_reg(pred_l, y_batch)
                
                # Unlabeled Consistency Loss
                loss_unsup = torch.tensor(0.0, device=self.device)
                
                ul_interval = int(1.0 / self.unlabeled_batch_ratio) if self.unlabeled_batch_ratio > 0 else 0
                do_unlabeled = (ul_interval > 0) and ((batch_idx % ul_interval) == 0)

                if use_unlabeled and do_unlabeled and X_unlbl_t is not None:
                    idx_u = torch.randperm(len(X_unlbl_t))[:x_batch.size(0)].to(self.device)
                    x_batch_u = X_unlbl_t[idx_u]
                    
                    # Generate Corrupted version
                    mask = torch.bernoulli(torch.full(x_batch_u.shape, self.mask_p, device=self.device))
                    x_corrupt_u = vime_feature_corrupt(x_batch_u, mask, self.empirical_dist)
                    
                    # Enforce consistency between clean and corrupted representations
                    pred_clean = self.model(x_batch_u)
                    pred_corrupt = self.model(x_corrupt_u)
                    
                    # The model should be robust against self-corrupted noise
                    loss_unsup = criterion_reg(pred_corrupt, pred_clean.detach())
                
                loss = loss_sup + 0.5 * loss_unsup  # typical weight for consistency
                
                optimizer_sup.zero_grad()
                loss.backward()
                optimizer_sup.step()
                
                epoch_loss += loss.item()
                
            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs_sup} | Loss: {epoch_loss / len(sup_loader):.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch in loader:
                pred_batch = self.model(batch[0])
                preds.append(pred_batch.cpu().numpy())
                
        return np.concatenate(preds).flatten()


def run_vime():
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
    model = VIME(input_dim=X.shape[1], verbose=True)
    model.fit(X_labeled, y_labeled, X_unlabeled)

    # Evaluate
    pred = model.predict(Xtest)
    rmse = np.sqrt(mean_squared_error(ytest, pred))
    print(f'[VIME] RMSE (10% label): {rmse:.4f}')


if __name__ == "__main__":
    run_vime()
