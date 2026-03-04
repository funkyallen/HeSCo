
# file: rankup.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, List
from utils import set_seed

# ==================== Configuration ====================
DATA_PATH = "data/California_Housing.csv"
LABELED_RATIO = 0.1
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# RankUp specific hyperparameters
RANKING_WEIGHT = 1.0        # Weight for ranking loss
CONSISTENCY_WEIGHT = 1.0    # Weight for consistency loss
CONF_THRESHOLD = 0.9        # Confidence threshold for pseudo-labels
RANK_PAIRS_PER_BATCH = 64   # Number of pairs per batch

# ==================== Model Architecture ====================
class RankUpNet(nn.Module):
    """
    RankUp Network Architecture.
    
    Components:
        - encoder: Shared feature extractor
        - reg_head: Regression head (predicts continuous value)
        - rank_head: Ranking classification head (predicts P(y_i > y_j))
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Ranking classification head
        # Input is concatenation of two feature vectors [f(x_i), f(x_j)]
        self.rank_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: logit for P(y_i > y_j)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass: Returns regression prediction."""
        features = self.encoder(x)
        return self.reg_head(features).squeeze()
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns encoded features."""
        return self.encoder(x)
    
    def forward_ranking(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        Ranking classifier forward pass.
        
        Args:
            x_i, x_j: Input tensors for pair comparison
        
        Returns:
            logit: Logit output for P(y_i > y_j)
        """
        f_i = self.encoder(x_i)
        f_j = self.encoder(x_j)
        pair_features = torch.cat([f_i, f_j], dim=1)
        return self.rank_head(pair_features).squeeze()


# ==================== Loss Functions ====================
def ranking_loss(rank_logits: torch.Tensor, rank_labels: torch.Tensor) -> torch.Tensor:
    """Binary Cross Entropy for ranking."""
    return F.binary_cross_entropy_with_logits(rank_logits, rank_labels)


def consistency_loss(pred_rank_prob: torch.Tensor, pred_reg_i: torch.Tensor, pred_reg_j: torch.Tensor) -> torch.Tensor:
    """
    Consistency Loss: Enforce regression and ranking predictions to match.
    Rank Head (P(i>j)) guides Regression Head (reg_i - reg_j).
    """
    # Soft Consistency: Convert regression difference to probability via Sigmoid
    # We want sigmoid(reg_i - reg_j) to approximate pred_rank_prob
    # detach() rank_prob because Ranking is the 'teacher' task here (more robust)
    
    diff = pred_reg_i - pred_reg_j
    prob_from_reg = torch.sigmoid(diff)
    
    # Minimize distance between Reg-induced prob and Rank-predicted prob
    return F.mse_loss(prob_from_reg, pred_rank_prob.detach())


# ==================== Data Pairing Utilities ====================
def create_ranking_pairs(X: torch.Tensor, y: torch.Tensor, n_pairs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Efficiently create ranking pairs from labeled data (Tensor-based).
    """
    n_samples = len(X)
    if n_samples < 2:
        return X, X, torch.zeros(len(X), device=X.device)

    # Vectorized random sampling
    idx_i = torch.randint(0, n_samples, (n_pairs,), device=X.device)
    idx_j = torch.randint(0, n_samples, (n_pairs,), device=X.device)
    
    X_i = X[idx_i]
    X_j = X[idx_j]
    
    # Calculate ground truth ranking labels
    # 1.0 if y_i > y_j, else 0.0
    labels = (y[idx_i] > y[idx_j]).float()
    
    return X_i, X_j, labels


def create_pseudo_ranking_pairs(model: nn.Module, 
                                X_labeled: torch.Tensor, y_labeled: torch.Tensor, 
                                X_unlabeled: torch.Tensor, 
                                n_pairs: int, 
                                conf_threshold: float,
                                device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Create ranking pairs using Pseudo-Labels with MC Dropout Uncertainty.
    """
    # 1. Uses the current batch directly
    X_u_cand = X_unlabeled
    
    # 2. MC Dropout for Uncertainty Estimation without OOM
    model.train() # Enable Dropout
    pseudo_sum = 0
    pseudo_sq_sum = 0
    with torch.no_grad():
        for _ in range(10): # 10 passes for uncertainty estimation
            # RegHead prediction
            p = model(X_u_cand)
            if p.dim() == 0:
                p = p.unsqueeze(0)
            pseudo_sum += p
            pseudo_sq_sum += p ** 2
            
    pseudo_mean = pseudo_sum / 10.0
    pseudo_var = (pseudo_sq_sum / 10.0) - (pseudo_mean ** 2)
    pseudo_std = torch.sqrt(torch.clamp(pseudo_var, min=1e-8))
    
    # 3. Filter by Uncertainty (mask = uncertainty < threshold)
    mask = pseudo_std < conf_threshold
    
    if mask.sum() == 0:
        return None, None, None
        
    X_u_good = X_u_cand[mask]
    y_u_good = pseudo_mean[mask]
    
    # 4. Create Pairs (Unlabeled vs Labeled)
    n_good = len(X_u_good)
    n_labeled = len(X_labeled)
    
    # Sample indices for U and L to form n_pairs
    idx_u = torch.randint(0, n_good, (n_pairs,), device=device)
    idx_l = torch.randint(0, n_labeled, (n_pairs,), device=device)
    
    X_u_sample = X_u_good[idx_u]
    y_u_sample = y_u_good[idx_u]
    
    X_l_sample = X_labeled[idx_l]
    y_l_sample = y_labeled[idx_l]
    
    # Labels: 1.0 if y_u > y_l
    labels = (y_u_sample > y_l_sample).float()
    
    return X_u_sample, X_l_sample, labels


# ==================== Main Training Class ====================
class RankUp:
    def __init__(self, 
                 input_dim: int, 
                 lr: float = LR, 
                 batch_size: int = BATCH_SIZE, 
                 epochs: int = EPOCHS,
                 ranking_weight: float = RANKING_WEIGHT, 
                 consistency_weight: float = CONSISTENCY_WEIGHT,
                 conf_threshold: float = CONF_THRESHOLD, 
                 unlabeled_batch_ratio: float = 0.5,
                 device: torch.device = DEVICE, 
                 random_state: int = 42,
                 verbose: bool = False):
        self.input_dim = input_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.ranking_weight = ranking_weight
        self.consistency_weight = consistency_weight
        self.conf_threshold = conf_threshold
        self.unlabeled_batch_ratio = unlabeled_batch_ratio
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        
        set_seed(self.random_state)
        self.model = None
        self.optimizer = None
        self.scheduler = None
    
    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, X_unlabeled: Optional[np.ndarray] = None):
        """Fit RankUp model."""
        set_seed(self.random_state)
        
        # --- Speed Optimization: Cap unlabeled samples to prevent excessive computation ---
        if X_unlabeled is not None and len(X_unlabeled) > 5000:
            idx = np.random.choice(len(X_unlabeled), 5000, replace=False)
            X_unlabeled = X_unlabeled[idx]
            
        self.model = RankUpNet(self.input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        # Pre-convert all data to Tensors to minimize CPU-GPU transfer inside loops
        # Keep labeled data on CPU for DataLoader
        X_lbl_t = torch.tensor(X_labeled, dtype=torch.float32).to(self.device)
        y_lbl_t = torch.tensor(y_labeled, dtype=torch.float32).to(self.device)
        
        if X_unlabeled is not None and len(X_unlabeled) > 0:
            # Unlabeled data
            X_unlbl_t = torch.tensor(X_unlabeled, dtype=torch.float32).to(self.device)
            use_unlabeled = True
        else:
            X_unlbl_t = None
            use_unlabeled = False
            
        labeled_dataset = TensorDataset(X_lbl_t, y_lbl_t)
        labeled_loader = DataLoader(
            labeled_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        if self.verbose:
            print(f"Training RankUp with {len(X_labeled)} labeled samples" +
                  (f" and {len(X_unlabeled)} unlabeled samples" if use_unlabeled else ""))
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_reg_loss = 0.0
            epoch_rank_loss = 0.0
            epoch_cons_loss = 0.0
            
            steps = len(labeled_loader)

            for batch_idx, (batch_x, batch_y) in enumerate(labeled_loader):
                
                # 1. Regression Loss
                pred_reg = self.model(batch_x)
                loss_reg = F.mse_loss(pred_reg, batch_y)
                
                # 2. Ranking Loss (Labeled Pairs)
                # Dynamic n_pairs scaling
                n_pairs_dynamic = min(256, len(X_lbl_t) * 2)
                
                # Create pairs directly on GPU memory
                X_i, X_j, rank_labels = create_ranking_pairs(X_lbl_t, y_lbl_t, n_pairs_dynamic)
                
                rank_logits = self.model.forward_ranking(X_i, X_j)
                loss_rank_sup = ranking_loss(rank_logits, rank_labels)
                
                # 3. Unsupervised Ranking Loss (Pseudo-labels)
                loss_rank_unsup = torch.tensor(0.0, device=self.device)
                
                ul_interval = int(1.0 / self.unlabeled_batch_ratio) if self.unlabeled_batch_ratio > 0 else 0
                do_unlabeled = (ul_interval > 0) and ((batch_idx % ul_interval) == 0)

                if use_unlabeled and do_unlabeled and X_unlbl_t is not None:
                    idx_u = torch.randperm(len(X_unlbl_t))[:batch_x.size(0)].to(self.device)
                    batch_x_u = X_unlbl_t[idx_u]
                    
                    X_i_p, X_j_p, rank_labels_p = create_pseudo_ranking_pairs(
                        self.model, X_lbl_t, y_lbl_t, batch_x_u,
                        n_pairs_dynamic, self.conf_threshold, self.device
                    )
                    
                    if X_i_p is not None:
                        rank_logits_p = self.model.forward_ranking(X_i_p, X_j_p)
                        loss_rank_unsup = ranking_loss(rank_logits_p, rank_labels_p)
                
                # 4. Consistency Loss (on paired data)
                # Combine Labeled pairs + Pseudo pairs for consistency? 
                # For simplicity, calculate consistency on the Labeled Pairs used for Ranking
                pred_reg_i = self.model(X_i)
                pred_reg_j = self.model(X_j)
                rank_prob = torch.sigmoid(rank_logits)
                loss_cons = consistency_loss(rank_prob, pred_reg_i, pred_reg_j)
                
                # Total Loss
                total_loss = (
                    loss_reg + 
                    self.ranking_weight * (loss_rank_sup + loss_rank_unsup) +
                    self.consistency_weight * loss_cons
                )
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_reg_loss += loss_reg.item()
                epoch_rank_loss += (loss_rank_sup.item() + loss_rank_unsup.item())
                epoch_cons_loss += loss_cons.item()
            
            self.scheduler.step()
            
            if self.verbose and (epoch + 1) % 20 == 0:
                n_batches = max(1, steps)
                print(f"Epoch {epoch+1}/{self.epochs} | Loss: {epoch_loss/n_batches:.4f} | "
                      f"Reg: {epoch_reg_loss/n_batches:.4f} | Rank: {epoch_rank_loss/n_batches:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch_x in loader:
                pred_batch = self.model(batch_x[0])
                preds.append(pred_batch.cpu().numpy())
                
        return np.concatenate(preds).flatten()


def run_rankup():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}")
        return
    
    data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)
    X, y = data[:, :-1], data[:, -1]
    
    # Split First
    Xtr, Xtest, ytr, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing (Fit on Train)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xtest = scaler.transform(Xtest)
    
    N_labeled = int(len(Xtr) * LABELED_RATIO)
    idx = np.random.permutation(len(Xtr))
    
    X_labeled = Xtr[idx[:N_labeled]]
    y_labeled = ytr[idx[:N_labeled]]
    X_unlabeled = Xtr[idx[N_labeled:]]
    
    model = RankUp(input_dim=X.shape[1], verbose=True)
    model.fit(X_labeled, y_labeled, X_unlabeled)
    
    pred = model.predict(Xtest)
    rmse = np.sqrt(mean_squared_error(ytest, pred))
    print(f'[RankUp] RMSE (10% labeled): {rmse:.4f}')


if __name__ == "__main__":
    run_rankup()
