# file: ft_transformer.py
"""
FT-Transformer: Feature Tokenizer Transformer for Tabular Data
Reference: Gorishniy et al., 2021
"Revisiting Deep Learning Models for Tabular Data"

This implementation provides a production-ready FT-Transformer for semi-supervised
tabular regression that matches the RankUp interface.
"""

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
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FT-Transformer specific hyperparameters (Gorishniy et al. 2021)
D_TOKEN = 192           # Token embedding dimension
N_BLOCKS = 3            # Number of transformer blocks
N_HEADS = 8             # Number of attention heads
ATTENTION_DROPOUT = 0.2 # Attention dropout rate
RESIDUAL_DROPOUT = 0.1  # Residual connection dropout
FF_DROPOUT = 0.1        # Feed-forward dropout
D_FFN_FACTOR = 4/3      # FFN hidden dimension factor
PATIENCE = 20           # Early stopping patience
GRAD_CLIP = 1.0         # Gradient clipping norm

# ==================== Model Architecture ====================

class FeatureTokenizer(nn.Module):
    """
    Feature Tokenizer for FT-Transformer.
    Transforms each numerical feature into a learnable token representation.
    
    Follows Gorishniy et al. (2021) approach:
    - Each feature gets its own learned embedding weights and bias
    - Applies element-wise multiplication and addition
    - Adds a special [CLS] token at the beginning
    """
    def __init__(self, input_dim: int, d_token: int):
        super().__init__()
        self.input_dim = input_dim
        self.d_token = d_token
        
        # Per-feature projection weights: (input_dim, d_token)
        self.weight = nn.Parameter(torch.empty(input_dim, d_token))
        self.bias = nn.Parameter(torch.empty(input_dim, d_token))
        
        # [CLS] token for classification/regression
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.cls_token, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            tokens: (batch_size, input_dim + 1, d_token)  [CLS] + feature tokens
        """
        batch_size = x.shape[0]
        
        # Element-wise transformation: x_i * weight_i + bias_i
        # x: (batch, input_dim) -> (batch, input_dim, 1)
        # weight: (input_dim, d_token) -> (1, input_dim, d_token)
        # Result: (batch, input_dim, d_token)
        x_expanded = x.unsqueeze(-1)  # (batch, input_dim, 1)
        tokens = x_expanded * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_token)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (batch, input_dim+1, d_token)
        
        return tokens


class FTTransformerNet(nn.Module):
    """
    FT-Transformer Network Architecture.
    
    Components:
    1. FeatureTokenizer: Converts input features to tokens
    2. TransformerEncoder: Attends over all feature tokens
    3. Head: Regression/classification head using [CLS] token
    
    Key design: Uses LayerNorm before attention (pre-norm), GELU activation
    """
    def __init__(self, 
                 input_dim: int, 
                 d_token: int = D_TOKEN, 
                 n_blocks: int = N_BLOCKS, 
                 n_heads: int = N_HEADS,
                 attention_dropout: float = ATTENTION_DROPOUT,
                 residual_dropout: float = RESIDUAL_DROPOUT,
                 ff_dropout: float = FF_DROPOUT,
                 d_ffn_factor: float = D_FFN_FACTOR):
        super().__init__()
        
        self.tokenizer = FeatureTokenizer(input_dim, d_token)
        
        # Calculate FFN dimension
        d_ffn = int(d_token * d_ffn_factor)
        
        # Transformer Encoder with pre-norm (norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=residual_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Apply LayerNorm before attention/FFN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        
        # Regression head: Uses [CLS] token output
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_token, 64),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            pred: (batch_size,) regression predictions
        """
        tokens = self.tokenizer(x)  # (batch, input_dim+1, d_token)
        x = self.transformer(tokens)  # (batch, input_dim+1, d_token)
        # Use [CLS] token (index 0) for prediction
        cls_output = x[:, 0, :]  # (batch, d_token)
        return self.head(cls_output).squeeze(-1)


# ==================== Main Training Class ====================

class FTTransformer:
    """
    FT-Transformer trainer for semi-supervised tabular regression.
    Matches the RankUp interface for easy comparison.
    """
    def __init__(self, 
                 input_dim: int, 
                 lr: float = LR, 
                 batch_size: int = BATCH_SIZE, 
                 epochs: int = EPOCHS,
                 weight_decay: float = WEIGHT_DECAY,
                 patience: int = PATIENCE,
                 grad_clip: float = GRAD_CLIP,
                 d_token: int = D_TOKEN,
                 n_blocks: int = N_BLOCKS,
                 n_heads: int = N_HEADS,
                 device: torch.device = DEVICE, 
                 random_state: int = 42,
                 verbose: bool = False):
        self.input_dim = input_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.grad_clip = grad_clip
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        
        set_seed(self.random_state)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.best_loss = float('inf')
        self.patience_counter = 0

    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, X_unlabeled: Optional[np.ndarray] = None):
        """
        Fit FT-Transformer model using labeled data (supervised learning).
        
        Note: Current implementation uses only labeled data. For semi-supervised learning,
        could implement consistency regularization or pseudo-labeling in the future.
        
        Args:
            X_labeled: (n_samples, n_features) labeled training data
            y_labeled: (n_samples,) labeled targets
            X_unlabeled: (m_samples, n_features) unlabeled data (currently unused)
        """
        set_seed(self.random_state)
        
        # Reset early stopping state for clean re-fit
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        self.model = FTTransformerNet(
            input_dim=self.input_dim,
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.epochs
        )
        
        # Convert to tensors and move to device
        X_lbl_t = torch.tensor(X_labeled, dtype=torch.float32).to(self.device)
        y_lbl_t = torch.tensor(y_labeled, dtype=torch.float32).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_lbl_t, y_lbl_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        if self.verbose:
            print(f"[FT-Transformer] Training with {len(X_labeled)} labeled samples")
            print(f"[FT-Transformer] Config: d_token={self.d_token}, n_blocks={self.n_blocks}, "
                  f"n_heads={self.n_heads}, lr={self.lr}, batch_size={self.batch_size}")

        # Training loop with early stopping and best model checkpointing
        best_state = None
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_x, batch_y in loader:
                pred = self.model(batch_x)
                loss = F.mse_loss(pred, batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / max(1, n_batches)
            self.scheduler.step()
            
            # Early stopping with best model checkpointing
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                if self.verbose and (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1:3d}/{self.epochs} | Loss: {avg_loss:.6f} ✓")
            else:
                self.patience_counter += 1
                if self.verbose and (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1:3d}/{self.epochs} | Loss: {avg_loss:.6f} (patience: {self.patience_counter}/{self.patience})")
                
                if self.patience_counter >= self.patience:
                    if self.verbose:
                        print(f"[FT-Transformer] Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            X: (n_samples, n_features) test data
        Returns:
            predictions: (n_samples,) regression predictions
        """
        if self.model is None:
            raise ValueError("[FT-Transformer] Model not fitted yet. Call fit() first.")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for (batch_x,) in loader:
                pred_batch = self.model(batch_x)
                preds.append(pred_batch.cpu().numpy())
        
        return np.concatenate(preds).flatten()


def run_ft_transformer():
    """
    Example script to run FT-Transformer on California Housing dataset.
    Requires: utils.set_seed, preprocess_utils.{load_data, preprocess_data}
    """
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}")
        return
    
    try:
        from preprocess_utils import load_data, preprocess_data
    except ImportError:
        print("preprocess_utils not found. Using basic preprocessing...")
        from sklearn.preprocessing import StandardScaler
        data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)
        X, y = data[:, :-1], data[:, -1]
        
        Xtr_full, Xtest, ytr_full, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_labeled = scaler.fit_transform(Xtr_full[:int(len(Xtr_full)*LABELED_RATIO)])
        Xtest = scaler.transform(Xtest)
        y_labeled = ytr_full[:int(len(ytr_full)*LABELED_RATIO)]
    else:
        X, y = load_data(DATA_PATH)
        Xtr_full, Xtest, ytr_full, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
        
        N_labeled = int(len(Xtr_full) * LABELED_RATIO)
        idx = np.random.permutation(len(Xtr_full))
        X_labeled_raw = Xtr_full.iloc[idx[:N_labeled]]
        y_labeled = ytr_full.iloc[idx[:N_labeled]].values if hasattr(ytr_full, 'iloc') else ytr_full[idx[:N_labeled]]
        X_unlabeled_raw = Xtr_full.iloc[idx[N_labeled:]]
        
        X_labeled, X_unlabeled, Xtest = preprocess_data(X_labeled_raw, X_unlabeled_raw, Xtest)
    
    model = FTTransformer(input_dim=X_labeled.shape[1], verbose=True)
    model.fit(X_labeled, y_labeled)
    
    pred = model.predict(Xtest)
    rmse = np.sqrt(mean_squared_error(ytest, pred))
    print(f'\n[FT-Transformer] Final RMSE (10% labeled): {rmse:.4f}')


if __name__ == "__main__":
    run_ft_transformer()
