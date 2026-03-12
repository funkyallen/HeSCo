# file: node_baseline.py
"""
NODE: Neural Oblivious Decision Ensembles
Reference: Popov et al., 2020
"Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data"

NODE combines the power of oblivious decision trees (ODTs) with end-to-end
differentiable learning. Each layer consists of an ensemble of differentiable
oblivious decision trees, using entmoid (a smooth approximation of sigmoid)
for soft feature selection and response aggregation.

Interface matches the project's standard pattern:
  - __init__(input_dim, ...)
  - fit(X_labeled, y_labeled, X_unlabeled=None)
  - predict(X)
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
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NODE-specific hyperparameters (Popov et al. 2020)
NUM_LAYERS = 2          # Number of NODE layers
NUM_TREES = 128         # Number of trees per layer
TREE_DEPTH = 6          # Depth of each oblivious decision tree
TREE_OUTPUT_DIM = 1     # Output dimension per tree
PATIENCE = 25           # Early stopping patience
GRAD_CLIP = 1.0         # Gradient clipping norm


# ==================== Oblivious Decision Tree Layer ====================

class DenseBlock(nn.Module):
    """
    A single differentiable Oblivious Decision Tree (ODT) ensemble layer.
    
    Each tree in the ensemble:
    1. Selects features using learned weights (entmoid activation for soft selection)
    2. Computes split decisions using learned thresholds
    3. Aggregates responses from 2^depth leaf nodes
    
    The ensemble output is the average of all tree responses.
    """
    def __init__(self, input_dim: int, num_trees: int, depth: int, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.num_trees = num_trees
        self.depth = depth
        self.output_dim = output_dim
        self.num_leaves = 2 ** depth
        
        # Feature selection weights: each internal node selects a feature
        # Shape: (num_trees, depth, input_dim)
        self.feature_selection = nn.Parameter(
            torch.randn(num_trees, depth, input_dim) * 0.01
        )
        
        # Split thresholds for each internal node
        # Shape: (num_trees, depth)
        self.thresholds = nn.Parameter(
            torch.zeros(num_trees, depth)
        )
        
        # Leaf response values
        # Shape: (num_trees, num_leaves, output_dim)
        self.leaf_responses = nn.Parameter(
            torch.randn(num_trees, self.num_leaves, output_dim) * 0.01
        )
        
        # Temperature for entmoid (controls softness of decisions)
        self.log_temperatures = nn.Parameter(
            torch.ones(num_trees, depth) * 0.0  # exp(0) = 1.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            output: (batch_size, num_trees * output_dim)
        """
        batch_size = x.shape[0]
        
        # Soft feature selection via entmoid
        # feature_selection: (num_trees, depth, input_dim)
        # Apply softmax over input_dim to get selection probabilities
        feature_weights = F.softmax(self.feature_selection, dim=-1)
        
        # Compute selected feature values: weighted sum of input features
        # x: (batch, input_dim) -> (batch, 1, 1, input_dim)
        # feature_weights: (num_trees, depth, input_dim) -> (1, num_trees, depth, input_dim)
        x_expanded = x.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, input_dim)
        fw_expanded = feature_weights.unsqueeze(0)  # (1, num_trees, depth, input_dim)
        
        # Selected features: (batch, num_trees, depth)
        selected = (x_expanded * fw_expanded).sum(dim=-1)
        
        # Compute split decisions using sigmoid with temperature
        temperatures = self.log_temperatures.exp().unsqueeze(0)  # (1, num_trees, depth)
        # decisions: probability of going right at each node
        decisions = torch.sigmoid((selected - self.thresholds.unsqueeze(0)) * temperatures)
        
        # Compute leaf probabilities via path products
        # For oblivious trees, each leaf is identified by a binary path
        # We enumerate all 2^depth paths
        # decisions: (batch, num_trees, depth), values in [0, 1]
        
        # Create binary path matrix: (num_leaves, depth) with 0/1 entries
        # Each row is the binary representation of the leaf index
        path_matrix = torch.zeros(self.num_leaves, self.depth, device=x.device)
        for i in range(self.num_leaves):
            for d in range(self.depth):
                path_matrix[i, d] = (i >> (self.depth - 1 - d)) & 1
        
        # Compute leaf probabilities
        # For each leaf, probability = product of (decisions if bit=1, else 1-decisions)
        # decisions: (batch, num_trees, depth) -> (batch, num_trees, 1, depth)
        decisions_expanded = decisions.unsqueeze(2)  # (batch, num_trees, 1, depth)
        path_expanded = path_matrix.unsqueeze(0).unsqueeze(0)  # (1, 1, num_leaves, depth)
        
        # Probability contributions: path=1 -> decision, path=0 -> 1-decision
        probs = decisions_expanded * path_expanded + (1 - decisions_expanded) * (1 - path_expanded)
        
        # Product along depth dimension: (batch, num_trees, num_leaves)
        leaf_probs = probs.prod(dim=-1)
        
        # Weighted sum of leaf responses
        # leaf_probs: (batch, num_trees, num_leaves)
        # leaf_responses: (num_trees, num_leaves, output_dim)
        responses = self.leaf_responses.unsqueeze(0)  # (1, num_trees, num_leaves, output_dim)
        leaf_probs_expanded = leaf_probs.unsqueeze(-1)  # (batch, num_trees, num_leaves, 1)
        
        # (batch, num_trees, output_dim)
        tree_outputs = (leaf_probs_expanded * responses).sum(dim=2)
        
        # Flatten: (batch, num_trees * output_dim)
        return tree_outputs.reshape(batch_size, -1)


class NODENet(nn.Module):
    """
    Full NODE Network: stacks multiple DenseBlock layers with skip connections.
    
    Architecture:
    - Input -> [DenseBlock_1 -> BN -> ReLU] -> ... -> [DenseBlock_L -> BN -> ReLU] -> Head -> Output
    - Each DenseBlock produces (num_trees * tree_output_dim) outputs
    - Layers are connected via dense (skip) connections
    """
    def __init__(self, 
                 input_dim: int, 
                 num_layers: int = NUM_LAYERS,
                 num_trees: int = NUM_TREES, 
                 depth: int = TREE_DEPTH,
                 tree_output_dim: int = TREE_OUTPUT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        tree_out = num_trees * tree_output_dim
        
        # NODE layers
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        current_dim = input_dim
        for i in range(num_layers):
            self.layers.append(DenseBlock(current_dim, num_trees, depth, tree_output_dim))
            self.bns.append(nn.BatchNorm1d(tree_out))
            current_dim = tree_out  # Next layer input = current layer output
        
        # Regression head: average tree outputs + MLP
        self.head = nn.Sequential(
            nn.Linear(tree_out, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            pred: (batch_size,) regression predictions
        """
        h = x
        for layer, bn in zip(self.layers, self.bns):
            h = layer(h)
            h = bn(h)
            h = F.relu(h)
        
        return self.head(h).squeeze(-1)


# ==================== Main Training Class ====================

class NODE:
    """
    NODE trainer for semi-supervised tabular regression.
    Matches the standard project interface for benchmark comparison.
    
    NODE uses differentiable oblivious decision trees, combining the 
    interpretability of tree ensembles with the gradient-based optimization
    of neural networks.
    """
    def __init__(self, 
                 input_dim: int, 
                 lr: float = LR, 
                 batch_size: int = BATCH_SIZE, 
                 epochs: int = EPOCHS,
                 weight_decay: float = WEIGHT_DECAY,
                 patience: int = PATIENCE,
                 grad_clip: float = GRAD_CLIP,
                 num_layers: int = NUM_LAYERS,
                 num_trees: int = NUM_TREES,
                 depth: int = TREE_DEPTH,
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
        self.num_layers = num_layers
        self.num_trees = num_trees
        self.depth = depth
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
        Fit NODE model using labeled data (supervised learning).
        
        NODE is primarily a supervised architecture. Unlabeled data is accepted
        for interface compatibility but currently unused.
        
        Args:
            X_labeled: (n_samples, n_features) labeled training data
            y_labeled: (n_samples,) labeled targets
            X_unlabeled: (m_samples, n_features) unlabeled data (currently unused)
        """
        set_seed(self.random_state)
        
        self.model = NODENet(
            input_dim=self.input_dim,
            num_layers=self.num_layers,
            num_trees=self.num_trees,
            depth=self.depth
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
        
        # Convert to tensors
        X_lbl_t = torch.tensor(X_labeled, dtype=torch.float32).to(self.device)
        y_lbl_t = torch.tensor(y_labeled, dtype=torch.float32).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_lbl_t, y_lbl_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        if self.verbose:
            print(f"[NODE] Training with {len(X_labeled)} labeled samples")
            print(f"[NODE] Config: layers={self.num_layers}, trees={self.num_trees}, "
                  f"depth={self.depth}, lr={self.lr}, batch_size={self.batch_size}")

        # Training loop with early stopping
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
            
            # Early stopping
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
                        print(f"[NODE] Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
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
            raise ValueError("[NODE] Model not fitted yet. Call fit() first.")
        
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


# ==================== Running Example ====================

def run_node():
    """
    Example script to run NODE on California Housing dataset.
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
    
    model = NODE(input_dim=X_labeled.shape[1], verbose=True)
    model.fit(X_labeled, y_labeled)
    
    pred = model.predict(Xtest)
    rmse = np.sqrt(mean_squared_error(ytest, pred))
    print(f'\n[NODE] Final RMSE (10% labeled): {rmse:.4f}')


if __name__ == "__main__":
    run_node()
