# HeSCo: Heterogeneous Semi-supervised Co-training

HeSCo is a powerful semi-supervised learning framework for regression tasks. It utilizes a **heterogeneous co-training** approach, combining a neural network (NN) with Gradient Boosting (XGBoost) to leverage unlabeled data and improve predictive performance.

## 🚀 Key Features

- **Heterogeneous Architecture**: Combines the strengths of deep learning (ResNet-based) and tree-based models (XGBoost).
- **Mutual Updates**: Uses a soft-update mechanism where models learn from each other's high-confidence predictions on unlabeled data.
- **Uncertainty-Aware Distillation**: Incorporates pinball loss for quantile regression to estimate prediction uncertainty and guide the distillation process.
- **Resilient Preprocessing**: Robust data handling including adaptive encoding and leakage-free scaling.

## 📦 Installation

Ensure you have Python 3.8+ installed. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## 🛠 Usage Example

```python
from hesco import HeSCo
import numpy as np

# Initialize HeSCo
model = HeSCo(
    epochs=100,
    batch_size=256,
    use_ensemble=True,
    verbose=True
)

# Fit the model with labeled and unlabeled data
# X_labeled: (n_samples, n_features)
# y_labeled: (n_samples,)
# X_unlabeled: (m_samples, n_features)
model.fit(X_labeled, y_labeled, X_unlabeled)

# Make predictions
predictions = model.predict(X_test)
```

## 📊 Benchmarking

You can compare HeSCo against other semi-supervised methods using the provided scripts:

```bash
# Run the main benchmark comparison
python run_benchmark.py

# Run sensitivity analysis for hyperparameters
python run_sensitivity.py

# Run ablation studies
python run_ablation.py
```

## 📂 Project Structure

- `hesco.py`: Core implementation of the HeSCo class and neural network structure.
- `run_benchmark.py`: Script to evaluate HeSCo against baselines (VIME, Drill, etc.).
- `preprocess_utils.py`: Data cleaning and preprocessing utilities.
- `data/`: Directory to place your CSV datasets for benchmarking.
- `logs/`: Directory where execution results and benchmark reports are saved.

## 📄 License
This project is for research purposes.
