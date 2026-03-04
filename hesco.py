import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import warnings
import os
from utils import set_seed

os.environ['LOKY_MAX_CPU_COUNT'] = '4'
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SmoothPinballLoss(nn.Module):
    def __init__(self, quantiles=[0.25, 0.5, 0.75], beta=0.1):
        super().__init__()
        self.quantiles = quantiles
        self.beta = beta

    def forward(self, preds, target):
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            error = target - preds[:, i]
            huber_penalty = nn.functional.huber_loss(
                preds[:, i],
                target,
                reduction='none',
                delta=self.beta,
            )
            multiplier = torch.where(
                error > 0,
                torch.tensor(q, device=preds.device),
                torch.tensor(1.0 - q, device=preds.device),
            )
            loss += (multiplier * huber_penalty).mean()
        return loss


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        return self.relu(x + out)


class RegressionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.pred_quantiles = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.pred_quantiles(h)


class HeSCo(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        use_mutual_update=True,
        use_ensemble=True,
        xgb_params=None,
        hidden_dim=128,
        num_blocks=2,
        dropout=0.1,
        lr=1e-3,
        batch_size=256,
        epochs=100,
        adversarial_weight=1.0,
        unlabeled_weight_ratio=0.5,
        unlabeled_batch_ratio=0.5,
        inc_trees=30,
        unc_sample_size=5000,
        update_schedule='mid_late',
        standardize_input=True,
        pinball_beta=0.1,
        conf_percentile=30,
        random_state=42,
        verbose=False,
    ):
        self.use_mutual_update = use_mutual_update
        self.use_ensemble = use_ensemble
        self.xgb_params = xgb_params if xgb_params else {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'n_jobs': 4,
            'tree_method': 'hist',
            'random_state': random_state,
        }
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.adversarial_weight = adversarial_weight
        self.unlabeled_weight_ratio = unlabeled_weight_ratio
        self.unlabeled_batch_ratio = unlabeled_batch_ratio
        self.inc_trees = inc_trees
        self.unc_sample_size = unc_sample_size
        self.update_schedule = update_schedule
        self.standardize_input = standardize_input
        self.pinball_beta = pinball_beta
        self.conf_percentile = conf_percentile
        self.random_state = random_state
        self.verbose = verbose
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.xgb_model = None
        self.nn_model = None
        self.xgb_r2 = 0.0
        self.nn_r2 = 0.0

    def _train_xgboost(self, X, y, sample_weight=None, init_model=None, update_trees=0):
        params = self.xgb_params.copy()
        params['random_state'] = self.random_state

        if init_model is not None and update_trees > 0:
            current_trees = init_model.get_booster().num_boosted_rounds()
            params['n_estimators'] = current_trees + update_trees
            model = xgb.XGBRegressor(**params)
            model.fit(X, y, sample_weight=sample_weight, xgb_model=init_model.get_booster())
        else:
            model = xgb.XGBRegressor(**params)
            model.fit(X, y, sample_weight=sample_weight)
        return model

    def _estimate_nn_reliability(self, X_tensor):
        self.nn_model.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(DEVICE)
            preds = self.nn_model(X_tensor)
            pred_base = preds[:, 1].cpu().numpy().flatten()
            nn_instability = torch.abs(preds[:, 2] - preds[:, 0]).cpu().numpy().flatten()
        return pred_base, nn_instability

    def _soft_update_xgboost(
        self,
        X_L,
        y_L,
        X_U_sc,
        nn_pred_U,
        nn_instability,
        xgb_pred_current,
        X_U_all=None,
    ):
        divergence = np.abs(nn_pred_U - xgb_pred_current)
        nn_conf_thresh = np.percentile(nn_instability, self.conf_percentile)
        mask_nn_trustworthy = nn_instability < nn_conf_thresh

        weights_U = np.exp(-nn_instability) * np.log1p(divergence)
        weights_U = weights_U * mask_nn_trustworthy.astype(float)

        if weights_U.max() > 0:
            weights_U = (weights_U / weights_U.max()) * self.unlabeled_weight_ratio

        valid_idx = weights_U > 1e-3
        if valid_idx.sum() > 0:
            blend_alpha = 0.7
            pseudo_labels = blend_alpha * nn_pred_U + (1.0 - blend_alpha) * xgb_pred_current
            X_mix = np.vstack([X_L, X_U_sc[valid_idx]])
            y_mix = np.concatenate([y_L, pseudo_labels[valid_idx]])
            w_mix = np.concatenate([np.ones(len(X_L)), weights_U[valid_idx]])
        else:
            X_mix, y_mix, w_mix = X_L, y_L, np.ones(len(X_L))

        self.xgb_model = self._train_xgboost(
            X_mix,
            y_mix,
            sample_weight=w_mix,
            init_model=self.xgb_model,
            update_trees=self.inc_trees,
        )
        return self.xgb_model.predict(X_U_all if X_U_all is not None else X_U_sc)

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        set_seed(self.random_state)

        X_all = np.vstack([X_labeled, X_unlabeled])
        if self.standardize_input:
            self.scaler_x.fit(X_all)
            X_L_sc = self.scaler_x.transform(X_labeled)
            X_U_sc = self.scaler_x.transform(X_unlabeled)
        else:
            X_L_sc = X_labeled
            X_U_sc = X_unlabeled

        y_L_sc = self.scaler_y.fit_transform(y_labeled.reshape(-1, 1)).flatten()

        self.xgb_model = self._train_xgboost(X_L_sc, y_L_sc)
        xgb_pred_U_sc = self.xgb_model.predict(X_U_sc)

        self.nn_model = RegressionNetwork(
            input_dim=X_labeled.shape[1],
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
        ).to(DEVICE)

        optimizer = optim.AdamW(self.nn_model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        quantile_criterion = SmoothPinballLoss(
            quantiles=[0.25, 0.5, 0.75],
            beta=self.pinball_beta,
        ).to(DEVICE)

        t_X_L = torch.tensor(X_L_sc, dtype=torch.float32)
        t_y_L = torch.tensor(y_L_sc, dtype=torch.float32)
        t_X_U = torch.tensor(X_U_sc, dtype=torch.float32).to(DEVICE)
        t_xgb_U = torch.tensor(xgb_pred_U_sc, dtype=torch.float32).to(DEVICE)

        labeled_loader = DataLoader(
            TensorDataset(t_X_L, t_y_L),
            batch_size=min(self.batch_size, len(t_X_L)),
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        update_epochs = (
            [int(self.epochs * 0.4), int(self.epochs * 0.8)] if self.update_schedule == 'mid_late' else []
        )
        self.nn_model.train()

        for epoch in range(self.epochs):
            if self.use_mutual_update and epoch in update_epochs and len(X_unlabeled) > 0:
                n_u = len(t_X_U)
                sample_size = min(self.unc_sample_size, n_u)
                if sample_size < n_u:
                    idx_sample = np.random.choice(n_u, sample_size, replace=False)
                    t_X_U_sub = t_X_U[idx_sample]
                    X_U_sc_sub = X_U_sc[idx_sample]
                    xgb_pred_current_sub = t_xgb_U[idx_sample].cpu().numpy()
                else:
                    t_X_U_sub, X_U_sc_sub, xgb_pred_current_sub = t_X_U, X_U_sc, t_xgb_U.cpu().numpy()

                nn_pred_np, nn_instability = self._estimate_nn_reliability(t_X_U_sub)
                new_anchors = self._soft_update_xgboost(
                    X_L_sc,
                    y_L_sc,
                    X_U_sc_sub,
                    nn_pred_np,
                    nn_instability,
                    xgb_pred_current_sub,
                    X_U_all=X_U_sc,
                )
                t_xgb_U = torch.tensor(new_anchors, dtype=torch.float32).to(DEVICE)
                self.nn_model.train()

            for batch_idx, (batch_x_l, batch_y_l) in enumerate(labeled_loader):
                batch_x_l, batch_y_l = batch_x_l.to(DEVICE), batch_y_l.to(DEVICE)
                optimizer.zero_grad()

                preds_l = self.nn_model(batch_x_l)
                loss_sup = quantile_criterion(preds_l, batch_y_l)

                loss_distill = torch.tensor(0.0).to(DEVICE)

                ul_interval = int(1.0 / self.unlabeled_batch_ratio) if self.unlabeled_batch_ratio > 0 else 0
                if len(t_X_U) > 0 and (ul_interval > 0) and ((batch_idx % ul_interval) == 0):
                    idx_u = torch.randperm(len(t_X_U))[:batch_x_l.size(0)].to(DEVICE)
                    batch_x_u, batch_xgb_u = t_X_U[idx_u], t_xgb_U[idx_u]

                    preds_u = self.nn_model(batch_x_u)

                    nn_instability_u = torch.abs(preds_u[:, 2] - preds_u[:, 0]).detach()
                    tau = torch.median(nn_instability_u) + 1e-6
                    trust_xgb_weight = torch.tanh(nn_instability_u / tau)

                    raw_distill_loss = quantile_criterion(preds_u, batch_xgb_u)
                    loss_distill = (raw_distill_loss * trust_xgb_weight).mean()

                loss = loss_sup + self.adversarial_weight * loss_distill
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

        if self.verbose:
            xgb_cv_scores = cross_val_score(self.xgb_model, X_L_sc, y_L_sc, cv=5, scoring='r2')
            self.xgb_r2 = xgb_cv_scores.mean()

            self.nn_model.eval()
            with torch.no_grad():
                nn_p = self.nn_model(t_X_L.to(DEVICE))[:, 1].cpu().numpy().flatten()
                ss_res = ((y_L_sc - nn_p) ** 2).sum()
                ss_tot = ((y_L_sc - y_L_sc.mean()) ** 2).sum()
                self.nn_r2 = 1 - ss_res / (ss_tot + 1e-8)

            print(f"Eval R2 -> NN: {self.nn_r2:.4f}, XGB (OOF): {self.xgb_r2:.4f}")

        return self

    def predict(self, X):
        X_sc = self.scaler_x.transform(X) if self.standardize_input else X

        if self.xgb_model is None:
            raise RuntimeError("Not fitted")

        self.nn_model.eval()
        with torch.no_grad():
            preds_sc = self.nn_model(torch.tensor(X_sc, dtype=torch.float32).to(DEVICE))
            nn_pred_sc = preds_sc[:, 1].cpu().numpy().flatten()

        nn_pred = self.scaler_y.inverse_transform(nn_pred_sc.reshape(-1, 1)).flatten()

        if not self.use_ensemble:
            return nn_pred

        xgb_pred = self.scaler_y.inverse_transform(
            self.xgb_model.predict(X_sc).reshape(-1, 1),
        ).flatten()
        return 0.5 * nn_pred + 0.5 * xgb_pred

