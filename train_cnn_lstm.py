"""
train_cnn_lstm.py
=================
PyTorch training script for the Phase-Aware CNN-LSTM model.

Paper: "Overcoming the Phase Discontinuity Problem in RIS Optimization:
        A Robust and Scalable Phase-Aware Deep Regression Framework
        via Hybrid CNN-LSTM Architecture"
Journal: Advanced Intelligent Systems, 2026
Authors: Seda Savaşçı Şen, Ali Çalhan, Murtaza Cicioğlu, Oğuzhan Demiryürek

Architecture (Section 3.2):
    Input:  X_feat   ∈ R^{B × 4N}  — [Re(F), Im(F), Re(G), Im(G)]  Eq.(5)
    Output: Y_sincos ∈ R^{B × 2N}  — [sin(θ), cos(θ)]              Eq.(11)
    Phase recovery: θ̂ = atan2(Y[:,:N], Y[:,N:])                   Eq.(12)

Requirements:
    pip install torch scipy numpy matplotlib tqdm

Usage:
    # Train with default settings (N=32):
    python train_cnn_lstm.py

    # Train with different array size:
    python train_cnn_lstm.py --N 64 --data_path RIS_dataset_N64.mat

    # Evaluate only:
    python train_cnn_lstm.py --eval_only --checkpoint best_model_N32.pth
"""

import argparse
import os
import time
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET
# ══════════════════════════════════════════════════════════════════════════════

class RISDataset(Dataset):
    """
    Loads the RIS channel dataset from a .mat file.

    X_feat   : [S × 4N]  NN input features  = [Re(F), Im(F), Re(G), Im(G)]
    Y_sincos : [S × 2N]  Sine-cosine labels = [sin(θ*), cos(θ*)]

    The split indices (idx_tr, idx_val, idx_te) are loaded from the .mat file
    and used to select the appropriate subset.
    """

    def __init__(self, mat_path: str, split: str = 'train'):
        """
        Args:
            mat_path : Path to RIS_dataset_N*.mat file
            split    : 'train', 'val', or 'test'
        """
        assert split in ('train', 'val', 'test'), \
            "split must be 'train', 'val', or 'test'"

        print(f"  Loading {split} data from {mat_path} ...")
        data = sio.loadmat(mat_path)

        X_all = data['X_feat'].astype(np.float32)    # [S × 4N]
        Y_all = data['Y_sincos'].astype(np.float32)  # [S × 2N]

        # MATLAB indices are 1-based → convert to 0-based
        idx_map = {'train': 'idx_tr', 'val': 'idx_val', 'test': 'idx_te'}
        idx = data[idx_map[split]].flatten().astype(int) - 1  # 0-indexed

        self.X = torch.from_numpy(X_all[idx])   # [n_split × 4N]
        self.Y = torch.from_numpy(Y_all[idx])   # [n_split × 2N]

        # Z-score normalization using training set statistics
        # (applied to all splits using train mean/std)
        if split == 'train':
            self.mean = self.X.mean(dim=0, keepdim=True)
            self.std  = self.X.std(dim=0, keepdim=True) + 1e-9
        # For val/test, call set_normalization() after creating train dataset

        print(f"  {split}: {len(self.X)} samples  |  "
              f"X: {self.X.shape}  Y: {self.Y.shape}")

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """Apply training-set normalization statistics."""
        self.mean = mean
        self.std  = std

    def normalize(self):
        """Apply z-score normalization. Call after set_normalization()."""
        self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL — Phase-Aware CNN-LSTM (Section 3.2)
# ══════════════════════════════════════════════════════════════════════════════

class PhaseAwareCNNLSTM(nn.Module):
    """
    Phase-Aware CNN-LSTM Architecture (Figure 2 in paper).

    Blocks:
      (i)  Convolutional Block  — local spatial feature extraction (Eq.14)
      (ii) LSTM Block           — inter-element dependency modeling (Eq.15)
      (iii)Fully Connected Block— sine-cosine phase regression

    Input  shape: [B, 4N]       (batch × features)
    Reshape to  : [B, 4, N]     (batch × channels × sequence)
    Output shape: [B, 2N]       [sin(θ₁..θ_N), cos(θ₁..θ_N)]
    """

    def __init__(self, N: int = 32):
        """
        Args:
            N : Number of RIS elements (default 32)
        """
        super().__init__()
        self.N = N

        # ── Convolutional Block (Section 3.2.2) ───────────────────────
        # Two 1D-CNN layers with kernel_size=3, same padding
        # W^(1) ∈ R^{64 × 4 × 3}  — captures 3-element receptive field
        # W^(2) ∈ R^{64 × 64 × 3} — effective receptive field: 5 elements
        self.conv_block = nn.Sequential(
            # Conv1D Layer 1 — Eq.(14)
            nn.Conv1d(in_channels=4, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            # Conv1D Layer 2
            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # ── LSTM Block (Section 3.2.3) — Eq.(15) ─────────────────────
        # Two-layer stacked LSTM: hidden_size 128 → 64
        # Processes CNN features sequentially (row-major element order)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.lstm_proj = nn.Linear(128, 64)  # 128 → 64 projection

        # ── Fully Connected Block (Section 3.2.4) ────────────────────
        # FC256 → FC128 → FC64 → FC(2N)
        fc_input = 64 * N
        self.fc_block = nn.Sequential(
            nn.Linear(fc_input, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 2 * N),   # Output: [sin(θ₁..N), cos(θ₁..N)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 4N]  — input features
        Returns:
            y : [B, 2N]  — [sin(θ), cos(θ)] predictions
        """
        B = x.shape[0]

        # Reshape: [B, 4N] → [B, 4, N]  (channels × sequence length)
        x = x.view(B, 4, self.N)

        # Convolutional block: [B, 4, N] → [B, 64, N]
        x = self.conv_block(x)

        # LSTM expects: [B, seq_len, features] = [B, N, 64]
        x = x.permute(0, 2, 1)                         # [B, N, 64]
        x, _ = self.lstm(x)                             # [B, N, 128]
        x = self.lstm_proj(x)                           # [B, N, 64]

        # Flatten for FC: [B, N, 64] → [B, N×64]
        x = x.contiguous().view(B, -1)

        # FC block: [B, N×64] → [B, 2N]
        y = self.fc_block(x)
        return y

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOSS FUNCTION — Phase-Aware MSE (Section 3.3, Eq.16)
# ══════════════════════════════════════════════════════════════════════════════

class PhaseAwareLoss(nn.Module):
    """
    Phase-Aware MSE Loss — Eq.(16):

    L = (1/BN) Σ_s Σ_n [(sin_pred - sin_true)² + (cos_pred - cos_true)²]

    Smooth and differentiable on R², encodes periodicity naturally.
    Avoids the ±π discontinuity of direct phase regression.
    """

    def forward(self,
                y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(y_pred, y_true)


# ══════════════════════════════════════════════════════════════════════════════
# 4. METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_rmse_degrees(y_pred: torch.Tensor,
                         y_true: torch.Tensor,
                         N: int) -> float:
    """
    Phase-Aware RMSE — Eq.(9):
    RMSE = sqrt( (1/S·N) Σ_s Σ_n [∠(exp(j(θ_pred - θ_true)))]² )
    reported in degrees.

    Args:
        y_pred : [B, 2N]  — predicted [sin, cos]
        y_true : [B, 2N]  — true [sin, cos]
        N      : number of RIS elements
    Returns:
        RMSE in degrees (scalar float)
    """
    with torch.no_grad():
        # Recover phases via atan2
        sin_pred = y_pred[:, :N];   cos_pred = y_pred[:, N:]
        sin_true = y_true[:, :N];   cos_true = y_true[:, N:]

        theta_pred = torch.atan2(sin_pred, cos_pred)   # [B, N]
        theta_true = torch.atan2(sin_true, cos_true)   # [B, N]

        # Wrap phase difference to [-π, π]
        diff = theta_pred - theta_true
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))

        rmse_rad = torch.sqrt((diff ** 2).mean())
        rmse_deg = rmse_rad * 180 / torch.pi
    return rmse_deg.item()


def compute_success_rate(y_pred: torch.Tensor,
                         y_true: torch.Tensor,
                         N: int,
                         threshold_deg: float = 5.0) -> float:
    """
    Success Rate — Eq.(10): SR(τ) = fraction of per-element errors < τ°
    Default τ = 5° as reported in Table 4.
    """
    with torch.no_grad():
        sin_pred = y_pred[:, :N];   cos_pred = y_pred[:, N:]
        sin_true = y_true[:, :N];   cos_true = y_true[:, N:]

        theta_pred = torch.atan2(sin_pred, cos_pred)
        theta_true = torch.atan2(sin_true, cos_true)

        diff_rad = torch.atan2(
            torch.sin(theta_pred - theta_true),
            torch.cos(theta_pred - theta_true)
        )
        diff_deg = diff_rad.abs() * 180 / torch.pi
        sr = (diff_deg < threshold_deg).float().mean()
    return sr.item() * 100  # percent


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device, N):
    model.train()
    total_loss = 0.0
    total_rmse = 0.0

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_hat = model(X)
        loss  = criterion(Y_hat, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        total_rmse += compute_rmse_degrees(Y_hat, Y, N)

    n = len(loader)
    return total_loss / n, total_rmse / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, N):
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    total_sr   = 0.0

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        Y_hat = model(X)
        total_loss += criterion(Y_hat, Y).item()
        total_rmse += compute_rmse_degrees(Y_hat, Y, N)
        total_sr   += compute_success_rate(Y_hat, Y, N, threshold_deg=5.0)

    n = len(loader)
    return total_loss / n, total_rmse / n, total_sr / n


# ══════════════════════════════════════════════════════════════════════════════
# 6. LEARNING RATE SCHEDULE (Section 3.4.1, Eq.17)
# ══════════════════════════════════════════════════════════════════════════════

def build_scheduler(optimizer, milestones=(15, 30, 45)):
    """
    Piecewise constant LR schedule — Eq.(17):
      α₀ = 1e-3  for epochs  1-15
      α₁ = 5e-4  for epochs 16-30
      α₂ = 2.5e-4 for epochs 31-45
      α₃ = 1.25e-4 for epochs 46-50
    """
    return optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(milestones), gamma=0.5
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. UNIT-CIRCLE PROJECTION (Section 3.1, Remark 3, Eq.13)
# ══════════════════════════════════════════════════════════════════════════════

def unit_circle_projection(y_pred: torch.Tensor, N: int) -> torch.Tensor:
    """
    L2-normalize (sin, cos) pairs to the unit circle — Eq.(13):
        (s̃, c̃) = (ŝ, ĉ) / sqrt(ŝ² + ĉ²)

    Applied at inference to guarantee valid phase values.
    Overhead: O(N) — negligible.
    """
    sin_hat = y_pred[:, :N]
    cos_hat = y_pred[:, N:]
    norm    = torch.sqrt(sin_hat**2 + cos_hat**2 + 1e-12)
    sin_hat = sin_hat / norm
    cos_hat = cos_hat / norm
    return torch.cat([sin_hat, cos_hat], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Train Phase-Aware CNN-LSTM for RIS phase optimization')
    p.add_argument('--data_path',  type=str,   default='RIS_dataset_N32.mat')
    p.add_argument('--N',          type=int,   default=32,
                   help='Number of RIS elements (32/64/128/256)')
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--batch_size', type=int,   default=256)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--checkpoint', type=str,   default=None,
                   help='Path to checkpoint for evaluation')
    p.add_argument('--eval_only',  action='store_true')
    p.add_argument('--out_dir',    type=str,   default='results')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Phase-Aware CNN-LSTM Training")
    print(f"  N = {args.N}  |  Device: {device}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = RISDataset(args.data_path, split='train')
    val_ds   = RISDataset(args.data_path, split='val')
    test_ds  = RISDataset(args.data_path, split='test')

    # Apply training normalization to all splits
    train_ds.normalize()
    val_ds.set_normalization(train_ds.mean, train_ds.std); val_ds.normalize()
    test_ds.set_normalization(train_ds.mean, train_ds.std); test_ds.normalize()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ────────────────────────────────────────────────
    model     = PhaseAwareCNNLSTM(N=args.N).to(device)
    criterion = PhaseAwareLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.999), eps=1e-8)
    scheduler = build_scheduler(optimizer, milestones=(15, 30, 45))

    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"  CNN Block : {sum(p.numel() for n,p in model.named_parameters() if 'conv' in n):,}")
    print(f"  LSTM Block: {sum(p.numel() for n,p in model.named_parameters() if 'lstm' in n):,}")
    print(f"  FC Block  : {sum(p.numel() for n,p in model.named_parameters() if 'fc' in n):,}")

    # ── Evaluation Only ───────────────────────────────────────
    if args.eval_only:
        assert args.checkpoint, "Provide --checkpoint for --eval_only"
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        _, rmse, sr = evaluate(model, test_loader, criterion, device, args.N)
        print(f"\nTest Results:")
        print(f"  RMSE       : {rmse:.4f}°")
        print(f"  SR@5°      : {sr:.2f}%")
        return

    # ── Training ──────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs ...\n")
    history = {'train_loss': [], 'val_loss': [],
               'train_rmse': [], 'val_rmse': [],
               'val_sr': [], 'lr': []}

    best_val_rmse = float('inf')
    best_ckpt_path = os.path.join(args.out_dir, f'best_model_N{args.N}.pth')

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        tr_loss, tr_rmse = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.N)

        # Validate
        val_loss, val_rmse, val_sr = evaluate(
            model, val_loader, criterion, device, args.N)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Save history
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_rmse'].append(tr_rmse)
        history['val_rmse'].append(val_rmse)
        history['val_sr'].append(val_sr)
        history['lr'].append(current_lr)

        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'val_rmse':    val_rmse,
                'val_sr':      val_sr,
                'N':           args.N,
                'args':        vars(args),
            }, best_ckpt_path)
            marker = ' ← best'
        else:
            marker = ''

        elapsed = (time.time() - t_start) / 60
        print(f"Epoch [{epoch:3d}/{args.epochs}]  "
              f"LR={current_lr:.2e}  "
              f"Train RMSE={tr_rmse:.4f}°  "
              f"Val RMSE={val_rmse:.4f}°  "
              f"SR@5°={val_sr:.1f}%  "
              f"[{elapsed:.1f} min]{marker}")

    # ── Test Evaluation ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Final Evaluation on Test Set")
    print(f"{'='*60}")

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    # Apply unit-circle projection at inference (Remark 3, Eq.13)
    model.eval()
    all_preds, all_true = [], []
    t_inf = []

    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(device)
            t0 = time.perf_counter()
            Y_hat = model(X)
            Y_hat = unit_circle_projection(Y_hat, args.N)  # Eq.(13)
            t_inf.append((time.perf_counter() - t0) / X.shape[0] * 1000)
            all_preds.append(Y_hat.cpu())
            all_true.append(Y)

    Y_pred = torch.cat(all_preds)
    Y_true = torch.cat(all_true)
    test_rmse = compute_rmse_degrees(Y_pred, Y_true, args.N)
    test_sr   = compute_success_rate(Y_pred, Y_true, args.N, threshold_deg=5.0)
    avg_inf   = np.mean(t_inf)

    print(f"  Test RMSE       : {test_rmse:.4f}°  "
          f"(Paper Table 4: 2.77° for N=32)")
    print(f"  Test SR@5°      : {test_sr:.2f}%   "
          f"(Paper Table 4: 95%  for N=32)")
    print(f"  Avg inference   : {avg_inf:.3f} ms/sample  "
          f"(Paper: 0.08 ms, CPU)")
    print(f"  Best checkpoint : {best_ckpt_path}")

    # ── Learning Curves ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Training Curves — N={args.N}', fontsize=12)

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'],   label='Val Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Phase-Aware Loss (Eq.16)')
    axes[0].legend(); axes[0].grid(True)

    # RMSE
    axes[1].plot(history['train_rmse'], label='Train RMSE')
    axes[1].plot(history['val_rmse'],   label='Val RMSE')
    axes[1].axhline(y=2.77, color='r', linestyle='--',
                    label='Paper result (2.77°)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('RMSE [°]')
    axes[1].set_title('Phase-Aware RMSE (Eq.9)')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    curve_path = os.path.join(args.out_dir, f'training_curves_N{args.N}.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    print(f"\nLearning curves saved: {curve_path}")

    print(f"\nTotal training time: {(time.time()-t_start)/60:.1f} minutes")
    print("Done.")


if __name__ == '__main__':
    main()
