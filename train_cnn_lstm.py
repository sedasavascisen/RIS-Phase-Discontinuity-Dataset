"""
train_cnn_lstm.py
-----------------
PyTorch implementation of the Phase-Aware CNN-LSTM model for RIS phase optimization.

The model predicts [sin(theta), cos(theta)] for each RIS element and recovers
the phase via atan2. This avoids the +-pi discontinuity that hurts direct
phase regression (see Section 3.3 in the paper).

Architecture:
    Input  [B, 4N] -> reshape [B, 4, N]
    Two Conv1D blocks (kernel=3, 64 channels, BN + ReLU)
    Two-layer LSTM (128 -> 64 hidden)
    FC block (256 -> 128 -> 64 -> 2N output)

Usage:
    python train_cnn_lstm.py                          # train N=32
    python train_cnn_lstm.py --N 64 --data_path ...  # train N=64
    python train_cnn_lstm.py --eval_only --checkpoint best_model_N32.pth

Requirements:
    pip install torch scipy numpy matplotlib
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

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True


# ---- Dataset ---------------------------------------------------------------

class RISDataset(Dataset):
    """
    Loads X_feat [S, 4N] and Y_sincos [S, 2N] from a .mat file
    produced by theta_NN_multioutput.m (or the Python preprocessing script).
    Split indices (idx_tr / idx_val / idx_te) are embedded in the .mat file.
    """

    def __init__(self, mat_path, split='train'):
        assert split in ('train', 'val', 'test')
        data  = sio.loadmat(mat_path)
        X_all = data['X_feat'].astype(np.float32)
        Y_all = data['Y_sincos'].astype(np.float32)

        key_map = {'train': 'idx_tr', 'val': 'idx_val', 'test': 'idx_te'}
        idx = data[key_map[split]].flatten().astype(int) - 1   # MATLAB -> 0-based

        self.X = torch.from_numpy(X_all[idx])
        self.Y = torch.from_numpy(Y_all[idx])
        print(f'  {split}: {len(self.X)} samples')

    def set_normalization(self, mean, std):
        self.mean = mean
        self.std  = std

    def normalize(self):
        if not hasattr(self, 'mean'):
            self.mean = self.X.mean(0, keepdim=True)
            self.std  = self.X.std(0,  keepdim=True) + 1e-9
        self.X = (self.X - self.mean) / self.std

    def __len__(self):  return len(self.X)
    def __getitem__(self, i):  return self.X[i], self.Y[i]


# ---- Model -----------------------------------------------------------------

class PhaseAwareCNNLSTM(nn.Module):
    """
    CNN-LSTM architecture (Fig. 2 in paper).
    Input: [B, 4N], Output: [B, 2N]  ->  [sin(theta_1..N), cos(theta_1..N)]
    """

    def __init__(self, N=32):
        super().__init__()
        self.N = N

        self.cnn = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.lstm      = nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.lstm_proj = nn.Linear(128, 64)

        self.fc = nn.Sequential(
            nn.Linear(64 * N, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2 * N),
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 4, self.N)           # [B, 4, N]
        x = self.cnn(x)                     # [B, 64, N]
        x = x.permute(0, 2, 1)             # [B, N, 64]
        x, _ = self.lstm(x)                 # [B, N, 128]
        x = self.lstm_proj(x)              # [B, N, 64]
        x = x.contiguous().view(B, -1)     # [B, N*64]
        return self.fc(x)                   # [B, 2N]

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---- Loss and metrics ------------------------------------------------------

def phase_aware_loss(y_pred, y_true):
    """MSE on (sin, cos) pairs -- smooth everywhere, no +-pi jump."""
    return nn.functional.mse_loss(y_pred, y_true)


def rmse_deg(y_pred, y_true, N):
    """Phase-aware RMSE in degrees (Eq. 9 in paper)."""
    with torch.no_grad():
        tp = torch.atan2(y_pred[:, :N], y_pred[:, N:])
        tt = torch.atan2(y_true[:, :N], y_true[:, N:])
        diff = torch.atan2(torch.sin(tp - tt), torch.cos(tp - tt))
        return (torch.sqrt((diff**2).mean()) * 180 / torch.pi).item()


def success_rate(y_pred, y_true, N, tau_deg=5.0):
    """SR(tau): fraction of per-element errors below tau degrees."""
    with torch.no_grad():
        tp = torch.atan2(y_pred[:, :N], y_pred[:, N:])
        tt = torch.atan2(y_true[:, :N], y_true[:, N:])
        diff = torch.atan2(torch.sin(tp - tt), torch.cos(tp - tt)).abs()
        return ((diff * 180 / torch.pi) < tau_deg).float().mean().item() * 100


# ---- Train / eval loops ----------------------------------------------------

def train_epoch(model, loader, optimizer, device, N):
    model.train()
    loss_sum = rmse_sum = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Yh = model(X)
        L  = phase_aware_loss(Yh, Y)
        L.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += L.item()
        rmse_sum += rmse_deg(Yh, Y, N)
    n = len(loader)
    return loss_sum / n, rmse_sum / n


@torch.no_grad()
def eval_epoch(model, loader, device, N):
    model.eval()
    loss_sum = rmse_sum = sr_sum = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        Yh = model(X)
        loss_sum += phase_aware_loss(Yh, Y).item()
        rmse_sum += rmse_deg(Yh, Y, N)
        sr_sum   += success_rate(Yh, Y, N)
    n = len(loader)
    return loss_sum / n, rmse_sum / n, sr_sum / n


# ---- Main ------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',  default='RIS_dataset_N32.mat')
    p.add_argument('--N',          type=int,   default=32)
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--batch_size', type=int,   default=256)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--eval_only',  action='store_true')
    p.add_argument('--out_dir',    default='results')
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nPhase-Aware CNN-LSTM  N={args.N}  device={device}\n')

    # data
    train_ds = RISDataset(args.data_path, 'train')
    val_ds   = RISDataset(args.data_path, 'val')
    test_ds  = RISDataset(args.data_path, 'test')

    train_ds.normalize()
    val_ds.set_normalization(train_ds.mean, train_ds.std);  val_ds.normalize()
    test_ds.set_normalization(train_ds.mean, train_ds.std); test_ds.normalize()

    kw = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    train_dl = DataLoader(train_ds, shuffle=True,  **kw)
    val_dl   = DataLoader(val_ds,   shuffle=False, **kw)
    test_dl  = DataLoader(test_ds,  shuffle=False, **kw)

    # model
    model     = PhaseAwareCNNLSTM(N=args.N).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45], gamma=0.5)
    ckpt_path = os.path.join(args.out_dir, f'best_model_N{args.N}.pth')

    print(f'Parameters: {model.num_params():,}')

    if args.eval_only:
        assert args.checkpoint, 'Provide --checkpoint'
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model'])
        _, r, sr = eval_epoch(model, test_dl, device, args.N)
        print(f'Test RMSE={r:.4f}°  SR@5°={sr:.2f}%');  return

    # training loop
    best_val = float('inf')
    hist = {'tr': [], 'val': [], 'lr': []}
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_rmse = train_epoch(model, train_dl, optimizer, device, args.N)
        vl_loss, vl_rmse, vl_sr = eval_epoch(model, val_dl, device, args.N)
        scheduler.step()

        hist['tr'].append(tr_rmse);  hist['val'].append(vl_rmse)
        hist['lr'].append(scheduler.get_last_lr()[0])

        tag = ''
        if vl_rmse < best_val:
            best_val = vl_rmse
            torch.save({'model': model.state_dict(), 'epoch': ep, 'val_rmse': vl_rmse}, ckpt_path)
            tag = ' *'

        if ep % 5 == 0 or ep == 1:
            print(f'Epoch {ep:3d}/{args.epochs}  '
                  f'lr={hist["lr"][-1]:.1e}  '
                  f'tr={tr_rmse:.3f}°  val={vl_rmse:.3f}°  SR@5={vl_sr:.1f}%'
                  f'{tag}')

    # test evaluation
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    preds, trues, times = [], [], []
    model.eval()
    with torch.no_grad():
        for X, Y in test_dl:
            X = X.to(device)
            t1 = time.perf_counter()
            Yh = model(X)
            times.append((time.perf_counter() - t1) / X.shape[0] * 1000)
            preds.append(Yh.cpu());  trues.append(Y)

    Yp = torch.cat(preds);  Yt = torch.cat(trues)
    test_rmse = rmse_deg(Yp, Yt, args.N)
    test_sr   = success_rate(Yp, Yt, args.N)

    print(f'\n=== Test results ===')
    print(f'RMSE         : {test_rmse:.4f}°  (paper: 2.77° for N=32)')
    print(f'SR@5°        : {test_sr:.2f}%')
    print(f'Avg inference: {np.mean(times):.3f} ms/sample  (paper: 0.08 ms, CPU)')
    print(f'Training time: {(time.time()-t0)/60:.1f} min')

    # learning curve
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hist['tr'], label='Train');  a1.plot(hist['val'], label='Val')
    a1.set_xlabel('Epoch');  a1.set_ylabel('RMSE [°]');  a1.set_title('Training curves')
    a1.axhline(2.77, color='r', linestyle='--', label='Paper (2.77°)')
    a1.legend();  a1.grid(True)
    a2.plot(hist['lr']);  a2.set_xlabel('Epoch');  a2.set_ylabel('Learning rate')
    a2.set_title('LR schedule');  a2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f'curves_N{args.N}.png'), dpi=150)
    print(f'\nPlot saved to {args.out_dir}/curves_N{args.N}.png')


if __name__ == '__main__':
    main()
