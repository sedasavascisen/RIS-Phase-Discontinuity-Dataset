 **Overcoming the Phase Discontinuity Problem in RIS Optimization: A Robust and Scalable Phase-Aware Deep Regression Framework via Hybrid CNN-LSTM Architecture**  
> Seda Savaşçı Şen, Ali Çalhan, Murtaza Cicioğlu, Oğuzhan Demiryürek  
> *Advanced Intelligent Systems*, 2026

---

## Overview

Reconfigurable Intelligent Surface (RIS) phase optimization requires predicting a phase angle per element. Training a neural network to regress angles directly hits the ±π discontinuity: two almost-equal phases near +π and −π produce a large loss, confusing the optimizer.

We fix this by predicting `[sin(θ), cos(θ)]` instead of `θ` directly and recovering the angle via `atan2`. This is smooth everywhere on the unit circle, and the gradient is bounded by 4 regardless of the phase value (Proposition 1 in the paper).

The network itself is a hybrid CNN-LSTM: 1D convolutions extract local element correlations, and a stacked LSTM captures longer-range inter-element dependencies that arise from the spatial layout of the panel.

**Main result:** Phase-aware RMSE of **2.77°** on a 32-element RIS, at **0.08 ms** inference per sample on a standard CPU — well within the 6G URLLC latency budget of 1 ms.

---

## Repository structure

```
├── RIS_dataset_large_2D.m          # dataset generation (run first)
├── theta_NN_multioutput.m          # CNN-LSTM training and evaluation (MATLAB)
├── train_cnn_lstm.py               # same model in PyTorch
│
├── ris_CD.m                        # Coordinate Descent baseline
├── ris_AO.m                        # Alternating Optimization baseline
├── ris_WMMSE.m                     # Weighted MMSE baseline
├── ris_SPA.m                       # Successive Phase Alignment baseline
├── ris_PSO.m                       # Particle Swarm Optimization baseline
├── ris_PGD.m                       # Projected Gradient Descent baseline
│
├── transformer_baseline.m          # Transformer and MLP comparison (Table 4)
├── theoretical_gradient_analysis.m # Proposition 1 verification
├── scalability_analysis.m          # latency projection for N = {32, 64, 128}
├── baseline_fairness_K_analysis.m  # CD with K = {5, 20, 50} iterations
│
├── Figure5_comparison.m            # bar chart (all methods, Table 4)
├── Figure9_fairness.m              # CD RMSE and latency vs K
├── Figure10_scalability.m          # latency vs N (log scale)
└── plot_results.m                  # visualize test results after training
```

---

## Requirements

### MATLAB
- MATLAB R2022b or later
- Deep Learning Toolbox
- GPU recommended but not required (set `ExecutionEnvironment='cpu'` in training options)

### Python (optional)
```
pip install torch scipy numpy matplotlib
```
Python ≥ 3.9, PyTorch ≥ 2.0.

---

## Quickstart

### Step 1 — Generate the dataset

```matlab
run('RIS_dataset_large_2D.m')
```

This creates `RIS_dataset_large_2D.mat` (~2.5 GB) with 40 000 samples, each containing the BS→RIS channel `F`, RIS→UE channel `G`, direct channel `h_d`, and the optimal phase labels `THETA`. Takes roughly 10 minutes on a standard laptop.

### Step 2 — Train the CNN-LSTM

**MATLAB:**
```matlab
run('theta_NN_multioutput.m')
```

**Python:**
```bash
python train_cnn_lstm.py --data_path RIS_dataset_N32.mat --N 32
```

Expected output (N=32, GPU):
```
Phase-aware RMSE : 2.77°
Avg inference    : 0.08 ms/sample
SR@5 deg         : 95.2%
```

### Step 3 — Run baselines

```matlab
[Theta, h_eff, rmse, t] = ris_CD(F, G, h_d, 5);      % Coordinate Descent
[Theta, h_eff, MSE, RMSE] = ris_AO(F, G, h_d, 5);    % Alternating Optimization
[Theta, h_eff, MSE, RMSE] = ris_PGD(F, G, h_d);       % Projected Gradient Descent
```

### Step 4 — Reproduce figures

```matlab
run('Figure5_comparison.m')       % Table 4 bar chart
run('Figure9_fairness.m')         % baseline fairness (K iterations)
run('Figure10_scalability.m')     % latency vs N
run('theoretical_gradient_analysis.m')  % Proposition 1
```

---

## Dataset

The dataset simulates a 3 GHz indoor environment (40 m × 20 m × 6 m room) with:
- 8 × 4 RIS panel (N = 32 elements, λ/2 spacing)
- 6 random scatterers per sample
- Rayleigh fading on all links
- TX and RX positions randomized uniformly over the room

Optimal phase labels use the closed-form solution θ\* = −∠G − ∠F (Eq. 8 in the paper).

For scalability experiments (N = 64, 128), change `M` and `K` in `RIS_dataset_large_2D.m` before running.

---

## Results summary (Table 6 in the paper)

| Method | RMSE [°] | SR@5° | Latency [ms] | Complexity | URLLC |
|---|---|---|---|---|---|
| **CNN-LSTM (proposed)** | **2.77 ± 1.15** | **95%** | **0.08** | O(N) | ✓ |
| CD | 3.50 ± 1.42 | 78% | 0.15 | O(KN²) | ✓ |
| AO | 3.50 ± 1.38 | 78% | 0.18 | O(KN²) | ✓ |
| PGD | 4.00 ± 1.68 | 71% | 0.50 | O(IN) | ✓ |
| WMMSE | 4.00 ± 1.65 | 71% | 0.20 | O(KN²) | ✓ |
| PSO | 4.50 ± 1.88 | 64% | 15.0 | O(PIN) | ✗ |
| SPA | 5.00 ± 2.15 | 57% | 0.05 | O(N) | ✓ |
| FC-DNN (direct θ) | 3.95 ± 1.71 | 82% | 0.04 | O(N) | ✓ |
| FC-DNN (sin/cos) | 3.41 ± 1.50 | 89% | 0.04 | O(N) | ✓ |
| CNN-Transformer | 2.91 ± 1.22 | 93% | 0.11†  | O(N²) | ✓† |
| GCN (spatial) | 3.12 ± 1.33 | 91% | 0.09 | O(N) | ✓ |

CNN-LSTM achieves the lowest RMSE while staying under the 1 ms URLLC constraint.

> **† Note on CNN-Transformer latency:** Table 6 in the published paper
> lists 0.11 ms for this value, while the Figure 1 caption and Table 8
> note in the same paper report 1.1 ms for N=32 — a 10× discrepancy,
> likely a decimal-point transcription error. This repository's own
> scalability script previously used ~1.14 ms, consistent with the
> Figure 1/Table 8 values. We are verifying against raw experiment logs
> and will update this note (and, if warranted, submit a corrigendum)
> accordingly. This does not affect the paper's central conclusions.
>
> 
## Citation

If you use this code, please cite:

```bibtex
@article{savascisen2026ris,
  title   = {Overcoming the Phase Discontinuity Problem in RIS Optimization:
             A Robust and Scalable Phase-Aware Deep Regression Framework
             via Hybrid CNN-LSTM Architecture},
  author  = {Sava{\c{s}}{\c{c}}{\i} {\c{S}}en, Seda and {\c{C}}alhan, Ali and
             Cicioğlu, Murtaza and Demiryürek, Oğuzhan},
  journal = {Advanced Intelligent Systems},
  year    = {2026}
}
```

---

## Contact

Seda Savaşçı Şen — questions or issues, please open a GitHub issue.

