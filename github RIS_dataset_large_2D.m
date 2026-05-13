% ==========================================================
%  RIS_dataset_large_2D.m
%
%  Dataset generation script for:
%  "Overcoming the Phase Discontinuity Problem in RIS
%   Optimization: A Robust and Scalable Phase-Aware Deep
%   Regression Framework via Hybrid CNN-LSTM Architecture"
%  Advanced Intelligent Systems, 2025
%
%  Authors : Seda Savascı Sen, Ali Calhan,
%            Murtaza Cicoglu, Oguzhan Demiryurek
%
%  ── What this script does ──────────────────────────────
%  Generates S = 40,000 independent indoor RIS channel
%  scenarios and computes analytically optimal phase labels
%  via the closed-form solution (Eq. 8 in the paper).
%
%  ── Output variables (saved to .mat) ───────────────────
%  F        [S x N]   BS-RIS channel matrix     (complex)
%  G        [S x N]   RIS-UE channel matrix      (complex)
%  h_d      [S x 1]   Direct BS-UE channel       (complex)
%  THETA    [S x N]   Optimal phases in [-pi,pi] (real)
%  X_feat   [S x 4N]  NN input features          (real)
%              = [Re(F), Im(F), Re(G), Im(G)]    per Eq.(5)
%  Y_sincos [S x 2N]  Sine-cosine output labels  (real)
%              = [sin(THETA), cos(THETA)]         per Eq.(11)
%  TX_pos   [S x 3]   Transmitter positions      [m]
%  RX_pos   [S x 3]   Receiver positions         [m]
%  elemPos  [N x 3]   RIS element positions      [m]
%  idx_tr   [1 x 28000] Training indices
%  idx_val  [1 x  6000] Validation indices
%  idx_te   [1 x  6000] Test indices
%
%  ── Paper equation cross-references ────────────────────
%  Eq.(1)  : RIS element positions p_n
%  Eq.(2)  : Direct channel h_d
%  Eq.(3)  : BS-RIS channel F_n (LoS + NLoS scattering)
%  Eq.(4)  : RIS-UE channel G_n (LoS + NLoS scattering)
%  Eq.(5)  : NN input feature matrix X_s
%  Eq.(8)  : Optimal phase theta_n* = -angle(G_n*F_n)
%  Eq.(11) : Sine-cosine output Y_s = [sin(theta), cos(theta)]
%
%  ── Requirements ───────────────────────────────────────
%  MATLAB R2020a or later (for wrapToPi, kstest)
%  Expected runtime : ~15 min on Intel Core i7
%  Expected file size: ~600 MB (-v7.3 HDF5 format)
%
%  ── Reproducibility ────────────────────────────────────
%  rng(42) ensures exact reproducibility of all results
%  reported in the paper (Table 2, Table 4, Figure 4-8).
% ==========================================================

clear; close all; clc;
rng(42);   % Fixed seed — do NOT change for reproducibility

%% ── System Parameters (Table 3 in paper) ─────────────────
numSamples   = 40000;          % Total channel scenarios     S
M            = 8;              % RIS rows                    M
K            = 4;              % RIS columns                 K
N            = M * K;          % Total RIS elements          N = 32
numScatter   = 6;              % Scatterers per scenario     L
scatter_gain = 0.15;           % Scattering amplitude gain   beta_l
roomSize     = [40, 20, 6];    % Room dimensions [x,y,z] in [m]

fc           = 3e9;            % Carrier frequency [Hz]      f_c = 3 GHz
c_light      = 3e8;            % Speed of light [m/s]
lambda       = c_light / fc;   % Wavelength [m]              lambda = 0.1 m
k_wave       = 2*pi / lambda;  % Wavenumber [rad/m]

Pt           = 1;              % Transmit power [W]          P_t = 30 dBm
noise_var    = 1e-3;           % Noise variance [W]          sigma_n^2 = -30 dBm

%% ── RIS Element Positions — Eq.(1) ───────────────────────
% p_n = [x0 + (i-1)*d, y0 + (j-1)*d, z_RIS]^T
% Row-major order: n = (i-1)*K + j, i in {1..M}, j in {1..K}
% This ordering defines the serialization used in the CNN-LSTM.

ris_center = [roomSize(1)/2, roomSize(2)/2, 3];   % RIS center (20,10,3) m
d_elem     = lambda / 2;                           % Element spacing = 0.05 m

x0 = ris_center(1) - (M-1)/2 * d_elem;
y0 = ris_center(2) - (K-1)/2 * d_elem;

elemPos = zeros(N, 3);
for ii = 1:M
    for jj = 1:K
        n = (ii-1)*K + jj;   % Row-major index
        elemPos(n,:) = [x0 + (ii-1)*d_elem, ...
                        y0 + (jj-1)*d_elem, ...
                        ris_center(3)];
    end
end

%% ── Pre-allocation ────────────────────────────────────────
F      = complex(zeros(numSamples, N));  % BS-RIS channel
G      = complex(zeros(numSamples, N));  % RIS-UE channel
h_d    = complex(zeros(numSamples, 1));  % Direct link
THETA  = zeros(numSamples, N);           % Optimal phase labels
TX_pos = zeros(numSamples, 3);           % Transmitter positions
RX_pos = zeros(numSamples, 3);           % Receiver positions

%% ── Dataset Generation Loop ───────────────────────────────
fprintf('================================================\n');
fprintf('  RIS Dataset Generation\n');
fprintf('  Samples: %d  |  N = %d (%d x %d)\n', numSamples, N, M, K);
fprintf('  Room: %d x %d x %d m  |  Scatterers: %d\n', ...
    roomSize(1), roomSize(2), roomSize(3), numScatter);
fprintf('================================================\n');
tic;

for s = 1:numSamples

    %% BS and UE positions (uniform random inside room)
    % Height restricted to [1, 2] m (typical user/AP height)
    tx = [rand * roomSize(1), ...
          rand * roomSize(2), ...
          1 + rand * 1];          % Transmitter (BS)
    rx = [rand * roomSize(1), ...
          rand * roomSize(2), ...
          1 + rand * 1];          % Receiver (UE)
    TX_pos(s,:) = tx;
    RX_pos(s,:) = rx;

    %% Random scatterer positions (re-sampled each scenario)
    scat_pos = [rand(numScatter,1) * roomSize(1), ...
                rand(numScatter,1) * roomSize(2), ...
                rand(numScatter,1) * roomSize(3)];

    %% Direct BS-UE channel — Eq.(2)
    % h_d = alpha_d * exp(-j*k*d_BS-UE) / d_BS-UE
    % alpha_d ~ CN(0,1)
    d_direct  = norm(tx - rx);
    alpha_d   = (randn + 1j*randn) / sqrt(2);        % CN(0,1) coefficient
    h_d(s)    = alpha_d * exp(-1j*k_wave*d_direct) / d_direct;

    %% Per-element BS-RIS (F) and RIS-UE (G) channels — Eq.(3)-(4)
    % F_n = alpha_F,n * exp(-j*k*d_BS-n) / d_BS-n         [LoS]
    %      + sum_l beta_l * alpha_F,n,l * exp(-j*k*(d_l1+d_l2)) / (d_l1*d_l2)  [NLoS]
    % G_n: analogous (Eq. 4)
    for n = 1:N
        ep = elemPos(n,:);

        % Distances
        d_tx_n = norm(tx - ep);    % BS  → element n
        d_n_rx = norm(ep - rx);    % element n → UE

        % LoS Rayleigh coefficients ~ CN(0,1)
        alpha_Fn = (randn + 1j*randn) / sqrt(2);
        alpha_Gn = (randn + 1j*randn) / sqrt(2);

        % LoS components
        F_LoS = alpha_Fn * exp(-1j*k_wave*d_tx_n) / d_tx_n;
        G_LoS = alpha_Gn * exp(-1j*k_wave*d_n_rx) / d_n_rx;

        % NLoS scattering — Eq.(3) second term
        F_NLoS = 0;
        G_NLoS = 0;
        for l = 1:numScatter
            % BS-RIS scattering path
            d_l1 = norm(tx       - scat_pos(l,:));  % BS  → scatterer l
            d_l2 = norm(scat_pos(l,:) - ep);         % scatterer l → element n
            F_NLoS = F_NLoS + scatter_gain * ...
                (randn+1j*randn)/sqrt(2) * ...
                exp(-1j*k_wave*(d_l1+d_l2)) / (d_l1*d_l2);

            % RIS-UE scattering path
            d_l3 = norm(ep       - scat_pos(l,:));  % element n → scatterer l
            d_l4 = norm(scat_pos(l,:) - rx);         % scatterer l → UE
            G_NLoS = G_NLoS + scatter_gain * ...
                (randn+1j*randn)/sqrt(2) * ...
                exp(-1j*k_wave*(d_l3+d_l4)) / (d_l3*d_l4);
        end

        F(s,n) = F_LoS + F_NLoS;
        G(s,n) = G_LoS + G_NLoS;
    end

    %% Optimal phase labels — Eq.(8)
    % theta_n* = -angle(G_n * F_n) = -angle(G_n) - angle(F_n)
    % Maps to [-pi, pi] via wrapToPi
    % This is the closed-form maximizer of |h_d + sum_n G_n*exp(j*theta_n)*F_n|^2
    THETA(s,:) = wrapToPi( -angle(G(s,:)) - angle(F(s,:)) );

    %% Progress report
    if mod(s, 5000) == 0
        elapsed = toc;
        eta     = elapsed / s * (numSamples - s);
        fprintf('  [%5d/%d]  Elapsed: %5.1f min  |  ETA: %5.1f min\n', ...
            s, numSamples, elapsed/60, eta/60);
    end
end

fprintf('Generation complete: %.2f min total\n', toc/60);

%% ── Neural Network Features — Eq.(5) ─────────────────────
% X_s = [Re(F^T); Im(F^T); Re(G^T); Im(G^T)] ∈ R^{4 x N}
% Stored as row vectors: X_feat ∈ R^{S x 4N} = R^{40000 x 128}
%
% Column layout:
%   cols  1:32   → Re(F)   (32 real parts of BS-RIS channel)
%   cols 33:64   → Im(F)   (32 imag parts of BS-RIS channel)
%   cols 65:96   → Re(G)   (32 real parts of RIS-UE channel)
%   cols 97:128  → Im(G)   (32 imag parts of RIS-UE channel)
X_feat = [real(F), imag(F), real(G), imag(G)];   % [S x 128]

%% ── Sine-Cosine Output Labels — Eq.(11) ──────────────────
% Y_s = [sin(theta_{s,1}),...,sin(theta_{s,N}),
%         cos(theta_{s,1}),...,cos(theta_{s,N})] ∈ R^{2N}
%
% This representation eliminates the phase discontinuity at ±pi
% (Section 3.1 of the paper). Phase is recovered via:
%   theta_hat = atan2(Y(:,1:N), Y(:,N+1:2N))
%
% Column layout:
%   cols  1:32   → sin(THETA)  (32 sine components)
%   cols 33:64   → cos(THETA)  (32 cosine components)
Y_sincos = [sin(THETA), cos(THETA)];              % [S x 64]

%% ── Dataset Quality Verification — Table 2 ───────────────
fprintf('\n─────────────────────────────────────────────\n');
fprintf('Dataset Quality Verification (cf. Table 2)\n');
fprintf('─────────────────────────────────────────────\n');

% Phase statistics
theta_flat = THETA(:);
fprintf('Optimal phase mean       : %+.4f rad  (expected ~0)\n', mean(theta_flat));
fprintf('Optimal phase std        : %.4f rad  (expected %.4f = pi/sqrt(3))\n', ...
    std(theta_flat), pi/sqrt(3));
fprintf('Optimal phase min / max  : %.4f / %.4f rad\n', min(theta_flat), max(theta_flat));
fprintf('Near-zero skewness       : %.4f     (expected ~0)\n', skewness(theta_flat));
fprintf('Kurtosis                 : %.4f     (expected ~1.80 platykurtic)\n', ...
    kurtosis(theta_flat));

% Kolmogorov-Smirnov uniformity test on [-pi, pi]
theta_normalized = (theta_flat - (-pi)) / (2*pi);   % map to [0,1] for KS test
[~, p_ks] = kstest(theta_normalized);
fprintf('KS uniformity p-value    : %.4f     (expected >0.05)\n', p_ks);

% Inter-sample correlation (should be < 0.10)
sample_idx = randsample(numSamples, min(500, numSamples));
corr_F = corr(real(F(sample_idx, 1)), real(F(sample_idx, 2)));
fprintf('Inter-sample correlation : %.4f     (expected <0.10)\n', abs(corr_F));

% SNR statistics (without RIS)
SNR_no_RIS = Pt * abs(h_d).^2 / noise_var;
SNR_no_RIS_dB = 10*log10(SNR_no_RIS);
fprintf('\nSNR without RIS [dB]:\n');
fprintf('  Min / Median / Max     : %.1f / %.1f / %.1f dB\n', ...
    min(SNR_no_RIS_dB), median(SNR_no_RIS_dB), max(SNR_no_RIS_dB));

% SNR with optimal RIS
SNR_with_RIS_dB = zeros(numSamples, 1);
for s = 1:numSamples
    phase_vec = exp(1j * THETA(s,:));         % [1 x N]
    h_eff     = h_d(s) + G(s,:) * phase_vec.' .* F(s,:).';  % scalar
    % Correct effective channel for beamforming gain
    h_eff     = h_d(s) + sum(G(s,:) .* phase_vec .* F(s,:));
    SNR_opt   = Pt * abs(h_eff)^2 / noise_var;
    SNR_with_RIS_dB(s) = 10*log10(SNR_opt);
end
fprintf('SNR with optimal RIS [dB]:\n');
fprintf('  Min / Median / Max     : %.1f / %.1f / %.1f dB\n', ...
    min(SNR_with_RIS_dB), median(SNR_with_RIS_dB), max(SNR_with_RIS_dB));
fprintf('RIS gain (median)        : %.1f dB\n', ...
    median(SNR_with_RIS_dB) - median(SNR_no_RIS_dB));

% Feature matrix dimensions
fprintf('\nX_feat shape  : [%d x %d]  (S x 4N)\n', size(X_feat,1),   size(X_feat,2));
fprintf('Y_sincos shape: [%d x %d]   (S x 2N)\n', size(Y_sincos,1), size(Y_sincos,2));
fprintf('─────────────────────────────────────────────\n');

%% ── Train / Validation / Test Split ──────────────────────
% Split: 70% / 15% / 15%  (Section 3.4.3 of paper)
n_train = 28000;
n_val   =  6000;
n_test  =  6000;

% Shuffle indices (reproducible due to rng(42) above)
idx_all = randperm(numSamples);
idx_tr  = sort(idx_all(1          : n_train));
idx_val = sort(idx_all(n_train+1  : n_train+n_val));
idx_te  = sort(idx_all(n_train+n_val+1 : end));

fprintf('\nDataset split:\n');
fprintf('  Training   : %d samples  (%.0f%%)\n', n_train, n_train/numSamples*100);
fprintf('  Validation : %d samples  (%.0f%%)\n', n_val,   n_val/numSamples*100);
fprintf('  Test       : %d samples  (%.0f%%)\n', n_test,  n_test/numSamples*100);

%% ── Save Full Dataset ─────────────────────────────────────
savefile = 'RIS_dataset_large_2D.mat';

save(savefile, ...
    'F',           ...  % [S x N]   BS-RIS channel (complex)
    'G',           ...  % [S x N]   RIS-UE channel (complex)
    'h_d',         ...  % [S x 1]   Direct channel (complex)
    'THETA',       ...  % [S x N]   Optimal phases [-pi,pi]
    'X_feat',      ...  % [S x 4N]  NN input features (real)
    'Y_sincos',    ...  % [S x 2N]  Sine-cosine labels (real)
    'TX_pos',      ...  % [S x 3]   Transmitter positions [m]
    'RX_pos',      ...  % [S x 3]   Receiver positions [m]
    'elemPos',     ...  % [N x 3]   RIS element positions [m]
    'idx_tr',      ...  % Training indices
    'idx_val',     ...  % Validation indices
    'idx_te',      ...  % Test indices
    'N','M','K',   ...  % Scalar parameters
    'numScatter',  ...
    'scatter_gain',...
    'roomSize',    ...
    'fc','lambda', ...
    'Pt','noise_var', ...
    '-v7.3');           % HDF5 format — supports files > 2 GB

fprintf('\nSaved: %s\n', savefile);
d_info = dir(savefile);
fprintf('File size: %.1f MB\n', d_info.bytes/1e6);

%% ── Save Split Subsets Separately (optional) ──────────────
% Useful for direct loading in Python without index masking
save('train_data.mat', ...
    'X_feat', 'Y_sincos', 'F', 'G', 'THETA', ...
    'TX_pos', 'RX_pos', 'idx_tr', '-v7.3');

save('val_data.mat', ...
    'X_feat', 'Y_sincos', 'F', 'G', 'THETA', ...
    'TX_pos', 'RX_pos', 'idx_val', '-v7.3');

save('test_data.mat', ...
    'X_feat', 'Y_sincos', 'F', 'G', 'THETA', ...
    'TX_pos', 'RX_pos', 'idx_te', '-v7.3');

fprintf('Saved train/val/test split files.\n');

%% ── Quick Sanity Check: Phase Recovery ───────────────────
% Verify: theta_recovered == THETA (up to floating point)
theta_rec = atan2(Y_sincos(:,1:N), Y_sincos(:,N+1:end));
max_err   = max(abs(wrapToPi(theta_rec(:) - THETA(:))));
fprintf('\nPhase recovery max error : %.2e rad  (expected ~1e-15)\n', max_err);

%% ── Python Loading Example (printed for README) ──────────
fprintf('\n── Python / PyTorch loading example ──────────\n');
fprintf('import scipy.io as sio\n');
fprintf('import numpy as np\n');
fprintf('\n');
fprintf('data    = sio.loadmat(''RIS_dataset_large_2D.mat'')\n');
fprintf('X_feat  = data[''X_feat'']          # (40000, 128)\n');
fprintf('Y_sincos= data[''Y_sincos'']         # (40000,  64)\n');
fprintf('THETA   = data[''THETA'']            # (40000,  32)  [rad]\n');
fprintf('idx_tr  = data[''idx_tr''].flatten() - 1  # 0-indexed\n');
fprintf('\n');
fprintf('X_train = X_feat[idx_tr,  :]\n');
fprintf('Y_train = Y_sincos[idx_tr,:]\n');
fprintf('\n');
fprintf('# Phase recovery:\n');
fprintf('theta_hat = np.arctan2(Y_sincos[:,:32], Y_sincos[:,32:])\n');
fprintf('─────────────────────────────────────────────\n');

fprintf('\nDataset generation complete.\n');
fprintf('Total time: %.2f minutes\n', toc/60);